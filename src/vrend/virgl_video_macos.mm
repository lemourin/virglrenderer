#include "virgl_video.h"

#include <CoreFoundation/CFBase.h>
#include <CoreMedia/CMBlockBuffer.h>
#include <CoreMedia/CMFormatDescription.h>
#include <CoreMedia/CMSampleBuffer.h>
#include <CoreMedia/CMTime.h>
#include <CoreVideo/CVBase.h>
#include <CoreVideo/CVPixelBuffer.h>
#include <CoreVideo/CVPixelBufferPool.h>
#include <CoreVideo/CoreVideo.h>
#include <MacTypes.h>
#import <Metal/Metal.h>
#include <VideoToolbox/VTCompressionSession.h>
#include <VideoToolbox/VTDecompressionSession.h>
#include <VideoToolbox/VTSession.h>
#include <sys/_types.h>
#include <mutex>
#include <ostream>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <span>
#include <sstream>

#include "pipe/p_video_enums.h"
#include "pipe/p_video_state.h"
#include "u_formats.h"
#include "virgl_video_hw.h"

namespace {

template <typename T>
auto WrapCf(T *obj) {
  auto deleter = [](T *obj) { CFRelease(obj); };
  return std::unique_ptr<T, decltype(deleter)>(obj);
}

auto WrapCf(CVPixelBufferRef buffer) {
  auto deleter = [](CVPixelBufferRef buffer) { CVPixelBufferRelease(buffer); };
  return std::unique_ptr<std::remove_pointer_t<CVPixelBufferRef>, decltype(deleter)>(buffer);
}

using CMFormatDescriptionPtr = decltype(WrapCf(CMFormatDescriptionRef()));
using CFDictionaryPtr = decltype(WrapCf(CFDictionaryRef()));
using VTDecompressionSessionPtr = decltype(WrapCf(VTDecompressionSessionRef()));
using VTCompressionSessionPtr = decltype(WrapCf(VTCompressionSessionRef()));
using CMSampleBufferPtr = decltype(WrapCf(CMSampleBufferRef()));
using CVMetalTexturePtr = decltype(WrapCf(CVMetalTextureRef()));
using CVImageBufferPtr = decltype(WrapCf(CVImageBufferRef()));
using CVPixelBufferPtr = decltype(WrapCf(CVPixelBufferRef()));

struct H264ParameterSet {
  std::vector<uint8_t> sps;
  std::vector<uint8_t> pps;

  auto operator<=>(const H264ParameterSet &) const = default;
};

struct FramePacket {
  uint32_t index;
  CMSampleBufferPtr buffer;

  auto operator<=>(const FramePacket &) const = default;
};

}  // namespace

struct virgl_video_codec {
  int width;
  int height;
  int level;
  void *opaque;
  pipe_video_profile profile;
  pipe_video_entrypoint entrypoint;
  pipe_video_chroma_format chroma_format;
  CMFormatDescriptionPtr format_description;
  VTDecompressionSessionPtr decompression_session;
  VTCompressionSessionPtr compression_session;
  std::optional<H264ParameterSet> h264_parameter_set;
  int nalu_header_length;

  std::mutex mutex;
  std::vector<CMSampleBufferPtr> frame_queue;
  std::vector<uint8_t> encode_parameter_set;
};

struct virgl_video_buffer {
  void *opaque;
  uint32_t id;
  CVImageBufferPtr image;
  CMTime pts;
  CMTime duration;
  uint32_t frame_index;
};

namespace {

std::ofstream *gLogFile;
std::ofstream *gVideoFile;
virgl_video_callbacks *gCallbacks;
id<MTLDevice> gMetalDevice;
CVMetalTextureCacheRef gTextureCache;
uint32_t gNextBufferId = 1;

struct BufferView {
  std::vector<uint8_t *> data;
  std::vector<unsigned> size;
};

struct BitAddress {
  uint8_t *id;
  uint8_t bit;
};

CMVideoCodecType GetCodecType(pipe_video_profile profile) {
  switch (profile) {
    case PIPE_VIDEO_PROFILE_MPEG4_AVC_BASELINE:
    case PIPE_VIDEO_PROFILE_MPEG4_AVC_CONSTRAINED_BASELINE:
    case PIPE_VIDEO_PROFILE_MPEG4_AVC_MAIN:
    case PIPE_VIDEO_PROFILE_MPEG4_AVC_EXTENDED:
    case PIPE_VIDEO_PROFILE_MPEG4_AVC_HIGH:
    case PIPE_VIDEO_PROFILE_MPEG4_AVC_HIGH10:
    case PIPE_VIDEO_PROFILE_MPEG4_AVC_HIGH422:
    case PIPE_VIDEO_PROFILE_MPEG4_AVC_HIGH444:
      return kCMVideoCodecType_H264;
    case PIPE_VIDEO_PROFILE_HEVC_MAIN:
    case PIPE_VIDEO_PROFILE_HEVC_MAIN_10:
    case PIPE_VIDEO_PROFILE_HEVC_MAIN_STILL:
    case PIPE_VIDEO_PROFILE_HEVC_MAIN_12:
    case PIPE_VIDEO_PROFILE_HEVC_MAIN_444:
      return kCMVideoCodecType_HEVC;
    default:
      return 0;
  }
}

template <typename... Ts>
void Log(const Ts &...args) {
  static std::mutex mutex;
  std::unique_lock lock{mutex};
  (*gLogFile << ... << args) << std::endl;
}

std::string ToString(std::span<const uint8_t> data) {
  std::stringstream stream;
  for (size_t i = 0; i < data.size(); i++) {
    stream << std::hex << std::setw(2) << std::setfill('0') << uint32_t(data[i]) << ' ';
  }
  return std::move(stream).str();
}

void WriteByte(uint8_t byte, int bit_count, BitAddress &address) {
  if (bit_count == 0) {
    return;
  }
  *address.id |= ((byte << (8 - bit_count)) >> address.bit);
  address.bit += bit_count;
  if (address.bit >= 8) {
    address.bit -= 8;
    address.id++;
    *address.id = (byte << (8 - address.bit));
  }
}

int GetBitCount(uint8_t d) {
  int idx = 0;
  while (d > 0) {
    d /= 2;
    idx++;
  }
  return idx;
}

void WriteGolomb(uint8_t source, BitAddress &address) {
  int bit_count = GetBitCount(source + 1);
  WriteByte(0, bit_count - 1, address);
  WriteByte(source + 1, bit_count, address);
}

void WriteSignedGolomb(int32_t source, BitAddress &address) {
  if (source <= 0) {
    source *= -2;
  } else {
    source *= 2;
    source--;
  }
  WriteGolomb(source, address);
}

CFDictionaryPtr CreateDecoderSpec(CMVideoCodecType codec_type,
                                  std::span<const uint8_t> extra_data) {
  auto config_info = WrapCf(CFDictionaryCreateMutable(
      kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks));

  CFDictionarySetValue(config_info.get(),
                       codec_type == kCMVideoCodecType_HEVC
                           ? kVTVideoDecoderSpecification_EnableHardwareAcceleratedVideoDecoder
                           : kVTVideoDecoderSpecification_RequireHardwareAcceleratedVideoDecoder,
                       kCFBooleanTrue);

  auto avc_info = WrapCf(CFDictionaryCreateMutable(
      kCFAllocatorDefault, 1, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks));

  auto data = WrapCf(CFDataCreate(kCFAllocatorDefault, extra_data.data(),
                                  static_cast<CFIndex>(extra_data.size())));
  switch (codec_type) {
    case kCMVideoCodecType_MPEG4Video:
      CFDictionarySetValue(avc_info.get(), CFSTR("esds"), data.get());
      break;
    case kCMVideoCodecType_H264:
      CFDictionarySetValue(avc_info.get(), CFSTR("avcC"), data.get());
      break;
    case kCMVideoCodecType_HEVC:
      CFDictionarySetValue(avc_info.get(), CFSTR("hvcC"), data.get());
      break;
    case kCMVideoCodecType_VP9:
      CFDictionarySetValue(avc_info.get(), CFSTR("vpcC"), data.get());
    case kCMVideoCodecType_AV1:
      CFDictionarySetValue(avc_info.get(), CFSTR("av1C"), data.get());
    default:
      break;
  }
  CFDictionarySetValue(config_info.get(),
                       kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms,
                       avc_info.get());
  return WrapCf(static_cast<CFDictionaryRef>(config_info.release()));
}

auto CreateBufferAttributes(int width, int height, OSType pix_fmt) {
  auto buffer_attributes = WrapCf(CFDictionaryCreateMutable(
      kCFAllocatorDefault, 4, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks));
  auto io_surface_properties = WrapCf(CFDictionaryCreateMutable(
      kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks));
  auto cv_pix_fmt = WrapCf(CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &pix_fmt));
  auto w = WrapCf(CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &width));
  auto h = WrapCf(CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &height));

  if (pix_fmt) {
    CFDictionarySetValue(buffer_attributes.get(), kCVPixelBufferPixelFormatTypeKey,
                         cv_pix_fmt.get());
  }
  CFDictionarySetValue(buffer_attributes.get(), kCVPixelBufferIOSurfacePropertiesKey,
                       io_surface_properties.get());
  CFDictionarySetValue(buffer_attributes.get(), kCVPixelBufferWidthKey, w.get());
  CFDictionarySetValue(buffer_attributes.get(), kCVPixelBufferHeightKey, h.get());
  CFDictionarySetValue(buffer_attributes.get(), kCVPixelBufferMetalCompatibilityKey,
                       kCFBooleanTrue);

  return buffer_attributes;
}

CMSampleBufferPtr CreateSampleBuffer(CMFormatDescriptionRef fmt_desc, const void *buffer,
                                     size_t size) {
  CMBlockBufferRef block_buf = nullptr;
  if (OSStatus status = CMBlockBufferCreateWithMemoryBlock(
          /*structureAllocator=*/kCFAllocatorDefault,
          /*memoryBlock=*/const_cast<void *>(buffer),
          /*blockLength=*/size,
          /*blockAllocator=*/kCFAllocatorNull,
          /*customBlockSource=*/nullptr,
          /*offsetToData=*/0,
          /*dataLength=*/size,
          /*flags=*/0, &block_buf);
      status != 0) {
    return nullptr;
  }

  CMSampleBufferRef sample_buf = nullptr;
  if (OSStatus status = CMSampleBufferCreate(kCFAllocatorDefault,
                                             /*dataBuffer=*/block_buf,
                                             /*dataReady=*/true,
                                             /*makeDataReadyCallback=*/nullptr,
                                             /*makeDataReadyRefcon=*/nullptr,
                                             /*formatDescription=*/fmt_desc,
                                             /*numSamples=*/1,
                                             /*numSampleTimingEntries=*/0,
                                             /*sampleTimingArray=*/nullptr,
                                             /*numSampleSizeEntries=*/0,
                                             /*sampleSizeArray=*/nullptr, &sample_buf);
      status != 0) {
    CFRelease(block_buf);
    return nullptr;
  };

  CFRelease(block_buf);
  return WrapCf(sample_buf);
}

H264ParameterSet GetH264ParameterSet(const virgl_video_codec *codec,
                                     const virgl_h264_picture_desc *desc,
                                     uint8_t num_ref_idx_l0_default_active_minus1,
                                     uint8_t num_ref_idx_l1_default_active_minus1) {
  std::vector<uint8_t> sps_data(128);
  const auto &sps = desc->pps.sps;
  BitAddress address{sps_data.data(), 0};
  WriteByte(0x67, 8, address);          // SPS NALU
  WriteByte(0x64, 8, address);          // PIPE_VIDEO_PROFILE_MPEG4_AVC_HIGH
  WriteByte(0x00, 8, address);          // constraint_set_flag
  WriteByte(codec->level, 8, address);  // level_idc
  WriteGolomb(0x0, address);            // seq_parameter_set_id
  WriteGolomb(sps.chroma_format_idc, address);
  if (sps.chroma_format_idc == 3) {
    WriteByte(sps.separate_colour_plane_flag, 1, address);
  }
  WriteGolomb(sps.bit_depth_luma_minus8, address);
  WriteGolomb(sps.bit_depth_chroma_minus8, address);
  WriteByte(0, 1, address);  // qpprime_y_zero_transform_bypass_flag
  WriteByte(sps.seq_scaling_matrix_present_flag, 1, address);
  if (sps.seq_scaling_matrix_present_flag) {
    for (int i = 0; i < ((sps.chroma_format_idc != 3) ? 8 : 12); i++) {
      WriteByte(0, 1, address);  // seq_scaling_list_present_flag[i]
    }
  }
  WriteGolomb(sps.log2_max_frame_num_minus4, address);
  WriteGolomb(sps.pic_order_cnt_type, address);
  if (sps.pic_order_cnt_type == 0) {
    WriteGolomb(sps.log2_max_pic_order_cnt_lsb_minus4, address);
  } else if (sps.pic_order_cnt_type == 1) {
    WriteByte(sps.delta_pic_order_always_zero_flag, 1, address);
    WriteSignedGolomb(sps.offset_for_non_ref_pic, address);
    WriteSignedGolomb(sps.offset_for_top_to_bottom_field, address);
    WriteGolomb(sps.num_ref_frames_in_pic_order_cnt_cycle, address);
    for (int i = 0; i < sps.num_ref_frames_in_pic_order_cnt_cycle; i++) {
      WriteSignedGolomb(sps.offset_for_ref_frame[i], address);
    }
  }
  WriteGolomb(desc->num_ref_frames, address);
  WriteByte(0, 1, address);                     // gaps_in_frame_num_value_allowed_flag
  WriteGolomb(codec->width / 16 - 1, address);  // pic_width_in_mbs_minus1
  WriteGolomb(codec->height / (16 * (2 - sps.frame_mbs_only_flag)) - 1,
              address);  // pic_height_in_map_units_minus1
  WriteByte(sps.frame_mbs_only_flag, 1, address);
  if (!sps.frame_mbs_only_flag) {
    WriteByte(sps.mb_adaptive_frame_field_flag, 1, address);
  }
  WriteByte(sps.direct_8x8_inference_flag, 1, address);
  WriteByte(0, 1, address);  // frame_cropping_flag
  WriteByte(0, 1, address);  // vui_parameters_present_flag
  sps_data.resize(static_cast<size_t>(address.id - sps_data.data()) + 1);

  std::vector<uint8_t> pps_data(128);
  const auto &pps = desc->pps;
  BitAddress address_pps{pps_data.data(), 0};
  WriteByte(0x68, 8, address_pps);
  WriteGolomb(0, address_pps);  // pic_parameter_set_id
  WriteGolomb(0, address_pps);  // seq_parameter_set_id
  WriteByte(pps.entropy_coding_mode_flag, 1, address_pps);
  WriteByte(pps.bottom_field_pic_order_in_frame_present_flag, 1, address_pps);
  WriteGolomb(0 /*pps.num_slice_groups_minus1*/, address_pps);
  WriteGolomb(num_ref_idx_l0_default_active_minus1, address_pps);
  WriteGolomb(num_ref_idx_l1_default_active_minus1, address_pps);
  WriteByte(pps.weighted_pred_flag, 1, address_pps);
  WriteByte(pps.weighted_bipred_idc, 2, address_pps);
  WriteSignedGolomb(pps.pic_init_qp_minus26, address_pps);
  WriteSignedGolomb(pps.pic_init_qs_minus26, address_pps);
  WriteSignedGolomb(pps.chroma_qp_index_offset, address_pps);
  WriteByte(pps.deblocking_filter_control_present_flag, 1, address_pps);
  WriteByte(pps.constrained_intra_pred_flag, 1, address_pps);
  WriteByte(pps.redundant_pic_cnt_present_flag, 1, address_pps);
  WriteByte(pps.transform_8x8_mode_flag, 1, address_pps);
  WriteByte(0, 1, address_pps);  // pic_scaling_matrix_present_flag
  WriteSignedGolomb(pps.second_chroma_qp_index_offset, address_pps);
  pps_data.resize(static_cast<size_t>(address_pps.id - pps_data.data()) + 1);

  Log("SPS = ", ToString(sps_data));
  Log("PPS = ", ToString(pps_data));

  return {sps_data, pps_data};
}

H264ParameterSet GetH264ParameterSet(const virgl_video_codec *codec,
                                     const virgl_h264_picture_desc *desc) {
  return GetH264ParameterSet(codec, desc, desc->pps.num_ref_idx_l0_default_active_minus1,
                             desc->pps.num_ref_idx_l1_default_active_minus1);
}

CMFormatDescriptionPtr CreateFormatDescription(const H264ParameterSet &parameter_set) {
  const auto &[sps_data, pps_data] = parameter_set;

  CMFormatDescriptionRef format_description;
  const uint8_t *parameter_set_pointers[] = {sps_data.data(), pps_data.data()};
  const size_t parameter_set_sizes[] = {sps_data.size(), pps_data.size()};
  if (OSStatus status = CMVideoFormatDescriptionCreateFromH264ParameterSets(
          kCFAllocatorDefault, /*parameterSetCount=*/2, parameter_set_pointers, parameter_set_sizes,
          /*NALUnitHeaderLength=*/4, &format_description);
      status != 0) {
    Log("CMVideoFormatDescriptionCreateFromH264ParameterSets error: ", status);
    return nullptr;
  }
  return WrapCf(format_description);
}

[[maybe_unused]] CMFormatDescriptionPtr CreateFormatDescription2(
    int width, int height, const H264ParameterSet &parameter_set) {
  const auto &[sps_data, pps_data] = parameter_set;
  std::vector<uint8_t> data(sps_data.size() + pps_data.size() + 11);
  uint32_t p = 0;
  data[p++] = 0x01;
  data[p++] = sps_data[1];
  data[p++] = sps_data[2];
  data[p++] = sps_data[3];
  data[p++] = 0xff;
  data[p++] = 0xe1;
  data[p++] = sps_data.size() >> 8;
  data[p++] = sps_data.size();
  memcpy(data.data() + p, sps_data.data(), sps_data.size());
  p += sps_data.size();
  data[p++] = 0x01;
  data[p++] = pps_data.size() >> 8;
  data[p++] = pps_data.size();
  memcpy(data.data() + p, pps_data.data(), pps_data.size());

  auto decoder_spec = CreateDecoderSpec(kCMVideoCodecType_H264, data);

  CMFormatDescriptionRef format_description;
  if (OSStatus status =
          CMVideoFormatDescriptionCreate(kCFAllocatorDefault, kCMVideoCodecType_H264, width, height,
                                         decoder_spec.get(), &format_description)) {
    Log("CMVideoFormatDescriptionCreate error: ", status);
    return nullptr;
  }
  return WrapCf(format_description);
}

CVMetalTexturePtr GetMtlTexture(CVImageBufferRef image, int width, int height, int plane_index) {
  CVMetalTextureRef texture;
  if (auto status = CVMetalTextureCacheCreateTextureFromImage(
          kCFAllocatorDefault, gTextureCache, image,
          /*textureAttributes=*/nullptr, MTLPixelFormatR8Unorm, width, height, plane_index,
          &texture)) {
    Log("Error converting frame: ", status, " plane_index: ", plane_index);
    return nullptr;
  }
  return WrapCf(texture);
}

void DecoderCallback(void * /*opaque*/, void *source_frame_ref_con, OSStatus status,
                     VTDecodeInfoFlags /*flags*/, CVImageBufferRef image_buffer, CMTime pts,
                     CMTime duration) {
  auto *target = reinterpret_cast<virgl_video_buffer *>(source_frame_ref_con);
  Log("Received frame ", status, ' ', image_buffer, " PREV ", target->image.get());
  if (status == 0 && image_buffer) {
    target->image.reset(CVPixelBufferRetain(image_buffer));
    target->pts = pts;
    target->duration = duration;
  }
}

VTDecompressionSessionPtr CreateDecompressionSession(const virgl_video_codec *codec,
                                                     CMFormatDescriptionRef format_description,
                                                     int width, int height) {
  auto buffer_attributes =
      CreateBufferAttributes(width, height, kCVPixelFormatType_420YpCbCr8Planar);
  VTDecompressionOutputCallbackRecord decoder_cb = {
      .decompressionOutputCallback = DecoderCallback,
      .decompressionOutputRefCon = const_cast<virgl_video_codec *>(codec)};
  VTDecompressionSessionRef decompression_session;
  if (OSStatus status =
          VTDecompressionSessionCreate(kCFAllocatorDefault, format_description,
                                       /*videoDecoderSpecification=*/nullptr,
                                       /*destinationImageBufferAttributes=*/buffer_attributes.get(),
                                       /*outputCallback=*/&decoder_cb, &decompression_session);
      status != 0) {
    Log("CMVideoFormatDescriptionCreate error: ", status);
    return nullptr;
  };
  return WrapCf(decompression_session);
}

void EncoderCallback(void *opaque, void * /*frame*/, int status, VTEncodeInfoFlags /*info*/,
                     CMSampleBufferRef sample_buffer) {
  auto *codec = reinterpret_cast<virgl_video_codec *>(opaque);
  if (status == 0 && sample_buffer) {
    auto buffer =
        WrapCf(reinterpret_cast<CMSampleBufferRef>(const_cast<void *>(CFRetain(sample_buffer))));
    std::unique_lock lock{codec->mutex};
    codec->frame_queue.push_back(std::move(buffer));
  } else {
    Log("EncoderCallback status = ", status);
  }
}

VTCompressionSessionPtr CreateCompressionSession(virgl_video_codec *codec,
                                                 const virgl_picture_desc *desc) {
  auto encoder_spec = WrapCf(CFDictionaryCreateMutable(kCFAllocatorDefault, /*capacity=*/20,
                                                       &kCFCopyStringDictionaryKeyCallBacks,
                                                       &kCFTypeDictionaryValueCallBacks));
  CFDictionarySetValue(encoder_spec.get(),
                       kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder,
                       kCFBooleanTrue);

  CMVideoCodecType codec_type = GetCodecType(codec->profile);
  VTCompressionSessionRef session;
  if (OSStatus status = VTCompressionSessionCreate(
          kCFAllocatorDefault, codec->width, codec->height, codec_type, encoder_spec.get(),
          /*sourceImageBufferAttributes=*/nullptr, kCFAllocatorDefault,
          /*outputCallback=*/EncoderCallback,
          /*outputCallbackRefCon=*/codec, &session)) {
    Log("VTCompressionSessionCreate error: ", status);
  }
  auto session_p = WrapCf(session);

  auto set_bitrate = [&](int target_bitrate) {
    Log("SETTING BITRATE ", target_bitrate);
    auto bitrate =
        WrapCf(CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &target_bitrate));
    if (OSStatus status = VTSessionSetProperty(session, kVTCompressionPropertyKey_AverageBitRate,
                                               bitrate.get())) {
      Log("VTSessionSetProperty AverageBitRate ", status);
      return -1;
    }
    return 0;
  };

  auto set_max_rate = [&](int max_bitrate) {
    int64_t bytes_per_second_value = max_bitrate >> 3;
    auto bytes_per_second =
        WrapCf(CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt64Type, &bytes_per_second_value));
    if (!bytes_per_second) {
      return -1;
    }
    int64_t one_second_value = 1;
    auto one_second =
        WrapCf(CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt64Type, &one_second_value));
    if (!one_second) {
      return -1;
    }
    const void *nums[2] = {reinterpret_cast<const void *>(bytes_per_second.get()),
                           reinterpret_cast<const void *>(one_second.get())};
    auto data_rate_limits =
        WrapCf(CFArrayCreate(kCFAllocatorDefault, (const void **)nums, 2, &kCFTypeArrayCallBacks));

    if (!data_rate_limits) {
      return -1;
    }
    if (OSStatus status = VTSessionSetProperty(session, kVTCompressionPropertyKey_DataRateLimits,
                                               data_rate_limits.get())) {
      Log("VTSessionSetProperty DataRateLimits: ", status);
      return -1;
    }
    return 0;
  };

  if (codec_type == kCMVideoCodecType_H264) {
    if (desc->h264_enc.pic_ctrl.enc_cabac_enable) {
      if (OSStatus status = VTSessionSetProperty(session, kVTCompressionPropertyKey_H264EntropyMode,
                                                 kVTH264EntropyMode_CABAC)) {
        Log("VTSessionSetProperty H264EntropyMode ", status);
        return nullptr;
      }
    }
    if (desc->h264_enc.rate_ctrl[0].target_bitrate > 0) {
      if (set_bitrate(desc->h264_enc.rate_ctrl[0].target_bitrate) != 0) {
        return nullptr;
      }
    }
    if (desc->h264_enc.rate_ctrl[0].peak_bitrate > 0) {
      if (set_max_rate(desc->h264_enc.rate_ctrl[0].peak_bitrate) != 0) {
        return nullptr;
      }
    }
  }

  if (codec_type == kCMVideoCodecType_HEVC) {
    if (desc->h265_enc.rc.target_bitrate > 0) {
      if (set_bitrate(desc->h265_enc.rc.target_bitrate) != 0) {
        return nullptr;
      }
    }
    if (desc->h265_enc.rc.peak_bitrate > 0) {
      if (set_max_rate(desc->h265_enc.rc.peak_bitrate) != 0) {
        return nullptr;
      }
    }
  }

  if (OSStatus status = VTSessionSetProperty(
          session, kVTCompressionPropertyKey_AllowFrameReordering, kCFBooleanFalse)) {
    Log("VTSessionSetProperty AllowFrameReordering ", status);
    return nullptr;
  }

  return session_p;
}

bool IsKeyFrame(CMVideoCodecType codec_type, const virgl_picture_desc *desc) {
  return (codec_type == kCMVideoCodecType_H264 &&
          (desc->h264_enc.picture_type == PIPE_H2645_ENC_PICTURE_TYPE_I ||
           desc->h264_enc.picture_type == PIPE_H2645_ENC_PICTURE_TYPE_IDR)) ||
         (codec_type == kCMVideoCodecType_HEVC &&
          (desc->h265_enc.picture_type == PIPE_H2645_ENC_PICTURE_TYPE_I ||
           desc->h265_enc.picture_type == PIPE_H2645_ENC_PICTURE_TYPE_IDR));
}

CFDictionaryPtr CreateEncoderDict(CMVideoCodecType codec_type, const virgl_picture_desc *desc) {
  if (IsKeyFrame(codec_type, desc)) {
    const void *keys[] = {kVTEncodeFrameOptionKey_ForceKeyFrame};
    const void *vals[] = {kCFBooleanTrue};

    return WrapCf(CFDictionaryCreate(kCFAllocatorDefault, keys, vals, /*numValues=*/1,
                                     /*keyCallBacks=*/nullptr,
                                     /*valueCallBacks=*/nullptr));
  }

  return nullptr;
}

std::vector<uint8_t> GetParameterSet(virgl_video_codec *codec, CMSampleBufferRef sample_buffer) {
  std::vector<uint8_t> buffer;
  CMFormatDescriptionRef format_description = CMSampleBufferGetFormatDescription(sample_buffer);
  const uint8_t *parameter_set;
  size_t parameter_set_size;
  size_t parameter_set_count;
  size_t parameter_set_index = 0;
  CMVideoCodecType codec_type = GetCodecType(codec->profile);
  auto get_param_set = codec_type == kCMVideoCodecType_H264
                           ? CMVideoFormatDescriptionGetH264ParameterSetAtIndex
                           : CMVideoFormatDescriptionGetHEVCParameterSetAtIndex;
  do {
    if (OSStatus status =
            get_param_set(format_description, parameter_set_index, &parameter_set,
                          &parameter_set_size, &parameter_set_count, &codec->nalu_header_length)) {
      Log("GetParameterSetAtIndex error: ", status);
      return {};
    }

    size_t offset = buffer.size();
    buffer.resize(buffer.size() + parameter_set_size + 4);

    buffer[offset + 3] = 1;
    memcpy(buffer.data() + offset + 4, parameter_set, parameter_set_size);

    Log("PARAMETER SET ", codec->nalu_header_length, ' ', parameter_set_size, ' ',
        ToString(std::span(parameter_set, parameter_set_size)));
    parameter_set_index++;
  } while (parameter_set_index < parameter_set_count);
  return buffer;
}

BufferView GetSampleBytes(CMSampleBufferRef sample_buffer, int nalu_header_length) {
  if (!sample_buffer) {
    return {};
  }
  CMBlockBufferRef block = CMSampleBufferGetDataBuffer(sample_buffer);
  if (!block) {
    return {};
  }

  size_t offset = 0;
  size_t length;
  size_t chunk_length;
  char *data_p;
  if (OSStatus status =
          CMBlockBufferGetDataPointer(block, offset, &chunk_length, &length, &data_p)) {
    Log("CMBlockBufferGetDataPointer error: ", status);
    return {};
  }
  if (chunk_length != length) {
    return {};
  }

  uint8_t *data = reinterpret_cast<uint8_t *>(data_p);
  Log("Encodec chunk: ", ToString(std::span<const uint8_t>(data, std::min<size_t>(64, length))));
  BufferView result;
  while (data - reinterpret_cast<uint8_t *>(data_p) < static_cast<uint32_t>(length)) {
    uint32_t chunk_length = 0;
    for (int i = 0; i < nalu_header_length; i++) {
      chunk_length <<= 8;
      chunk_length += data[i];
    }
    data[1] = 0;
    data[2] = 0;
    data[3] = 1;
    result.data.push_back(data + 1);
    result.size.push_back(chunk_length + 3);
    data += chunk_length + 4;
    Log("OUTPUT BUFFER ", chunk_length);
  }

  return result;
}

std::vector<CMSampleBufferPtr> GetFrameQueue(virgl_video_codec *codec) {
  std::unique_lock lock{codec->mutex};
  auto queue = std::move(codec->frame_queue);
  codec->frame_queue.clear();
  return queue;
}

void CallEncodeCompleted(virgl_video_codec *codec, std::span<const CMSampleBufferPtr> frame_queue) {
  BufferView bytes;
  for (const CMSampleBufferPtr &sample : frame_queue) {
    std::vector<uint8_t> parameter_buffer;
    if (parameter_buffer = GetParameterSet(codec, sample.get());
        codec->encode_parameter_set != parameter_buffer) {
      codec->encode_parameter_set = std::move(parameter_buffer);
      bytes.data.push_back(codec->encode_parameter_set.data());
      bytes.size.push_back(codec->encode_parameter_set.size());
    }
    BufferView current = GetSampleBytes(sample.get(), codec->nalu_header_length);
    std::copy(current.data.begin(), current.data.end(), std::back_inserter(bytes.data));
    std::copy(current.size.begin(), current.size.end(), std::back_inserter(bytes.size));
  }
  for (size_t i = 0; i < bytes.data.size(); i++) {
    gVideoFile->write(reinterpret_cast<const char *>(bytes.data[i]), bytes.size[i]);
    gVideoFile->flush();
  }
  gCallbacks->encode_completed(codec, /*src_buf=*/nullptr, /*ref_buf=*/nullptr, bytes.data.size(),
                               reinterpret_cast<void **>(bytes.data.data()), bytes.size.data());
}

}  // namespace

int virgl_video_init(int /*drm_fd*/, struct virgl_video_callbacks *cbs, unsigned int /*flags*/) {
  gLogFile = new std::ofstream("virgl-renderer.txt");
  if (!*gLogFile) {
    return -1;
  }
  gVideoFile = new std::ofstream("video.h264", std::ios::binary);
  if (!*gVideoFile) {
    return -1;
  }
  Log("virgl_video_init");
  gCallbacks = cbs;
  gMetalDevice = MTLCreateSystemDefaultDevice();
  if (auto status = CVMetalTextureCacheCreate(kCFAllocatorDefault,
                                              /*cacheAttributes=*/nullptr, gMetalDevice,
                                              /*textureAttributes=*/nullptr, &gTextureCache)) {
    Log("CVMetalTextureCacheCreate: ", status);
    [gMetalDevice release];
    gMetalDevice = nullptr;
    return -1;
  }
  return 0;
}

void virgl_video_destroy() {
  Log("virgl_video_destroy");

  delete gLogFile;
  gLogFile = nullptr;

  delete gVideoFile;
  gVideoFile = nullptr;

  CFRelease(gTextureCache);
  gTextureCache = nullptr;

  [gMetalDevice release];
  gMetalDevice = nullptr;
}

int virgl_video_fill_caps(union virgl_caps *caps) {
  Log("virgl_video_fill_caps ", caps->max_version);

  auto add_cap = [&](pipe_video_profile profile, pipe_video_entrypoint entrypoint) {
    virgl_video_caps *vcaps = &caps->v2.video_caps[caps->v2.num_video_caps++];
    vcaps->profile = profile;
    vcaps->entrypoint = entrypoint;
    vcaps->max_level = 0;
    vcaps->stacked_frames = 0;
    vcaps->max_width = 3840;
    vcaps->max_height = 2160;
    vcaps->prefered_format = PIPE_FORMAT_IYUV;
    vcaps->max_macroblocks = 1;
    vcaps->npot_texture = 1;
    vcaps->supports_progressive = 1;
    vcaps->supports_interlaced = 0;
    vcaps->prefers_interlaced = 0;
  };

  add_cap(PIPE_VIDEO_PROFILE_MPEG4_AVC_HIGH, PIPE_VIDEO_ENTRYPOINT_BITSTREAM);
  add_cap(PIPE_VIDEO_PROFILE_MPEG4_AVC_HIGH, PIPE_VIDEO_ENTRYPOINT_ENCODE);
  add_cap(PIPE_VIDEO_PROFILE_HEVC_MAIN, PIPE_VIDEO_ENTRYPOINT_ENCODE);

  return 0;
}

struct virgl_video_codec *virgl_video_create_codec(
    const struct virgl_video_create_codec_args *args) {
  Log("virgl_video_create_codec ", args->width, ' ', args->height);
  auto *codec = new (std::nothrow) virgl_video_codec{};
  if (!codec) {
    return nullptr;
  }
  codec->width = args->width;
  codec->height = args->height;
  codec->opaque = args->opaque;
  codec->entrypoint = args->entrypoint;
  codec->profile = args->profile;
  codec->chroma_format = args->chroma_format;
  return codec;
}

void virgl_video_destroy_codec(struct virgl_video_codec *codec) {
  Log("virgl_video_destroy_codec");
  if (codec->decompression_session) {
    VTDecompressionSessionInvalidate(codec->decompression_session.get());
  }
  if (codec->compression_session) {
    VTCompressionSessionInvalidate(codec->compression_session.get());
  }
  delete codec;
}

enum pipe_video_profile virgl_video_codec_profile(const struct virgl_video_codec *codec) {
  Log("virgl_video_codec_profile");
  return codec->profile;
}

void *virgl_video_codec_opaque_data(struct virgl_video_codec *codec) { return codec->opaque; }

struct virgl_video_buffer *virgl_video_create_buffer(
    const struct virgl_video_create_buffer_args *args) {
  Log("virgl_video_create_buffer ", args->format, ' ', args->width, ' ', args->height);

  virgl_video_buffer *buffer = new (std::nothrow) virgl_video_buffer{};
  if (!buffer) {
    return nullptr;
  }
  buffer->opaque = args->opaque;
  buffer->id = gNextBufferId++;
  return buffer;
}

void virgl_video_destroy_buffer(struct virgl_video_buffer *buffer) {
  Log("virgl_video_destroy_buffer");
  delete buffer;
}

uint32_t virgl_video_buffer_id(const struct virgl_video_buffer *buffer) {
  return buffer ? buffer->id : -1;
}

void *virgl_video_buffer_opaque_data(struct virgl_video_buffer *buffer) { return buffer->opaque; }

int virgl_video_begin_frame(struct virgl_video_codec * /*codec*/,
                            struct virgl_video_buffer * /*target*/) {
  Log("virgl_video_begin_frame");
  return 0;
}

int virgl_video_decode_bitstream(struct virgl_video_codec *codec, struct virgl_video_buffer *target,
                                 const union virgl_picture_desc *desc, unsigned num_buffers,
                                 const void *const *buffers, const unsigned *sizes) {
  Log("virgl_video_decode_bitstream ", num_buffers, " ", sizes[0], ' ', sizeof(desc->h264), ' ',
      uint32_t(desc->h264.pps.sps.level_idc));

  H264ParameterSet h264_parameter_set = GetH264ParameterSet(codec, &desc->h264);

  if (codec->h264_parameter_set != h264_parameter_set) {
    if (codec->h264_parameter_set) {
      VTDecompressionSessionInvalidate(codec->decompression_session.get());
      codec->decompression_session = nullptr;
      Log("VTDecompressionSession recreate");
    }
    codec->h264_parameter_set = std::move(h264_parameter_set);
    codec->format_description = CreateFormatDescription(*codec->h264_parameter_set);
    if (codec->format_description == nullptr) {
      return -1;
    }
    codec->decompression_session = CreateDecompressionSession(
        codec, codec->format_description.get(), codec->width, codec->height);
    if (!codec->decompression_session) {
      return -1;
    }
  }

  for (unsigned i = 0; i < num_buffers; i++) {
    const auto *data = static_cast<const uint8_t *>(buffers[i]);
    const unsigned input_size = sizes[i];

    std::vector<uint8_t> buffer;
    {
      buffer.resize(input_size + 1);
      memcpy(buffer.data() + 4, data + 3, input_size - 3);

      uint32_t size = htonl(input_size - 3);
      memcpy(buffer.data(), &size, sizeof(size));
    }

    Log("DECODE BUF = ", ToString(std::span<const uint8_t>(data, 8u)));

    const size_t size = buffer.size();

    auto sample_buf = CreateSampleBuffer(
        CreateFormatDescription(GetH264ParameterSet(codec, &desc->h264,
                                                    desc->h264.num_ref_idx_l0_active_minus1,
                                                    desc->h264.num_ref_idx_l1_active_minus1))
            .get(),
        buffer.data(), size);

    target->image.reset();
    if (OSStatus status = VTDecompressionSessionDecodeFrame(
            codec->decompression_session.get(), sample_buf.get(),
            /*decodeFlags=*/0,
            /*sourceFrameRefCon=*/target, /*infoFlagsOut=*/nullptr);
        status != 0) {
      Log("VTDecompressionSessionDecodeFrame error: ", status);
      return -1;
    }
  }
  return 0;
}

int virgl_video_encode_bitstream(struct virgl_video_codec *codec, struct virgl_video_buffer *source,
                                 const union virgl_picture_desc *desc) {
  Log("virgl_video_encode_bitstream");

  if (!codec->compression_session) {
    codec->compression_session = CreateCompressionSession(codec, desc);
    if (!codec->compression_session) {
      return -1;
    }
  }

  CMVideoCodecType codec_type = GetCodecType(codec->profile);
  CMTime pts = kCMTimeZero;
  if (codec_type == kCMVideoCodecType_H264) {
    pts =
        CMTimeMake((desc->h264_enc.frame_num_cnt - 1) * desc->h264_enc.rate_ctrl[0].frame_rate_num,
                   desc->h264_enc.rate_ctrl[0].frame_rate_den * 1000);
  } else if (codec_type == kCMVideoCodecType_HEVC) {
    pts = CMTimeMake((desc->h265_enc.frame_num - 1) * desc->h265_enc.rc.frame_rate_num,
                     desc->h265_enc.rc.frame_rate_den * 1000);
  }
  if (OSStatus status = VTCompressionSessionEncodeFrame(
          codec->compression_session.get(), source->image.get(), pts, /*duration=*/kCMTimeInvalid,
          CreateEncoderDict(GetCodecType(codec->profile), desc).get(),
          /*sourceFrameRefcon=*/nullptr,
          /*infoFlagsOut=*/nullptr)) {
    Log("VTCompressionSessionEncodeFrame: ", status);
    return -1;
  }

  return 0;
}

int virgl_video_end_frame(struct virgl_video_codec *codec, struct virgl_video_buffer *target) {
  if (codec->decompression_session) {
    if (target->image) {
      Log("decode completed ", target->image.get());
      auto luma = GetMtlTexture(target->image.get(), codec->width, codec->height, 0);
      auto chroma_x = GetMtlTexture(target->image.get(), codec->width / 2, codec->height / 2, 1);
      auto chroma_y = GetMtlTexture(target->image.get(), codec->width / 2, codec->height / 2, 2);
      virgl_video_dma_buf buffer = {
          .buf = target,
          .width = static_cast<uint32_t>(codec->width),
          .height = static_cast<uint32_t>(codec->height),
          .flags = VIRGL_VIDEO_DMABUF_READ_ONLY,
          .num_planes = 3,
          .planes = {{.mtl_texture = CVMetalTextureGetTexture(luma.get())},
                     {.mtl_texture = CVMetalTextureGetTexture(chroma_x.get())},
                     {.mtl_texture = CVMetalTextureGetTexture(chroma_y.get())}}};

      gCallbacks->decode_completed(codec, &buffer);
    } else {
      Log("FRAME NOT READY! ", target);
    }
    return 0;
  } else if (codec->compression_session) {
    if (OSStatus status =
            VTCompressionSessionCompleteFrames(codec->compression_session.get(), kCMTimeInvalid)) {
      Log("VTCompressionSessionCompleteFrames: ", status);
      return -1;
    }
    Log("virgl_video_end_frame");
    CallEncodeCompleted(codec, GetFrameQueue(codec));
    return 0;
  }

  return -1;
}
