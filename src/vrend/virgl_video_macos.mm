#include <CoreFoundation/CFBase.h>
#include <CoreMedia/CMFormatDescription.h>
#include <CoreVideo/CVPixelBuffer.h>
#include <MacTypes.h>
#import <Metal/Metal.h>
#include <VideoToolbox/VTDecompressionSession.h>
#include <sys/_types.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <span>
#include <sstream>

#include "pipe/p_video_enums.h"
#include "u_formats.h"
#include "virgl_video.h"
#include "virgl_video_hw.h"

namespace {

template <typename T>
auto WrapCf(T *obj) {
  auto deleter = [](T *obj) { CFRelease(obj); };
  return std::unique_ptr<T, decltype(deleter)>(obj);
}

using CMFormatDescriptionPtr = decltype(WrapCf(CMFormatDescriptionRef()));
using CFDictionaryPtr = decltype(WrapCf(CFDictionaryRef()));
using VTDecompressionSessionPtr = decltype(WrapCf(VTDecompressionSessionRef()));
using CMSampleBufferPtr = decltype(WrapCf(CMSampleBufferRef()));
using CVMetalTexturePtr = decltype(WrapCf(CVMetalTextureRef()));

}  // namespace

struct virgl_video_codec {
  int width;
  int height;
  void *opaque;
  CMFormatDescriptionPtr format_description;
  VTDecompressionSessionPtr decompression_session;
};

struct virgl_video_buffer {
  void *opaque;
  uint32_t id;
  CVMetalTexturePtr luma;
  CVMetalTexturePtr chroma_x;
  CVMetalTexturePtr chroma_y;
  virgl_video_dma_buf buffer;
};

namespace {

std::ofstream *gLogFile;
virgl_video_callbacks *gCallbacks;
id<MTLDevice> gMetalDevice;
CVMetalTextureCacheRef gTextureCache;
uint32_t gNextBufferId = 1;

template <typename... Ts>
void Log(const Ts &...args) {
  (*gLogFile << ... << args) << std::endl;
}

struct BitAddress {
  uint8_t *id;
  uint8_t bit;
};

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
      kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks,
      &kCFTypeDictionaryValueCallBacks));

  CFDictionarySetValue(
      config_info.get(),
      codec_type == kCMVideoCodecType_HEVC
          ? kVTVideoDecoderSpecification_EnableHardwareAcceleratedVideoDecoder
          : kVTVideoDecoderSpecification_RequireHardwareAcceleratedVideoDecoder,
      kCFBooleanTrue);

  auto avc_info = WrapCf(CFDictionaryCreateMutable(
      kCFAllocatorDefault, 1, &kCFTypeDictionaryKeyCallBacks,
      &kCFTypeDictionaryValueCallBacks));

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
  CFDictionarySetValue(
      config_info.get(),
      kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms,
      avc_info.get());
  return WrapCf(static_cast<CFDictionaryRef>(config_info.release()));
}

auto CreateBufferAttributes(int width, int height, OSType pix_fmt) {
  auto buffer_attributes = WrapCf(CFDictionaryCreateMutable(
      kCFAllocatorDefault, 4, &kCFTypeDictionaryKeyCallBacks,
      &kCFTypeDictionaryValueCallBacks));
  auto io_surface_properties = WrapCf(CFDictionaryCreateMutable(
      kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks,
      &kCFTypeDictionaryValueCallBacks));
  auto cv_pix_fmt = WrapCf(
      CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &pix_fmt));
  auto w =
      WrapCf(CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &width));
  auto h =
      WrapCf(CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &height));

  if (pix_fmt) {
    CFDictionarySetValue(buffer_attributes.get(),
                         kCVPixelBufferPixelFormatTypeKey, cv_pix_fmt.get());
  }
  CFDictionarySetValue(buffer_attributes.get(),
                       kCVPixelBufferIOSurfacePropertiesKey,
                       io_surface_properties.get());
  CFDictionarySetValue(buffer_attributes.get(), kCVPixelBufferWidthKey,
                       w.get());
  CFDictionarySetValue(buffer_attributes.get(), kCVPixelBufferHeightKey,
                       h.get());
  CFDictionarySetValue(buffer_attributes.get(),
                       kCVPixelBufferMetalCompatibilityKey, kCFBooleanTrue);

  return buffer_attributes;
}

CMSampleBufferPtr CreateSampleBuffer(CMFormatDescriptionRef fmt_desc,
                                     const void *buffer, int size) {
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
  if (OSStatus status =
          CMSampleBufferCreate(kCFAllocatorDefault,
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

CMFormatDescriptionPtr CreateFormatDescription(
    int width, int height, const virgl_h264_picture_desc *desc) {
  std::vector<uint8_t> sps_data(128);
  const auto &sps = desc->pps.sps;
  BitAddress address{sps_data.data(), 0};
  WriteByte(0x67, 8, address);  // SPS NALU
  WriteByte(0x64, 8, address);  // PIPE_VIDEO_PROFILE_MPEG4_AVC_HIGH
  WriteByte(0x00, 8, address);  // constraint_set_flag
  WriteByte(sps.MinLumaBiPredSize8x8 ? 31 : 0, 8, address);  // level_idc
  WriteGolomb(0x0, address);  // seq_parameter_set_id
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
  WriteGolomb(4 /*sps.max_num_ref_frames*/, address);
  WriteByte(0, 1, address);              // gaps_in_frame_num_value_allowed_flag
  WriteGolomb(width / 16 - 1, address);  // pic_width_in_mbs_minus1
  WriteGolomb(height / (16 * (2 - sps.frame_mbs_only_flag)) - 1,
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
  WriteGolomb(pps.num_ref_idx_l0_default_active_minus1, address_pps);
  WriteGolomb(pps.num_ref_idx_l1_default_active_minus1, address_pps);
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

  CMFormatDescriptionRef format_description;
  const uint8_t *parameter_set_pointers[] = {sps_data.data(), pps_data.data()};
  const size_t parameter_set_sizes[] = {sps_data.size(), pps_data.size()};
  if (OSStatus status = CMVideoFormatDescriptionCreateFromH264ParameterSets(
          kCFAllocatorDefault, /*parameterSetCount=*/2, parameter_set_pointers,
          parameter_set_sizes,
          /*NALUnitHeaderLength=*/4, &format_description);
      status != 0) {
    Log("CMVideoFormatDescriptionCreateFromH264ParameterSets error: ", status);
    return nullptr;
  }
  return WrapCf(format_description);

  //  std::vector<uint8_t> data(sps_data.size() + pps_data.size() + 11);
  //  uint32_t p = 0;
  //  data[p++] = 0x01;
  //  data[p++] = sps_data[1];
  //  data[p++] = sps_data[2];
  //  data[p++] = sps_data[3];
  //  data[p++] = 0xff;
  //  data[p++] = 0xe1;
  //  data[p++] = sps_data.size() >> 8;
  //  data[p++] = sps_data.size();
  //  memcpy(data.data() + p, sps_data.data(), sps_data.size());
  //  p += sps_data.size();
  //  data[p++] = 0x01;
  //  data[p++] = pps_data.size() >> 8;
  //  data[p++] = pps_data.size();
  //  memcpy(data.data() + p, pps_data.data(), pps_data.size());
  //
  //  auto decoder_spec = CreateDecoderSpec(kCMVideoCodecType_H264, data);
  //
  //  CMFormatDescriptionRef format_description;
  //  if (OSStatus status = CMVideoFormatDescriptionCreate(
  //          kCFAllocatorDefault, kCMVideoCodecType_H264, width, height,
  //          decoder_spec.get(), &format_description)) {
  //    Log("CMVideoFormatDescriptionCreateFromH264ParameterSets error: ",
  //    status); return nullptr;
  //  }
  //  return WrapCf(format_description);
}

CVMetalTexturePtr GetMtlTexture(CVImageBufferRef image, int width, int height,
                                int plane_index) {
  CVMetalTextureRef texture;
  if (auto status = CVMetalTextureCacheCreateTextureFromImage(
          kCFAllocatorDefault, gTextureCache, image,
          /*textureAttributes=*/nullptr, MTLPixelFormatR8Unorm, width, height,
          plane_index, &texture)) {
    Log("Error converting frame: ", status, " plane_index: ", plane_index);
    return nullptr;
  }
  return WrapCf(texture);
}

void DecoderCallback(void *opaque, void *source_frame_ref_con, OSStatus status,
                     VTDecodeInfoFlags flags, CVImageBufferRef image_buffer,
                     CMTime pts, CMTime duration) {
  auto *codec = reinterpret_cast<virgl_video_codec *>(opaque);
  auto *target = reinterpret_cast<virgl_video_buffer *>(source_frame_ref_con);
  Log("Received frame ", status, ' ', codec);
  if (status == 0 && image_buffer) {
    target->luma = GetMtlTexture(image_buffer, codec->width, codec->height, 0);
    target->chroma_x =
        GetMtlTexture(image_buffer, codec->width / 2, codec->height / 2, 1);
    target->chroma_y =
        GetMtlTexture(image_buffer, codec->width / 2, codec->height / 2, 2);
    target->buffer = {
        .buf = target,
        .width = static_cast<uint32_t>(codec->width),
        .height = static_cast<uint32_t>(codec->height),
        .flags = VIRGL_VIDEO_DMABUF_READ_ONLY,
        .num_planes = 3,
        .planes = {
            {.mtl_texture = CVMetalTextureGetTexture(target->luma.get())},
            {.mtl_texture = CVMetalTextureGetTexture(target->chroma_x.get())},
            {.mtl_texture = CVMetalTextureGetTexture(target->chroma_y.get())}}};
  }
}

VTDecompressionSessionPtr CreateDecompressionSession(
    virgl_video_codec *codec, CMFormatDescriptionRef format_description,
    int width, int height) {
  auto buffer_attributes = CreateBufferAttributes(
      width, height, kCVPixelFormatType_420YpCbCr8Planar);
  VTDecompressionOutputCallbackRecord decoder_cb = {
      .decompressionOutputCallback = DecoderCallback,
      .decompressionOutputRefCon = codec};
  VTDecompressionSessionRef decompression_session;
  if (OSStatus status = VTDecompressionSessionCreate(
          kCFAllocatorDefault, format_description,
          /*videoDecoderSpecification=*/nullptr,
          /*destinationImageBufferAttributes=*/buffer_attributes.get(),
          /*outputCallback=*/&decoder_cb, &decompression_session);
      status != 0) {
    Log("CMVideoFormatDescriptionCreate error: ", status);
    return nullptr;
  };
  return WrapCf(decompression_session);
}

}  // namespace

int virgl_video_init(int drm_fd, struct virgl_video_callbacks *cbs,
                     unsigned int flags) {
  gLogFile = new std::ofstream("virgl-renderer.txt");
  if (!gLogFile) {
    return -1;
  }
  Log("virgl_video_init");
  gCallbacks = cbs;
  gMetalDevice = MTLCreateSystemDefaultDevice();
  if (auto status = CVMetalTextureCacheCreate(
          kCFAllocatorDefault,
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

  CFRelease(gTextureCache);
  gTextureCache = nullptr;

  [gMetalDevice release];
  gMetalDevice = nullptr;
}

int virgl_video_fill_caps(union virgl_caps *caps) {
  Log("virgl_video_fill_caps ", caps->max_version);

  {
    virgl_video_caps *vcaps = &caps->v2.video_caps[caps->v2.num_video_caps++];
    vcaps->profile = PIPE_VIDEO_PROFILE_MPEG4_AVC_HIGH;
    vcaps->entrypoint = PIPE_VIDEO_ENTRYPOINT_BITSTREAM;
    vcaps->max_level = 0;
    vcaps->stacked_frames = 0;
    vcaps->max_width = 1920;
    vcaps->max_height = 1080;
    vcaps->prefered_format = PIPE_FORMAT_R8G8B8_UNORM;
    vcaps->max_macroblocks = 1;
    vcaps->npot_texture = 1;
    vcaps->supports_progressive = 1;
    vcaps->supports_interlaced = 0;
    vcaps->prefers_interlaced = 0;
    vcaps->max_temporal_layers = 0;
  }
  {
    virgl_video_caps *vcaps = &caps->v2.video_caps[caps->v2.num_video_caps++];
    vcaps->profile = PIPE_VIDEO_PROFILE_HEVC_MAIN;
    vcaps->entrypoint = PIPE_VIDEO_ENTRYPOINT_ENCODE;
    vcaps->max_level = 0;
    vcaps->stacked_frames = 0;
    vcaps->max_width = 1920;
    vcaps->max_height = 1080;
    vcaps->prefered_format = PIPE_FORMAT_NONE;
    vcaps->max_macroblocks = 1;
    vcaps->npot_texture = 1;
    vcaps->supports_progressive = 1;
    vcaps->supports_interlaced = 0;
    vcaps->prefers_interlaced = 0;
    vcaps->max_temporal_layers = 0;
  }
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
  return codec;
}

void virgl_video_destroy_codec(struct virgl_video_codec *codec) {
  Log("virgl_video_create_codec");
  if (codec->decompression_session) {
    VTDecompressionSessionInvalidate(codec->decompression_session.get());
  }
  delete codec;
}

enum pipe_video_profile virgl_video_codec_profile(
    const struct virgl_video_codec *codec) {
  Log("virgl_video_codec_profile");
  return PIPE_VIDEO_PROFILE_MPEG4_AVC_HIGH;
}

void *virgl_video_codec_opaque_data(struct virgl_video_codec *codec) {
  return codec->opaque;
}

struct virgl_video_buffer *virgl_video_create_buffer(
    const struct virgl_video_create_buffer_args *args) {
  Log("virgl_video_create_buffer ", args->format, ' ', args->width, ' ',
      args->height);

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

void *virgl_video_buffer_opaque_data(struct virgl_video_buffer *buffer) {
  return buffer->opaque;
}

int virgl_video_begin_frame(struct virgl_video_codec *codec,
                            struct virgl_video_buffer *target) {
  Log("virgl_video_begin_frame");
  target->buffer = {};
  target->luma.reset();
  target->chroma_x.reset();
  target->chroma_y.reset();
  return 0;
}

int virgl_video_decode_bitstream(struct virgl_video_codec *codec,
                                 struct virgl_video_buffer *target,
                                 const union virgl_picture_desc *desc,
                                 unsigned num_buffers,
                                 const void *const *buffers,
                                 const unsigned *sizes) {
  Log("virgl_video_decode_bitstream ", num_buffers, " ", sizes[0], ' ',
      sizeof(desc->h264), ' ', uint32_t(desc->h264.pps.sps.level_idc));

  if (!codec->format_description) {
    codec->format_description =
        CreateFormatDescription(codec->width, codec->height, &desc->h264);
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
    const uint8_t *data = static_cast<const uint8_t *>(buffers[i]);
    const unsigned input_size = sizes[i];

    std::vector<uint8_t> buffer;
    {
      buffer.resize(input_size + 1);
      memcpy(buffer.data() + 4, data + 3, input_size - 3);

      uint32_t size = htonl(input_size - 3);
      memcpy(buffer.data(), &size, sizeof(size));
    }

    const unsigned size = buffer.size();

    auto sample_buf = CreateSampleBuffer(codec->format_description.get(),
                                         buffer.data(), size);

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

int virgl_video_encode_bitstream(struct virgl_video_codec *codec,
                                 struct virgl_video_buffer *source,
                                 const union virgl_picture_desc *desc) {
  Log("virgl_video_encode_bitstream");
  return 0;
}

int virgl_video_end_frame(struct virgl_video_codec *codec,
                          struct virgl_video_buffer *target) {
  Log("virgl_video_end_frame");
  if (OSStatus status = VTDecompressionSessionWaitForAsynchronousFrames(
          codec->decompression_session.get());
      status != 0) {
    Log("VTDecompressionSessionWaitForAsynchronousFrames error: ", status);
    return -1;
  }

  if (target->buffer.num_planes > 0) {
    Log("decode completed ", target->buffer.num_planes);
    gCallbacks->decode_completed(codec, &target->buffer);
  } else {
    Log("FRAME NOT READY! ", target);
  }

  return 0;
}
