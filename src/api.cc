// Copyright 2017 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//  Public entry points: plain-C calls, EncoderParam, and Encode()
//
// Author: Skal (pascal.massimino@gmail.com)

#include <assert.h>
#include <stdint.h>

#include <cstdlib>
#include <memory>
#include <new>
#include <string>

#include "sjpegi.h"

////////////////////////////////////////////////////////////////////////////////
// public plain-C functions

size_t SjpegEncode(const uint8_t* rgb, int width, int height, int stride,
                   uint8_t** out_data, float quality, int method,
                   SjpegYUVMode yuv_mode) {
  if (rgb == nullptr || out_data == nullptr) return 0;
  if (width <= 0 || height <= 0 || std::abs(stride) < 3 * width) return 0;
  *out_data = nullptr;  // safety

  sjpeg::MemorySink sink((size_t)width * height / 4);
  sjpeg::Encoder* const enc = sjpeg::EncoderFactory(rgb, width, height, stride,
                                                    yuv_mode, &sink);
  if (enc == nullptr) return 0;
  enc->SetQuality(quality);
  enc->SetCompressionMethod(method);
  size_t size = 0;
  if (enc->Encode()) sink.Release(out_data, &size);
  delete enc;
  return size;
}

////////////////////////////////////////////////////////////////////////////////

size_t SjpegCompress(const uint8_t* rgb, int width, int height, float quality,
                     uint8_t** out_data) {
  return SjpegEncode(rgb, width, height, 3 * width, out_data,
                     quality, 4, SJPEG_YUV_AUTO);
}

void SjpegFreeBuffer(const uint8_t* buffer) {
  delete[] buffer;
}

////////////////////////////////////////////////////////////////////////////////

uint32_t SjpegVersion() {
  return SJPEG_VERSION;
}

////////////////////////////////////////////////////////////////////////////////
// Parametrized call

namespace sjpeg {

EncoderParam::EncoderParam() : search_hook(nullptr), memory(nullptr) {
  Init(kDefaultQuality);
}

EncoderParam::EncoderParam(float quality_factor)
    : search_hook(nullptr), memory(nullptr) {
  Init(quality_factor);
}

void EncoderParam::Init(float quality_factor) {
  Huffman_compress = true;
  adaptive_quantization = true;
  use_trellis = false;
  yuv_mode = SJPEG_YUV_AUTO;
  quantization_bias = kDefaultBias;
  qdelta_max_luma = kDefaultDeltaMaxLuma;
  qdelta_max_chroma = kDefaultDeltaMaxChroma;
  adaptive_bias = false;
  SetLimitQuantization(false);
  min_quant_tolerance_ = 0;
  SetQuality(quality_factor);
  target_mode = TARGET_NONE;
  target_value = 0;
  passes = 1;
  tolerance = 1.;
  qmin = 0.;
  qmax = 100.;
}

void EncoderParam::SetQuality(float quality_factor) {
  const float q = GetQFactor(quality_factor);
  SetQuantMatrix(kDefaultMatrices[0], q, quant_[0]);
  SetQuantMatrix(kDefaultMatrices[1], q, quant_[1]);
}

void EncoderParam::SetQuantization(const uint8_t m[2][64],
                                       float reduction) {
  if (reduction <= 1.f) reduction = 1.f;
  if (m == nullptr) return;
  for (int c = 0; c < 2; ++c) {
    for (size_t i = 0; i < 64; ++i) {
      const int v = static_cast<int>(m[c][i] * 100. / reduction + .5);
      quant_[c][i] = (v > 255) ? 255u : (v < 1) ? 1u : v;
    }
  }
}

void EncoderParam::SetLimitQuantization(bool limit_quantization,
                                            int min_quant_tolerance) {
  use_min_quant_ = limit_quantization;
  if (limit_quantization) SetMinQuantization(quant_, min_quant_tolerance);
}

void EncoderParam::SetMinQuantization(const uint8_t m[2][64],
                                          int min_quant_tolerance) {
  use_min_quant_ = true;
  CopyQuantMatrix(m[0], min_quant_[0]);
  CopyQuantMatrix(m[1], min_quant_[1]);
  min_quant_tolerance_ = (min_quant_tolerance < 0) ? 0
                       : (min_quant_tolerance > 100) ? 100
                       : min_quant_tolerance;
}

void EncoderParam::ResetMetadata() {
  iccp.clear();
  exif.clear();
  app_markers.clear();
  xmp.clear();
  xmp_split_point = 0u;
}

bool Encoder::InitFromParam(const EncoderParam& param) {
  SetQuantMatrices(param.quant_);
  if (param.use_min_quant_) {
    SetMinQuantMatrices(param.min_quant_, param.min_quant_tolerance_);
  } else {
    SetDefaultMinQuantMatrices();
  }

  int method = param.Huffman_compress ? 1 : 0;
  if (param.adaptive_quantization) method += 3;
  if (param.use_trellis) {
    method = (method == 4) ? 7 : (method == 6) ? 8 : method;
  }

  SetCompressionMethod(method);
  SetQuantizationBias(param.quantization_bias, param.adaptive_bias);
  SetQuantizationDeltas(param.qdelta_max_luma, param.qdelta_max_chroma);

  SetMetadata(param.iccp, Encoder::ICC);
  SetMetadata(param.exif, Encoder::EXIF);
  SetMetadata(param.app_markers, Encoder::MARKERS);
  SetMetadata(param.xmp, Encoder::XMP);
  xmp_split_ = param.xmp_split_point;

  passes_ = (param.passes < 1) ? 1 : (param.passes > 20) ? 20 : param.passes;
  if (passes_ > 1) {
    use_extra_memory_ = true;
    reuse_run_levels_ = true;
    search_hook_ = (param.search_hook == nullptr) ? &default_hook_
                                                  : param.search_hook;
    if (!search_hook_->Setup(param)) return false;
  }

  assert(memory_hook_ == (param.memory == nullptr ? GetDefaultMemoryManager()
                                                  : param.memory));
  return true;
}

bool Encode(const uint8_t* rgb, int width, int height, int stride,
            const EncoderParam& param, ByteSink* sink) {
  if (rgb == nullptr || sink == nullptr) return false;
  if (width <= 0 || height <= 0 || std::abs(stride) < 3 * width) return false;

  Encoder* const enc = EncoderFactory(rgb, width, height, stride,
                                      param.yuv_mode, sink, kRGBInput,
                                      param.memory);
  return FinishEncoding(enc, param);
}

size_t Encode(const uint8_t* rgb, int width, int height, int stride,
              const EncoderParam& param, uint8_t** out_data) {
  MemorySink sink((size_t)width * height / 4);    // estimation of output size
  if (!Encode(rgb, width, height, stride, param, &sink)) return 0;
  size_t size;
  sink.Release(out_data, &size);
  return size;
}

bool EncodeBGRA(const uint8_t* bgra, int width, int height, int stride,
                const EncoderParam& param, ByteSink* sink) {
  if (bgra == nullptr || sink == nullptr) return false;
  if (width <= 0 || height <= 0 || std::abs(stride) < 4 * width) return false;
  const SjpegYUVMode mode = param.yuv_mode;
  if (mode == SJPEG_YUV_AUTO || mode == SJPEG_YUV_SHARP) {
    // Fallback: convert to a scratch RGB plane and reuse the RGB path.
    const int rgb_stride = 3 * width;
    std::unique_ptr<uint8_t[]> rgb(new (std::nothrow)
                                       uint8_t[(size_t)rgb_stride * height]);
    if (rgb == nullptr) return false;
    for (int y = 0; y < height; ++y) {
      const uint8_t* s = bgra + (size_t)y * stride;
      uint8_t* d = rgb.get() + (size_t)y * rgb_stride;
      for (int x = 0; x < width; ++x, s += 4, d += 3) {
        d[0] = s[2];
        d[1] = s[1];
        d[2] = s[0];
      }
    }
    return Encode(rgb.get(), width, height, rgb_stride, param, sink);
  }
  Encoder* const enc = EncoderFactory(bgra, width, height, stride, mode, sink,
                                      kBGRAInput, param.memory);
  return FinishEncoding(enc, param);
}

bool EncodeRGBA(const uint8_t* rgba, int width, int height, int stride,
                const EncoderParam& param, ByteSink* sink) {
  if (rgba == nullptr || sink == nullptr) return false;
  if (width <= 0 || height <= 0 || std::abs(stride) < 4 * width) return false;
  const SjpegYUVMode mode = param.yuv_mode;
  if (mode == SJPEG_YUV_AUTO || mode == SJPEG_YUV_SHARP) {
    // Fallback: convert to a scratch RGB plane and reuse the RGB path.
    const int rgb_stride = 3 * width;
    std::unique_ptr<uint8_t[]> rgb(new (std::nothrow)
                                       uint8_t[(size_t)rgb_stride * height]);
    if (rgb == nullptr) return false;
    for (int y = 0; y < height; ++y) {
      const uint8_t* s = rgba + (size_t)y * stride;
      uint8_t* d = rgb.get() + (size_t)y * rgb_stride;
      for (int x = 0; x < width; ++x, s += 4, d += 3) {
        d[0] = s[0];
        d[1] = s[1];
        d[2] = s[2];
      }
    }
    return Encode(rgb.get(), width, height, rgb_stride, param, sink);
  }
  Encoder* const enc = EncoderFactory(rgba, width, height, stride, mode, sink,
                                      kRGBAInput, param.memory);
  return FinishEncoding(enc, param);
}

bool EncodeGray(const uint8_t* gray, int width, int height, int stride,
                const EncoderParam& param, ByteSink* sink) {
  if (gray == nullptr || sink == nullptr) return false;
  if (width <= 0 || height <= 0 || std::abs(stride) < width) return false;

  Encoder* const enc = GrayEncoderFactory(gray, width, height, stride, sink,
                                          param.memory);
  return FinishEncoding(enc, param);
}

////////////////////////////////////////////////////////////////////////////////
// std::string variants

bool Encode(const uint8_t* rgb, int width, int height, int stride,
            const EncoderParam& param, std::string* output) {
  if (output == nullptr) return false;
  output->clear();
  output->reserve((size_t)width * height / 4);
  StringSink sink(output);
  return Encode(rgb, width, height, stride, param, &sink);
}

bool EncodeBGRA(const uint8_t* bgra, int width, int height, int stride,
                const EncoderParam& param, std::string* output) {
  if (output == nullptr) return false;
  output->clear();
  output->reserve((size_t)width * height / 4);
  StringSink sink(output);
  return EncodeBGRA(bgra, width, height, stride, param, &sink);
}

bool EncodeRGBA(const uint8_t* rgba, int width, int height, int stride,
                const EncoderParam& param, std::string* output) {
  if (output == nullptr) return false;
  output->clear();
  output->reserve((size_t)width * height / 4);
  StringSink sink(output);
  return EncodeRGBA(rgba, width, height, stride, param, &sink);
}

bool EncodeGray(const uint8_t* gray, int width, int height, int stride,
                const EncoderParam& param, std::string* output) {
  if (output == nullptr) return false;
  output->clear();
  output->reserve((size_t)width * height / 4);
  StringSink sink(output);
  return EncodeGray(gray, width, height, stride, param, &sink);
}

}  // namespace sjpeg

