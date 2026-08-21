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
//  Per-colorspace Encoder sub-classes, and their factory
//
// Author: Skal (pascal.massimino@gmail.com)

#include <assert.h>
#include <stdint.h>
#include <string.h>

#include <cstdlib>
#include <new>

#include "sjpegi.h"

namespace sjpeg {

////////////////////////////////////////////////////////////////////////////////

void Encoder::InitComponents() {
  switch (yuv_mode_) {
    case SJPEG_YUV_444:
      nb_comps_ = 3;

      quant_idx_[0] = 0;
      quant_idx_[1] = 1;
      quant_idx_[2] = 1;

      nb_blocks_[0] = 1;
      nb_blocks_[1] = 1;
      nb_blocks_[2] = 1;
      mcu_blocks_ = 3;
      block_w_ = 8;
      block_h_ = 8;

      block_dims_[0] = 0x11;
      block_dims_[1] = 0x11;
      block_dims_[2] = 0x11;
    break;
    case SJPEG_YUV_420:
      nb_comps_ = 3;

      quant_idx_[0] = 0;
      quant_idx_[1] = 1;
      quant_idx_[2] = 1;

      nb_blocks_[0] = 4;
      nb_blocks_[1] = 1;
      nb_blocks_[2] = 1;
      mcu_blocks_ = 6;

      block_w_ = 16;
      block_h_ = 16;
      block_dims_[0] = 0x22;
      block_dims_[1] = 0x11;
      block_dims_[2] = 0x11;
    break;
    case SJPEG_YUV_400:
      nb_comps_ = 1;

      quant_idx_[0] = 0;

      nb_blocks_[0] = 1;
      mcu_blocks_ = 1;

      block_w_ = 8;
      block_h_ = 8;
      block_dims_[0] = 0x11;
    break;
    case SJPEG_YUV_AUTO:
    case SJPEG_YUV_SHARP:
    default:
      assert(0);   // shouldn't happen
    break;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Edge replication

namespace {

int GetAverage(const int16_t* const out) {
  int DC = 0;
  for (int i = 0; i < 64; ++i) DC += out[i];
  return (DC + 32) >> 6;
}

void SetAverage(int DC, int16_t* const out) {
  for (int i = 0; i < 64; ++i) out[i] = DC;
}

}   // namespace

void Encoder::AverageExtraLuma(int sub_w, int sub_h, int16_t* out) {
  // out[] points to four 8x8 blocks. When one of these blocks is totally
  // outside of the frame, we set it flat to the average value of the previous
  // block ("DC"), in order to help compressibility.
  int DC = GetAverage(out);
  if (sub_w <= 8) {   // set block #1 to block #0's average value
    SetAverage(DC, out + 1 * 64);
  }
  if (sub_h <= 8) {   // Need to flatten block #2 and #3
    if (sub_w > 8) {  // block #1 was not flattened, so get its real DC
      DC = GetAverage(out + 1 * 64);
    }
    SetAverage(DC, out + 2 * 64);
    SetAverage(DC, out + 3 * 64);
  } else if (sub_w <= 8) {   // set block #3 to the block #2's average value
    DC = GetAverage(out + 2 * 64);
    SetAverage(DC, out + 3 * 64);
  }
}

////////////////////////////////////////////////////////////////////////////////

const uint8_t* Encoder::GetReplicatedSamples(const uint8_t* rgb,
                                             int rgb_step,
                                             int sub_w, int sub_h,
                                             int w, int h) {
  Replicate8b(rgb, rgb_step, replicated_buffer_, pix_step_ * w, sub_w, sub_h, w,
              h, pix_step_);
  return replicated_buffer_;
}

const uint8_t* Encoder::GetReplicatedYSamples(const uint8_t* in,
                                              int step, int sub_w, int sub_h) {
  Replicate8b(in, step, replicated_buffer_, 16, sub_w, sub_h, 16, 16, 1);
  return replicated_buffer_;
}

// useful common function. Declared in sjpegi.h: api.cc calls it too.
bool FinishEncoding(Encoder* const enc, const EncoderParam& param) {
  const bool ok = (enc != nullptr) &&
                  enc->Ok() &&
                  enc->InitFromParam(param) &&
                  enc->Encode();
  delete enc;
  return ok;
}

////////////////////////////////////////////////////////////////////////////////
// sub-class for YUV 4:2:0 version

class Encoder420 final : public Encoder {
 public:
  Encoder420(int W, int H, const uint8_t* const rgb, int step,
             ByteSink* const sink, PixelFormat fmt = kRGBInput,
             MemoryManager* const memory = nullptr)
      : Encoder(SJPEG_YUV_420, W, H, sink, memory), rgb_(rgb), step_(step) {
    ok_ = (rgb != nullptr);
    if (fmt != kRGBInput) {
      pix_step_ = 4;
      get_yuv_block_ = GetBlockFunc(yuv_mode_, fmt);
    }
  }
  ~Encoder420() override {}
  void GetSamples(int mb_x, int mb_y, bool clipped, int16_t* out) override {
    const uint8_t* rgb = rgb_ + (pix_step_ * mb_x + mb_y * step_) * 16;
    int step = step_;
    if (clipped) {
      rgb = GetReplicatedSamples(rgb, step,
                                 W_ - mb_x * 16, H_ - mb_y * 16, 16, 16);
      step = pix_step_ * 16;
    }
    get_yuv_block_(rgb, step, out);
    if (clipped) {
      AverageExtraLuma(W_ - mb_x * 16, H_ - mb_y * 16, out);
    }
  }

 protected:
  const uint8_t* const rgb_;   // input samples
  int step_;
};

////////////////////////////////////////////////////////////////////////////////
// sub-class for YUV 4:4:4 version

class Encoder444 final : public Encoder {
 public:
  Encoder444(int W, int H, const uint8_t* const rgb, int step,
             ByteSink* const sink, PixelFormat fmt = kRGBInput,
             MemoryManager* const memory = nullptr)
      : Encoder(SJPEG_YUV_444, W, H, sink, memory), rgb_(rgb), step_(step) {
    ok_ = (rgb != nullptr);
    if (fmt != kRGBInput) {
      pix_step_ = 4;
      get_yuv_block_ = GetBlockFunc(yuv_mode_, fmt);
    }
  }
  ~Encoder444() override {}

  void GetSamples(int mb_x, int mb_y, bool clipped, int16_t* out) override {
    const uint8_t* rgb = rgb_ + (pix_step_ * mb_x + mb_y * step_) * 8;
    int step = step_;
    if (clipped) {
      rgb = GetReplicatedSamples(rgb, step,
                                 W_ - mb_x * 8, H_ - mb_y * 8, 8, 8);
      step = pix_step_ * 8;
    }
    get_yuv_block_(rgb, step, out);
  }

 protected:
  const uint8_t* const rgb_;   // input samples
  int step_;
};

////////////////////////////////////////////////////////////////////////////////
// sub-class for YUV 4:0:0 version

class Encoder400 final : public Encoder {
 public:
  Encoder400(int W, int H, const uint8_t* const src, int step,
             ByteSink* const sink, PixelFormat fmt = kRGBInput,
             MemoryManager* const memory = nullptr)
      : Encoder(SJPEG_YUV_400, W, H, sink, memory), rgb_(src), step_(step) {
    ok_ = (src != nullptr);
    if (fmt != kRGBInput) {
      pix_step_ = 4;
      get_yuv_block_ = GetBlockFunc(yuv_mode_, fmt);
    }
  }
  ~Encoder400() override {}

  void GetSamples(int mb_x, int mb_y, bool clipped, int16_t* out) override {
    const uint8_t* rgb = rgb_ + (pix_step_ * mb_x + mb_y * step_) * 8;
    int step = step_;
    if (clipped) {
      rgb = GetReplicatedSamples(rgb, step_,
                                 W_ - mb_x * 8, H_ - mb_y * 8, 8, 8);
      step = pix_step_ * 8;
    }
    get_yuv_block_(rgb, step, out);
  }

 protected:
  const uint8_t* const rgb_;   // input samples
  int step_;
};

// This variant takes luma plane as input directly.
class Encoder400G final : public Encoder {
 public:
  Encoder400G(int W, int H, const uint8_t* const gray, int step,
              ByteSink* const sink, MemoryManager* const memory = nullptr)
      : Encoder(SJPEG_YUV_400, W, H, sink, memory),
        gray_(gray), step_(step) {}
  ~Encoder400G() override {}

  void GetSamples(int mb_x, int mb_y, bool clipped, int16_t* out) override {
    const uint8_t* data = gray_ + (mb_x + mb_y * step_) * 8;
    if (clipped) {
      Convert8To16bClipped(data, step_, out, W_ - mb_x * 8, H_ - mb_y * 8);
    } else {
      Convert8To16b(data, step_, out);
    }
  }

 protected:
  const uint8_t* const gray_;   // input samples
  int step_;
};

////////////////////////////////////////////////////////////////////////////////
// Ad-hoc functions for NV21/NV12

class EncoderNV12 final : public Encoder {
 public:
  EncoderNV12(const uint8_t* y, int y_step, const uint8_t* uv, int uv_step,
              int W, int H, ByteSink* const sink, bool is_nv12,
              MemoryManager* const memory = nullptr)
      : Encoder(SJPEG_YUV_420, W, H, sink, memory),
        y_(y), y_step_(y_step), uv_(uv), uv_step_(uv_step), is_nv12_(is_nv12) {
    assert(sink != nullptr);
  }

  void GetSamples(int mb_x, int mb_y, bool clipped, int16_t* out) override {
    GetYSamples(mb_x, mb_y, clipped, out);
    GetUVSamples(mb_x, mb_y, clipped, out + 4 * 64, out + 5 * 64);
  }

 protected:
  void GetYSamples(int mb_x, int mb_y, bool clipped, int16_t* out) {
    const uint8_t* Y1 = y_ + (mb_x + mb_y * y_step_) * 16;
    int y_step = y_step_;
    if (clipped) {
      Y1 = GetReplicatedYSamples(Y1, y_step, W_ - mb_x * 16, H_ - mb_y * 16);
      y_step = 16;
    }
    const uint8_t* Y2 = Y1 + 8 * y_step;
    Convert8To16b(Y1 + 0, y_step, out + 0 * 64);
    Convert8To16b(Y1 + 8, y_step, out + 1 * 64);
    Convert8To16b(Y2 + 0, y_step, out + 2 * 64);
    Convert8To16b(Y2 + 8, y_step, out + 3 * 64);
    if (clipped) {
      AverageExtraLuma(W_ - mb_x * 16, H_ - mb_y * 16, out);
    }
  }
  void GetUVSamples(int mb_x, int mb_y, bool clipped,
                    int16_t* const U, int16_t* const V) {
    const uint8_t* UV = uv_ + (2 * mb_x + mb_y * uv_step_) * 8;
    int uv_step = uv_step_;
    uint8_t tmp_uv[2 * 8 * 8];
    if (clipped) {
      const int uv_w = ((W_ + 1) >> 1) - mb_x * 8;
      const int uv_h = ((H_ + 1) >> 1) - mb_y * 8;
      Replicate8b(UV, uv_step_, tmp_uv, 16, uv_w, uv_h, 8, 8, 2);
      UV = tmp_uv;
      uv_step = 16;
    }
    // input samples are U/V/U/V/... for NV12 and V/U/V/U... for NV21
    const uint8_t* u = &UV[is_nv12_ ? 0 : 1];
    const uint8_t* v = &UV[is_nv12_ ? 1 : 0];
    for (int y = 0; y < 8; ++y) {
      for (int x = 0; x < 8; ++x) {
        U[x + y * 8] = (int16_t)u[2 * x] - 128;
        V[x + y * 8] = (int16_t)v[2 * x] - 128;
      }
      u += uv_step;
      v += uv_step;
    }
  }

 private:
  const uint8_t* y_;
  int y_step_;
  const uint8_t* uv_;
  int uv_step_;
  bool is_nv12_;
};

// Common implementation for NV12 (U/V/U/V...) and NV21 (V/U/V/U...).
// Arguments are checked here: the base class uses 'output' at construction.
static bool EncodeNV(const uint8_t* y, int y_stride,
                     const uint8_t* uv, int uv_stride,
                     int width, int height, bool is_nv12,
                     const EncoderParam& param, ByteSink* output) {
  if (y == nullptr || uv == nullptr || output == nullptr) return false;
  if (width <= 0 || height <= 0) return false;
  if (std::abs(y_stride) < width) return false;
  if (std::abs(uv_stride) < 2 * ((width + 1) / 2)) return false;
  Encoder* const enc =
      new (std::nothrow) EncoderNV12(y, y_stride, uv, uv_stride,
                                     width, height, output, is_nv12,
                                     param.memory);
  return FinishEncoding(enc, param);
}

// Encode from NV12 samples, using YUV420 format
bool EncodeNV12(const uint8_t* y, int y_stride,
                const uint8_t* uv, int uv_stride,
                int width, int height,
                const EncoderParam& param, ByteSink* output) {
  return EncodeNV(y, y_stride, uv, uv_stride, width, height, true,
                  param, output);
}

// Encode from NV21 samples, using YUV420 format
bool EncodeNV21(const uint8_t* y, int y_stride,
                const uint8_t* vu, int vu_stride,
                int width, int height,
                const EncoderParam& param, ByteSink* output) {
  return EncodeNV(y, y_stride, vu, vu_stride, width, height, false,
                  param, output);
}

////////////////////////////////////////////////////////////////////////////////
// Direct YUV444 encoder

class EncoderYUV444 final : public Encoder {
 public:
  EncoderYUV444(const uint8_t* y, int y_step,
                const uint8_t* u, int u_step,
                const uint8_t* v, int v_step,
                int W, int H, ByteSink* const sink,
                MemoryManager* const memory = nullptr)
      : Encoder(SJPEG_YUV_444, W, H, sink, memory),
        y_(y), u_(u), v_(v), y_step_(y_step), u_step_(u_step), v_step_(v_step) {
    ok_ = (y_ != nullptr) && (u_ != nullptr) && (v_ != nullptr);
  }
  ~EncoderYUV444() override {}

  void GetSamples(int mb_x, int mb_y, bool clipped, int16_t* out) override {
    const uint8_t* const y = y_ + (mb_x + mb_y * y_step_) * 8;
    const uint8_t* const u = u_ + (mb_x + mb_y * u_step_) * 8;
    const uint8_t* const v = v_ + (mb_x + mb_y * v_step_) * 8;
    if (clipped) {
      const int sub_w = W_ - mb_x * 8;
      const int sub_h = H_ - mb_y * 8;
      Convert8To16bClipped(y, y_step_, out + 0 * 64, sub_w, sub_h);
      Convert8To16bClipped(u, u_step_, out + 1 * 64, sub_w, sub_h);
      Convert8To16bClipped(v, v_step_, out + 2 * 64, sub_w, sub_h);
    } else {
      Convert8To16b(y, y_step_, out + 0 * 64);
      Convert8To16b(u, u_step_, out + 1 * 64);
      Convert8To16b(v, v_step_, out + 2 * 64);
    }
  }

 private:
  const uint8_t* y_;
  const uint8_t* u_;
  const uint8_t* v_;
  int y_step_, u_step_, v_step_;
};

// Encode bitstream using Y/U/V input in YUV444 format.
bool EncodeYUV444(const uint8_t* Y, int Y_stride,
                  const uint8_t* U, int U_stride,
                  const uint8_t* V, int V_stride,
                  int width, int height,
                  const EncoderParam& param, ByteSink* output) {
  if (Y == nullptr || U == nullptr || V == nullptr) return false;
  if (output == nullptr) return false;
  if (width <= 0 || height <= 0) return false;
  if (std::abs(Y_stride) < width) return false;
  if (std::abs(U_stride) < width) return false;
  if (std::abs(V_stride) < width) return false;
  Encoder* const enc =
      new (std::nothrow) EncoderYUV444(Y, Y_stride, U, U_stride, V, V_stride,
                                       width, height, output, param.memory);
  return FinishEncoding(enc, param);
}

////////////////////////////////////////////////////////////////////////////////
// Direct YUV420 encoder

class EncoderYUV420 : public Encoder {
 public:
  EncoderYUV420(const uint8_t* y, int y_step,
                const uint8_t* u, int u_step,
                const uint8_t* v, int v_step,
                int W, int H, ByteSink* const sink,
                MemoryManager* const memory = nullptr)
      : Encoder(SJPEG_YUV_420, W, H, sink, memory),
        y_(y), u_(u), v_(v), y_step_(y_step), u_step_(u_step), v_step_(v_step) {
    ok_ = (y_ != nullptr) && (u_ != nullptr) && (v_ != nullptr);
  }
  ~EncoderYUV420() override {}

  void GetSamples(int mb_x, int mb_y, bool clipped, int16_t* out) override {
    // Luma
    const uint8_t* Y1 = y_ + (mb_x + mb_y * y_step_) * 16;
    int y_step = y_step_;
    if (clipped) {
      Y1 = GetReplicatedYSamples(Y1, y_step,  W_ - mb_x * 16, H_ - mb_y * 16);
      y_step = 16;
    }
    const uint8_t* const Y2 = Y1 + 8 * y_step;
    Convert8To16b(Y1 + 0, y_step, out + 0 * 64);
    Convert8To16b(Y1 + 8, y_step, out + 1 * 64);
    Convert8To16b(Y2 + 0, y_step, out + 2 * 64);
    Convert8To16b(Y2 + 8, y_step, out + 3 * 64);
    if (clipped) {
      AverageExtraLuma(W_ - mb_x * 16, H_ - mb_y * 16, out);
    }
    // U/V
    const uint8_t* U = u_ + (mb_x + mb_y * u_step_) * 8;
    const uint8_t* V = v_ + (mb_x + mb_y * v_step_) * 8;
    if (clipped) {
      const int sub_w = ((W_ + 1) >> 1) - mb_x * 8;
      const int sub_h = ((H_ + 1) >> 1) - mb_y * 8;
      Convert8To16bClipped(U, u_step_, out + 4 * 64, sub_w, sub_h);
      Convert8To16bClipped(V, v_step_, out + 5 * 64, sub_w, sub_h);
    } else {
      Convert8To16b(U, u_step_, out + 4 * 64);
      Convert8To16b(V, v_step_, out + 5 * 64);
    }
  }

 protected:
  const uint8_t* y_;
  const uint8_t* u_;
  const uint8_t* v_;
  int y_step_, u_step_, v_step_;
};

bool EncodeYUV420(const uint8_t* Y, int Y_stride,
                  const uint8_t* U, int U_stride,
                  const uint8_t* V, int V_stride,
                  int width, int height,
                  const EncoderParam& param, ByteSink* output) {
  if (Y == nullptr || U == nullptr || V == nullptr) return false;
  if (output == nullptr) return false;
  if (width <= 0 || height <= 0) return false;
  if (std::abs(Y_stride) < width) return false;
  if (std::abs(U_stride) < (width + 1) / 2) return false;
  if (std::abs(V_stride) < (width + 1) / 2) return false;
  Encoder* const enc =
      new (std::nothrow) EncoderYUV420(Y, Y_stride, U, U_stride, V, V_stride,
                                       width, height, output, param.memory);
  return FinishEncoding(enc, param);
}

////////////////////////////////////////////////////////////////////////////////
// sub-class for the sharp YUV 4:2:0 version

class EncoderSharp420 final : public EncoderYUV420 {
 public:
  EncoderSharp420(int W, int H, const uint8_t* const rgb, int step,
                  ByteSink* const sink, MemoryManager* const memory = nullptr)
      : EncoderYUV420(nullptr, 0, nullptr, 0, nullptr, 0, W, H, sink, memory),
        yuv_memory_(nullptr) {
    const int uv_w = (W + 1) >> 1;
    const int uv_h = (H + 1) >> 1;
    const size_t y_size = (size_t)W * H;
    const size_t uv_size = (size_t)uv_w * uv_h;
    yuv_memory_ = Alloc<uint8_t>(y_size + 2 * uv_size);
    ok_ = (yuv_memory_ != nullptr);
    if (ok_) {
      y_ = yuv_memory_;
      y_step_ = W;
      u_ = yuv_memory_ + y_size;
      v_ = u_ + uv_size;
      u_step_ = uv_w;
      v_step_ = uv_w;
      ApplySharpYUVConversion(rgb, W, H, step,
                              const_cast<uint8_t*>(y_),
                              const_cast<uint8_t*>(u_),
                              const_cast<uint8_t*>(v_));
    }
  }
  ~EncoderSharp420() override { Free(yuv_memory_); }

 protected:
  uint8_t* yuv_memory_;
};

////////////////////////////////////////////////////////////////////////////////
// all-in-one factory to pickup the right encoder instance

Encoder* EncoderFactory(const uint8_t* rgb, int W, int H, int stride,
                        SjpegYUVMode yuv_mode, ByteSink* const sink,
                        PixelFormat fmt, MemoryManager* const memory) {
  if (yuv_mode == SJPEG_YUV_AUTO) {
    yuv_mode = SjpegRiskiness(rgb, W, H, stride, nullptr);
  }

  Encoder* enc = nullptr;
  if (yuv_mode == SJPEG_YUV_420) {
    enc = new (std::nothrow) Encoder420(W, H, rgb, stride, sink, fmt, memory);
  } else if (yuv_mode == SJPEG_YUV_SHARP) {
    enc = new (std::nothrow) EncoderSharp420(W, H, rgb, stride, sink, memory);
  } else if (yuv_mode == SJPEG_YUV_444) {
    enc = new (std::nothrow) Encoder444(W, H, rgb, stride, sink, fmt, memory);
  } else if (yuv_mode == SJPEG_YUV_400) {
    enc = new (std::nothrow) Encoder400(W, H, rgb, stride, sink, fmt, memory);
  }
  if (enc == nullptr || !enc->Ok()) {
    delete enc;
    enc = nullptr;
  }
  return enc;
}

Encoder* GrayEncoderFactory(const uint8_t* gray, int W, int H, int stride,
                            ByteSink* const sink,
                            MemoryManager* const memory) {
  return new (std::nothrow) Encoder400G(W, H, gray, stride, sink, memory);
}

}    // namespace sjpeg

