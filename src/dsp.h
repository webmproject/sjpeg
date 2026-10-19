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
//  DSP: transforms, color conversion, scoring and quantization helpers.
//  Internal header: not installed.
//
// Author: Skal (pascal.massimino@gmail.com)

#ifndef SJPEG_DSP_H_
#define SJPEG_DSP_H_

#include <stdint.h>

#include "common.h"
#include "sjpeg.h"

namespace sjpeg {

// Forward 8x8 Fourier transforms, in-place.
typedef void (*FdctFunc)(int16_t* coeffs, int num_blocks);
FdctFunc GetFdct();

// these are the default luma/chroma matrices (JPEG spec section K.1)
extern const uint8_t kDefaultMatrices[2][64];
extern const uint8_t kZigzag[64];
extern const uint8_t kInvZigzag[64];

// riskiness index = y + 7 * u + 35 * v: 7 luma levels, 5 chroma levels
// (7 levels with the two outer ones merged on each side)
static const int kRiskYLevels = 7;
static const int kRiskUVLevels = 5;
static const int kRiskColors = kRiskYLevels * kRiskUVLevels * kRiskUVLevels;
// first gray index (u = v = 2), followed by the other luma levels
static const int kRiskGrayMin = 2 * kRiskYLevels * (1 + kRiskUVLevels);

// pair scores, kRiskColors x kRiskColors, in score_175.cc
extern const uint8_t kSharpnessScore[];

// internal riskiness scoring functions:
extern double DCTRiskinessScore(const int16_t yuv[3 * 64],
                                int16_t scores[8 * 8]);
extern double BlockRiskinessScore(const uint8_t* rgb, int stride,
                                  int16_t scores[8 * 8]);
enum class BlockActivityTier;  // see definition below
extern BlockActivityTier BlockActivityScore(const uint8_t* rgb, int stride,
                                            uint32_t* activity = nullptr);
extern int YUVToRiskIdx(int16_t y, int16_t u, int16_t v);
typedef void (*RiskinessScoreRowFunc)(const uint16_t* row1,
                                      const uint16_t* row2, int size,
                                      int noise_level, int64_t* score_sum,
                                      int64_t* score_num, int64_t* gray_num);
extern RiskinessScoreRowFunc GetRiskinessScoreRowFunc();
// scalar C reference, also used for the remainder of the SIMD variants
extern void RiskinessScoreRow_C(const uint16_t* row1, const uint16_t* row2,
                                int size, int noise_level, int64_t* score_sum,
                                int64_t* score_num, int64_t* gray_num);

///////////////////////////////////////////////////////////////////////////////
// RGB->YUV conversion

// Internal pixel format for the input samples fed to the block functions.
enum PixelFormat { kRGBInput = 0, kBGRAInput = 1, kRGBAInput = 2 };

// convert 16x16 RGB block into YUV420, or 8x8 RGB block into YUV444 or YUV400
typedef void (*RGBToYUVBlockFunc)(const uint8_t* src, int src_stride,
                                  int16_t* blocks);
extern RGBToYUVBlockFunc GetBlockFunc(SjpegYUVMode mode,
                                      PixelFormat fmt = kRGBInput);

// convert a row of RGB samples to YUV444
typedef void (*RGBToIndexRowFunc)(const uint8_t* src, int width, uint16_t* dst);
extern RGBToIndexRowFunc GetRowFunc();

class Encoder;

// Enhanced slower RGB->YUV conversion:
//  y_plane[] has dimension W x H, whereas u_plane[] and v_plane[] have
//  dimension (W + 1)/2 x (H + 1)/2.
bool ApplySharpYUVConversion(const uint8_t* rgb, int W, int H, int stride,
                             uint8_t* y_plane, uint8_t* u_plane,
                             uint8_t* v_plane,
                             const Encoder* encoder = nullptr);

// Shared by yuv_convert.cc and yuv_convert_avx2.cc.
typedef int16_t fixed_t;
typedef uint16_t fixed_y_t;

// Import one RGB row into planar fixed_y_t format starting at column start_x.
void ImportOneRow_C(const uint8_t* rgb, int start_x, int pic_width,
                    fixed_y_t* dst);
void ImportOneRow_C(const uint8_t* rgb, int pic_width, fixed_y_t* dst);

///////////////////////////////////////////////////////////////////////////////
// Generic sample-replication function. Replicate sub_w x sub_h area of 'src'
// into 'dst', assuming the individual samples are 'x_step' bytes each.
void Replicate8b(const uint8_t* src, int src_stride, uint8_t* dst,
                 int dst_stride, int sub_w, int sub_h, int w, int h,
                 int x_step);

// This variant will replicate src[] into a 16b output dst[], subtracting 128.
// This function operates on a 8x8 block only.
void Convert8To16bClipped(const uint8_t* src, int src_step, int16_t dst[8 * 8],
                          int sub_w, int sub_h);

// Convert a 8x8 block of unsigned-8b values to signed-16b, subtracting 128.
void Convert8To16b(const uint8_t* src, int src_step, int16_t dst[8 * 8]);

///////////////////////////////////////////////////////////////////////////////
// some useful helper functions around quant matrices

extern float GetQFactor(float q);  // convert quality factor -> scale factor
extern void CopyQuantMatrix(const uint8_t in[64], uint8_t out[64]);
extern void SetQuantMatrix(const uint8_t in[64], float q_factor,
                           uint8_t out[64]);
extern void SetMinQuantMatrix(const uint8_t* m, uint8_t out[64], int tolerance);
extern void SetDefaultMinQuantMatrix(uint8_t out[64]);

// Block activity classification for perceptual adaptive-bias quantization.
enum class BlockActivityTier {
  kFlatBlock = -1,
  kNormalBlock = 0,
  kBusyBlock = 1
};
enum {
  kActivityLo = 28 << AC_BITS,
  kActivityHi = 360 << AC_BITS,
};

// Returns the flat/normal/busy classification from an activity score.
BlockActivityTier ClassifyActivity(uint32_t activity);

// Classify an 8x8 block from its DCT coefficients.
BlockActivityTier ClassifyBlockActivity(const int16_t in[64],
                                        uint32_t* score = nullptr);

}  // namespace sjpeg

#endif  // SJPEG_DSP_H_
