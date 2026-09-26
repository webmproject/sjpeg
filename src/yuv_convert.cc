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
//  Enhanced RGB->YUV conversion functions
//
// Author: Skal (pascal.massimino@gmail.com)

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <climits>
#include <functional>
#include <memory>
#include <mutex>  // NOLINT
#include <new>
#include <vector>
using std::vector;

#define SJPEG_NEED_ASM_HEADERS
#include "sjpegi.h"

namespace sjpeg {

// We could use SFIX=0 and only uint8_t for fixed_y_t, but it produces some
// banding sometimes. Better use extra precision.
#define SFIX 2                // fixed-point precision of RGB and Y/W
#define SHALF (1 << SFIX >> 1)
#define MAX_Y_T ((256 << SFIX) - 1)

static fixed_y_t clip_y(int y) {
  return (!(y & ~MAX_Y_T)) ? (fixed_y_t)y : (y < 0) ? 0 : MAX_Y_T;
}

////////////////////////////////////////////////////////////////////////////////
// Helper functions for Y/U/V fixed-point calculations.

// The following functions convert r/g/b values in SFIX fixed-point precision
// to 8b values, clipped:
#define YUV_FIX 16
#define TFIX (YUV_FIX + SFIX)
#define TROUNDER (1 << TFIX >> 1)

static uint8_t clip_8b(int v) {
  return (!(v & ~0xff)) ? (uint8_t)v : (v < 0) ? 0u : 255u;
}

static uint8_t ConvertRGBToY(int r, int g, int b) {
  const int luma = 19595 * r + 38469 * g + 7471 * b + TROUNDER;
  return clip_8b(luma >> TFIX);
}

static uint8_t ConvertRGBToU(int r, int g, int b) {
  const int u =  -11058 * r - 21709 * g + 32768 * b + TROUNDER;
  return clip_8b(128 + (u >> TFIX));
}

static uint8_t ConvertRGBToV(int r, int g, int b) {
  const int v = +32768 * r - 27439 * g - 5328 * b + TROUNDER;
  return clip_8b(128 + (v >> TFIX));
}

// convert to luma using 16b precision:
static void ConvertRowToY(const uint8_t* row, int w, uint8_t* const dst) {
  for (int i = 0; i < w; i += 1, row += 3) {
    const int r = row[0], g = row[1], b = row[2];
    const int y = 19595 * r + 38469 * g + 7471 * b;
    dst[i] = (y + (1 << YUV_FIX >> 1)) >> YUV_FIX;
  }
}

static void ConvertRowToUV(const uint8_t* row1, const uint8_t* row2,
                           int w, uint8_t* u, uint8_t* v) {
  for (int i = 0; i < (w & ~1); i += 2, row1 += 6, row2 += 6) {
    const int r = row1[0] + row1[3] + row2[0] + row2[3];
    const int g = row1[1] + row1[4] + row2[1] + row2[4];
    const int b = row1[2] + row1[5] + row2[2] + row2[5];
    *u++ = ConvertRGBToU(r, g, b);
    *v++ = ConvertRGBToV(r, g, b);
  }
  if (w & 1) {
    const int r = 2 * (row1[0] + row2[0]);
    const int g = 2 * (row1[1] + row2[1]);
    const int b = 2 * (row1[2] + row2[2]);
    *u++ = ConvertRGBToU(r, g, b);
    *v++ = ConvertRGBToV(r, g, b);
  }
}

#undef TFIX
#undef ROUNDER

////////////////////////////////////////////////////////////////////////////////
// Sharp RGB->YUV conversion

static const int kNumIterations = 4;
static const int kMinDimensionIterativeConversion = 4;

// size of the interpolation table for linear-to-gamma
#define GAMMA_TABLE_SIZE 32
uint32_t kLinearToGammaTab[GAMMA_TABLE_SIZE + 2];
alignas(32) uint32_t kPackedLinearToGammaTab[GAMMA_TABLE_SIZE + 2];
#define GAMMA_TO_LINEAR_BITS 14
uint32_t kGammaToLinearTab[MAX_Y_T + 1];   // size scales with Y_FIX

static void InitGammaTablesF() {
  static std::once_flag once;
  assert(2 * GAMMA_TO_LINEAR_BITS < 32);  // we use uint32_t intermediate values
  std::call_once(once, []() {
    int v;
    const double norm = 1. / MAX_Y_T;
    const double scale = 1. / GAMMA_TABLE_SIZE;
    const double a = 0.099;
    const double thresh = 0.018;
    const double gamma = 1. / 0.45;
    const double final_scale = 1 << GAMMA_TO_LINEAR_BITS;
    for (v = 0; v <= MAX_Y_T; ++v) {
      const double g = norm * v;
      double value;
      if (g <= thresh * 4.5) {
        value = g / 4.5;
      } else {
        const double a_rec = 1. / (1. + a);
        value = pow(a_rec * (g + a), gamma);
      }
      kGammaToLinearTab[v] = (uint32_t)(value * final_scale + .5);
    }
    for (v = 0; v <= GAMMA_TABLE_SIZE; ++v) {
      const double g = scale * v;
      double value;
      if (g <= thresh) {
        value = 4.5 * g;
      } else {
        value = (1. + a) * pow(g, 1. / gamma) - a;
      }
      kLinearToGammaTab[v] = (uint32_t)(MAX_Y_T * value);
    }
    // to prevent small rounding errors to cause read-overflow:
    kLinearToGammaTab[GAMMA_TABLE_SIZE + 1] =
        kLinearToGammaTab[GAMMA_TABLE_SIZE];
    for (v = 0; v <= GAMMA_TABLE_SIZE; ++v) {
      const uint32_t diff = kLinearToGammaTab[v + 1] - kLinearToGammaTab[v];
      kPackedLinearToGammaTab[v] =
          (diff << 16) | (kLinearToGammaTab[v] & 0xffff);
    }
    kPackedLinearToGammaTab[GAMMA_TABLE_SIZE + 1] =
        kPackedLinearToGammaTab[GAMMA_TABLE_SIZE];
  });
}

// return value has a fixed-point precision of GAMMA_TO_LINEAR_BITS
// (used by yuv_convert_avx2.cc too)
uint32_t GammaToLinear(int v) { return kGammaToLinearTab[v]; }

// return value is in [0, MAX_Y_T]
uint32_t LinearToGamma(uint32_t value) {
  // 'value' is in GAMMA_TO_LINEAR_BITS fractional precision
  const uint32_t v = value * GAMMA_TABLE_SIZE;
  const uint32_t tab_pos = v >> GAMMA_TO_LINEAR_BITS;
  // fractional part, in GAMMA_TO_LINEAR_BITS fixed-point precision
  const uint32_t x = v - (tab_pos << GAMMA_TO_LINEAR_BITS);  // fractional part
  // v0 / v1 are in GAMMA_TO_LINEAR_BITS fixed-point precision (range [0..1])
  const uint32_t v0 = kLinearToGammaTab[tab_pos + 0];
  const uint32_t v1 = kLinearToGammaTab[tab_pos + 1];
  // Final interpolation.
  const uint32_t v2 = (v1 - v0) * x;    // note: v1 >= v0.
  const uint32_t result = v0 + (v2 >> GAMMA_TO_LINEAR_BITS);
  return result;
}

//------------------------------------------------------------------------------

uint32_t RGBToGray(uint32_t r, uint32_t g, uint32_t b) {
  const uint32_t luma = 13933 * r + 46871 * g + 4732 * b + (1u << YUV_FIX >> 1);
  return (luma >> YUV_FIX);
}

static fixed_y_t UpLift(uint8_t a) {  // 8bit -> SFIX
  return ((fixed_y_t)a << SFIX) | SHALF;
}

static void StoreGray_C(const fixed_y_t* const rgb, fixed_y_t* const y, int w) {
  for (int i = 0; i < w; ++i) {
    y[i] = RGBToGray(rgb[0 * w + i], rgb[1 * w + i], rgb[2 * w + i]);
  }
}

void ImportOneRow_C(const uint8_t* const rgb, int start_x, int pic_width,
                    fixed_y_t* const dst) {
  const int w = (pic_width + 1) & ~1;
  for (int i = start_x; i < pic_width; ++i) {
    const int off = i * 3;
    dst[i + 0 * w] = UpLift(rgb[off + 0]);
    dst[i + 1 * w] = UpLift(rgb[off + 1]);
    dst[i + 2 * w] = UpLift(rgb[off + 2]);
  }
  if (pic_width & 1) {  // replicate rightmost pixel
    dst[pic_width + 0 * w] = dst[pic_width + 0 * w - 1];
    dst[pic_width + 1 * w] = dst[pic_width + 1 * w - 1];
    dst[pic_width + 2 * w] = dst[pic_width + 2 * w - 1];
  }
}

void ImportOneRow_C(const uint8_t* const rgb, int pic_width,
                    fixed_y_t* const dst) {
  ImportOneRow_C(rgb, 0, pic_width, dst);
}

//------------------------------------------------------------------------------

static uint64_t SharpUpdateY_C(const uint16_t* ref, const uint16_t* src,
                               uint16_t* dst, int len) {
  uint64_t diff = 0;
  for (int i = 0; i < len; ++i) {
    const int diff_y = ref[i] - src[i];
    const int new_y = (int)dst[i] + diff_y;
    dst[i] = clip_y(new_y);
    diff += (uint64_t)abs(diff_y);
  }
  return diff;
}

static void SharpUpdateRGB_C(const int16_t* ref, const int16_t* src,
                             int16_t* dst, int len) {
  for (int i = 0; i < len; ++i) {
    const int diff_uv = ref[i] - src[i];
    dst[i] += diff_uv;
  }
}

static void SharpFilterRow_C(const int16_t* A, const int16_t* B, int len,
                             const uint16_t* best_y, uint16_t* out) {
  for (int i = 0; i < len; ++i, ++A, ++B) {
    const int v0 = (A[0] * 9 + A[1] * 3 + B[0] * 3 + B[1] + 8) >> 4;
    const int v1 = (A[1] * 9 + A[0] * 3 + B[1] * 3 + B[0] + 8) >> 4;
    out[2 * i + 0] = clip_y(best_y[2 * i + 0] + v0);
    out[2 * i + 1] = clip_y(best_y[2 * i + 1] + v1);
  }
}

#if defined(SJPEG_HAVE_AVX2)
uint64_t SharpUpdateY_AVX2(const uint16_t* ref, const uint16_t* src,
                           uint16_t* dst, int len);
void SharpUpdateRGB_AVX2(const int16_t* ref, const int16_t* src,
                         int16_t* dst, int len);
void SharpFilterRow_AVX2(const int16_t* A, const int16_t* B, int len,
                         const uint16_t* best_y, uint16_t* out);
void StoreGray_AVX2(const fixed_y_t* const rgb, fixed_y_t* const y, int w);
void ImportOneRow_AVX2(const uint8_t* const rgb, int pic_width,
                       fixed_y_t* const dst);
void UpdateW_AVX2(const fixed_y_t* src, fixed_y_t* dst, int w);
void UpdateChroma_AVX2(const fixed_y_t* src1, const fixed_y_t* src2,
                       fixed_t* dst, size_t uv_w);
#endif

#if defined(SJPEG_USE_SSE2)

static uint64_t SharpUpdateY_SSE2(const uint16_t* ref, const uint16_t* src,
                                  uint16_t* dst, int len) {
  uint64_t diff = 0;
  uint32_t tmp[4];
  int i;
  const __m128i zero = _mm_setzero_si128();
  const __m128i max = _mm_set1_epi16(MAX_Y_T);
  const __m128i one = _mm_set1_epi16(1);
  __m128i sum = zero;

  for (i = 0; i + 8 <= len; i += 8) {
    const __m128i A = LOAD_16(ref + i);
    const __m128i B = LOAD_16(src + i);
    const __m128i C = LOAD_16(dst + i);
    const __m128i D = _mm_sub_epi16(A, B);       // diff_y
    const __m128i E = _mm_cmpgt_epi16(zero, D);  // sign (-1 or 0)
    const __m128i F = _mm_add_epi16(C, D);       // new_y
    const __m128i G = _mm_or_si128(E, one);      // -1 or 1
    const __m128i H = CLAMP_16(F, zero, max);
    const __m128i I = _mm_madd_epi16(D, G);      // sum(abs(...))
    STORE_16(H, dst + i);
    sum = _mm_add_epi32(sum, I);
  }
  STORE_16(sum, tmp);
  diff = tmp[3] + tmp[2] + tmp[1] + tmp[0];
  for (; i < len; ++i) {
    const int diff_y = ref[i] - src[i];
    const int new_y = (int)dst[i] + diff_y;
    dst[i] = clip_y(new_y);
    diff += (uint64_t)abs(diff_y);
  }
  return diff;
}

static void SharpUpdateRGB_SSE2(const int16_t* ref, const int16_t* src,
                                int16_t* dst, int len) {
  int i = 0;
  for (i = 0; i + 8 <= len; i += 8) {
    const __m128i A = LOAD_16(ref + i);
    const __m128i B = LOAD_16(src + i);
    const __m128i C = LOAD_16(dst + i);
    const __m128i D = _mm_sub_epi16(A, B);   // diff_uv
    const __m128i E = _mm_add_epi16(C, D);   // new_uv
    STORE_16(E, dst + i);
  }
  for (; i < len; ++i) {
    const int diff_uv = ref[i] - src[i];
    dst[i] += diff_uv;
  }
}

static void SharpFilterRow_SSE2(const int16_t* A, const int16_t* B, int len,
                                const uint16_t* best_y, uint16_t* out) {
  int i;
  const __m128i kCst8 = _mm_set1_epi16(8);
  const __m128i max = _mm_set1_epi16(MAX_Y_T);
  const __m128i zero = _mm_setzero_si128();
  for (i = 0; i + 8 <= len; i += 8) {
    const __m128i a0 = LOAD_16(A + i + 0);
    const __m128i a1 = LOAD_16(A + i + 1);
    const __m128i b0 = LOAD_16(B + i + 0);
    const __m128i b1 = LOAD_16(B + i + 1);
    const __m128i a0b1 = _mm_add_epi16(a0, b1);
    const __m128i a1b0 = _mm_add_epi16(a1, b0);
    const __m128i a0a1b0b1 = _mm_add_epi16(a0b1, a1b0);  // A0+A1+B0+B1
    const __m128i a0a1b0b1_8 = _mm_add_epi16(a0a1b0b1, kCst8);
    const __m128i a0b1_2 = _mm_add_epi16(a0b1, a0b1);    // 2*(A0+B1)
    const __m128i a1b0_2 = _mm_add_epi16(a1b0, a1b0);    // 2*(A1+B0)
    const __m128i c0 = _mm_srai_epi16(_mm_add_epi16(a0b1_2, a0a1b0b1_8), 3);
    const __m128i c1 = _mm_srai_epi16(_mm_add_epi16(a1b0_2, a0a1b0b1_8), 3);
    const __m128i d0 = _mm_add_epi16(c1, a0);
    const __m128i d1 = _mm_add_epi16(c0, a1);
    const __m128i e0 = _mm_srai_epi16(d0, 1);
    const __m128i e1 = _mm_srai_epi16(d1, 1);
    const __m128i f0 = _mm_unpacklo_epi16(e0, e1);
    const __m128i f1 = _mm_unpackhi_epi16(e0, e1);
    const __m128i g0 = LOAD_16(best_y + 2 * i + 0);
    const __m128i g1 = LOAD_16(best_y + 2 * i + 8);
    const __m128i h0 = _mm_add_epi16(g0, f0);
    const __m128i h1 = _mm_add_epi16(g1, f1);
    const __m128i i0 = CLAMP_16(h0, zero, max);
    const __m128i i1 = CLAMP_16(h1, zero, max);
    STORE_16(i0, out + 2 * i + 0);
    STORE_16(i1, out + 2 * i + 8);
  }
  for (; i < len; ++i) {
    //   (9 * A0 + 3 * A1 + 3 * B0 + B1 + 8) >> 4 =
    // = (8 * A0 + 2 * (A1 + B0) + (A0 + A1 + B0 + B1 + 8)) >> 4
    // We reuse the common sub-expressions.
    const int a0b1 = A[i + 0] + B[i + 1];
    const int a1b0 = A[i + 1] + B[i + 0];
    const int a0a1b0b1 = a0b1 + a1b0 + 8;
    const int v0 = (8 * A[i + 0] + 2 * a1b0 + a0a1b0b1) >> 4;
    const int v1 = (8 * A[i + 1] + 2 * a0b1 + a0a1b0b1) >> 4;
    out[2 * i + 0] = clip_y(best_y[2 * i + 0] + v0);
    out[2 * i + 1] = clip_y(best_y[2 * i + 1] + v1);
  }
}

#elif defined(SJPEG_USE_NEON)

static uint64_t SharpUpdateY_NEON(const uint16_t* ref, const uint16_t* src,
                                  uint16_t* dst, int len) {
  int i;
  const int16x8_t zero = vdupq_n_s16(0);
  const int16x8_t max = vdupq_n_s16(MAX_Y_T);
  uint64x2_t sum = vdupq_n_u64(0);

  for (i = 0; i + 8 <= len; i += 8) {
    const int16x8_t A = LOAD_16(ref + i);
    const int16x8_t B = LOAD_16(src + i);
    const int16x8_t C = LOAD_16(dst + i);
    const int16x8_t D = vsubq_s16(A, B);       // diff_y
    const int16x8_t F = vaddq_s16(C, D);       // new_y
    const int16x8_t H = CLAMP_16(F, zero, max);
    const int16x8_t I = ABS_16(D);  // abs(diff_y)
    STORE_16(H, dst + i);
    sum = vpadalq_u32(sum, vpaddlq_u16(vreinterpretq_u16_s16(I)));
  }
  uint64_t diff = vgetq_lane_u64(sum, 0) + vgetq_lane_u64(sum, 1);
  for (; i < len; ++i) {
    const int diff_y = ref[i] - src[i];
    const int new_y = (int)dst[i] + diff_y;
    dst[i] = clip_y(new_y);
    diff += (uint64_t)abs(diff_y);
  }
  return diff;
}

static void SharpUpdateRGB_NEON(const int16_t* ref, const int16_t* src,
                                int16_t* dst, int len) {
  int i;
  for (i = 0; i + 8 <= len; i += 8) {
    const int16x8_t A = LOAD_16(ref + i);
    const int16x8_t B = LOAD_16(src + i);
    const int16x8_t C = LOAD_16(dst + i);
    const int16x8_t D = vsubq_s16(A, B);   // diff_uv
    const int16x8_t E = vaddq_s16(C, D);   // new_uv
    STORE_16(E, dst + i);
  }
  for (; i < len; ++i) {
    const int diff_uv = ref[i] - src[i];
    dst[i] += diff_uv;
  }
}

static void SharpFilterRow_NEON(const int16_t* A, const int16_t* B, int len,
                                const uint16_t* best_y, uint16_t* out) {
  int i;
  const int16x8_t max = vdupq_n_s16(MAX_Y_T);
  const int16x8_t zero = vdupq_n_s16(0);
  for (i = 0; i + 8 <= len; i += 8) {
    const int16x8_t a0 = LOAD_16(A + i + 0);
    const int16x8_t a1 = LOAD_16(A + i + 1);
    const int16x8_t b0 = LOAD_16(B + i + 0);
    const int16x8_t b1 = LOAD_16(B + i + 1);
    const int16x8_t a0b1 = vaddq_s16(a0, b1);
    const int16x8_t a1b0 = vaddq_s16(a1, b0);
    const int16x8_t a0a1b0b1 = vaddq_s16(a0b1, a1b0);  // A0+A1+B0+B1
    const int16x8_t a0b1_2 = vaddq_s16(a0b1, a0b1);    // 2*(A0+B1)
    const int16x8_t a1b0_2 = vaddq_s16(a1b0, a1b0);    // 2*(A1+B0)
    const int16x8_t c0 = vshrq_n_s16(vaddq_s16(a0b1_2, a0a1b0b1), 3);
    const int16x8_t c1 = vshrq_n_s16(vaddq_s16(a1b0_2, a0a1b0b1), 3);
    const int16x8_t d0 = vaddq_s16(c1, a0);
    const int16x8_t d1 = vaddq_s16(c0, a1);
    const int16x8_t e0 = vrshrq_n_s16(d0, 1);
    const int16x8_t e1 = vrshrq_n_s16(d1, 1);
    const int16x8x2_t f = vzipq_s16(e0, e1);
    const int16x8_t g0 = LOAD_16(best_y + 2 * i + 0);
    const int16x8_t g1 = LOAD_16(best_y + 2 * i + 8);
    const int16x8_t h0 = vaddq_s16(g0, f.val[0]);
    const int16x8_t h1 = vaddq_s16(g1, f.val[1]);
    const int16x8_t i0 = CLAMP_16(h0, zero, max);
    const int16x8_t i1 = CLAMP_16(h1, zero, max);
    STORE_16(i0, out + 2 * i + 0);
    STORE_16(i1, out + 2 * i + 8);
  }
  for (; i < len; ++i) {
    const int a0b1 = A[i + 0] + B[i + 1];
    const int a1b0 = A[i + 1] + B[i + 0];
    const int a0a1b0b1 = a0b1 + a1b0 + 8;
    const int v0 = (8 * A[i + 0] + 2 * a1b0 + a0a1b0b1) >> 4;
    const int v1 = (8 * A[i + 1] + 2 * a0b1 + a0a1b0b1) >> 4;
    out[2 * i + 0] = clip_y(best_y[2 * i + 0] + v0);
    out[2 * i + 1] = clip_y(best_y[2 * i + 1] + v1);
  }
}

#endif    // SJPEG_USE_NEON

// forward decls for InitFunctionPointers() below
static void UpdateW(const fixed_y_t* src, fixed_y_t* dst, int w);
static void UpdateChroma(const fixed_y_t* src1, const fixed_y_t* src2,
                         fixed_t* dst, size_t uv_w);

static uint64_t (*kSharpUpdateY)(const uint16_t* src, const uint16_t* ref,
                                 uint16_t* dst, int len);
static void (*kSharpUpdateRGB)(const int16_t* src, const int16_t* ref,
                               int16_t* dst, int len);
static void (*kSharpFilterRow)(const int16_t* A, const int16_t* B,
                               int len, const uint16_t* best_y, uint16_t* out);
static void (*kUpdateW)(const fixed_y_t* src, fixed_y_t* dst, int w);
static void (*kUpdateChroma)(const fixed_y_t* src1, const fixed_y_t* src2,
                             fixed_t* dst, size_t uv_w);
static void (*kStoreGray)(const fixed_y_t* const rgb, fixed_y_t* const y,
                          int w);
static void (*kImportOneRow)(const uint8_t* const rgb, int pic_width,
                             fixed_y_t* const dst);

static void InitFunctionPointers() {
  static std::once_flag once;
  std::call_once(once, []() {
    kSharpUpdateY = SharpUpdateY_C;
    kSharpUpdateRGB = SharpUpdateRGB_C;
    kSharpFilterRow = SharpFilterRow_C;
    kStoreGray = StoreGray_C;
    kImportOneRow = ImportOneRow_C;
    kUpdateW = UpdateW;
    kUpdateChroma = UpdateChroma;
#if defined(SJPEG_HAVE_AVX2)
    if (sjpeg::SupportsAVX2()) {
      kSharpUpdateY = SharpUpdateY_AVX2;
      kSharpUpdateRGB = SharpUpdateRGB_AVX2;
      kSharpFilterRow = SharpFilterRow_AVX2;
      kStoreGray = StoreGray_AVX2;
      kImportOneRow = ImportOneRow_AVX2;
      kUpdateW = UpdateW_AVX2;
      kUpdateChroma = UpdateChroma_AVX2;
      return;
    }
#endif
#if defined(SJPEG_USE_SSE2)
    if (sjpeg::SupportsSSE2()) {
      kSharpUpdateY = SharpUpdateY_SSE2;
      kSharpUpdateRGB = SharpUpdateRGB_SSE2;
      kSharpFilterRow = SharpFilterRow_SSE2;
    }
#endif
#if defined(SJPEG_USE_NEON)
    if (sjpeg::SupportsNEON()) {
      kSharpUpdateY = SharpUpdateY_NEON;
      kSharpUpdateRGB = SharpUpdateRGB_NEON;
      kSharpFilterRow = SharpFilterRow_NEON;
    }
#endif
  });
}

//------------------------------------------------------------------------------

uint32_t ScaleDown(int a, int b, int c, int d) {
  const uint32_t A = GammaToLinear(a);
  const uint32_t B = GammaToLinear(b);
  const uint32_t C = GammaToLinear(c);
  const uint32_t D = GammaToLinear(d);
  return LinearToGamma((A + B + C + D + 2) >> 2);
}

static void UpdateChroma(const fixed_y_t* src1, const fixed_y_t* src2,
                         fixed_t* dst, size_t uv_w) {
  for (size_t i = 0; i < uv_w; ++i) {
    const uint32_t r = ScaleDown(src1[0 * uv_w + 0], src1[0 * uv_w + 1],
                                 src2[0 * uv_w + 0], src2[0 * uv_w + 1]);
    const uint32_t g = ScaleDown(src1[2 * uv_w + 0], src1[2 * uv_w + 1],
                                 src2[2 * uv_w + 0], src2[2 * uv_w + 1]);
    const uint32_t b = ScaleDown(src1[4 * uv_w + 0], src1[4 * uv_w + 1],
                                 src2[4 * uv_w + 0], src2[4 * uv_w + 1]);
    const int W = RGBToGray(r, g, b);
    dst[0 * uv_w] = (fixed_t)(r - W);
    dst[1 * uv_w] = (fixed_t)(g - W);
    dst[2 * uv_w] = (fixed_t)(b - W);
    dst  += 1;
    src1 += 2;
    src2 += 2;
  }
}

static void UpdateW(const fixed_y_t* src, fixed_y_t* dst, int w) {
  for (int i = 0; i < w; ++i) {
    const uint32_t R = GammaToLinear(src[0 * w + i]);
    const uint32_t G = GammaToLinear(src[1 * w + i]);
    const uint32_t B = GammaToLinear(src[2 * w + i]);
    const uint32_t Y = RGBToGray(R, G, B);
    dst[i] = (fixed_y_t)LinearToGamma(Y);
  }
}

static fixed_y_t Filter2(int A, int B, int W0) {
  const int v0 = (A * 3 + B + 2) >> 2;
  return clip_y(v0 + W0);
}

static void InterpolateTwoRows(const fixed_y_t* const best_y,
                               const fixed_t* prev_uv,
                               const fixed_t* cur_uv,
                               const fixed_t* next_uv,
                               int w,
                               fixed_y_t* out1, fixed_y_t* out2) {
  const int uv_w = w >> 1;
  const int len = (w - 1) >> 1;   // length to filter
  for (int k = 3; k > 0; --k) {  // process each R/G/B segments in turn
    // special boundary case for i==0
    out1[0] = Filter2(cur_uv[0], prev_uv[0], best_y[0]);
    out2[0] = Filter2(cur_uv[0], next_uv[0], best_y[w]);

    kSharpFilterRow(cur_uv, prev_uv, len, best_y + 0 + 1, out1 + 1);
    kSharpFilterRow(cur_uv, next_uv, len, best_y + w + 1, out2 + 1);

    // special boundary case for i == w - 1 when w is even
    if (!(w & 1)) {
      out1[w - 1] = Filter2(cur_uv[uv_w - 1], prev_uv[uv_w - 1],
                            best_y[w - 1 + 0]);
      out2[w - 1] = Filter2(cur_uv[uv_w - 1], next_uv[uv_w - 1],
                            best_y[w - 1 + w]);
    }
    out1 += w;
    out2 += w;
    prev_uv += uv_w;
    cur_uv  += uv_w;
    next_uv += uv_w;
  }
}

static void ConvertWRGBToYUVSlice(const fixed_y_t* best_y,
                                  const fixed_t* best_uv,
                                  int width, int height,
                                  size_t j_uv_start, size_t j_uv_end,
                                  uint8_t* y_plane,
                                  uint8_t* u_plane, uint8_t* v_plane) {
  const size_t w = ((size_t)width + 1) & ~1ULL;
  const size_t uv_w = w >> 1;
  const size_t row_elems = 3 * uv_w;
  const size_t j_y_start = j_uv_start * 2;
  const size_t j_y_end = std::min((size_t)height, j_uv_end * 2);

  for (size_t j = j_y_start; j < j_y_end; ++j) {
    const size_t off = (j >> 1) * row_elems;
    uint8_t* const dst_y = y_plane + j * width;
    for (size_t i = 0; i < (size_t)width; ++i) {
      const int W = best_y[i + j * w];
      const int r = best_uv[off + (i >> 1) + 0 * uv_w] + W;
      const int g = best_uv[off + (i >> 1) + 1 * uv_w] + W;
      const int b = best_uv[off + (i >> 1) + 2 * uv_w] + W;
      dst_y[i] = ConvertRGBToY(r, g, b);
    }
  }
  for (size_t j = j_uv_start; j < j_uv_end; ++j) {
    uint8_t* const dst_u = u_plane + j * uv_w;
    uint8_t* const dst_v = v_plane + j * uv_w;
    for (size_t i = 0; i < uv_w; ++i) {
      const size_t off = i + j * row_elems;
      const int r = best_uv[off + 0 * uv_w];
      const int g = best_uv[off + 1 * uv_w];
      const int b = best_uv[off + 2 * uv_w];
      dst_u[i] = ConvertRGBToU(r, g, b);
      dst_v[i] = ConvertRGBToV(r, g, b);
    }
  }
}

//------------------------------------------------------------------------------
// Main function

static bool PreprocessARGB(const uint8_t* const rgb, int width, int height,
                           int stride, uint8_t* y_plane, uint8_t* u_plane,
                           uint8_t* v_plane, const Encoder* encoder) {
  if (width <= 0 || height <= 0 ||
      width > kMaxDimension || height > kMaxDimension) {
    return false;
  }
  // We expand the right/bottom border if needed.
  const size_t w = ((size_t)width + 1) & ~1ULL;
  const size_t h = ((size_t)height + 1) & ~1ULL;
  const size_t uv_w = w >> 1;
  const size_t uv_h = h >> 1;
  const size_t num_pairs = (height + 1) / 2;
  uint64_t prev_diff_y_sum = ~0ULL;

#if !defined(SJPEG_NO_MULTITHREADING)
  const int max_threads = (int)num_pairs;
  const int num_threads =
      (encoder != nullptr)
          ? std::max(1, std::min(encoder->num_threads(), max_threads))
          : 1;
#else
  const int num_threads = 1;
#endif

  InitGammaTablesF();
  InitFunctionPointers();

  const size_t row_elems = 3 * uv_w;
  const size_t per_thread_elems = (8 * w) + (3 * row_elems);

  // Single memory chunk allocation to avoid allocator lock contention and heap
  // fragmentation.
  // We allocate 4 extra rows (kNumIterations) for best_uv to perform shifted
  // iterations without a second buffer.
  const uint64_t total_elems64 =
      ((uint64_t)w * h) +                                // best_y
      ((uint64_t)w * h) +                                // target_y
      ((uint64_t)(kNumIterations + uv_h) * row_elems) +  // best_uv
      ((uint64_t)uv_h * row_elems) +                     // target_uv
      ((uint64_t)num_threads * per_thread_elems);        // per-thread scratch
  if (total_elems64 > SIZE_MAX / sizeof(uint16_t)) return false;
  const size_t total_elems = (size_t)total_elems64;
  std::unique_ptr<uint16_t[]> arena(new (std::nothrow) uint16_t[total_elems]);
  if (arena == nullptr) return false;

  uint16_t* ptr = arena.get();
  fixed_y_t* const best_y = ptr;
  ptr += w * h;
  fixed_y_t* const target_y = ptr;
  ptr += w * h;
  fixed_t* const best_uv_alloc = (fixed_t*)ptr;
  ptr += (kNumIterations + uv_h) * row_elems;
  fixed_t* const target_uv = (fixed_t*)ptr;
  ptr += uv_h * row_elems;
  uint16_t* const thread_scratch_base = ptr;
  ptr += num_threads * per_thread_elems;
  assert(ptr == arena.get() + total_elems);

  // Origin for pass 0: shifted by kNumIterations rows so negative row offsets
  // remain in-bounds.
  fixed_t* const best_uv_origin = best_uv_alloc + kNumIterations * row_elems;
  const uint64_t diff_y_threshold = 3ULL * w * h;

  assert(width >= kMinDimensionIterativeConversion);
  assert(height >= kMinDimensionIterativeConversion);

  auto get_thread_scratch = [&](int t,
                                fixed_y_t** src1,
                                fixed_y_t** src2,
                                fixed_y_t** best_rgb_y,
                                fixed_t** best_rgb_uv,
                                fixed_t** head_buffer) {
    uint16_t* p =
        thread_scratch_base + (size_t)t * per_thread_elems;
    *src1 = p;
    p += 3 * w;
    *src2 = p;
    p += 3 * w;
    *best_rgb_y = p;
    p += 2 * w;
    *best_rgb_uv = reinterpret_cast<fixed_t*>(p);
    p += row_elems;
    *head_buffer = reinterpret_cast<fixed_t*>(p);
  };

  auto run_parallel = [&](int total,
                          const std::function<void(int, int, int)>& fn) {
#if !defined(SJPEG_NO_MULTITHREADING)
    if (encoder != nullptr && num_threads > 1) {
      encoder->RunParallel(num_threads, total, fn);
      return;
    }
#endif
    fn(0, 0, total);
  };

  // Import RGB samples to W/RGB representation.
  run_parallel(num_pairs, [&](int t, int p_start, int p_end) {
    const size_t j_start = p_start * 2;
    const size_t j_end =
        std::min((size_t)height, (size_t)(p_end * 2));

    fixed_y_t *src1, *src2, *best_rgb_y;
    fixed_t *best_rgb_uv, *head_buffer;
    get_thread_scratch(t, &src1, &src2, &best_rgb_y, &best_rgb_uv,
                       &head_buffer);

    for (size_t j = j_start; j < j_end; j += 2) {
      const bool is_last_row = (j == (size_t)height - 1);
      const ptrdiff_t rgb_off = (ptrdiff_t)j * stride;
      const size_t y_off = j * w;
      const size_t uv_off = (j >> 1) * row_elems;

      kImportOneRow(rgb + rgb_off, width, src1);
      if (is_last_row) {
        memcpy(src2, src1, 3 * w * sizeof(*src2));
      } else {
        kImportOneRow(rgb + rgb_off + stride, width, src2);
      }
      kStoreGray(src1, &best_y[y_off + 0], w);
      kStoreGray(src2, &best_y[y_off + w], w);
      kUpdateW(src1, &target_y[y_off + 0], w);
      kUpdateW(src2, &target_y[y_off + w], w);
      kUpdateChroma(src1, src2, &target_uv[uv_off], uv_w);
      memcpy(&best_uv_origin[uv_off], &target_uv[uv_off],
             row_elems * sizeof(best_uv_origin[0]));
    }
  });

  const ptrdiff_t row_stride = (ptrdiff_t)row_elems;
  int current_offset_rows = 0;
  struct SliceRange {
    int p_start;
    int p_end;
  };
  std::vector<SliceRange> slices(num_threads, {0, 0});
  std::vector<uint64_t> thread_diffs(num_threads, 0);

  // Iterate and resolve clipping conflicts using shifted iterations.
  for (int iter = 0; iter < kNumIterations; ++iter) {
    const int in_offset = current_offset_rows;
    const int out_offset = current_offset_rows - 1;

    fixed_t* const cur_in = best_uv_origin + in_offset * row_stride;
    fixed_t* const cur_out = best_uv_origin + out_offset * row_stride;

    std::fill(thread_diffs.begin(), thread_diffs.end(), 0);

    run_parallel(num_pairs, [&](int t, int p_start, int p_end) {
      slices[t] = {p_start, p_end};
      const size_t J_start = p_start;
      const size_t J_end = p_end;

      fixed_y_t *src1, *src2, *best_rgb_y;
      fixed_t *best_rgb_uv, *head_buffer;
      get_thread_scratch(t, &src1, &src2, &best_rgb_y, &best_rgb_uv,
                         &head_buffer);

      uint64_t local_diff = 0;
      for (size_t J = J_start; J < J_end; ++J) {
        const size_t prev_J = (J > 0) ? J - 1 : 0;
        const size_t next_J = (J < uv_h - 1) ? J + 1 : J;
        const fixed_t* const prev_uv = &cur_in[prev_J * row_elems];
        const fixed_t* const cur_uv  = &cur_in[J      * row_elems];
        const fixed_t* const next_uv = &cur_in[next_J * row_elems];

        const size_t uv_off = J * row_elems;
        const size_t y_off = (J << 1) * w;

        InterpolateTwoRows(&best_y[y_off], prev_uv, cur_uv, next_uv, (int)w,
                           src1, src2);
        kUpdateW(src1, &best_rgb_y[0 * w], (int)w);
        kUpdateW(src2, &best_rgb_y[1 * w], (int)w);
        kUpdateChroma(src1, src2, &best_rgb_uv[0], (int)uv_w);

        local_diff += kSharpUpdateY(&target_y[y_off], &best_rgb_y[0],
                                    &best_y[y_off], (int)(2 * w));

        // For threads t > 0, buffer the first 2 rows of the slice in local
        // scratch to prevent overwriting boundary rows needed by thread t - 1.
        fixed_t* dst_ptr;
        if (t > 0 && (J - J_start < 2)) {
          dst_ptr = head_buffer + (J - J_start) * row_elems;
        } else {
          dst_ptr = &cur_out[uv_off];
        }

        memcpy(dst_ptr, &cur_in[uv_off], row_elems * sizeof(fixed_t));
        kSharpUpdateRGB(&target_uv[uv_off], &best_rgb_uv[0], dst_ptr,
                        (int)row_elems);
      }
      thread_diffs[t] = local_diff;
    });

    // Barrier complete. Flush buffered head rows for helper threads.
    if (num_threads > 1) {
      for (int t = 1; t < num_threads; ++t) {
        const int count = std::min(2, slices[t].p_end - slices[t].p_start);
        if (count > 0) {
          fixed_y_t *src1, *src2, *best_rgb_y;
          fixed_t *best_rgb_uv, *head_buffer;
          get_thread_scratch(t, &src1, &src2, &best_rgb_y, &best_rgb_uv,
                             &head_buffer);
          memcpy(&cur_out[(size_t)slices[t].p_start * row_elems],
                 head_buffer,
                 (size_t)count * row_elems * sizeof(fixed_t));
        }
      }
    }

    uint64_t diff_y_sum = 0;
    for (int t = 0; t < num_threads; ++t) {
      diff_y_sum += thread_diffs[t];
    }

    current_offset_rows = out_offset;
    if (diff_y_sum < diff_y_threshold) break;
    if (iter > 0 && diff_y_sum > prev_diff_y_sum) break;
    prev_diff_y_sum = diff_y_sum;
  }

  // Final reconstruction.
  fixed_t* const final_best_uv =
      best_uv_origin + current_offset_rows * row_stride;
  run_parallel(uv_h, [&](int /*t*/, int p_start, int p_end) {
    ConvertWRGBToYUVSlice(&best_y[0], final_best_uv, width, height,
                          p_start, p_end, y_plane, u_plane, v_plane);
  });
  return true;
}

}  // namespace sjpeg

////////////////////////////////////////////////////////////////////////////////
// Entry point

bool sjpeg::ApplySharpYUVConversion(const uint8_t* const rgb, int W, int H,
                                    int stride, uint8_t* y_plane,
                                    uint8_t* u_plane, uint8_t* v_plane,
                                    const Encoder* encoder) {
  if (rgb == nullptr || y_plane == nullptr || u_plane == nullptr ||
      v_plane == nullptr) {
    return false;
  }
  if (W <= 0 || H <= 0 || W > kMaxDimension || H > kMaxDimension) {
    return false;
  }
  if (stride == INT_MIN) return false;
  const int abs_stride = std::abs(stride);
  if (abs_stride < 3 * W) return false;

  if (W <= kMinDimensionIterativeConversion ||
      H <= kMinDimensionIterativeConversion) {
    const int uv_w = (W + 1) >> 1;
    for (int y = 0; y < H; y += 2) {
      const uint8_t* const rgb1 = rgb + (ptrdiff_t)y * stride;
      const uint8_t* const rgb2 = (y < H - 1) ? rgb1 + stride : rgb1;
      ConvertRowToY(rgb1, W, &y_plane[(size_t)y * W]);
      if (y < H - 1) {
        ConvertRowToY(rgb2, W, &y_plane[(size_t)(y + 1) * W]);
      }
      ConvertRowToUV(rgb1, rgb2, W,
                     &u_plane[(size_t)(y >> 1) * uv_w],
                     &v_plane[(size_t)(y >> 1) * uv_w]);
    }
    return true;
  } else {
    return PreprocessARGB(rgb, W, H, stride, y_plane, u_plane, v_plane,
                          encoder);
  }
}

////////////////////////////////////////////////////////////////////////////////
