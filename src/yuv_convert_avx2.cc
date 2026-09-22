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
// AVX2 variant of the Sharp RGB->YUV conversion's gamma-table
// lookups and SIMD implementations of preprocessing and refinement passes,
// compiled with -mavx2 only for this file.
//
// GammaToLinear()/LinearToGamma() lookups are vectorized without gather using
// scalar loads and in-register table permutation.
//
// Author: Skal (pascal.massimino@gmail.com)

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define SJPEG_NEED_ASM_HEADERS
#include "sjpegi.h"

#if defined(SJPEG_USE_AVX2)

namespace sjpeg {

#define SFIX 2
#define SHALF (1 << SFIX >> 1)
#define MAX_Y_T ((256 << SFIX) - 1)
#define YUV_FIX 16
typedef int16_t fixed_t;
typedef uint16_t fixed_y_t;

static fixed_y_t clip_y(int y) {
  return (!(y & ~MAX_Y_T)) ? (fixed_y_t)y : (y < 0) ? 0 : MAX_Y_T;
}

extern uint32_t RGBToGray(uint32_t r, uint32_t g, uint32_t b);

uint64_t SharpUpdateY_AVX2(const uint16_t* ref, const uint16_t* src,
                           uint16_t* dst, int len) {
  uint64_t diff = 0;
  uint32_t tmp[8];
  int i;
  const __m256i zero = _mm256_setzero_si256();
  const __m256i max = _mm256_set1_epi16(MAX_Y_T);
  const __m256i one = _mm256_set1_epi16(1);
  __m256i sum = zero;

  for (i = 0; i + 16 <= len; i += 16) {
    const __m256i A = LOAD_32(ref + i);
    const __m256i B = LOAD_32(src + i);
    const __m256i C = LOAD_32(dst + i);
    const __m256i D = _mm256_sub_epi16(A, B);   // diff_y
    const __m256i abs_D = ABS_32(D);            // |diff_y|
    const __m256i F = _mm256_add_epi16(C, D);   // new_y
    const __m256i H = _mm256_max_epi16(_mm256_min_epi16(F, max), zero);
    const __m256i I = _mm256_madd_epi16(abs_D, one);  // sum(abs(...))
    STORE_32(H, dst + i);
    sum = _mm256_add_epi32(sum, I);
  }
  STORE_32(sum, tmp);
  diff = (uint64_t)tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] +
         tmp[6] + tmp[7];
  for (; i < len; ++i) {
    const int diff_y = ref[i] - src[i];
    const int new_y = static_cast<int>(dst[i]) + diff_y;
    dst[i] = clip_y(new_y);
    diff += static_cast<uint64_t>(abs(diff_y));
  }
  return diff;
}

void SharpUpdateRGB_AVX2(const int16_t* ref, const int16_t* src,
                         int16_t* dst, int len) {
  int i = 0;
  for (i = 0; i + 16 <= len; i += 16) {
    const __m256i A = LOAD_32(ref + i);
    const __m256i B = LOAD_32(src + i);
    const __m256i C = LOAD_32(dst + i);
    const __m256i D = _mm256_sub_epi16(A, B);  // diff_uv
    const __m256i E = _mm256_add_epi16(C, D);  // new_uv
    STORE_32(E, dst + i);
  }
  for (; i < len; ++i) {
    const int diff_uv = ref[i] - src[i];
    dst[i] += diff_uv;
  }
}

void SharpFilterRow_AVX2(const int16_t* A, const int16_t* B, int len,
                         const uint16_t* best_y, uint16_t* out) {
  int i;
  const __m256i kCst8 = _mm256_set1_epi16(8);
  const __m256i max = _mm256_set1_epi16(MAX_Y_T);
  const __m256i zero = _mm256_setzero_si256();

  for (i = 0; i + 16 <= len; i += 16) {
    const __m256i a0 = LOAD_32(A + i + 0);
    const __m256i a1 = LOAD_32(A + i + 1);
    const __m256i b0 = LOAD_32(B + i + 0);
    const __m256i b1 = LOAD_32(B + i + 1);

    const __m256i a0b1 = _mm256_add_epi16(a0, b1);
    const __m256i a1b0 = _mm256_add_epi16(a1, b0);
    const __m256i a0a1b0b1 = _mm256_add_epi16(a0b1, a1b0);
    const __m256i a0a1b0b1_8 = _mm256_add_epi16(a0a1b0b1, kCst8);
    const __m256i a0b1_2 = _mm256_add_epi16(a0b1, a0b1);
    const __m256i a1b0_2 = _mm256_add_epi16(a1b0, a1b0);

    const __m256i c0 =
        _mm256_srai_epi16(_mm256_add_epi16(a0b1_2, a0a1b0b1_8), 3);
    const __m256i c1 =
        _mm256_srai_epi16(_mm256_add_epi16(a1b0_2, a0a1b0b1_8), 3);

    const __m256i d0 = _mm256_add_epi16(c1, a0);
    const __m256i d1 = _mm256_add_epi16(c0, a1);

    const __m256i e0 = _mm256_srai_epi16(d0, 1);
    const __m256i e1 = _mm256_srai_epi16(d1, 1);

    const __m256i f_lo = _mm256_unpacklo_epi16(e0, e1);
    const __m256i f_hi = _mm256_unpackhi_epi16(e0, e1);

    const __m256i f0 = _mm256_permute2x128_si256(f_lo, f_hi, 0x20);
    const __m256i f1 = _mm256_permute2x128_si256(f_lo, f_hi, 0x31);

    const __m256i g0 = LOAD_32(best_y + 2 * i + 0);
    const __m256i g1 = LOAD_32(best_y + 2 * i + 16);

    const __m256i h0 = _mm256_add_epi16(g0, f0);
    const __m256i h1 = _mm256_add_epi16(g1, f1);

    const __m256i i0 = _mm256_max_epi16(_mm256_min_epi16(h0, max), zero);
    const __m256i i1 = _mm256_max_epi16(_mm256_min_epi16(h1, max), zero);

    STORE_32(i0, out + 2 * i + 0);
    STORE_32(i1, out + 2 * i + 16);
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

void StoreGray_AVX2(const fixed_y_t* const rgb, fixed_y_t* const y, int w) {
  int i = 0;
  const __m256i mult_rg = _mm256_set_epi16(
      46871 - 32768, 13933, 46871 - 32768, 13933, 46871 - 32768, 13933,
      46871 - 32768, 13933, 46871 - 32768, 13933, 46871 - 32768, 13933,
      46871 - 32768, 13933, 46871 - 32768, 13933);
  const __m256i mult_b = _mm256_set1_epi32(4732);
  const __m256i rounder = _mm256_set1_epi32(1 << 15);
  const __m256i zero = _mm256_setzero_si256();

  for (; i + 16 <= w; i += 16) {
    const __m256i r = LOAD_32(rgb + 0 * w + i);
    const __m256i g = LOAD_32(rgb + 1 * w + i);
    const __m256i b = LOAD_32(rgb + 2 * w + i);

    const __m256i rg_lo = _mm256_unpacklo_epi16(r, g);
    const __m256i rg_hi = _mm256_unpackhi_epi16(r, g);

    const __m256i b_lo = _mm256_unpacklo_epi16(b, zero);
    const __m256i b_hi = _mm256_unpackhi_epi16(b, zero);

    const __m256i m_rg_lo = _mm256_madd_epi16(rg_lo, mult_rg);
    const __m256i m_rg_hi = _mm256_madd_epi16(rg_hi, mult_rg);

    const __m256i g_lo_32 =
        _mm256_slli_epi32(_mm256_unpacklo_epi16(g, zero), 15);
    const __m256i g_hi_32 =
        _mm256_slli_epi32(_mm256_unpackhi_epi16(g, zero), 15);

    const __m256i m_b_lo = _mm256_mullo_epi32(b_lo, mult_b);
    const __m256i m_b_hi = _mm256_mullo_epi32(b_hi, mult_b);

    const __m256i luma_lo = _mm256_add_epi32(_mm256_add_epi32(m_rg_lo, g_lo_32),
                                             _mm256_add_epi32(m_b_lo, rounder));
    const __m256i luma_hi = _mm256_add_epi32(_mm256_add_epi32(m_rg_hi, g_hi_32),
                                             _mm256_add_epi32(m_b_hi, rounder));

    const __m256i y_lo = _mm256_srli_epi32(luma_lo, 16);
    const __m256i y_hi = _mm256_srli_epi32(luma_hi, 16);

    const __m256i y_packed = _mm256_packs_epi32(y_lo, y_hi);
    STORE_32(y_packed, y + i);
  }
  for (; i < w; ++i) {
    y[i] = RGBToGray(rgb[0 * w + i], rgb[1 * w + i], rgb[2 * w + i]);
  }
}

static inline void RGB24ToPlanar8(const uint8_t* const rgb, __m128i* const r,
                                  __m128i* const g, __m128i* const b) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i in0 = LOAD_16(rgb + 0);
  const __m128i in1 = LOAD_16(rgb + 8);
  static const int8_t kShufR0[16] = {0,  3,  6,  9,  12, 15, -1, -1,
                                     -1, -1, -1, -1, -1, -1, -1, -1};
  static const int8_t kShufR1[16] = {-1, -1, -1, -1, -1, -1, 10, 13,
                                     -1, -1, -1, -1, -1, -1, -1, -1};
  static const int8_t kShufG0[16] = {1,  4,  7,  10, 13, -1, -1, -1,
                                     -1, -1, -1, -1, -1, -1, -1, -1};
  static const int8_t kShufG1[16] = {-1, -1, -1, -1, -1, 8,  11, 14,
                                     -1, -1, -1, -1, -1, -1, -1, -1};
  static const int8_t kShufB0[16] = {2,  5,  8,  11, 14, -1, -1, -1,
                                     -1, -1, -1, -1, -1, -1, -1, -1};
  static const int8_t kShufB1[16] = {-1, -1, -1, -1, -1, 9,  12, 15,
                                     -1, -1, -1, -1, -1, -1, -1, -1};

  const __m128i mask_r0 = LOAD_16(kShufR0);
  const __m128i mask_r1 = LOAD_16(kShufR1);
  const __m128i mask_g0 = LOAD_16(kShufG0);
  const __m128i mask_g1 = LOAD_16(kShufG1);
  const __m128i mask_b0 = LOAD_16(kShufB0);
  const __m128i mask_b1 = LOAD_16(kShufB1);

  const __m128i r_bytes = _mm_or_si128(_mm_shuffle_epi8(in0, mask_r0),
                                       _mm_shuffle_epi8(in1, mask_r1));
  const __m128i g_bytes = _mm_or_si128(_mm_shuffle_epi8(in0, mask_g0),
                                       _mm_shuffle_epi8(in1, mask_g1));
  const __m128i b_bytes = _mm_or_si128(_mm_shuffle_epi8(in0, mask_b0),
                                       _mm_shuffle_epi8(in1, mask_b1));

  *r = _mm_unpacklo_epi8(r_bytes, zero);
  *g = _mm_unpacklo_epi8(g_bytes, zero);
  *b = _mm_unpacklo_epi8(b_bytes, zero);
}

void ImportOneRow_AVX2(const uint8_t* const rgb, int pic_width,
                       fixed_y_t* const dst) {
  const int w = (pic_width + 1) & ~1;
  int i = 0;
  const __m256i shalf = _mm256_set1_epi16(SHALF);

  for (; i + 16 <= pic_width; i += 16) {
    __m128i r0, g0, b0, r1, g1, b1;
    RGB24ToPlanar8(rgb + i * 3 + 0, &r0, &g0, &b0);
    RGB24ToPlanar8(rgb + i * 3 + 24, &r1, &g1, &b1);
    const __m256i r = _mm256_set_m128i(r1, r0);
    const __m256i g = _mm256_set_m128i(g1, g0);
    const __m256i b = _mm256_set_m128i(b1, b0);
    const __m256i r_up = _mm256_or_si256(_mm256_slli_epi16(r, SFIX), shalf);
    const __m256i g_up = _mm256_or_si256(_mm256_slli_epi16(g, SFIX), shalf);
    const __m256i b_up = _mm256_or_si256(_mm256_slli_epi16(b, SFIX), shalf);
    STORE_32(r_up, dst + 0 * w + i);
    STORE_32(g_up, dst + 1 * w + i);
    STORE_32(b_up, dst + 2 * w + i);
  }
  if (i < pic_width) {
    ImportOneRow_C(rgb, i, pic_width, dst);
  }
}

static const int kGammaToLinearBits = 14;  // must match GAMMA_TO_LINEAR_BITS

// Filled by InitGammaTablesF()
extern uint32_t kGammaToLinearTab[];
extern uint32_t kPackedLinearToGammaTab[];

// C-version for left-over tails.
extern uint32_t GammaToLinear(int v);
extern uint32_t LinearToGamma(uint32_t value);
extern uint32_t ScaleDown(int a, int b, int c, int d);

//------------------------------------------------------------------------------

static inline __m256i GammaToLinear8_Direct(const fixed_y_t* p) {
  return _mm256_set_epi32(
      kGammaToLinearTab[p[7]], kGammaToLinearTab[p[6]],
      kGammaToLinearTab[p[5]], kGammaToLinearTab[p[4]],
      kGammaToLinearTab[p[3]], kGammaToLinearTab[p[2]],
      kGammaToLinearTab[p[1]], kGammaToLinearTab[p[0]]);
}

// 'value' is in kGammaToLinearBits fractional precision.
static inline __m256i LinearToGamma8(__m256i value) {
  const __m256i v = _mm256_slli_epi32(value, 5);
  const __m256i tab_pos = _mm256_srli_epi32(v, kGammaToLinearBits);
  const __m256i x =
      _mm256_and_si256(v, _mm256_set1_epi32((1 << kGammaToLinearBits) - 1));

  const __m256i T0 = LOAD_32(kPackedLinearToGammaTab + 0);
  const __m256i T1 = LOAD_32(kPackedLinearToGammaTab + 8);
  const __m256i T2 = LOAD_32(kPackedLinearToGammaTab + 16);
  const __m256i T3 = LOAD_32(kPackedLinearToGammaTab + 24);

  const __m256i p0 = _mm256_permutevar8x32_epi32(T0, tab_pos);
  const __m256i p1 = _mm256_permutevar8x32_epi32(T1, tab_pos);
  const __m256i p2 = _mm256_permutevar8x32_epi32(T2, tab_pos);
  const __m256i p3 = _mm256_permutevar8x32_epi32(T3, tab_pos);

  const __m256 mask_bit3 = _mm256_castsi256_ps(
      _mm256_slli_epi32(_mm256_and_si256(tab_pos, _mm256_set1_epi32(8)), 28));
  const __m256 mask_bit4 = _mm256_castsi256_ps(
      _mm256_slli_epi32(_mm256_and_si256(tab_pos, _mm256_set1_epi32(16)), 27));

  const __m256 p01 = _mm256_blendv_ps(_mm256_castsi256_ps(p0),
                                      _mm256_castsi256_ps(p1), mask_bit3);
  const __m256 p23 = _mm256_blendv_ps(_mm256_castsi256_ps(p2),
                                      _mm256_castsi256_ps(p3), mask_bit3);
  __m256 packed_f = _mm256_blendv_ps(p01, p23, mask_bit4);

  const __m256 is_32 = _mm256_castsi256_ps(
      _mm256_cmpgt_epi32(tab_pos, _mm256_set1_epi32(31)));
  packed_f = _mm256_blendv_ps(
      packed_f,
      _mm256_castsi256_ps(
          _mm256_set1_epi32(kPackedLinearToGammaTab[32])),
      is_32);
  const __m256i packed = _mm256_castps_si256(packed_f);

  const __m256i v0 = _mm256_and_si256(packed, _mm256_set1_epi32(0xffff));
  const __m256i diff = _mm256_srli_epi32(packed, 16);
  const __m256i v2 = _mm256_mullo_epi32(diff, x);
  return _mm256_add_epi32(v0, _mm256_srli_epi32(v2, kGammaToLinearBits));
}

static inline __m256i RGBToGray8(__m256i r, __m256i g, __m256i b) {
  const __m256i round = _mm256_set1_epi32(1 << 15);
  const __m256i rr = _mm256_mullo_epi32(r, _mm256_set1_epi32(13933));
  const __m256i gg = _mm256_mullo_epi32(g, _mm256_set1_epi32(46871));
  const __m256i bb = _mm256_mullo_epi32(b, _mm256_set1_epi32(4732));
  const __m256i luma =
      _mm256_add_epi32(round, _mm256_add_epi32(rr, _mm256_add_epi32(gg, bb)));
  return _mm256_srli_epi32(luma, 16);
}

// Truncating narrow of 8x int32 lanes to 8x 16b, matching the plain C
// `(fixed_y_t)v` / `(fixed_t)v` casts.
static inline void Store8TruncTo16(void* p, __m256i v) {
  const __m256i masked = _mm256_and_si256(v, _mm256_set1_epi32(0xffff));
  const __m256i packed = _mm256_packus_epi32(masked, masked);
  const __m256i fixed = _mm256_permute4x64_epi64(packed, 0xd8);
  STORE_16(_mm256_castsi256_si128(fixed), p);
}

//------------------------------------------------------------------------------

void UpdateW_AVX2(const fixed_y_t* src, fixed_y_t* dst, int w) {
  int i = 0;
  for (; i + 8 <= w; i += 8) {
    const __m256i R = GammaToLinear8_Direct(src + 0 * w + i);
    const __m256i G = GammaToLinear8_Direct(src + 1 * w + i);
    const __m256i B = GammaToLinear8_Direct(src + 2 * w + i);
    const __m256i Y = RGBToGray8(R, G, B);
    Store8TruncTo16(dst + i, LinearToGamma8(Y));
  }
  for (; i < w; ++i) {
    const uint32_t R = GammaToLinear(src[0 * w + i]);
    const uint32_t G = GammaToLinear(src[1 * w + i]);
    const uint32_t B = GammaToLinear(src[2 * w + i]);
    const uint32_t Y = RGBToGray(R, G, B);
    dst[i] = (fixed_y_t)LinearToGamma(Y);
  }
}

//------------------------------------------------------------------------------

static inline __m256i ScaleDownChannel8(const fixed_y_t* src1,
                                        const fixed_y_t* src2,
                                        size_t channel_off) {
  const fixed_y_t* p1 = src1 + channel_off;
  const fixed_y_t* p2 = src2 + channel_off;

  const __m256i A = _mm256_set_epi32(
      kGammaToLinearTab[p1[14]], kGammaToLinearTab[p1[12]],
      kGammaToLinearTab[p1[10]], kGammaToLinearTab[p1[8]],
      kGammaToLinearTab[p1[6]],  kGammaToLinearTab[p1[4]],
      kGammaToLinearTab[p1[2]],  kGammaToLinearTab[p1[0]]);
  const __m256i B = _mm256_set_epi32(
      kGammaToLinearTab[p1[15]], kGammaToLinearTab[p1[13]],
      kGammaToLinearTab[p1[11]], kGammaToLinearTab[p1[9]],
      kGammaToLinearTab[p1[7]],  kGammaToLinearTab[p1[5]],
      kGammaToLinearTab[p1[3]],  kGammaToLinearTab[p1[1]]);
  const __m256i C = _mm256_set_epi32(
      kGammaToLinearTab[p2[14]], kGammaToLinearTab[p2[12]],
      kGammaToLinearTab[p2[10]], kGammaToLinearTab[p2[8]],
      kGammaToLinearTab[p2[6]],  kGammaToLinearTab[p2[4]],
      kGammaToLinearTab[p2[2]],  kGammaToLinearTab[p2[0]]);
  const __m256i D = _mm256_set_epi32(
      kGammaToLinearTab[p2[15]], kGammaToLinearTab[p2[13]],
      kGammaToLinearTab[p2[11]], kGammaToLinearTab[p2[9]],
      kGammaToLinearTab[p2[7]],  kGammaToLinearTab[p2[5]],
      kGammaToLinearTab[p2[3]],  kGammaToLinearTab[p2[1]]);

  __m256i sum =
      _mm256_add_epi32(_mm256_add_epi32(A, B), _mm256_add_epi32(C, D));
  sum = _mm256_srli_epi32(_mm256_add_epi32(sum, _mm256_set1_epi32(2)), 2);
  return LinearToGamma8(sum);
}

void UpdateChroma_AVX2(const fixed_y_t* src1, const fixed_y_t* src2,
                       fixed_t* dst, size_t uv_w) {
  size_t i = 0;
  for (; i + 8 <= uv_w; i += 8, dst += 8, src1 += 16, src2 += 16) {
    const __m256i r = ScaleDownChannel8(src1, src2, 0 * uv_w);
    const __m256i g = ScaleDownChannel8(src1, src2, 2 * uv_w);
    const __m256i b = ScaleDownChannel8(src1, src2, 4 * uv_w);
    const __m256i W = RGBToGray8(r, g, b);
    Store8TruncTo16(dst + 0 * uv_w, _mm256_sub_epi32(r, W));
    Store8TruncTo16(dst + 1 * uv_w, _mm256_sub_epi32(g, W));
    Store8TruncTo16(dst + 2 * uv_w, _mm256_sub_epi32(b, W));
  }
  for (; i < uv_w; ++i, ++dst, src1 += 2, src2 += 2) {
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
  }
}

#undef SFIX
#undef SHALF
#undef MAX_Y_T
#undef YUV_FIX

}  // namespace sjpeg

#endif  // SJPEG_USE_AVX2
