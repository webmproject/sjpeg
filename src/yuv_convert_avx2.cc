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
// Gather-based AVX2 variant of the Sharp RGB->YUV conversion's gamma-table
// lookups compiled with -mavx2 only for this file.
//
// GammaToLinear()/LinearToGamma() are per-lane lookups at data-dependent
// indices => fit the use of 'gather'.
//
// Author: Skal (pascal.massimino@gmail.com)

#define SJPEG_NEED_ASM_HEADERS
#include "sjpegi.h"

#if defined(SJPEG_USE_AVX2) && defined(SJPEG_USE_AVX2_YUV_GATHER)

namespace sjpeg {

static const int kGammaToLinearBits = 14;  // must match GAMMA_TO_LINEAR_BITS

// Filled by InitGammaTablesF()
extern uint32_t kGammaToLinearTab[];
extern uint32_t kLinearToGammaTab[];

// C-version for left-over tails.
extern uint32_t GammaToLinear(int v);
extern uint32_t LinearToGamma(uint32_t value);
extern uint32_t RGBToGray(uint32_t r, uint32_t g, uint32_t b);
extern uint32_t ScaleDown(int a, int b, int c, int d);

//------------------------------------------------------------------------------

static inline __m256i GammaToLinear8(__m256i idx) {
  return _mm256_i32gather_epi32(
      reinterpret_cast<const int*>(kGammaToLinearTab), idx, 4);
}

// 'value' is in kGammaToLinearBits fractional precision.
static inline __m256i LinearToGamma8(__m256i value) {
  const __m256i v = _mm256_slli_epi32(value, 5);
  const __m256i tab_pos = _mm256_srli_epi32(v, kGammaToLinearBits);
  const __m256i x =
      _mm256_and_si256(v, _mm256_set1_epi32((1 << kGammaToLinearBits) - 1));
  const __m256i v0 = _mm256_i32gather_epi32(
      reinterpret_cast<const int*>(kLinearToGammaTab), tab_pos, 4);
  const __m256i v1 = _mm256_i32gather_epi32(
      reinterpret_cast<const int*>(kLinearToGammaTab),
      _mm256_add_epi32(tab_pos, _mm256_set1_epi32(1)), 4);
  const __m256i v2 = _mm256_mullo_epi32(_mm256_sub_epi32(v1, v0), x);
  return _mm256_add_epi32(v0, _mm256_srli_epi32(v2, kGammaToLinearBits));
}

static inline __m256i RGBToGray8(__m256i r, __m256i g, __m256i b) {
  const __m256i round = _mm256_set1_epi32(1 << 16 >> 1);
  const __m256i rr = _mm256_mullo_epi32(r, _mm256_set1_epi32(13933));
  const __m256i gg = _mm256_mullo_epi32(g, _mm256_set1_epi32(46871));
  const __m256i bb = _mm256_mullo_epi32(b, _mm256_set1_epi32(4732));
  const __m256i luma =
      _mm256_add_epi32(round, _mm256_add_epi32(rr, _mm256_add_epi32(gg, bb)));
  return _mm256_srli_epi32(luma, 16);
}

static inline __m256i Load8U16AsI32(const fixed_y_t* p) {
  return _mm256_cvtepu16_epi32(
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}

// Truncating narrow of 8x int32 lanes to 8x 16b, matching the plain C
// `(fixed_y_t)v` / `(fixed_t)v` casts.
static inline void Store8TruncTo16(void* p, __m256i v) {
  const __m256i masked = _mm256_and_si256(v, _mm256_set1_epi32(0xffff));
  const __m256i packed = _mm256_packus_epi32(masked, masked);
  const __m256i fixed = _mm256_permute4x64_epi64(packed, 0xd8);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(p),
                    _mm256_castsi256_si128(fixed));
}

//------------------------------------------------------------------------------

void UpdateWAVX2(const fixed_y_t* src, fixed_y_t* dst, int w) {
  int i = 0;
  for (; i + 8 <= w; i += 8) {
    const __m256i R = GammaToLinear8(Load8U16AsI32(src + 0 * w + i));
    const __m256i G = GammaToLinear8(Load8U16AsI32(src + 1 * w + i));
    const __m256i B = GammaToLinear8(Load8U16AsI32(src + 2 * w + i));
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

// Deinterleaves 8 pairs into 'a'/'b' lanes
static inline void LoadPairs8(const fixed_y_t* p, __m256i* a, __m256i* b) {
  const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
  *a = _mm256_and_si256(v, _mm256_set1_epi32(0xffff));
  *b = _mm256_srli_epi32(v, 16);
}

static inline __m256i ScaleDown8(__m256i a, __m256i b, __m256i c, __m256i d) {
  const __m256i A = GammaToLinear8(a);
  const __m256i B = GammaToLinear8(b);
  const __m256i C = GammaToLinear8(c);
  const __m256i D = GammaToLinear8(d);
  __m256i sum = _mm256_add_epi32(_mm256_add_epi32(A, B), _mm256_add_epi32(C, D));
  sum = _mm256_srli_epi32(_mm256_add_epi32(sum, _mm256_set1_epi32(2)), 2);
  return LinearToGamma8(sum);
}

static inline __m256i ScaleDownChannel8(const fixed_y_t* src1,
                                        const fixed_y_t* src2,
                                        size_t channel_off) {
  __m256i a1, b1, a2, b2;
  LoadPairs8(src1 + channel_off, &a1, &b1);
  LoadPairs8(src2 + channel_off, &a2, &b2);
  return ScaleDown8(a1, b1, a2, b2);
}

void UpdateChromaAVX2(const fixed_y_t* src1, const fixed_y_t* src2,
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

}  // namespace sjpeg

#endif  // SJPEG_USE_AVX2 && SJPEG_USE_AVX2_YUV_GATHER
