// Copyright 2026 Google Inc.
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
//  Common SIMD load/store helpers for SSE2, AVX2, and NEON.
//
// Author: Skal (pascal.massimino@gmail.com)

#ifndef SJPEG_SIMD_H_
#define SJPEG_SIMD_H_

#include <stddef.h>
#include <stdint.h>

#if !defined(SJPEG_USE_SSE2) && defined(__SSE2__)
#define SJPEG_USE_SSE2
#endif

#if !defined(SJPEG_USE_SSSE3) && defined(__SSSE3__)
#define SJPEG_USE_SSSE3
#endif

#if !defined(SJPEG_USE_AVX2) && defined(__AVX2__)
#define SJPEG_USE_AVX2
#endif

#if !defined(SJPEG_USE_NEON) && (defined(__ARM_NEON__) || defined(__aarch64__))
#define SJPEG_USE_NEON
#endif

// Architecture-specific intrinsic headers.
#if defined(SJPEG_USE_AVX2)
#include <immintrin.h>
#elif defined(SJPEG_USE_SSSE3)
#include <tmmintrin.h>
#elif defined(SJPEG_USE_SSE2)
#include <emmintrin.h>
#endif

#if defined(SJPEG_USE_NEON)
#include <arm_neon.h>
#endif

//------------------------------------------------------------------------------
// SSE2 (128-bit) load/store macros

#if defined(SJPEG_USE_SSE2)

// Load 16 bytes unaligned / aligned into __m128i.
#define LOAD_16(src) _mm_loadu_si128(reinterpret_cast<const __m128i*>(src))
#define LOAD_ALIGNED_16(src) \
  _mm_load_si128(reinterpret_cast<const __m128i*>(src))

// Store 16 bytes unaligned / aligned from __m128i.
#define STORE_16(V, dst) _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), (V))
#define STORE_ALIGNED_16(V, dst) \
  _mm_store_si128(reinterpret_cast<__m128i*>(dst), (V))

// Load / store 8 bytes (64 bits) into lower half of __m128i.
#define LOAD_64(src) _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src))
#define STORE_64(V, dst) _mm_storel_epi64(reinterpret_cast<__m128i*>(dst), (V))

// 16-bit integer absolute value (SSSE3 has hardware instruction; SSE2
// fallbacks).
#if defined(SJPEG_USE_SSSE3)
#define ABS_16(V) _mm_abs_epi16(V)
#else
static inline __m128i Abs16_SSE2(__m128i v) {
  const __m128i s = _mm_srai_epi16(v, 15);
  return _mm_sub_epi16(_mm_xor_si128(v, s), s);
}
#define ABS_16(V) Abs16_SSE2(V)
#endif

// Horizontal-sum reduction of a 128-bit accumulator down to a scalar. Only
// two widths are needed: 32-bit (score sums) and 16-bit (counts, kept
// signed even for conceptually-unsigned counters, since values are bounded
// well under 2^15 and the bit pattern is the same either way -- this avoids
// a third, unsigned variant).
static inline int32_t HorizontalSumS32(__m128i v) {
  const __m128i hi = _mm_unpackhi_epi64(v, v);
  const __m128i sum = _mm_add_epi32(v, hi);
  const __m128i hi2 = _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 1, 1, 1));
  return _mm_cvtsi128_si32(_mm_add_epi32(sum, hi2));
}

static inline int32_t HorizontalSumS16(__m128i v) {
  const __m128i ones = _mm_set1_epi16(1);
  return HorizontalSumS32(_mm_madd_epi16(v, ones));
}

#endif  // SJPEG_USE_SSE2

//------------------------------------------------------------------------------
// AVX2 (256-bit) load/store macros

#if defined(SJPEG_USE_AVX2)

// Load 32 bytes unaligned / aligned into __m256i.
#define LOAD_32(src) _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src))
#define LOAD_ALIGNED_32(src) \
  _mm256_load_si256(reinterpret_cast<const __m256i*>(src))

// Store 32 bytes unaligned / aligned from __m256i.
#define STORE_32(V, dst) \
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), (V))
#define STORE_ALIGNED_32(V, dst) \
  _mm256_store_si256(reinterpret_cast<__m256i*>(dst), (V))

// 16-bit integer absolute value for 256-bit vectors.
#define ABS_32(V) _mm256_abs_epi16(V)

#endif  // SJPEG_USE_AVX2

//------------------------------------------------------------------------------
// ARM NEON load/store macros

#if defined(SJPEG_USE_NEON)

// NEON loads/stores have no separate aligned-access instructions (unlike
// SSE2/AVX2), so the ALIGNED variants below just alias the unaligned ones.
struct NeonVector16 {
  uint8x16_t v;
  operator uint8x16_t() const { return v; }
#define SJPEG_NEON16_OP(T, S) \
  operator T() const { return vreinterpretq_##S##_u8(v); }
  SJPEG_NEON16_OP(int8x16_t, s8)
  SJPEG_NEON16_OP(uint16x8_t, u16)
  SJPEG_NEON16_OP(int16x8_t, s16)
  SJPEG_NEON16_OP(uint32x4_t, u32)
  SJPEG_NEON16_OP(int32x4_t, s32)
  SJPEG_NEON16_OP(uint64x2_t, u64)
  SJPEG_NEON16_OP(int64x2_t, s64)
#undef SJPEG_NEON16_OP
};
static inline NeonVector16 Load16_NEON(const void* src) {
  return NeonVector16{vld1q_u8(reinterpret_cast<const uint8_t*>(src))};
}

struct NeonVector8 {
  uint8x8_t v;
  operator uint8x8_t() const { return v; }
#define SJPEG_NEON8_OP(T, S) \
  operator T() const { return vreinterpret_##S##_u8(v); }
  SJPEG_NEON8_OP(int8x8_t, s8)
  SJPEG_NEON8_OP(uint16x4_t, u16)
  SJPEG_NEON8_OP(int16x4_t, s16)
  SJPEG_NEON8_OP(uint32x2_t, u32)
  SJPEG_NEON8_OP(int32x2_t, s32)
  SJPEG_NEON8_OP(uint64x1_t, u64)
  SJPEG_NEON8_OP(int64x1_t, s64)
#undef SJPEG_NEON8_OP
};
static inline NeonVector8 Load8_NEON(const void* src) {
  return NeonVector8{vld1_u8(reinterpret_cast<const uint8_t*>(src))};
}

#define SJPEG_STORE_NEON(T16, T8, SUFFIX, PTR_T)       \
  static inline void Store16_NEON(T16 v, void* dst) {  \
    vst1q_##SUFFIX(reinterpret_cast<PTR_T*>(dst), v);  \
  }                                                    \
  static inline void Store8_NEON(T8 v, void* dst) {    \
    vst1_##SUFFIX(reinterpret_cast<PTR_T*>(dst), v);   \
  }

SJPEG_STORE_NEON(uint8x16_t, uint8x8_t, u8, uint8_t)
SJPEG_STORE_NEON(int8x16_t, int8x8_t, s8, int8_t)
SJPEG_STORE_NEON(uint16x8_t, uint16x4_t, u16, uint16_t)
SJPEG_STORE_NEON(int16x8_t, int16x4_t, s16, int16_t)
SJPEG_STORE_NEON(uint32x4_t, uint32x2_t, u32, uint32_t)
SJPEG_STORE_NEON(int32x4_t, int32x2_t, s32, int32_t)
SJPEG_STORE_NEON(uint64x2_t, uint64x1_t, u64, uint64_t)
SJPEG_STORE_NEON(int64x2_t, int64x1_t, s64, int64_t)
#undef SJPEG_STORE_NEON

static inline void Store16_NEON(NeonVector16 v, void* dst) {
  vst1q_u8(reinterpret_cast<uint8_t*>(dst), v.v);
}
static inline void Store8_NEON(NeonVector8 v, void* dst) {
  vst1_u8(reinterpret_cast<uint8_t*>(dst), v.v);
}

#define LOAD_16(src) Load16_NEON(src)
#define LOAD_ALIGNED_16(src) Load16_NEON(src)
#define STORE_16(V, dst) Store16_NEON((V), (dst))
#define STORE_ALIGNED_16(V, dst) Store16_NEON((V), (dst))

#define LOAD_64(src) Load8_NEON(src)
#define STORE_64(V, dst) Store8_NEON((V), (dst))

static inline int16x8_t Abs16_NEON(int16x8_t v) { return vabsq_s16(v); }
static inline int16x8_t Abs16_NEON(NeonVector16 v) {
  return vabsq_s16(static_cast<int16x8_t>(v));
}
#define ABS_16(V) Abs16_NEON(V)

// Horizontal-sum reduction of a NEON accumulator down to a scalar. Only two
// widths are needed: 32-bit (score sums) and 16-bit (counts, kept signed
// even for conceptually-unsigned counters, since values are bounded well
// under 2^15 and the bit pattern is the same either way -- this avoids a
// third, unsigned variant). vaddvq_* is aarch64-only; armv7 NEON falls back
// to a pairwise-add reduction.
static inline int32_t HorizontalSumS32(int32x4_t v) {
#if defined(SJPEG_AARCH64)
  return vaddvq_s32(v);
#else
  int32x2_t sum = vadd_s32(vget_low_s32(v), vget_high_s32(v));
  sum = vpadd_s32(sum, sum);
  return vget_lane_s32(sum, 0);
#endif
}

static inline int32_t HorizontalSumS16(int16x8_t v) {
#if defined(SJPEG_AARCH64)
  return vaddvq_s16(v);
#else
  int16x4_t sum = vadd_s16(vget_low_s16(v), vget_high_s16(v));
  sum = vpadd_s16(sum, sum);
  sum = vpadd_s16(sum, sum);
  return vget_lane_s16(sum, 0);
#endif
}

#endif  // SJPEG_USE_NEON

#endif  // SJPEG_SIMD_H_
