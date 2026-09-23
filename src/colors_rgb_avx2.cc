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
//  AVX2 variants for RGB -> YUV conversion
//

#define SJPEG_NEED_ASM_HEADERS
#include <stdint.h>

#include "sjpegi.h"

#if defined(SJPEG_USE_AVX2)

namespace sjpeg {

// global fixed-point precision
enum {
  FRAC = 16,
  HALF = (1 << FRAC) >> 1,
  ROUND_UV = (HALF << 2),
  ROUND_Y = HALF - (128 << FRAC)
};

// AVX2 256-bit fixed-point dot-product and descaling macro.
#define TRANSFORM_AVX2(RG_LO, RG_HI, GB_LO, GB_HI, MULT_RG, MULT_GB, ROUNDER, \
                       DESCALE_FIX, ADD_OR_SUB, OUT)                          \
  do {                                                                        \
    const __m256i V0_lo = _mm256_madd_epi16(RG_LO, MULT_RG);                  \
    const __m256i V0_hi = _mm256_madd_epi16(RG_HI, MULT_RG);                  \
    const __m256i V1_lo = _mm256_madd_epi16(GB_LO, MULT_GB);                  \
    const __m256i V1_hi = _mm256_madd_epi16(GB_HI, MULT_GB);                  \
    const __m256i V2_lo = ADD_OR_SUB(V0_lo, V1_lo);                           \
    const __m256i V2_hi = ADD_OR_SUB(V0_hi, V1_hi);                           \
    const __m256i V3_lo = _mm256_add_epi32(V2_lo, ROUNDER);                   \
    const __m256i V3_hi = _mm256_add_epi32(V2_hi, ROUNDER);                   \
    const __m256i V5_lo = _mm256_srai_epi32(V3_lo, DESCALE_FIX);              \
    const __m256i V5_hi = _mm256_srai_epi32(V3_hi, DESCALE_FIX);              \
    (OUT) = _mm256_packs_epi32(V5_lo, V5_hi);                                 \
  } while (0)

// Helper to broadcast alternating 16-bit constants (B, A) across a 256-bit
// register (adapted from MK_CST_16 in colors_rgb.cc).
#define MK_CST_256_16(A, B)                                                    \
  _mm256_set_epi16((B), (A), (B), (A), (B), (A), (B), (A), (B), (A), (B), (A), \
                   (B), (A), (B), (A))

// Convert 16 RGB samples to 16 Y samples.
static inline void ConvertRGBToY_AVX2(const __m256i* const R,
                                      const __m256i* const G,
                                      const __m256i* const B,
                                      __m256i* const Y) {
  const __m256i kRG_y = MK_CST_256_16(19595, 38469 - 16384);
  const __m256i kGB_y = MK_CST_256_16(16384, 7471);
  const __m256i kRound_Y = _mm256_set1_epi32(ROUND_Y);
  const __m256i RG_lo = _mm256_unpacklo_epi16(*R, *G);
  const __m256i RG_hi = _mm256_unpackhi_epi16(*R, *G);
  const __m256i GB_lo = _mm256_unpacklo_epi16(*G, *B);
  const __m256i GB_hi = _mm256_unpackhi_epi16(*G, *B);
  TRANSFORM_AVX2(RG_lo, RG_hi, GB_lo, GB_hi, kRG_y, kGB_y, kRound_Y, FRAC,
                 _mm256_add_epi32, *Y);
}

// 256-bit asymmetric dual-lane shuffle masks for 2-load zero-over-read RGB24
// unpacking. Load 0 reads bytes 0..31, Load 1 reads bytes 16..47 (exact 48-byte
// length).
alignas(32) static const int8_t kRGB24ShufA2[6][32] = {
    /* r0 */ {0, 3,  6,  9,  12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
              8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    /* r1 */ {-1, -1, -1, -1, -1, -1, 2,  5,  -1, -1, -1, -1, -1, -1, -1, -1,
              -1, -1, -1, 1,  4,  7,  10, 13, -1, -1, -1, -1, -1, -1, -1, -1},
    /* g0 */ {1, 4,  7,  10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
              9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    /* g1 */ {-1, -1, -1, -1, -1, 0, 3,  6,  -1, -1, -1, -1, -1, -1, -1, -1,
              -1, -1, -1, 2,  5,  8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1},
    /* b0 */ {2,  5,  8,  11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
              10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    /* b1 */ {-1, -1, -1, -1, -1, 1, 4,  7,  -1, -1, -1, -1, -1, -1, -1, -1,
              -1, -1, 0,  3,  6,  9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1},
};

// Convert 16 packed RGB samples to planar r, g, b.
static inline void RGB24PackedToPlanar_AVX2(const uint8_t* const rgb,
                                            __m256i* const r, __m256i* const g,
                                            __m256i* const b) {
  const __m256i zero = _mm256_setzero_si256();
  const __m256i L0 = LOAD_32(rgb + 0);
  const __m256i L1 = LOAD_32(rgb + 16);

  const __m256i mr0 = LOAD_ALIGNED_32(kRGB24ShufA2[0]);
  const __m256i mr1 = LOAD_ALIGNED_32(kRGB24ShufA2[1]);
  const __m256i mg0 = LOAD_ALIGNED_32(kRGB24ShufA2[2]);
  const __m256i mg1 = LOAD_ALIGNED_32(kRGB24ShufA2[3]);
  const __m256i mb0 = LOAD_ALIGNED_32(kRGB24ShufA2[4]);
  const __m256i mb1 = LOAD_ALIGNED_32(kRGB24ShufA2[5]);

  const __m256i r_bytes = _mm256_or_si256(_mm256_shuffle_epi8(L0, mr0),
                                          _mm256_shuffle_epi8(L1, mr1));
  const __m256i g_bytes = _mm256_or_si256(_mm256_shuffle_epi8(L0, mg0),
                                          _mm256_shuffle_epi8(L1, mg1));
  const __m256i b_bytes = _mm256_or_si256(_mm256_shuffle_epi8(L0, mb0),
                                          _mm256_shuffle_epi8(L1, mb1));

  *r = _mm256_unpacklo_epi8(r_bytes, zero);
  *g = _mm256_unpacklo_epi8(g_bytes, zero);
  *b = _mm256_unpacklo_epi8(b_bytes, zero);
}

// 4-channel single-mask byte shuffle for RGBA32 and BGRA32 packed-to-planar
// de-interleaving.
alignas(32) static const int8_t kShuf4Chan[32] = {
    0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
    0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};

// 16 packed BGRA/RGBA -> r/g/b (alpha dropped).
template <bool is_bgra>
static inline void XGXA32PackedToPlanar_AVX2(const uint8_t* const data,
                                             __m256i* const r, __m256i* const g,
                                             __m256i* const b) {
  const __m256i in0 = LOAD_32(data + 0);
  const __m256i in1 = LOAD_32(data + 32);
  const __m256i zero = _mm256_setzero_si256();
  const __m256i shuf_mask = LOAD_ALIGNED_32(kShuf4Chan);

  const __m256i p0 = _mm256_permute2x128_si256(in0, in1, 0x20);
  const __m256i p1 = _mm256_permute2x128_si256(in0, in1, 0x31);

  const __m256i s0 = _mm256_shuffle_epi8(p0, shuf_mask);
  const __m256i s1 = _mm256_shuffle_epi8(p1, shuf_mask);

  const __m256i lo = _mm256_unpacklo_epi32(s0, s1);
  const __m256i hi = _mm256_unpackhi_epi32(s0, s1);

  *g = _mm256_unpackhi_epi8(lo, zero);
  if (is_bgra) {
    *b = _mm256_unpacklo_epi8(lo, zero);
    *r = _mm256_unpacklo_epi8(hi, zero);
  } else {
    *r = _mm256_unpacklo_epi8(lo, zero);
    *b = _mm256_unpacklo_epi8(hi, zero);
  }
}

// Convert 16 accumulated R/G/B samples into 16 U and 16 V samples (2 full
// rows). The descaling factor is FRAC + 2 because four 2x2 samples are summed.
static inline void ConvertRGBToUVAccumulated_AVX2(const __m256i* const R,
                                                  const __m256i* const G,
                                                  const __m256i* const B,
                                                  __m256i* const U,
                                                  __m256i* const V) {
  const __m256i kRG_u = MK_CST_256_16(-11059, -21709);
  const __m256i kGB_u = MK_CST_256_16(0, -32768);
  const __m256i kRG_v = MK_CST_256_16(-32768, 0);
  const __m256i kGB_v = MK_CST_256_16(-27439, -5329);
  const __m256i kRound_UV = _mm256_set1_epi32(ROUND_UV);

  const __m256i RG_lo = _mm256_unpacklo_epi16(*R, *G);
  const __m256i RG_hi = _mm256_unpackhi_epi16(*R, *G);
  const __m256i GB_lo = _mm256_unpacklo_epi16(*G, *B);
  const __m256i GB_hi = _mm256_unpackhi_epi16(*G, *B);

  TRANSFORM_AVX2(RG_lo, RG_hi, GB_lo, GB_hi, kRG_u, kGB_u, kRound_UV, FRAC + 2,
                 _mm256_sub_epi32, *U);
  TRANSFORM_AVX2(GB_lo, GB_hi, RG_lo, RG_hi, kGB_v, kRG_v, kRound_UV, FRAC + 2,
                 _mm256_sub_epi32, *V);
}

// Convert 16x16 packed pixels to planar YUV420 blocks using pure 256-bit AVX2.
template <typename PackedToPlanarFunc>
static inline void Get16x16Block_AVX2_Impl(
    const uint8_t* data, int step, int16_t* blocks,
    PackedToPlanarFunc packed_to_planar) {
  const __m256i one = _mm256_set1_epi16(1);
  int16_t* y_left = blocks + 0 * 64;   // y0, then y2
  int16_t* y_right = blocks + 1 * 64;  // y1, then y3
  int16_t* u_out = blocks + 4 * 64;
  int16_t* v_out = blocks + 5 * 64;

  for (int half = 0; half < 2; ++half) {
    for (int pair = 0; pair < 2; ++pair) {
      const uint8_t* const src = data + (half * 8 + pair * 4) * step;
      const int y_off = pair * 4 * 8;

      __m256i r0, g0, b0, r1, g1, b1;
      __m256i Y0, Y1;

      // Pair 0 (Rows 0 + 1)
      packed_to_planar(src + 0 * step, &r0, &g0, &b0);
      ConvertRGBToY_AVX2(&r0, &g0, &b0, &Y0);

      packed_to_planar(src + 1 * step, &r1, &g1, &b1);
      ConvertRGBToY_AVX2(&r1, &g1, &b1, &Y1);

      // Paired 256-bit Y stores: saves 4x 128-bit stores + 2x vextracti128
      const __m256i Y_left_01 = _mm256_permute2x128_si256(Y0, Y1, 0x20);
      const __m256i Y_right_01 = _mm256_permute2x128_si256(Y0, Y1, 0x31);
      STORE_32(Y_left_01, y_left + y_off + 0 * 8);
      STORE_32(Y_right_01, y_right + y_off + 0 * 8);

      const __m256i r_madd0 = _mm256_madd_epi16(_mm256_add_epi16(r0, r1), one);
      const __m256i g_madd0 = _mm256_madd_epi16(_mm256_add_epi16(g0, g1), one);
      const __m256i b_madd0 = _mm256_madd_epi16(_mm256_add_epi16(b0, b1), one);

      // Pair 1 (Rows 2 + 3)
      packed_to_planar(src + 2 * step, &r0, &g0, &b0);
      ConvertRGBToY_AVX2(&r0, &g0, &b0, &Y0);

      packed_to_planar(src + 3 * step, &r1, &g1, &b1);
      ConvertRGBToY_AVX2(&r1, &g1, &b1, &Y1);

      const __m256i Y_left_23 = _mm256_permute2x128_si256(Y0, Y1, 0x20);
      const __m256i Y_right_23 = _mm256_permute2x128_si256(Y0, Y1, 0x31);
      STORE_32(Y_left_23, y_left + y_off + 2 * 8);
      STORE_32(Y_right_23, y_right + y_off + 2 * 8);

      const __m256i r_madd1 = _mm256_madd_epi16(_mm256_add_epi16(r0, r1), one);
      const __m256i g_madd1 = _mm256_madd_epi16(_mm256_add_epi16(g0, g1), one);
      const __m256i b_madd1 = _mm256_madd_epi16(_mm256_add_epi16(b0, b1), one);

      // Post-permute: pack without pre-permuting R/G/B, saving 4 permutes per
      // block
      const __m256i r_pack = _mm256_packs_epi32(r_madd0, r_madd1);
      const __m256i g_pack = _mm256_packs_epi32(g_madd0, g_madd1);
      const __m256i b_pack = _mm256_packs_epi32(b_madd0, b_madd1);

      __m256i U256, V256;
      ConvertRGBToUVAccumulated_AVX2(&r_pack, &g_pack, &b_pack, &U256, &V256);

      STORE_32(_mm256_permute4x64_epi64(U256, _MM_SHUFFLE(3, 1, 2, 0)), u_out);
      STORE_32(_mm256_permute4x64_epi64(V256, _MM_SHUFFLE(3, 1, 2, 0)), v_out);

      u_out += 16;
      v_out += 16;
    }
    y_left += 2 * 64;
    y_right += 2 * 64;
  }
}

// Convert 16x16 RGB samples to YUV420
void Get16x16Block_AVX2(const uint8_t* data, int step, int16_t* blocks) {
  Get16x16Block_AVX2_Impl(data, step, blocks, RGB24PackedToPlanar_AVX2);
}

// Convert 16x16 BGRA samples to YUV420
void Get16x16Block_BGRA_AVX2(const uint8_t* data, int step, int16_t* blocks) {
  Get16x16Block_AVX2_Impl(data, step, blocks,
                          XGXA32PackedToPlanar_AVX2</*is_bgra=*/true>);
}

// Convert 16x16 RGBA samples to YUV420
void Get16x16Block_RGBA_AVX2(const uint8_t* data, int step, int16_t* blocks) {
  Get16x16Block_AVX2_Impl(data, step, blocks,
                          XGXA32PackedToPlanar_AVX2</*is_bgra=*/false>);
}

// Fused color conversion sharing RG and GB 256-bit unpacks
static inline void ConvertRGBToYUV_Fused_AVX2(
    const __m256i* const R, const __m256i* const G, const __m256i* const B,
    __m256i* const Y, __m256i* const U, __m256i* const V) {
  const __m256i kRG_y = MK_CST_256_16(19595, 38469 - 16384);
  const __m256i kGB_y = MK_CST_256_16(16384, 7471);
  const __m256i kRound_Y = _mm256_set1_epi32(HALF);
  const __m256i kRG_u = MK_CST_256_16(-11059, -21709);
  const __m256i kGB_u = MK_CST_256_16(0, -32768);
  const __m256i kRG_v = MK_CST_256_16(-32768, 0);
  const __m256i kGB_v = MK_CST_256_16(-27439, -5329);
  const __m256i kRound_UV = _mm256_set1_epi32((128 << FRAC) + HALF);

  const __m256i RG_lo = _mm256_unpacklo_epi16(*R, *G);
  const __m256i RG_hi = _mm256_unpackhi_epi16(*R, *G);
  const __m256i GB_lo = _mm256_unpacklo_epi16(*G, *B);
  const __m256i GB_hi = _mm256_unpackhi_epi16(*G, *B);

  TRANSFORM_AVX2(RG_lo, RG_hi, GB_lo, GB_hi, kRG_y, kGB_y, kRound_Y, FRAC,
                 _mm256_add_epi32, *Y);
  TRANSFORM_AVX2(RG_lo, RG_hi, GB_lo, GB_hi, kRG_u, kGB_u, kRound_UV, FRAC,
                 _mm256_sub_epi32, *U);
  TRANSFORM_AVX2(GB_lo, GB_hi, RG_lo, RG_hi, kGB_v, kRG_v, kRound_UV, FRAC,
                 _mm256_sub_epi32, *V);
}

// Map 16 Y, U, V values to quantized riskiness histogram indices.
static inline __m256i YUVToIndices_AVX2(const __m256i& Y, const __m256i& U,
                                       const __m256i& V, const __m256i& mult,
                                       const __m256i& mult1,
                                       const __m256i& k255) {
  // Fast Clamping: Y strictly in [0, 255] requires no clamp; U, V in [1, 256]
  // require only min(255).
  const __m256i u1 = _mm256_min_epi16(U, k255);
  const __m256i v1 = _mm256_min_epi16(V, k255);

  const __m256i y2 = _mm256_mulhi_epi16(Y, mult);
  const __m256i u2 = _mm256_mulhi_epi16(u1, mult);
  const __m256i v2 = _mm256_mulhi_epi16(v1, mult);

  // Factored Index Arithmetic: y + 7 * (u + 7 * v)
  const __m256i v7 = _mm256_mullo_epi16(v2, mult1);
  const __m256i uv7 = _mm256_add_epi16(u2, v7);
  const __m256i uv = _mm256_mullo_epi16(uv7, mult1);
  return _mm256_add_epi16(y2, uv);
}

// Convert 16 RGB pixels to riskiness indices.
static inline void RowToIndex16_AVX2(const uint8_t* rgb, uint16_t* dst,
                                     const __m256i& mult, const __m256i& mult1,
                                     const __m256i& k255) {
  __m256i r, g, b, Y, U, V;
  RGB24PackedToPlanar_AVX2(rgb, &r, &g, &b);
  ConvertRGBToYUV_Fused_AVX2(&r, &g, &b, &Y, &U, &V);
  const __m256i idx = YUVToIndices_AVX2(Y, U, V, mult, mult1, k255);
  STORE_32(idx, dst);
}

// Reference scalar C implementation in colors_rgb.cc used for trailing pixels.
extern void RowToIndexC(const uint8_t* rgb, int width, uint16_t* dst);

void RowToIndexAVX2(const uint8_t* rgb, int width, uint16_t* dst) {
  const __m256i mult = _mm256_set1_epi16(0x0101u * (sjpeg::kRGBSize - 1));
  const __m256i mult1 = _mm256_set1_epi16(sjpeg::kRGBSize);
  const __m256i k255 = _mm256_set1_epi16(255);

  while (width >= 32) {
    RowToIndex16_AVX2(rgb + 0, dst + 0, mult, mult1, k255);
    RowToIndex16_AVX2(rgb + 48, dst + 16, mult, mult1, k255);
    rgb += 2 * 48;
    dst += 32;
    width -= 32;
  }
  while (width >= 16) {
    RowToIndex16_AVX2(rgb, dst, mult, mult1, k255);
    rgb += 48;
    dst += 16;
    width -= 16;
  }
  if (width > 0) {
    RowToIndexC(rgb, width, dst);
  }
}

#undef TRANSFORM_AVX2
#undef MK_CST_256_16

}  // namespace sjpeg

#endif  // SJPEG_USE_AVX2
