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
//  AVX2 variant of Fdct(), compiled with -mavx2 only for this file, mirror
//  of the SSE2 code, but just with wider registers.
//
// Author: Skal (pascal.massimino@gmail.com)

#include <stdint.h>

#define SJPEG_NEED_ASM_HEADERS
#include "sjpegi.h"

#if defined(SJPEG_USE_AVX2)

namespace sjpeg {

// Defined in fdct.cc, used here for the trailing block.
extern void ColumnDct_SSE2(int16_t* in);
extern void RowDct_SSE2(int16_t* in, const __m128i* table1,
                        const __m128i* table2);

///////////////////////////////////////////////////////////////////////////////
// Same constants as in fdct.cc

#define kTan1   (13036)   // = tan(pi/16)
#define kTan2   (27146)   // = tan(2.pi/16) = sqrt(2) - 1.
#define kTan3m1 (-21746)  // = tan(3.pi/16) - 1
#define k2Sqrt2 (23170)   // = 1 / 2.sqrt(2)

// shuffled transposed cosine table, borrowed from kfTables_SSE2[]
static const union {
  const uint16_t s[4 * 8];
  const __m128i m[4];
} kfTables[4] = {
    { { 0x4000, 0x4000, 0x58c5, 0x4b42, 0xdd5d, 0xac61, 0xa73b, 0xcdb7,
        0x4000, 0x4000, 0x3249, 0x11a8, 0x539f, 0x22a3, 0x4b42, 0xee58,
        0x4000, 0xc000, 0x3249, 0xa73b, 0x539f, 0xdd5d, 0x4b42, 0xa73b,
        0xc000, 0x4000, 0x11a8, 0x4b42, 0x22a3, 0xac61, 0x11a8, 0xcdb7 } },
    { { 0x58c5, 0x58c5, 0x7b21, 0x6862, 0xcff5, 0x8c04, 0x84df, 0xba41,
        0x58c5, 0x58c5, 0x45bf, 0x187e, 0x73fc, 0x300b, 0x6862, 0xe782,
        0x58c5, 0xa73b, 0x45bf, 0x84df, 0x73fc, 0xcff5, 0x6862, 0x84df,
        0xa73b, 0x58c5, 0x187e, 0x6862, 0x300b, 0x8c04, 0x187e, 0xba41 } },
    { { 0x539f, 0x539f, 0x73fc, 0x6254, 0xd2bf, 0x92bf, 0x8c04, 0xbe4d,
        0x539f, 0x539f, 0x41b3, 0x1712, 0x6d41, 0x2d41, 0x6254, 0xe8ee,
        0x539f, 0xac61, 0x41b3, 0x8c04, 0x6d41, 0xd2bf, 0x6254, 0x8c04,
        0xac61, 0x539f, 0x1712, 0x6254, 0x2d41, 0x92bf, 0x1712, 0xbe4d } },
    { { 0x4b42, 0x4b42, 0x6862, 0x587e, 0xd746, 0x9dac, 0x979e, 0xc4df,
        0x4b42, 0x4b42, 0x3b21, 0x14c3, 0x6254, 0x28ba, 0x587e, 0xeb3d,
        0x4b42, 0xb4be, 0x3b21, 0x979e, 0x6254, 0xd746, 0x587e, 0x979e,
        0xb4be, 0x4b42, 0x14c3, 0x587e, 0x28ba, 0x9dac, 0x14c3, 0xc4df } } };

///////////////////////////////////////////////////////////////////////////////
// Vertical (column) pass adapted from COLUMN_DCT8 in fdct.cc.

#define BUTTERFLY256(a, b) do {                    \
  (a) = _mm256_sub_epi16((a), (b));                \
  (b) = _mm256_add_epi16((b), (b));                \
  (b) = _mm256_add_epi16((b), (a));                \
} while (0)

static inline __m256i LoadRow(const int16_t* a, const int16_t* b, int row) {
  return _mm256_set_m128i(
      _mm_load_si128(reinterpret_cast<const __m128i*>(b + row * 8)),
      _mm_load_si128(reinterpret_cast<const __m128i*>(a + row * 8)));
}

static inline void StoreRow(int16_t* a, int16_t* b, int row, __m256i v) {
  _mm_store_si128(reinterpret_cast<__m128i*>(a + row * 8),
                  _mm256_castsi256_si128(v));
  _mm_store_si128(reinterpret_cast<__m128i*>(b + row * 8),
                  _mm256_extracti128_si256(v, 1));
}

static void ColumnDct_AVX2(int16_t* a, int16_t* b) {
  __m256i m0 = LoadRow(a, b, 0);
  __m256i m2 = LoadRow(a, b, 2);
  __m256i m7 = LoadRow(a, b, 7);
  __m256i m5 = LoadRow(a, b, 5);

  BUTTERFLY256(m0, m7);
  BUTTERFLY256(m2, m5);

  __m256i m3 = LoadRow(a, b, 3);
  __m256i m4 = LoadRow(a, b, 4);
  BUTTERFLY256(m3, m4);

  __m256i m6 = LoadRow(a, b, 6);
  __m256i m1 = LoadRow(a, b, 1);
  BUTTERFLY256(m1, m6);
  BUTTERFLY256(m7, m4);
  BUTTERFLY256(m6, m5);

  m4 = _mm256_slli_epi16(m4, 3);
  m5 = _mm256_slli_epi16(m5, 3);
  BUTTERFLY256(m4, m5);
  StoreRow(a, b, 0, m5);
  StoreRow(a, b, 4, m4);

  m7 = _mm256_slli_epi16(m7, 3);
  m6 = _mm256_slli_epi16(m6, 3);
  m3 = _mm256_slli_epi16(m3, 3);
  m0 = _mm256_slli_epi16(m0, 3);

  m4 = _mm256_set1_epi16(kTan2);
  m5 = m4;
  m4 = _mm256_mulhi_epi16(m4, m7);
  m5 = _mm256_mulhi_epi16(m5, m6);
  m4 = _mm256_sub_epi16(m4, m6);
  m5 = _mm256_add_epi16(m5, m7);
  StoreRow(a, b, 2, m5);
  StoreRow(a, b, 6, m4);

  m6 = _mm256_set1_epi16(k2Sqrt2);
  m2 = _mm256_slli_epi16(m2, 3 + 1);
  m1 = _mm256_slli_epi16(m1, 3 + 1);
  BUTTERFLY256(m1, m2);
  m2 = _mm256_mulhi_epi16(m2, m6);
  m1 = _mm256_mulhi_epi16(m1, m6);
  BUTTERFLY256(m3, m1);
  BUTTERFLY256(m0, m2);

  m4 = _mm256_set1_epi16(kTan3m1);
  m5 = _mm256_set1_epi16(kTan1);
  m7 = m3;
  m6 = m1;
  m3 = _mm256_mulhi_epi16(m3, m4);
  m1 = _mm256_mulhi_epi16(m1, m5);

  m3 = _mm256_add_epi16(m3, m7);
  m1 = _mm256_add_epi16(m1, m2);
  const __m256i kOne = _mm256_set1_epi16(1);
  m1 = _mm256_adds_epi16(m1, kOne);
  m3 = _mm256_adds_epi16(m3, kOne);
  m4 = _mm256_mulhi_epi16(m4, m0);
  m5 = _mm256_mulhi_epi16(m5, m2);
  m4 = _mm256_add_epi16(m4, m0);
  m0 = _mm256_sub_epi16(m0, m3);
  m7 = _mm256_add_epi16(m7, m4);
  m5 = _mm256_sub_epi16(m5, m6);

  StoreRow(a, b, 1, m1);
  StoreRow(a, b, 3, m0);
  StoreRow(a, b, 5, m7);
  StoreRow(a, b, 7, m5);
}

#undef BUTTERFLY256

///////////////////////////////////////////////////////////////////////////////
// Horizontal (row) pass, adapted from RowDct_SSE2 in fdct.cc

static void RowDct_AVX2(int16_t* a, int16_t* b, const __m128i* table1,
                        const __m128i* table2) {
  const __m256i t1_0 = _mm256_broadcastsi128_si256(table1[0]);
  const __m256i t1_1 = _mm256_broadcastsi128_si256(table1[1]);
  const __m256i t1_2 = _mm256_broadcastsi128_si256(table1[2]);
  const __m256i t1_3 = _mm256_broadcastsi128_si256(table1[3]);
  const __m256i t2_0 = _mm256_broadcastsi128_si256(table2[0]);
  const __m256i t2_1 = _mm256_broadcastsi128_si256(table2[1]);
  const __m256i t2_2 = _mm256_broadcastsi128_si256(table2[2]);
  const __m256i t2_3 = _mm256_broadcastsi128_si256(table2[3]);

  // load row [0123|4567] as [0123|7654], for both blocks
  __m256i m0 = _mm256_shufflehi_epi16(LoadRow(a, b, 0), 0x1b);
  __m256i m2 = _mm256_shufflehi_epi16(LoadRow(a, b, 1), 0x1b);

  // process two rows in parallel (per block)
  __m256i m4 = m0;
  m0 = _mm256_castps_si256(
      _mm256_shuffle_ps(_mm256_castsi256_ps(m0), _mm256_castsi256_ps(m2),
                        0x44));
  m4 = _mm256_castps_si256(
      _mm256_shuffle_ps(_mm256_castsi256_ps(m4), _mm256_castsi256_ps(m2),
                        0xee));

  // initial butterfly
  m2 = m0;
  m0 = _mm256_add_epi16(m0, m4);
  m2 = _mm256_sub_epi16(m2, m4);

  // prepare for scalar products (madd_epi16)
  __m256i m6;
  m4 = m0;
  m0 = _mm256_unpacklo_epi32(m0, m2);
  m4 = _mm256_unpackhi_epi32(m4, m2);
  m2 = _mm256_shuffle_epi32(m0, 0x4e);
  m6 = _mm256_shuffle_epi32(m4, 0x4e);

  __m256i m1, m3, m5, m7;
  m1 = _mm256_madd_epi16(m2, t1_1);
  m3 = _mm256_madd_epi16(m0, t1_2);
  m5 = _mm256_madd_epi16(m6, t2_1);
  m7 = _mm256_madd_epi16(m4, t2_2);

  m2 = _mm256_madd_epi16(m2, t1_3);
  m0 = _mm256_madd_epi16(m0, t1_0);
  m6 = _mm256_madd_epi16(m6, t2_3);
  m4 = _mm256_madd_epi16(m4, t2_0);

  // add the sub-terms
  m0 = _mm256_add_epi32(m0, m1);
  m4 = _mm256_add_epi32(m4, m5);
  m2 = _mm256_add_epi32(m2, m3);
  m6 = _mm256_add_epi32(m6, m7);

  // descale
  m0 = _mm256_srai_epi32(m0, 16);
  m4 = _mm256_srai_epi32(m4, 16);
  m2 = _mm256_srai_epi32(m2, 16);
  m6 = _mm256_srai_epi32(m6, 16);

  m0 = _mm256_packs_epi32(m0, m2);
  m4 = _mm256_packs_epi32(m4, m6);

  StoreRow(a, b, 0, m0);
  StoreRow(a, b, 1, m4);
}

///////////////////////////////////////////////////////////////////////////////

static void Dct2Blocks_AVX2(int16_t* a, int16_t* b) {
  ColumnDct_AVX2(a, b);
  RowDct_AVX2(a + 0 * 8, b + 0 * 8, kfTables[0].m, kfTables[1].m);
  RowDct_AVX2(a + 2 * 8, b + 2 * 8, kfTables[2].m, kfTables[3].m);
  RowDct_AVX2(a + 4 * 8, b + 4 * 8, kfTables[0].m, kfTables[3].m);
  RowDct_AVX2(a + 6 * 8, b + 6 * 8, kfTables[2].m, kfTables[1].m);
}

static void DctOneBlock_SSE2(int16_t* in) {
  ColumnDct_SSE2(in);
  RowDct_SSE2(in + 0 * 8, kfTables[0].m, kfTables[1].m);
  RowDct_SSE2(in + 2 * 8, kfTables[2].m, kfTables[3].m);
  RowDct_SSE2(in + 4 * 8, kfTables[0].m, kfTables[3].m);
  RowDct_SSE2(in + 6 * 8, kfTables[2].m, kfTables[1].m);
}

void FdctAVX2(int16_t* coeffs, int num_blocks) {
  while (num_blocks >= 2) {
    Dct2Blocks_AVX2(coeffs, coeffs + 64);
    coeffs += 2 * 64;
    num_blocks -= 2;
  }
  if (num_blocks > 0) {
    DctOneBlock_SSE2(coeffs);
  }
}

#undef kTan1
#undef kTan2
#undef kTan3m1
#undef k2Sqrt2

}  // namespace sjpeg

#endif  // SJPEG_USE_AVX2
