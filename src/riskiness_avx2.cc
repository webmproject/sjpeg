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
// AVX2/gather variant of the inner loop of SjpegRiskiness() (jpeg_tools.cc).
// This version uses 'gather' instruction, known to be slow on first-gen AVX2
// hardware (Haswell/Broadwell).
//
// kSharpnessScore[] is a byte table but AVX2 has no byte gather. This gathers
// dwords at scale=1 (byte-granular index, dword-aligned read) and masks the
// low byte -- see score_7.cc: 3 extra trailing bytes for valid access.
//
// Author: Skal (pascal.massimino@gmail.com)

#define SJPEG_NEED_ASM_HEADERS
#include "sjpegi.h"

#if defined(SJPEG_USE_AVX2) && defined(SJPEG_USE_AVX2_RISKINESS)

namespace sjpeg {

static inline int32_t HorizontalSumEpi32(__m256i v) {
  const __m128i lo = _mm256_castsi256_si128(v);
  const __m128i hi = _mm256_extracti128_si256(v, 1);
  const __m128i sum = _mm_add_epi32(lo, hi);
  int32_t tmp[4];
  _mm_storeu_si128(reinterpret_cast<__m128i*>(tmp), sum);
  return tmp[0] + tmp[1] + tmp[2] + tmp[3];
}

// Processes 'size' samples in [0, size), 8 at a time.
// Returns how many samples were actually consumed (a multiple of 8, <= size).
// Caller needs to handle the [return value, size) remainder using C-version.
int RiskinessScoreRowAVX2(const uint16_t* row1, const uint16_t* row2,
                          int size, int noise_level,
                          int64_t* const score_sum, int64_t* const score_num,
                          int64_t* const gray_num) {
  const int s = kRGBSize;
  const int K = s * s * s;
  const int gray = (s / 2) * (1 + s) * s;   // gray level for y=0,u=128,v=128
  const int gray_min = gray - gray % s;

  const int32_t* const table =
      reinterpret_cast<const int32_t*>(kSharpnessScore);
  const __m256i K_vec = _mm256_set1_epi32(K);
  const __m256i noise_vec = _mm256_set1_epi32(noise_level);
  const __m256i gray_min_m1_vec = _mm256_set1_epi32(gray_min - 1);
  const __m256i gray_max_vec = _mm256_set1_epi32(gray_min + s);
  const __m256i mask_ff = _mm256_set1_epi32(0xff);
  const __m256i zero = _mm256_setzero_si256();

  // 8-lane accumulators, reduced post-loop. Overflow-safe since kMaxDimension
  // is 65535 => even the worst case (max score on every iteration) stays under
  // 32bit limit (65534/8 * 765 ~= 6.3M)
  __m256i sum_vec = zero;    // scores above the noise level
  __m256i num_vec = zero;    // number of sum_vec
  __m256i gray_vec = zero;   // samples with neutral chroma
  int i = 0;
  for (; i + 8 <= size; i += 8) {
    const __m128i r1_0 =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(row1 + i));
    const __m128i r1_1 =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(row1 + i + 1));
    const __m128i r2_0 =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(row2 + i));
    const __m256i V0 = _mm256_cvtepu16_epi32(r1_0);   // idx0
    const __m256i V1 = _mm256_cvtepu16_epi32(r1_1);   // idx1
    const __m256i V2 = _mm256_cvtepu16_epi32(r2_0);   // idx2

    const __m256i V1K = _mm256_mullo_epi32(V1, K_vec);
    const __m256i V2K = _mm256_mullo_epi32(V2, K_vec);
    const __m256i A = _mm256_add_epi32(V0, V1K);   // idx0 + K*idx1
    const __m256i B = _mm256_add_epi32(V0, V2K);   // idx0 + K*idx2
    const __m256i C = _mm256_add_epi32(V1, V2K);   // idx1 + K*idx2

    const __m256i GA =
        _mm256_and_si256(_mm256_i32gather_epi32(table, A, 1), mask_ff);
    const __m256i GB =
        _mm256_and_si256(_mm256_i32gather_epi32(table, B, 1), mask_ff);
    const __m256i GC =
        _mm256_and_si256(_mm256_i32gather_epi32(table, C, 1), mask_ff);
    const __m256i score = _mm256_add_epi32(_mm256_add_epi32(GA, GB), GC);

    const __m256i score_mask = _mm256_cmpgt_epi32(score, noise_vec);
    const __m256i ge_mask = _mm256_cmpgt_epi32(V0, gray_min_m1_vec);
    const __m256i lt_mask = _mm256_cmpgt_epi32(gray_max_vec, V0);
    const __m256i gray_mask = _mm256_and_si256(ge_mask, lt_mask);

    sum_vec = _mm256_add_epi32(sum_vec, _mm256_and_si256(score, score_mask));
    num_vec = _mm256_sub_epi32(num_vec, score_mask);
    gray_vec = _mm256_sub_epi32(gray_vec, gray_mask);
  }
  *score_sum += HorizontalSumEpi32(sum_vec);
  *score_num += HorizontalSumEpi32(num_vec);
  *gray_num += HorizontalSumEpi32(gray_vec);
  return i;
}

}  // namespace sjpeg

#endif  // SJPEG_USE_AVX2 && SJPEG_USE_AVX2_RISKINESS
