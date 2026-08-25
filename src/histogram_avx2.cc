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
//  AVX2 variant of StoreHisto(), compiled with -mavx2 only for this file
//  (see Makefile's src/%_avx2.o rule) so the rest of the library doesn't
//  need to be built for an AVX2 target. 256-bit widening of StoreHistoSSE2
//  in histogram.cc: same per-lane abs/shift/min, no cross-lane shuffle, so
//  it doubles lane count 8 -> 16 with no algorithmic change. The trailing
//  per-coefficient histogram increment stays scalar either way (data-
//  dependent scatter, not vectorizable with AVX2).
//
// Author: Skal (pascal.massimino@gmail.com)

#define SJPEG_NEED_ASM_HEADERS
#include "sjpegi.h"

#include <assert.h>

#if defined(SJPEG_USE_AVX2)

namespace sjpeg {

#define LOAD_32(src) _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src))
#define STORE_32(V, dst) \
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), (V))

void StoreHistoAVX2(const int16_t in[64], Histo* const histos,
                    int nb_blocks) {
  assert(nb_blocks > 0);
  const __m256i kMaxHisto = _mm256_set1_epi16(MAX_HISTO_DCT_COEFF);
  int n = 0;
  do {
    uint16_t tmp[64];
    for (int i = 0; i < 64; i += 16) {
      const __m256i A = LOAD_32(in + i);
      const __m256i C = _mm256_abs_epi16(A);
      const __m256i D = _mm256_srli_epi16(C, HSHIFT);
      const __m256i E = _mm256_min_epi16(D, kMaxHisto);
      STORE_32(E, tmp + i);
    }
    for (int j = 0; j < 64; ++j) {
      const int k = tmp[j];
      ++histos->counts_[j][k];
    }
    in += 64;
  } while (++n < nb_blocks);
}

#undef LOAD_32
#undef STORE_32

}  // namespace sjpeg

#endif  // SJPEG_USE_AVX2
