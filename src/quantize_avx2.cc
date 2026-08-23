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
//  AVX2 variant of QuantizeBlock(), compiled with -mavx2 only for this file
//  (see Makefile's src/%_avx2.o rule). 256-bit widening of QuantizeBlockSSE2
//  in quantize.cc: same per-lane abs/bias/mulhi/shift/sign-restore, no
//  cross-lane shuffle needed there. The scalar tail (zig-zag remap + run/
//  level emission, the dominant cost of this function -- see the
//  SJPEG_HAVE_AVX2 commit message) is untouched, copied verbatim from
//  QuantizeBlockSSE2.
//
//  One AVX2-specific wrinkle: _mm256_packs_epi16/_mm256_movemask_epi8
//  operate per-128-bit-lane, not linearly across the full 256 bits (unlike
//  their SSE2 counterparts), so packing+movemask-ing 16 lanes' worth of
//  compare results does not give a contiguous 16-bit mask the way the
//  8-lane SSE2 version's _mm_movemask_epi8(...) & 0xff does. The 32-bit
//  result instead has the two real 8-bit halves (from the compare's low
//  and high 128-bit lanes) 8 bits apart with a duplicate byte pair in
//  between (m32 bits [8:15] and [24:31] are exact duplicates of [0:7] and
//  [16:23], an artifact of packing 'cmp' against itself); compacted back to
//  a real 16-bit mask with one shift+mask+or below.
//
// Author: Skal (pascal.massimino@gmail.com)

#define SJPEG_NEED_ASM_HEADERS
#include "sjpegi.h"

#if defined(SJPEG_USE_AVX2)

// Uses BMI2 pext to compact the non-zero mask (2 ops vs 5). Off by default:
// pext is microcoded and slower on AMD Zen1/Zen+. Needs -mbmi2 if enabled.
// #define SJPEG_USE_PEXT

namespace sjpeg {

int QuantizeBlockAVX2(const int16_t in[64], int idx, const Quantizer* const Q,
                      DCTCoeffs* const out, RunLevel* const rl) {
  const uint16_t* const bias = Q->bias_;
  const uint16_t* const iquant = Q->iquant_;
  int prev = 1;
  int nb = 0;
  int16_t tmp[64], masked[64];
  const __m256i zero = _mm256_setzero_si256();
  uint64_t nzn = 0;  // natural-order non-zero mask: bit j set iff tmp[j] != 0.
  for (int i = 0; i < 64; i += 16) {
    const __m256i m_bias =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias + i));
    const __m256i m_mult =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(iquant + i));
    const __m256i A =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i));
    const __m256i B = _mm256_srai_epi16(A, 15);  // sign (still needed below)
    const __m256i C = _mm256_abs_epi16(A);        // abs(A)
    const __m256i D = _mm256_adds_epi16(C, m_bias);            // v' = v+bias
    const __m256i E = _mm256_mulhi_epu16(D, m_mult);           // (v'*iq)>>16
    const __m256i F = _mm256_srli_epi16(E, AC_BITS);           // QUANTIZE(...)
    const __m256i G = _mm256_xor_si256(F, B);                  // v ^ mask
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp + i), F);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(masked + i), G);
    const __m256i cmp = _mm256_cmpgt_epi16(F, zero);
#if defined(SJPEG_USE_PEXT)
    const uint32_t m32 = static_cast<uint32_t>(_mm256_movemask_epi8(cmp));
    const uint32_t m16 = static_cast<uint32_t>(_pext_u32(m32, 0x55555555u));
#else
    // compact the doubled/interleaved pack+movemask result to a real mask
    const uint32_t m32 = static_cast<uint32_t>(
        _mm256_movemask_epi8(_mm256_packs_epi16(cmp, cmp)));
    const uint32_t m16 = (m32 & 0xff) | ((m32 >> 8) & 0xff00);
#endif
    nzn |= static_cast<uint64_t>(m16) << i;
  }
  // Emit run/level entries -- identical to QuantizeBlockSSE2/NEON/C, see
  // quantize.cc for the explanation of this bit-trick zig-zag scan.
  uint64_t zz = 0;
  for (uint64_t b = nzn & ~1ull; b != 0; b &= b - 1) {
    zz |= 1ull << kInvZigzag[TrailingZeros64(b)];
  }
  for (uint64_t b = zz; b != 0; b &= b - 1) {
    const int i = static_cast<int>(TrailingZeros64(b));
    const int j = kZigzag[i];
    const int n = CalcLog2(tmp[j]);
    const uint16_t code = masked[j] & ((1 << n) - 1);
    rl[nb].level_ = (code << 4) | n;
    rl[nb].run_ = i - prev;
    prev = i + 1;
    ++nb;
  }
  const int dc = (in[0] < 0) ? -tmp[0] : tmp[0];
  out->idx_ = idx;
  out->last_ = prev - 1;
  out->nb_coeffs_ = nb;
  return dc;
}

}  // namespace sjpeg

#endif  // SJPEG_USE_AVX2
