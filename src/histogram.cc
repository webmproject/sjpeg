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
//  Coefficient histograms, and quant matrices derived from them
//
// Author: Skal (pascal.massimino@gmail.com)

#include <assert.h>
#include <float.h>  // for FLT_MAX
#include <stdint.h>
#include <string.h>

#define SJPEG_NEED_ASM_HEADERS
#include "sjpegi.h"

namespace sjpeg {

// finer tuning of perceptual optimizations, all read by AnalyseHisto() only:

// Minimum average number of entries per bin required for performing histogram-
// -based optimization. Below this limit, the channel's histogram is declared
// under-populated and the corresponding optimization skipped.
static const double kDensityThreshold = 0.5;
// Rejection limit on the correlation factor when extrapolating the distortion
// from histograms. If the least-square fit has a squared correlation factor
// less than this threshold, the corresponding quantization scale will be
// kept unchanged.
static const double kCorrelationThreshold = 0.5;
// Bit-map of channels to omit during quantization matrix optimization.
// If the bit 'i + 8 * j' is set in this bit field, the matrix entry at
// position (i,j) will be kept unchanged during optimization.
// The default value is 0x103 = 1 + 2 + 256: the 3 entries in the top-left
// corner (with lowest-frequency) are not optimized, since it can lead to
// visual degradation of smooth gradients.
static const uint64_t kOmittedChannels = 0x0000000000000103ULL;

////////////////////////////////////////////////////////////////////////////////
// Histogram

void Encoder::ResetHisto() {
  memset(histos_, 0, sizeof(histos_));
}

#if defined(SJPEG_USE_SSE2)
void StoreHistoSSE2(const int16_t in[64], Histo* const histos, int nb_blocks) {
  const __m128i kMaxHisto = _mm_set1_epi16(MAX_HISTO_DCT_COEFF);
  for (int n = 0; n < nb_blocks; ++n, in += 64) {
    uint16_t tmp[64];
    for (int i = 0; i < 64; i += 8) {
      const __m128i A =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + i));
#if defined(SJPEG_USE_SSSE3)
      const __m128i C = _mm_abs_epi16(A);
#else
      const __m128i B = _mm_srai_epi16(A, 15);                  // sign extract
      const __m128i C = _mm_sub_epi16(_mm_xor_si128(A, B), B);  // abs(A)
#endif
      const __m128i D = _mm_srli_epi16(C, HSHIFT);              // >>= HSHIFT
      const __m128i E = _mm_min_epi16(D, kMaxHisto);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(tmp + i), E);
    }
    for (int j = 0; j < 64; ++j) {
      const int k = tmp[j];
      ++histos->counts_[j][k];
    }
  }
}
#elif defined(SJPEG_USE_NEON)
void StoreHistoNEON(const int16_t in[64], Histo* const histos, int nb_blocks) {
  const uint16x8_t kMaxHisto = vdupq_n_u16(MAX_HISTO_DCT_COEFF);
  for (int n = 0; n < nb_blocks; ++n, in += 64) {
    uint16_t tmp[64];
    for (int i = 0; i < 64; i += 8) {
      const int16x8_t A = vld1q_s16(in + i);
      const int16x8_t B = vabsq_s16(A);               // abs(in)
      const uint16x8_t C = vreinterpretq_u16_s16(B);  // signed->unsigned
      const uint16x8_t D = vshrq_n_u16(C, HSHIFT);    // >>= HSHIFT
      const uint16x8_t E = vminq_u16(D, kMaxHisto);   // min(.,kMaxHisto)
      vst1q_u16(tmp + i, E);
    }
    for (int j = 0; j < 64; ++j) {
      const int k = tmp[j];
      ++histos->counts_[j][k];
    }
  }
}
#endif

// This C-version is does not produce the same counts_[] output than the
// assembly above. But the extra entry counts_[MAX_HISTO_DCT_COEFF] is
// not used for the final computation, and the global result is unchanged.
void StoreHisto(const int16_t in[64], Histo* const histos, int nb_blocks) {
  for (int n = 0; n < nb_blocks; ++n, in += 64) {
    for (int i = 0; i < 64; ++i) {
      const int k = (in[i] < 0 ? -in[i] : in[i]) >> HSHIFT;
      if (k < MAX_HISTO_DCT_COEFF) {
        ++histos->counts_[i][k];
      }
    }
  }
}

Encoder::StoreHistoFunc Encoder::GetStoreHistoFunc() {
#if defined(SJPEG_USE_SSE2)
  if (SupportsSSE2()) return StoreHistoSSE2;
#elif defined(SJPEG_USE_NEON)
  if (SupportsNEON()) return StoreHistoNEON;
#endif
  return StoreHisto;  // default
}

const float Encoder::kHistoWeight[QSIZE] = {
  // Gaussian with sigma ~= 3
  0, 0, 0, 0, 0,
  1,   5,  16,  43,  94, 164, 228, 255, 228, 164,  94,  43,  16,   5,   1,
  0, 0, 0, 0, 0
};

void Encoder::AnalyseHisto() {
  // A bit of theory and background: for each sub-band i in [0..63], we pick a
  // quantization scale New_Qi close to the initial one Qi. We evaluate a cost
  // function associated with F({New_Qi}) = distortion + lambda . rate,
  // where rate and distortion depend on the quantizers set in a complex non-
  // analytic way. Just, for well-behaved regular histograms, we expect the
  // rate to scale as -log(Q), and the distortion as Q^2.
  // We want the cost function to be stationnary around the initial {Qi} set,
  // in order to achieve the best transfer between distortion and rate when we
  // displace a little the Qi values. Mainly we want to use bits as efficiently
  // as possible, where every bit we use has maximal impact in lowering
  // distortion (and vice versa: if we spend an extra bit of coding, we want to
  // have the best bang for this buck. The optimization works up-hill too).
  //
  // Hence, lambda is picked to minimize F around {Qi}, as:
  //    lambda = -d(distortion) / d(rate)
  // where the derivates are evaluated using a double least-square fit on both
  // the clouds of {delta, distortion} and {delta, size} points.
  //
  // Note1: The least-square fitted slope of a {x,y} cloud is expressed as:
  //    slope = (<xy> - <x><y>) / (<xx> - <x><x>) = Cov(x,y) / Cov(x,x)
  // where <.> is our gaussian-averaging operator.
  // But since we are eventually computing a quotient of such slopes, we can
  // factor out the common (<xx> - <x><x>) denominator (which is strictly
  // positive).
  // Note2: we use a Gaussian-weighted average around the center value Qi
  // instead of averaging over the whole [QDELTA_MIN, QDELTA_MAX] range.
  // This rules out fringe samples on noisy cases (like: when the source is
  // already JPEG-compressed!).
  // Note3: We fall back to some sane value HLAMBDA in case of ill-condition.
  //
  // We use use the correlation coefficient
  //       r = Cov(x,y) / sqrt(Cov(x,x) * Cov(y,y))
  // to detect bad cases with poorly extrapolated distortion. In such
  // occurrence, we skip the channel. This is particularly important for
  // already-compressed JPEG sources that give treacherous comb-like
  // histograms.
  //
  // Once this particular lambda has been picked, we loop over each channel
  // and optimize them separately, locally picking the best New_Qi for each.
  // The choice of lambda ensure a good balancing between size and distortion,
  // and prevent being too aggressive on file-size reduction for instance.
  //
  const double r_limit = kCorrelationThreshold;
  for (int c = (nb_comps_ > 1 ? 1 : 0); c >= 0; --c) {
    const int idx = quant_idx_[c];
    const Histo* const histo = &histos_[idx];
    // For chrominance, it can be visually damageable to be too
    // aggressive on the filesize. So with the default settings we
    // restrict the algorithm to mainly try to *increase* the bitrate
    // (and quality) by using a smaller qdelta_max_chroma_.
    // delta_max is only use during the second phase, but not during
    // the first phase of deriving an optimal lambda.
    assert(QDELTA_MAX >= qdelta_max_luma_);
    assert(QDELTA_MAX >= qdelta_max_chroma_);
    const int delta_max =
      ((idx == 0) ? qdelta_max_luma_ : qdelta_max_chroma_) - QDELTA_MIN;
    assert(delta_max < QSIZE);
    float sizes[64][QSIZE];
    float distortions[64][QSIZE];
    double num = 0.;  // accumulate d(distortion) around delta_q = 0
    double den = 0.;  // accumulate d(size) around delta_q = 0
    uint64_t omit_channels = kOmittedChannels;
    for (int pos = 0; pos < 64; ++pos) {
      if (omit_channels & (1ULL << pos)) {
        continue;
      }
      const int dq0 = quants_[idx].quant_[pos];
      const int min_dq0 = quants_[idx].min_quant_[pos];
      // We should be using the exact bias:
      //    const int bias = quants_[idx].bias_[pos] << (FP_BITS - AC_BITS);
      // but this value is too precise considering the other approximations
      // we're using (namely: HSHIFT). So we better use the a mid value of 0.5
      // for the bias. This have the advantage of making it possible to
      // use pre-calculated look-up tables for every quantities in the loop.
      // This is still a TODO(skal) below, though. Not sure the gain is big.
      const int bias = 1 << FP_BITS >> 1;
      const int* const h = histo->counts_[pos];
      int total = 0;
      int last = 0;
      for (int i = 0; i < MAX_HISTO_DCT_COEFF; ++i) {
        total += h[i];
        if (h[i]) last = i + 1;
      }
      if (total < kDensityThreshold * last) {
        omit_channels |= 1ULL << pos;
        continue;
      }
      // accumulators for averaged values.
      double sw = 0., sx = 0.;
      double sxx = 0., syy1 = 0.;
      double sy1 = 0., sxy1 = 0.;   // accumulators for distortion cloud
      double sy2 = 0., sxy2 = 0.;   // accumulators for size cloud
      for (int delta = 0; delta < QSIZE; ++delta) {
        double bsum = 0., dsum = 0.;
        const int dq = dq0 + (delta + QDELTA_MIN);
        if (dq >= min_dq0 && dq <= 255) {
          // TODO(skal): pre-compute idq and use it in FinalizeQuantMatrix too
          const int idq = ((1 << FP_BITS) + dq - 1) / dq;
          for (int i = 0; i < last; ++i) {
            if (h[i]) {
              // v = current bin's centroid in the histogram
              // qv = quantized value for the bin's representant 'v'
              // dqv = dequantized qv, to be compared against v (=> 'error')
              // bits = approximate bit-cost of quantized representant
              // h[i] = this bin's weight
              const int v = (i << HSHIFT) + HHALF;
              const int qv = (v * idq + bias) >> FP_BITS;
              // TODO(skal): for a given 'last' value, we know the upper limit
              // on dq that will make *all* quantized 'qv' values be zero.
              // => We can restrict the loop on 'dq' using 'last'.
              if (qv) {
                const int bits = CalcLog2(qv);
                const int dqv = qv * dq;
                const int error = (v - dqv) * (v - dqv);
                bsum += h[i] * bits;
                dsum += h[i] * error;
              } else {
                dsum += h[i] * v * v;
              }
            }
          }   // end of 'i' loop
          distortions[pos][delta] = static_cast<float>(dsum);
          sizes[pos][delta] = static_cast<float>(bsum);
          const double w = kHistoWeight[delta];   // Gaussian weight
          if (w > 0.) {
            const double x = static_cast<double>(delta + QDELTA_MIN);
            sw   += w;
            sx   += w * x;
            sxx  += w * x * x;
            sy1  += w * dsum;
            syy1 += w * dsum * dsum;
            sy2  += w * bsum;
            sxy1 += w * dsum * x;
            sxy2 += w * bsum * x;
          }
        } else {  // the new quantizer is out-of-range.
          distortions[pos][delta] = FLT_MAX;
          sizes[pos][delta] = 0;
        }
      }
      // filter channels according to correlation factor.
      const double cov_xy1 = sw * sxy1 - sx * sy1;
      if (cov_xy1 * cov_xy1 < r_limit *
                              (sw * sxx - sx * sx) * (sw * syy1 - sy1 * sy1)) {
        omit_channels |= 1ULL << pos;
        continue;
      }
      // accumulate numerator and denominator for the derivate calculation
      num += cov_xy1;
      den += sw * sxy2 - sx * sy2;
    }

    // we evaluate lambda =~ -d(distortion)/d(size) at dq=0
    double lambda = HLAMBDA;
    // When increasing Q, size should significantly decrease and distortion
    // increase. If they don't, we are ill-conditionned and should fall back
    // to a safe value HLAMBDA.
    if (num > 1000. && den < -10.) {
      // This is our approximation of -d(Distortion) / d(Rate)
      // We limit it to 1. below, to avoid degenerated cases
      lambda = -num / den;
      if (lambda < 1.) {
        lambda = 1.;
      }
    }
    // now, optimize each channel using the optimal lambda selection
    for (int pos = 0; pos < 64; ++pos) {
      if (omit_channels & (1ULL << pos)) {
        continue;
      }
      float best_score = FLT_MAX;
      int best_dq = 0;
      for (int delta = 0; delta <= delta_max; ++delta) {
        if (distortions[pos][delta] < FLT_MAX) {
          const float score = distortions[pos][delta]
                            + lambda * sizes[pos][delta];
          if (score < best_score) {
            best_score = score;
            best_dq = delta + QDELTA_MIN;
          }
        }
      }
      quants_[idx].quant_[pos] += best_dq;
      assert(quants_[idx].quant_[pos] >= 1);
    }
    FinalizeQuantMatrix(&quants_[idx], q_bias_);
    SetCostCodes(idx);
  }
}

void Encoder::CollectHistograms() {
  ResetHisto();
  int16_t* in = in_blocks_;
  const int mb_x_max = W_ / block_w_;
  const int mb_y_max = H_ / block_h_;
  const bool use_extra_memory = use_extra_memory_;
  for (int mb_y = 0; mb_y < mb_h_; ++mb_y) {
    const bool yclip = (mb_y == mb_y_max);
    for (int mb_x = 0; mb_x < mb_w_; ++mb_x) {
      if (!use_extra_memory) {
        in = in_blocks_;
      }
      GetSamples(mb_x, mb_y, yclip | (mb_x == mb_x_max), in);
      fDCT_(in, mcu_blocks_);
      for (int c = 0; c < nb_comps_; ++c) {
        const int num_blocks = nb_blocks_[c];
        store_histo_(in, &histos_[quant_idx_[c]], num_blocks);
        in += 64 * num_blocks;
      }
    }
  }
  have_coeffs_ = use_extra_memory_;
}

}    // namespace sjpeg
