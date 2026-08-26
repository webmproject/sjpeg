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
//  Misc tools for quickly parsing JPEG data
//
// Author: Skal (pascal.massimino@gmail.com)

#include <assert.h>
#include <math.h>     // for fabs
#include <stdint.h>
#include <string.h>   // for memset

#include <algorithm>  // for std::min, std::max
#include <cstdlib>
#include <utility>   // for std::swap
#include <vector>

#include "sjpegi.h"

///////////////////////////////////////////////////////////////////////////////
// Dimensions (SOF)

namespace {
// This function will quickly locate the first appareance of an SOF marker in
// the passed JPEG buffer. It assumes the streams starts wih an SOI marker,
// like any valid JPEG should. Returned value points to the beginning of the
// marker and is guarantied to contain a least 8 bytes of valid data.
// Positions are used instead of pointers, to never build one past the end.
const uint8_t* GetSOFData(const uint8_t* src, size_t size) {
  if (src == nullptr || size < 2 + 8) return nullptr;
  const size_t end = size - 8;   // 8 bytes of safety, for the marker
  size_t pos = 2;   // skip M_SOI
  while (pos < end && src[pos] != 0xff) { ++pos; }  // search first 0xff marker
  while (pos < end) {
    const uint32_t marker =
        static_cast<uint32_t>((src[pos] << 8) | src[pos + 1]);
    if (marker == M_SOF0 || marker == M_SOF1) return src + pos;
    pos += 2 + ((src[pos + 2] << 8) | src[pos + 3]);
  }
  return nullptr;  // No SOF marker found
}
}   // anonymous namespace

bool SjpegDimensions(const uint8_t* src0, size_t size,
                     int* width, int* height, int* is_yuv420) {
  const uint8_t* const src = GetSOFData(src0, size);
  if (src == nullptr) return false;
  const size_t left_over = size - static_cast<size_t>(src - src0);
  if (left_over < 8 + 3 * 1) return false;
  if (height != nullptr) *height = (src[5] << 8) | src[6];
  if (width != nullptr) *width = (src[7] << 8) | src[8];
  if (is_yuv420 != nullptr) {
    const size_t nb_comps = src[9];
    *is_yuv420 = (nb_comps == 3);
    if (left_over < 11 + 3 * nb_comps) return false;
    for (int c = 0; *is_yuv420 && c < 3; ++c) {
      const int expected_dim = (c == 0 ? 0x22 : 0x11);
      *is_yuv420 &= (src[11 + c * 3] == expected_dim);
    }
  }
  return true;
}

///////////////////////////////////////////////////////////////////////////////
// Quantizer marker (DQT)

int SjpegFindQuantizer(const uint8_t* src, size_t size,
                       uint8_t quant[2][64]) {
  memset(quant[0], 0, sizeof(quant[0]));
  memset(quant[1], 0, sizeof(quant[1]));
  // minimal size for 64 coeffs and the markers (5 bytes)
  if (src == nullptr || size < 69 || src[0] != 0xff || src[1] != 0xd8) {
    return 0;
  }
  const uint8_t* const end = src + size - 8;   // 8 bytes of safety, for marker
  src += 2;   // skip over the initial M_SOI
  for (; src < end && *src != 0xff; ++src) { /* search first 0xff marker */ }
  int nb_comp = 0;
  while (src < end) {
    const uint32_t marker = static_cast<uint32_t>((src[0] << 8) | src[1]);
    const int chunk_size = 2 + ((src[2] << 8) | src[3]);
    if (src + chunk_size > end) {
      break;
    }
    if (marker == M_SOS) {
      // we can stop searching at the first SOS marker encountered, to avoid
      // parsing the whole data
      break;
    } else if (marker == M_DQT) {
      // Jump over packets of 1 index + 64 coeffs
      int i = 4;
      while (i + 1 < chunk_size) {
        const int Pq = src[i] >> 4;
        const int Tq = src[i] & 0x0f;
        if (Pq > 1 || Tq > 3) return 0;    // invalid bitstream. See B.4.
        const int m_size = 64 * Pq + 65;
        if (i + m_size > chunk_size) return 0;
        if (Tq < 2) {
          for (int j = 0; j < 64; ++j) {
            int v;
            if (Pq == 0) {
              v = src[i + 1 + j];
            } else {
              // convert 16b->8b by clamping
              v = ((int)src[i + 1 + 2 * j + 0] << 8)
                      | src[i + 1 + 2 * j + 1];
              v = (v > 255) ? 255 : v;
            }
            quant[Tq][sjpeg::kZigzag[j]] = (v < 1) ? 1u : (uint8_t)v;
          }
        } else {
          // we don't store the pointer, but we record the component
        }
        nb_comp |= 1 << Tq;
        i += m_size;
      }
    }
    src += chunk_size;
  }
  return ((nb_comp & 1) != 0) + ((nb_comp & 2) != 0)
       + ((nb_comp & 4) != 0) + ((nb_comp & 8) != 0);
}

///////////////////////////////////////////////////////////////////////////////

void SjpegQuantMatrix(float quality, bool for_chroma, uint8_t matrix[64]) {
  const float q_factor = sjpeg::GetQFactor(quality) / 100.f;
  const uint8_t* const matrix0 = sjpeg::kDefaultMatrices[for_chroma];
  for (int i = 0; i < 64; ++i) {
    const int v = (int)(matrix0[i] * q_factor + .5f);
    matrix[i] = (v < 1) ? 1u : (v > 255) ? 255u : v;
  }
}

float SjpegEstimateQuality(const uint8_t matrix[64], bool for_chroma) {
  // There's a lot of way to speed up this search (dichotomy, Newton, ...)
  // but also a lot of way to fabricate a twisted input to fool it.
  // So we're better off trying all the 100 possibilities since it's not
  // a lot after all.
  int best_quality = 0;
  float best_score = 256 * 256 * 64 + 1;
  for (int quality = 0; quality <= 100; ++quality) {
    uint8_t m[64];
    SjpegQuantMatrix(quality, for_chroma, m);
    float score = 0;
    SJPEG_UNROLL(4)
    for (size_t i = 0; i < 64; ++i) {
      const float diff = m[i] - matrix[i];
      score += diff * diff;
      if (score > best_score) {
        break;
      }
    }
    if (score < best_score) {
      best_score = score;
      best_quality = quality;
    }
  }
  return best_quality;
}

////////////////////////////////////////////////////////////////////////////////
// Bluriness risk evaluation and YUV420 / sharp-YUV420 / YUV444 decision

static const int kNoiseLevel = 4;
static const double kThreshGray = 0.995;  // 1.00 = full gray (max)
static const double kThreshYU420 = 40.0;
static const double kThreshSharpYU420 = 70.0;

// Progressive refinement: score rows in bands spread in bit-reversed dispersed
// order. Check score at pre-defined increasingly sparse check-points.
// Equivalent to a sequential pass if there's no early-out (full coverage).
static const int kBandHeight = 8;
static const double kConfidenceMargin = 4.0;
static const int kNumCheckpoints = 4;
static const double kCheckpointFractions[kNumCheckpoints + 1] = {
  0.05, 0.15, 0.30, 0.50, /*sentinel*/1.0
};

// reverses the low 'bits' bits of v, bits in [0, 16]
static int BitReversal16b(int v, int bits) {
  int r = 0;
  for (int i = 0; i < bits; ++i) {
    r = (r << 1) | (v & 1);
    v >>= 1;
  }
  return r;
}

#if defined(SJPEG_HAVE_AVX2) && defined(SJPEG_USE_AVX2_RISKINESS)
namespace sjpeg {
// defined in riskiness_avx2.cc, built separately with -mavx2 (see Makefile)
// so this file itself doesn't need an AVX2 target.
extern int RiskinessScoreRowAVX2(const uint16_t* row1, const uint16_t* row2,
                                 int size, int noise_level,
                                 int64_t* const score_sum,
                                 int64_t* const score_num,
                                 int64_t* const gray_num);
}  // namespace sjpeg
#endif

SjpegYUVMode SjpegRiskiness(const uint8_t* rgb,
                            int width, int height, int stride, float* risk) {
  const sjpeg::RGBToIndexRowFunc cvrt_func = sjpeg::GetRowFunc();
#if defined(SJPEG_HAVE_AVX2) && defined(SJPEG_USE_AVX2_RISKINESS)
  const bool use_avx2_riskiness = sjpeg::SupportsAVX2();
#endif
  const int s = sjpeg::kRGBSize;  // shortcut
  const int kRGB3 = s * s * s;
  const int gray = (s / 2) * (1 + s) * s;   // gray level for y=0,u=128,v=128
  // idx packs y + s * (u + s * v), so the samples with neutral chroma are
  // exactly the ones in [gray_min, gray_min + s), whatever their luma.
  const int gray_min = gray - gray % s;

  // Use faster int64_t accumulation instead of double. A score is at most
  // 3 * 255, so even a 16k x 16k image stays under 2^38. The cast to double
  // below is safe.
  int64_t score_sum = 0;    // scores above the noise level
  int64_t score_num = 0;    // how many samples that was
  int64_t gray_num = 0;     // samples with neutral chroma
  int64_t rows_scored = 0;  // number of true adjacent row-pairs scored so far

  // derives best recommendation from the accumulators collected so far
  const auto Finalize = [&](float* const risk_out) -> SjpegYUVMode {
    const double count = (double)score_num;
    double gray_count = (double)gray_num;
    double total_score = (count > 0) ? score_sum / count : 0.;
    // rightmost pixel is excluded, hence the (width - 1.)
    const double num_samples = (width - 1.) * (double)rows_scored;
    if (num_samples > 0.) gray_count /= num_samples;

    // pixels evaluated, scaled by how many rows were actually scored
    const double effective_area = (rows_scored > 0)
        ? (double)width * height * rows_scored / (height - 1.) : 0.;
    // if less than 1% of pixels were evaluated -> below noise level.
    const double frac =
        (effective_area > 0.) ? 100. * count / effective_area : 0.;
    if (frac < 1.) total_score = 0.;

    // recommendation (TODO(skal): tune thresholds)
    total_score = (total_score > 25.) ? 100. : total_score * 100. / 25.;
    if (risk_out != nullptr) *risk_out = (float)total_score;
    return (gray_count > kThreshGray) ?        SJPEG_YUV_400 :
           (total_score < kThreshYU420) ?      SJPEG_YUV_420 :
           (total_score < kThreshSharpYU420) ? SJPEG_YUV_SHARP :
                                               SJPEG_YUV_444;
  };

  const int num_bands = (height - 1 + kBandHeight - 1) / kBandHeight;
  // height is 16-bit in practice (kMaxDimension, JPEG's SOF field width), so
  // num_bands stays under ~8192 and bits stays under 16 -- required for
  // BitReversal16b() above, not just a speed assumption.
  int bits = 0;
  while ((1 << bits) < num_bands) ++bits;
  assert(bits <= 16);

  std::vector<uint16_t> row1(width), row2(width);

  // scores one adjacent row-pair (row1 = above, row2 = below)
  const auto ScoreRow = [&]() {
    int i = 0;
#if defined(SJPEG_HAVE_AVX2) && defined(SJPEG_USE_AVX2_RISKINESS)
    if (use_avx2_riskiness) {
      i = sjpeg::RiskinessScoreRowAVX2(&row1[0], &row2[0], width - 1,
                                       kNoiseLevel, &score_sum, &score_num,
                                       &gray_num);
    }
#endif
    SJPEG_UNROLL(4)
    for (; i < width - 1; ++i) {
      const int idx0 = row1[i + 0];
      const int idx1 = row1[i + 1];
      const int idx2 = row2[i + 0];
      const int score = sjpeg::kSharpnessScore[idx0 + kRGB3 * idx1]
                      + sjpeg::kSharpnessScore[idx0 + kRGB3 * idx2]
                      + sjpeg::kSharpnessScore[idx1 + kRGB3 * idx2];
      if (score > kNoiseLevel) {
        score_sum += score;
        score_num += 1;
      }
      gray_num += ((uint32_t)(idx0 - gray_min) < (uint32_t)s);
    }
  };

  int next_checkpoint = 0;
  int target = (int)ceil(kCheckpointFractions[next_checkpoint] * num_bands);
  int cursor = 0;  // walks bit-reversed indices in [0, 1<<bits), skipping
                   // the ones larger than num_bands

  for (int k = 0; k < num_bands; ++k) {
    int band;
    do {
      band = BitReversal16b(cursor, bits);
      ++cursor;
    } while (band >= num_bands);
    int j = band * kBandHeight;  // current row
    const int last_band = std::min(j + 1 + kBandHeight, height);
    cvrt_func(rgb + (size_t)j * stride, width, &row1[0]);
    while (++j < last_band) {
      // note: cvrt_func() is called height/kBandHeight times too much,
      // but that's ok
      cvrt_func(rgb + (size_t)j * stride, width, &row2[0]);
      ScoreRow();
      std::swap(row1, row2);
      ++rows_scored;
    }

    if (k >= target) {
      float candidate_risk;
      const SjpegYUVMode candidate_mode = Finalize(&candidate_risk);
      // never trust an early gray verdict: unlike 420/SHARP/444
      // YUV400 discards chroma directly. Don't let a local flat gray area
      // derail the estimation!
      if (candidate_mode != SJPEG_YUV_400 &&
          fabs(candidate_risk - kThreshYU420) >= kConfidenceMargin &&
          fabs(candidate_risk - kThreshSharpYU420) >= kConfidenceMargin) {
        if (risk != nullptr) *risk = candidate_risk;
        return candidate_mode;
      }
      ++next_checkpoint;
      target = (int)ceil(kCheckpointFractions[next_checkpoint] * num_bands);
    }
  }
  return Finalize(risk);
}

namespace sjpeg {

// (X * 0x0101 >> 16) ~= X / 255
static uint32_t Convert(uint32_t v) {
  return (v * (0x0101u * (sjpeg::kRGBSize - 1))) >> 16;
}

// Convert 8b values y/u/v to index entry.
int YUVToRiskIdx(int16_t y, int16_t u, int16_t v) {
  const int idx = Convert(y + 128)
                + Convert(u + 128) * sjpeg::kRGBSize
                + Convert(v + 128) * sjpeg::kRGBSize * sjpeg::kRGBSize;
  return idx;
}

// return riskiness score on an 8x8 block. Input is YUV444 block
// of DCT coefficients (Y/U/V).
double DCTRiskinessScore(const int16_t yuv[3 * 64], int16_t scores[8 * 8]) {
  uint16_t idx[64];
  for (int k = 0; k < 64; ++k) {
    idx[k] = YUVToRiskIdx(yuv[k + 0 * 64], yuv[k + 1 * 64],  yuv[k + 2 * 64]);
  }
  const int kRGB3 = sjpeg::kRGBSize * sjpeg::kRGBSize * sjpeg::kRGBSize;
  double total_score = 0;
  double count = 0;
  for (size_t J = 0; J <= 7; ++J) {
    for (size_t I = 0; I <= 7; ++I) {
      const int k = I + J * 8;
      const int idx0 = idx[k + 0];
      const int idx1 = idx[k + (I < 7 ? 1 : -1)];
      const int idx2 = idx[k + (J < 7 ? 8 : -8)];
      int score = sjpeg::kSharpnessScore[idx0 + kRGB3 * idx1]
                + sjpeg::kSharpnessScore[idx0 + kRGB3 * idx2]
                + sjpeg::kSharpnessScore[idx1 + kRGB3 * idx2];
      if (score <= kNoiseLevel) {
        score = 0;
      } else {
        total_score += score;
        count += 1.0;
      }
      scores[I + J * 8] = static_cast<int16_t>(score);
    }
  }
  if (count > 0) total_score /= count;
  total_score = (total_score > 25.) ? 100. : total_score * 100. / 25.;
  return total_score;
}

// This function returns the raw per-pixel riskiness scores. The input rgb[]
// samples is a 8x8 block, the output is a 8x8 block.
// Not an official API, because a little too specific. But still accessible.
double BlockRiskinessScore(const uint8_t* rgb, int stride,
                           int16_t scores[8 * 8]) {
  const RGBToYUVBlockFunc get_block = GetBlockFunc(SJPEG_YUV_444);
  int16_t yuv444[3 * 64];
  get_block(rgb, stride, yuv444);
  return DCTRiskinessScore(yuv444, scores);
}

}   // namespace sjpeg
