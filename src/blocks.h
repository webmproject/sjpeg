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
//  Blocks of coefficients, and the quantizer.
//  Internal header: not installed.
//
// Author: Skal (pascal.massimino@gmail.com)

#ifndef SJPEG_BLOCKS_H_
#define SJPEG_BLOCKS_H_

#include <stdint.h>

#include "common.h"

namespace sjpeg {

////////////////////////////////////////////////////////////////////////////////
// main structs

// quantizer matrices
struct Quantizer {
  uint8_t quant_[64];      // direct quantizer matrix
  uint8_t min_quant_[64];  // min quantizer value allowed
  uint16_t iquant_[64];    // precalc'd reciprocal for divisor
  uint16_t qthresh_[64];   // minimal absolute value that produce non-zero coeff
  uint32_t qthresh2_[64];  // qthresh_[]^2 for fast comparison against coeff^2
  uint16_t bias_[64];      // bias, for coring
  // Alternate thresholds for adaptive-bias mode.
  uint16_t qthresh_flat_[64];
  uint16_t qthresh_busy_[64];
  const uint32_t* codes_;  // codes for bit-cost calculation
};

// Testing hook: exposes Encoder::GetQuantizeErrorFunc(), which is private.
typedef uint32_t (*QuantizeErrorTestFunc)(const int16_t in[64],
                                          const Quantizer* const Q);
QuantizeErrorTestFunc GetQuantizeErrorFuncForTest();

// compact Run/Level storage, separate from DCTCoeffs infos
// Run/Level Information is not yet entropy-coded, but just stored
struct RunLevel {
  int16_t run_;
  uint16_t level_;  // 4bits for length, 12bits for mantissa
};

// short infos about the block of quantized coefficients
struct DCTCoeffs {
  int16_t last_;       // last position (inclusive) of non-zero coeff
  int16_t nb_coeffs_;  // total number of non-zero AC coeffs
  uint16_t dc_code_;   // DC code (4bits for length, 12bits for suffix)
  int8_t idx_;         // component idx
};

// Histogram of transform coefficients, for adaptive quant matrices
// * HSHIFT controls the trade-off between storage size for counts[]
//   and precision: the fdct doesn't descale and returns coefficients as
//   signed 16bit value. We are only interested in the absolute values
//   of coefficients that are less than MAX_HISTO_DCT_COEFF, which are our
//   best contributors.
//   Still, storing histogram up to MAX_HISTO_DCT_COEFF can be costly, so
//   we further aggregate the statistics in bins of size 1 << HSHIFT to save
//   space.
// * HLAMBDA roughly measures how much you are willing to trade in distortion
//   for a 1-bit gain in filesize.
// * QDELTA_MIN / QDELTA_MAX control how much we allow wandering around the
//   initial point. This helps reducing the CPU cost, as long as keeping the
//   optimization around the initial desired quality-factor (HLAMBDA also
//   serve this purpose).
enum {
  HSHIFT = 2,  // size of bins is (1 << HSHIFT)
  HHALF = 1 << (HSHIFT - 1),
  MAX_HISTO_DCT_COEFF = (1 << 7),  // max coefficient, descaled by HSHIFT
  HLAMBDA = 0x80,
  // Limits on range of alternate quantizers explored around
  // the initial value.  (see details in AnalyseHisto())
  QDELTA_MIN = -12,
  QDELTA_MAX = 12,
  QSIZE = QDELTA_MAX + 1 - QDELTA_MIN,
};

struct Histo {
  // Reserve one extra entry for counting all coeffs greater than
  // MAX_HISTO_DCT_COEFF. Result isn't used, but it makes the loop easier.
  int counts_[64][MAX_HISTO_DCT_COEFF + 1];
};

}  // namespace sjpeg

#endif  // SJPEG_BLOCKS_H_
