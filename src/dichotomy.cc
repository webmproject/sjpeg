// Copyright 2018 Google Inc.
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
//  Dichotomy loop
//
// Author: Skal (pascal.massimino@gmail.com)

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "sjpegi.h"

using namespace sjpeg;

// convergence is considered reached if |dq| < kdQLimit
static const float kdQLimit = 0.2;
// maximal variation allowed on dQ
static const float kdQThresh = 200.;

namespace sjpeg {

////////////////////////////////////////////////////////////////////////////////
// dichotomy

#define DBG_PRINT 0

struct PassStats {  // struct for organizing convergence in either size or PSNR
  bool is_first;
  float dq, q, last_q;
  double value, last_value, target;
  bool do_size_search;

  PassStats(const Encoder& enc);
  bool ComputeNextQ(float result);   // returns true if finished
  void BackTrack() {
    q = last_q;
    dq /= 2;
    q += dq;
  }
};

PassStats::PassStats(const Encoder& enc) {
  is_first = true;
  dq = 50.f;
  q = last_q = 100.;
  value = last_value = 0;
  target = enc.target_value_;
  do_size_search = (enc.target_mode_ == SjpegEncodeParam::TARGET_SIZE);
}

static float Clamp(float v, float min, float max) {
  return (v < min) ? min : (v > max) ? max : v;
}

bool PassStats::ComputeNextQ(float result) {
  value = result;
  if (is_first) {
    if (do_size_search) {
      dq = (value < target) ? -dq : dq;
    } else {
      dq = (value < target) ? -dq : dq;
    }
    is_first = false;
  } else {
    if (fabs(value - last_value) > 0.01 * value) {
      const double slope = (target - value) / (last_value - value);
      dq = (float)(slope * (last_q - q));
    } else {
      dq = 0.;  // we're done?!
    }
  }
  // Limit variable to avoid large swings.
  dq = Clamp(dq, -kdQThresh, kdQThresh);
  last_q = q;
  last_value = value;
  q = Clamp(q + dq, 0.f, 2000.f);
  return fabs(q - last_q) < kdQLimit;
}

void Encoder::StoreRunLevels(DCTCoeffs* coeffs) {
  assert(use_extra_memory_);
  assert(reuse_run_levels_);

  const QuantizeBlockFunc quantize_block = use_trellis_ ? TrellisQuantizeBlock
                                                        : quantize_block_;

  ResetDCs();
  nb_run_levels_ = 0;
  int16_t* in = in_blocks_;
  for (int n = 0; n < mb_w_ * mb_h_; ++n) {
    CheckBuffers();
    for (int c = 0; c < nb_comps_; ++c) {
      for (int i = 0; i < nb_blocks_[c]; ++i) {
        RunLevel* const run_levels = all_run_levels_ + nb_run_levels_;
        const int dc = quantize_block(in, c, &quants_[quant_idx_[c]],
                                      coeffs, run_levels);
        coeffs->dc_code_ = GenerateDCDiffCode(dc, &DCs_[c]);
        nb_run_levels_ += coeffs->nb_coeffs_;
        ++coeffs;
        in += 64;
      }
    }
  }
}

void Encoder::LoopScan() {
  assert(use_extra_memory_);
  assert(reuse_run_levels_);

  if (use_adaptive_quant_) {
    CollectHistograms();
  } else {
    CollectCoeffs();   // we just need the coeffs
  }

  // We use the default Huffman tables as basis for bit-rate evaluation
  if (use_trellis_) InitCodes(true);

  const size_t nb_mbs = mb_w_ * mb_h_ * mcu_blocks_;
  DCTCoeffs* const base_coeffs = new DCTCoeffs[nb_mbs];

  uint8_t base_quant[2][64], opt_quants[2][64];
  for (int c = 0; c < 2; ++c) {
    CopyQuantMatrix(quants_[c].quant_, base_quant[c]);
  }

  // Dichotomy passes
  PassStats stats(*this);
  for (int p = 0; p < passes_; ++p) {
    for (int c = 0; c < 2; ++c) {
      SetQuantMatrix(base_quant[c], stats.q, quants_[c].quant_);
      FinalizeQuantMatrix(&quants_[c], q_bias_);
    }

    if (use_adaptive_quant_) {
      AnalyseHisto();   // adjust quant_[] matrices
    }

    if (stats.do_size_search) {
      // compute pass to store coeffs / runs / dc_code_
      StoreRunLevels(base_coeffs);
    } else {
      // TODO(skal): have a function that only sums error, not run/levels
      StoreRunLevels(base_coeffs);
    }

    if (optimize_size_) {
      // if we're just targetting PSNR, we don't need to optimize
      // for size within the loop.
      if (stats.do_size_search || use_trellis_) {
        StoreOptimalHuffmanTables(nb_mbs, base_coeffs);
        if (use_trellis_) InitCodes(true);
      }
    }

    const float result =
      stats.do_size_search ? ComputeSize(base_coeffs, all_run_levels_)
                           : ComputePSNR(base_coeffs, all_run_levels_);
    if (p > 0 && min_psnr_ > 0.) {
      const float psnr =
          stats.do_size_search ? ComputePSNR(base_coeffs, all_run_levels_)
                               : result;
      if (psnr < min_psnr_) {
        stats.BackTrack();
        continue;
      }
    }
    if (DBG_PRINT) printf("pass #%d: q=%f value:%.2f\n", p, stats.q, result);
    for (int c = 0; c < 2; ++c) {
      CopyQuantMatrix(quants_[c].quant_, opt_quants[c]);
    }
    if (stats.ComputeNextQ(result)) break;
  }
  SetQuantMatrices(opt_quants);   // set the final matrix

  // optimize Huffman table now, if we haven't already during the search
  if (optimize_size_ && !stats.do_size_search && !use_trellis_) {
    StoreOptimalHuffmanTables(nb_mbs, base_coeffs);
  }

  // finish bitstream
  WriteDQT();
  WriteSOF();
  WriteDHT();
  WriteSOS();
  FinalPassScan(nb_mbs, base_coeffs);
}

////////////////////////////////////////////////////////////////////////////////
// Size & PSNR computation, mostly for dichotomy

size_t Encoder::HeaderSize() const {
  size_t size = 0;
  size += 20;    // APP0
  size += app_markers_.size();
  if (exif_.size() > 0) {
    size += 8 + exif_.size();
  }
  if (iccp_.size() > 0) {
    const size_t chunk_size_max = 0xffff - 12 - 4;
    const size_t num_chunks = (iccp_.size() - 1) / chunk_size_max + 1;
    size += num_chunks * (12 + 4 + 2);
    size += iccp_.size();
  }
  if (xmp_.size() > 0) {
    size += 2 + 2 + 29 + xmp_.size();
  }
  size += 2 * 65 + 2 + 2;         // DQT
  size += 8 + 3 * nb_comps_ + 2;  // SOF
  size += 6 + 2 * nb_comps_ + 2;  // SOS
  size += 2;                      // EOI
  // DHT:
  for (int c = 0; c < (nb_comps_ == 1 ? 1 : 2); ++c) {   // luma, chroma
    for (int type = 0; type <= 1; ++type) {               // dc, ac
      const HuffmanTable* const h = Huffman_tables_[type * 2 + c];
      size += 2 + 3 + 16 + h->nb_syms_;
    }
  }
  return size * 8;
}

void Encoder::BlocksSize(int nb_mbs, const DCTCoeffs* coeffs,
                         const RunLevel* rl,
                         BitCounter* const bc) const {
  for (int n = 0; n < nb_mbs; ++n) {
    const DCTCoeffs& c = coeffs[n];
    const int idx = c.idx_;
    const int q_idx = quant_idx_[idx];

    // DC
    const int dc_len = c.dc_code_ & 0x0f;
    const uint32_t code = dc_codes_[q_idx][dc_len];
    bc->AddPackedCode(code);
    bc->AddBits(c.dc_code_ >> 4, dc_len);

    // AC
    const uint32_t* const codes = ac_codes_[q_idx];
    for (int i = 0; i < c.nb_coeffs_; ++i) {
      int run = rl[i].run_;
      while (run & ~15) {        // escapes
        bc->AddPackedCode(codes[0xf0]);
        run -= 16;
      }
      const uint32_t suffix = rl[i].level_;
      const size_t nbits = suffix & 0x0f;
      const int sym = (run << 4) | nbits;
      bc->AddPackedCode(codes[sym]);
      bc->AddBits(suffix >> 4, nbits);
    }
    if (c.last_ < 63) bc->AddPackedCode(codes[0x00]);  // EOB
    rl += c.nb_coeffs_;
  }
}

float Encoder::ComputeSize(const DCTCoeffs* coeffs,
                                 const RunLevel* rl) {
  InitCodes(false);
  size_t size = HeaderSize();
  BitCounter bc;
  BlocksSize(mb_w_ * mb_h_ * mcu_blocks_, coeffs, rl, &bc);
  size += bc.Size();
  return size / 8.f;
}

////////////////////////////////////////////////////////////////////////////////

static double GetPSNR(uint64_t err, uint64_t size) {
  return (err > 0 && size > 0) ? 10. * log10(255. * 255. * size / err) : 99.;
}

static const uint16_t kMasks[16 + 1][2] = {
  { 0x0000U, 0x0000U },  // fake extra entry, for robustness
  { 0x0001U, 0x0001U }, { 0x0002U, 0x0003U },
  { 0x0004U, 0x0007U }, { 0x0008U, 0x000fU },
  { 0x0010U, 0x001fU }, { 0x0020U, 0x003fU },
  { 0x0040U, 0x007fU }, { 0x0080U, 0x00ffU },
  { 0x0100U, 0x01ffU }, { 0x0200U, 0x03ffU },
  { 0x0400U, 0x07ffU }, { 0x0800U, 0x03ffU },
  { 0x1000U, 0x1fffU }, { 0x2000U, 0x3fffU },
  { 0x4000U, 0x7fffU }, { 0x8000U, 0xffffU }
};

static int16_t CodeToCoeff(uint16_t code) {
  const int nbits = (code & 0xf);
  int16_t qcoeff = (code >> 4);
  if (qcoeff < kMasks[nbits][0]) qcoeff -= kMasks[nbits][1];
  return qcoeff;
}

float Encoder::ComputePSNR(const DCTCoeffs* coeffs,
                           const RunLevel* rl) const {
  uint64_t error = 0;
  int16_t* in = in_blocks_;
  const size_t nb_mbs = mb_w_ * mb_h_ * mcu_blocks_;
  int32_t DCs[3] = { 0, 0, 0 };
  for (size_t n = 0; n < nb_mbs; ++n) {
    const DCTCoeffs& c = coeffs[n];
    const int c_idx = c.idx_;
    const int q_idx = quant_idx_[c_idx];
    const uint8_t* const Q = quants_[q_idx].quant_;
    int pos = 0;
    int32_t dq[64] = { 0 };
    for (int i = 0; i < c.nb_coeffs_; ++i, ++rl) {
      pos += rl->run_ + 1;
      const int j = kZigzag[pos];
      dq[j] = Q[j] * CodeToCoeff(rl->level_);
    }
    dq[0] = (DCs[c_idx] += CodeToCoeff(c.dc_code_)) * Q[0];
    for (int i = 0; i < 64; ++i) {
      const int16_t v0 = in[i] / 16;  // fDCT output is upscaled
      const int16_t v = dq[i];
      error += ((uint64_t)(v0 - v) * (v0 - v));
    }
    in += 64;
  }
  return GetPSNR(error, nb_mbs * 64ull);
}

}    // namespace sjpeg
