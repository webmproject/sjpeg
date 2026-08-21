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
//  Progressive (spectral-split-only) mode: raster-ordered per-component
//  planes, and the DC / AC scan drivers.
//
// Author: Skal (pascal.massimino@gmail.com)

#include <assert.h>
#include <stdint.h>
#include <string.h>  // for memset

#include "sjpegi.h"

#if !defined(SJPEG_NO_PROGRESSIVE)

namespace sjpeg {

// spec's 15-bit EOBn cap (class R=14 tops out at 2^15-1)
enum { kMaxEOBRun = 0x7fff };

////////////////////////////////////////////////////////////////////////////////
// handles progressive planes

bool Encoder::AllocateProgPlanes() {
  prog_planes_ = Alloc<ProgPlanes>(1);
  if (prog_planes_ == nullptr) return false;
  memset(prog_planes_, 0, sizeof(*prog_planes_));  // coeffs[]/run_levels[]=null

  const int Hmax = block_w_ / 8, Vmax = block_h_ / 8;  // max sampling factors
  for (int c = 0; c < nb_comps_; ++c) {
    const int h_samp = block_dims_[c] >> 4;
    const int v_samp = block_dims_[c] & 0xf;
    prog_planes_->plane_w[c] = mb_w_ * h_samp;
    prog_planes_->plane_h[c] = mb_h_ * v_samp;
    // true_w/h: see ProgPlanes in sjpegi.h
    const int samples_w = (W_ * h_samp + Hmax - 1) / Hmax;
    const int samples_h = (H_ * v_samp + Vmax - 1) / Vmax;
    prog_planes_->true_w[c] = (samples_w + 7) / 8;
    prog_planes_->true_h[c] = (samples_h + 7) / 8;
    const size_t nb_blocks =
        (size_t)prog_planes_->plane_w[c] * prog_planes_->plane_h[c];
    prog_planes_->coeffs[c] = Alloc<DCTCoeffs>(nb_blocks);
    if (prog_planes_->coeffs[c] == nullptr) return false;
    prog_planes_->run_levels[c] = Alloc<RunLevel>(nb_blocks * 63);
    if (prog_planes_->run_levels[c] == nullptr) return false;
  }
  return true;
}

void Encoder::DeallocateProgPlanes() {
  if (prog_planes_ == nullptr) return;
  for (int c = 0; c < MAX_COMP; ++c) {
    Free(prog_planes_->coeffs[c]);
    Free(prog_planes_->run_levels[c]);
  }
  Free(prog_planes_);
  prog_planes_ = nullptr;
}

int Encoder::ProgPlaneIndex(int c, int mb_x, int mb_y, int i) const {
  const int h_samp = block_dims_[c] >> 4;
  const int v_samp = block_dims_[c] & 0xf;
  const int block_row = mb_y * v_samp + i / h_samp;
  const int block_col = mb_x * h_samp + i % h_samp;
  return block_row * prog_planes_->plane_w[c] + block_col;
}

bool Encoder::CheckProgBuffers() {
  size_t chunk = (size_t)W_ * H_ / 4;
  if (chunk < 4096) chunk = 4096;
  if (chunk > (256 << 10)) chunk = 256 << 10;
  ok_ = ok_ && bw_.ReserveMore(2560, chunk);
  return ok_;
}

void Encoder::EncodeProgAC(int c, int split) {
  const bool has_high = (split < 63);
  const int nb_bands = has_high ? 2 : 1;
  const int bands[2][2] = { { 1, has_high ? split : 63 },
                            { split + 1, 63 } };
  const int stride = prog_planes_->plane_w[c];
  const int tw = prog_planes_->true_w[c];
  const int th = prog_planes_->true_h[c];
  RunLevel windowed[64];
  for (int b = 0; b < nb_bands; ++b) {
    const int Ss = bands[b][0];
    const int Se = bands[b][1];

    ResetEntropyStatsAC();
    int eobrun = 0;  // persists across blocks
    for (int row = 0; row < th; ++row) {
      for (int col = 0; col < tw; ++col) {
        const int n = row * stride + col;
        const DCTCoeffs* const coeffs = &prog_planes_->coeffs[c][n];
        const RunLevel* const rl = prog_planes_->run_levels[c] + n * 63;
        bool has_eob;
        const int nb_w = WindowRunLevels(coeffs, rl, Ss, Se, windowed,
                                         &has_eob);
        if (nb_w > 0) {
          if (eobrun > 0) {
            AddEntropyStatsEOBRun(eobrun);
            eobrun = 0;
          }
          AddEntropyStatsACWindowed(windowed, nb_w);
        }
        if (has_eob) {
          if (++eobrun == kMaxEOBRun) {
            AddEntropyStatsEOBRun(eobrun);
            eobrun = 0;
          }
        }
      }
    }
    if (eobrun > 0) AddEntropyStatsEOBRun(eobrun);  // flush trailing run
    CompileEntropyStatsAC();

    if (!CheckProgBuffers()) return;
    WriteOneDHT(/*table_class=*/1, /*table_id=*/0, &prog_planes_->opt_table_ac);
    const ScanComponent sc = { c, /*dc=*/0, /*ac=*/0 };
    WriteProgSOS(&sc, 1, Ss, Se);

    eobrun = 0;
    for (int row = 0; row < th; ++row) {
      for (int col = 0; col < tw; ++col) {
        // per-block: a whole row can exceed the 2560-byte guarantee
        if (!CheckProgBuffers()) return;
        const int n = row * stride + col;
        const DCTCoeffs* const coeffs = &prog_planes_->coeffs[c][n];
        const RunLevel* const rl = prog_planes_->run_levels[c] + n * 63;
        bool has_eob;
        const int nb_w = WindowRunLevels(coeffs, rl, Ss, Se, windowed,
                                         &has_eob);
        if (nb_w > 0) {
          if (eobrun > 0) {
            CodeEOBRun(eobrun);
            eobrun = 0;
          }
          CodeBlockACWindowed(windowed, nb_w);
        }
        if (has_eob) {
          if (++eobrun == kMaxEOBRun) {
            CodeEOBRun(eobrun);
            eobrun = 0;
          }
        }
      }
    }
    if (eobrun > 0) CodeEOBRun(eobrun);  // flush trailing run
    bw_.Flush();
  }
}

bool Encoder::EncodeProgressive() {
  ResetDCs();
  if (!AllocateProgPlanes()) return false;

  // 1. Quantize every block once into the raster-ordered planes.
  const QuantizeBlockFunc quantize_block =
      use_trellis_ ? TrellisQuantizeBlock : quantize_block_;
  // trellis needs ac_codes_[] seeded, like the baseline path does
  if (use_trellis_) InitCodes(true);
  for (int mb_y = 0; mb_y < mb_h_; ++mb_y) {
    for (int mb_x = 0; mb_x < mb_w_; ++mb_x) {
      int16_t* in = in_blocks_;
      TransformMCU(mb_x, mb_y, in);
      for (int c = 0; c < nb_comps_; ++c) {
        for (int i = 0; i < nb_blocks_[c]; ++i) {
          const int idx = ProgPlaneIndex(c, mb_x, mb_y, i);
          DCTCoeffs* const coeffs = &prog_planes_->coeffs[c][idx];
          RunLevel* const run_levels = prog_planes_->run_levels[c] + idx * 63;
          const int dc = quantize_block(in, c, &quants_[quant_idx_[c]],
                                        coeffs, run_levels);
          coeffs->dc_code_ = GenerateDCDiffCode(dc, &DCs_[c]);
          in += 64;
        }
      }
    }
  }
  if (!ok_) {
    DeallocateProgPlanes();
    return false;
  }

  // 2. DC scan: one interleaved scan, baseline MCU order.
  memset(freq_dc_, 0, sizeof(freq_dc_));
  for (int mb_y = 0; mb_y < mb_h_; ++mb_y) {
    for (int mb_x = 0; mb_x < mb_w_; ++mb_x) {
      for (int c = 0; c < nb_comps_; ++c) {
        for (int i = 0; i < nb_blocks_[c]; ++i) {
          const int idx = ProgPlaneIndex(c, mb_x, mb_y, i);
          AddEntropyStatsDC(&prog_planes_->coeffs[c][idx]);
        }
      }
    }
  }
  CompileEntropyStatsDC();
  if (!CheckProgBuffers()) return false;
  {
    ScanComponent scs[MAX_COMP];
    for (int c = 0; c < nb_comps_; ++c) {
      scs[c] = { c, quant_idx_[c], 0 };
    }
    WriteOneDHT(0, 0, &opt_tables_dc_[0]);
    if (nb_comps_ > 1) WriteOneDHT(0, 1, &opt_tables_dc_[1]);
    WriteProgSOS(scs, nb_comps_, 0, 0);
  }
  for (int mb_y = 0; mb_y < mb_h_; ++mb_y) {
    for (int mb_x = 0; mb_x < mb_w_; ++mb_x) {
      if (!CheckProgBuffers()) return false;
      for (int c = 0; c < nb_comps_; ++c) {
        for (int i = 0; i < nb_blocks_[c]; ++i) {
          const int idx = ProgPlaneIndex(c, mb_x, mb_y, i);
          CodeBlockDC(&prog_planes_->coeffs[c][idx]);
        }
      }
    }
  }
  bw_.Flush();

  // 3. AC scans: luma, then chroma.
  EncodeProgAC(0, prog_luma_split_);
  for (int c = 1; c < nb_comps_; ++c) {
    EncodeProgAC(c, prog_chroma_split_);
  }

  DeallocateProgPlanes();
  return ok_;
}

}    // namespace sjpeg

#endif  // !SJPEG_NO_PROGRESSIVE
