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
//  Progressive mode (spectral-split-only)
//
// Author: Skal (pascal.massimino@gmail.com)

#include <assert.h>
#include <stdint.h>
#include <string.h>  // for memset

#include <algorithm>

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
  memset(prog_planes_, 0, sizeof(*prog_planes_));  // pointers=null, rl_used[]=0

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
    // coeffs[] and rl_offset[] share one allocation (see ProgPlanes)
    static_assert(sizeof(DCTCoeffs) == 2 * sizeof(uint32_t), "packing");
    const size_t nb_slots = nb_blocks + (nb_blocks + 1) / 2;
    prog_planes_->coeffs[c] = Alloc<DCTCoeffs>(nb_slots);
    if (prog_planes_->coeffs[c] == nullptr) return false;
    prog_planes_->rl_offset[c] =
        (uint32_t*)(prog_planes_->coeffs[c] + nb_blocks);
    // Starting guess, like the baseline path. Grown by GrowProgRunLevels().
    prog_planes_->rl_capacity[c] = nb_blocks * 8 + 64;
    prog_planes_->run_levels[c] = Alloc<RunLevel>(prog_planes_->rl_capacity[c]);
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
  FreePtr(&prog_planes_);
}

// Doubles the capacity of component c's run_levels[], up to what the plane
// can ever need. Old and new buffers coexist during the copy: with more than
// 32 run/levels per block on average, that's a transient 95 per block.
bool Encoder::GrowProgRunLevels(int c) {
  ProgPlanes* const p = prog_planes_;
  const uint64_t nb_blocks = (uint64_t)p->plane_w[c] * p->plane_h[c];
  const uint64_t new_capacity =
      std::min(2 * (uint64_t)p->rl_capacity[c], nb_blocks * 63 + 64);
  if (new_capacity > SIZE_MAX / sizeof(RunLevel)) return SetError();
  RunLevel* const new_rl = Alloc<RunLevel>((size_t)new_capacity);
  if (new_rl == nullptr) return false;
  memcpy(new_rl, p->run_levels[c], p->rl_used[c] * sizeof(RunLevel));
  Free(p->run_levels[c]);
  p->run_levels[c] = new_rl;
  p->rl_capacity[c] = (size_t)new_capacity;
  return true;
}

int Encoder::ProgPlaneIndex(int c, int mb_x, int mb_y, int i) const {
  const int h_samp = block_dims_[c] >> 4;
  const int v_samp = block_dims_[c] & 0xf;
  const int block_row = mb_y * v_samp + i / h_samp;
  const int block_col = mb_x * h_samp + i % h_samp;
  return block_row * prog_planes_->plane_w[c] + block_col;
}

// Visits component c's blocks for band [Ss,Se], in raster order. Calls
// on_start() before each block (stops if it returns false), on_block() with
// the block's windowed run/levels if any, and on_eob() for each completed
// EOB run, across empty blocks.
template <class BlockFn, class EOBFn, class StartFn>
bool Encoder::WalkACBand(int c, int Ss, int Se, BlockFn on_block, EOBFn on_eob,
                         StartFn on_start) const {
  const int stride = prog_planes_->plane_w[c];
  const int tw = prog_planes_->true_w[c];
  const int th = prog_planes_->true_h[c];
  RunLevel windowed[64];
  int eobrun = 0;  // persists across blocks
  for (int row = 0; row < th; ++row) {
    for (int col = 0; col < tw; ++col) {
      if (!on_start()) return false;
      const int n = row * stride + col;
      const DCTCoeffs* const coeffs = &prog_planes_->coeffs[c][n];
      const RunLevel* const rl =
          prog_planes_->run_levels[c] + prog_planes_->rl_offset[c][n];
      bool has_eob;
      const int nb_w =
          ProgACCoder::WindowRunLevels(coeffs, rl, Ss, Se, windowed, &has_eob);
      if (nb_w > 0) {
        if (eobrun > 0) {
          on_eob(eobrun);
          eobrun = 0;
        }
        on_block(windowed, nb_w);
      }
      if (has_eob) {
        if (++eobrun == kMaxEOBRun) {
          on_eob(eobrun);
          eobrun = 0;
        }
      }
    }
  }
  if (eobrun > 0) on_eob(eobrun);  // flush trailing run
  return true;
}

void Encoder::EncodeProgAC(int c, int split) {
  const bool has_high = (split < 63);
  const int nb_bands = has_high ? 2 : 1;
  const int bands[2][2] = {{1, has_high ? split : 63}, {split + 1, 63}};
  ProgACCoder& ac = prog_planes_->ac;
  for (int b = 0; b < nb_bands; ++b) {
    const int Ss = bands[b][0];
    const int Se = bands[b][1];

    // pass 1: gather the stats and build the table
    ac.Reset();
    WalkACBand(
        c, Ss, Se, [&](const RunLevel* rl, int n) { ac.AddWindowed(rl, n); },
        [&](int run) { ac.AddEOBRun(run); }, [] { return true; });
    ac.Compile();

    if (!CheckProgBuffers()) return;
    hw_.WriteOneDHT(/*table_class=*/1, /*table_id=*/0, &ac.table);
    const HeaderWriter::ScanComponent sc = {c, /*dc=*/0, /*ac=*/0};
    hw_.WriteSOS(&sc, 1, Ss, Se);

    // pass 2: code the scan. Per-block buffer check: a whole row can exceed
    // the 2560-byte guarantee.
    if (!WalkACBand(
            c, Ss, Se,
            [&](const RunLevel* rl, int n) { ac.CodeWindowed(rl, n, &bw_); },
            [&](int run) { ac.CodeEOBRun(run, &bw_); },
            [&] { return CheckProgBuffers(); })) {
      return;
    }
    bw_.Flush();
  }
}

bool Encoder::EncodeProgressive() {
  ResetDCs();
  const bool ok = AllocateProgPlanes() && EncodeProgScans();
  DeallocateProgPlanes();
  return ok;
}

bool Encoder::EncodeProgScans() {
  // 1. Quantize every block once, gathering DC entropy stats along the way.
  // Same mb_y/mb_x/c/i nest as the DC-code pass below -- keep them in sync.
  const QuantizeBlockFunc quantize_block = GetActiveQuantizeBlockFunc();
  // trellis/rdo need the AC codes seeded, like the baseline path does
  InitCostCodes();
  stats_.Reset();
  int16_t* in = in_blocks_;
  for (int mb_y = 0; mb_y < mb_h_; ++mb_y) {
    for (int mb_x = 0; mb_x < mb_w_; ++mb_x) {
      MaybeTransformMCU(mb_x, mb_y, &in);
      for (int c = 0; c < nb_comps_; ++c) {
        for (int i = 0; i < nb_blocks_[c]; ++i) {
          const int idx = ProgPlaneIndex(c, mb_x, mb_y, i);
          DCTCoeffs* const coeffs = &prog_planes_->coeffs[c][idx];
          // room for one block: 63 run/levels, plus the quantizers' slack
          const size_t rl_pos = prog_planes_->rl_used[c];
          if (rl_pos + 64 > prog_planes_->rl_capacity[c] &&
              !GrowProgRunLevels(c)) {
            return false;
          }
          const int dc = quantize_block(in, c, quants_[quant_idx_[c]], coeffs,
                                        prog_planes_->run_levels[c] + rl_pos);
          prog_planes_->rl_offset[c][idx] = (uint32_t)rl_pos;
          prog_planes_->rl_used[c] = rl_pos + coeffs->nb_coeffs_;
          coeffs->dc_code_ = GenerateDCDiffCode(dc, &DCs_[c]);
          stats_.AddDC(quant_idx_[c], coeffs);
          in += 64;
        }
      }
    }
  }
  DeallocateBlocks();  // we can free up some coeffs memory at this point
  if (!ok_) return false;

  // 2. DC scan: one interleaved scan, baseline MCU order.
  entropy_.CompileDC(stats_);
  if (!CheckProgBuffers()) return false;
  {
    HeaderWriter::ScanComponent scs[MAX_COMP];
    for (int c = 0; c < nb_comps_; ++c) {
      scs[c] = {c, quant_idx_[c], 0};
    }
    for (int q = 0; q < entropy_.nb_tables(); ++q) {
      hw_.WriteOneDHT(0, q, entropy_.opt_table_dc(q));
    }
    hw_.WriteSOS(scs, nb_comps_, 0, 0);
  }
  for (int mb_y = 0; mb_y < mb_h_; ++mb_y) {
    for (int mb_x = 0; mb_x < mb_w_; ++mb_x) {
      if (!CheckProgBuffers()) return false;
      for (int c = 0; c < nb_comps_; ++c) {
        for (int i = 0; i < nb_blocks_[c]; ++i) {
          const int idx = ProgPlaneIndex(c, mb_x, mb_y, i);
          const DCTCoeffs* const coeffs = &prog_planes_->coeffs[c][idx];
          entropy_.CodeBlockDC(coeffs, quant_idx_[c], &bw_);
        }
      }
    }
  }
  bw_.Flush();

  // 3. AC scans: luma, then chroma; free each component right after its scan.
  EncodeProgAC(0, prog_luma_split_);
  FreePtr(&prog_planes_->coeffs[0]);
  FreePtr(&prog_planes_->run_levels[0]);
  for (int c = 1; c < nb_comps_; ++c) {
    EncodeProgAC(c, prog_chroma_split_);
    FreePtr(&prog_planes_->coeffs[c]);
    FreePtr(&prog_planes_->run_levels[c]);
  }
  return ok_;
}

}  // namespace sjpeg

#endif  // !SJPEG_NO_PROGRESSIVE
