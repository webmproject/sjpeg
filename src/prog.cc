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
#include <atomic>

#include "sjpegi.h"

#if !defined(SJPEG_NO_PROGRESSIVE)

namespace sjpeg {

// spec's 15-bit EOBn cap (class R=14 tops out at 2^15-1)
enum { kMaxEOBRun = 0x7fff };

////////////////////////////////////////////////////////////////////////////////
// handles progressive planes

bool Encoder::AllocateProgPlanes(int nb_slices) {
  prog_planes_ = Alloc<ProgPlanes>(1);
  if (prog_planes_ == nullptr) return false;
  memset(prog_planes_, 0, sizeof(*prog_planes_));  // pointers=null
  prog_planes_->slices = Alloc<ProgPlanes::Slice>(nb_slices);
  if (prog_planes_->slices == nullptr) return false;
  memset(prog_planes_->slices, 0, nb_slices * sizeof(ProgPlanes::Slice));
  prog_planes_->nb_slices = nb_slices;
  // The memory manager is only called from the calling thread: threaded
  // slices allocate from the C heap.
  prog_planes_->rl_memory =
      (nb_slices == 1) ? memory_hook_ : GetDefaultMemoryManager();

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
    assert(h_samp * v_samp <= 4);
    for (int i = 0; i < h_samp * v_samp; ++i) {
      prog_planes_->sub_offset[c][i] =
          (i / h_samp) * prog_planes_->plane_w[c] + i % h_samp;
    }
    const size_t nb_blocks =
        (size_t)prog_planes_->plane_w[c] * prog_planes_->plane_h[c];
    // coeffs[] and rl_offset[] share one allocation (see ProgPlanes)
    static_assert(sizeof(DCTCoeffs) == 2 * sizeof(uint32_t), "packing");
    const size_t nb_slots = nb_blocks + (nb_blocks + 1) / 2;
    prog_planes_->coeffs[c] = Alloc<DCTCoeffs>(nb_slots);
    if (prog_planes_->coeffs[c] == nullptr) return false;
    prog_planes_->rl_offset[c] =
        (uint32_t*)(prog_planes_->coeffs[c] + nb_blocks);
    // Need uint64_t, not size_t (which is 32bit on 32-bit build): the most a
    // plane can ever need, nb_blocks * 63, can overflow 32bit. It must also
    // fit rl_offset[]'s 32 bits (65535x65535: 4.23G, just below).
    const uint64_t rl_max = (uint64_t)nb_blocks * 63 + 64;
    if (rl_max > SIZE_MAX / sizeof(RunLevel) || rl_max > UINT32_MAX) {
      return SetError();
    }
  }
  return true;
}

RunLevel* Encoder::AllocProgRunLevels(size_t num) {
  // no overflow: num is at most 63 per block + 64, checked at allocation
  return (RunLevel*)prog_planes_->rl_memory->Alloc(num * sizeof(RunLevel));
}

void Encoder::FreeProgRunLevels(int c) {
  for (int t = 0; t < prog_planes_->nb_slices; ++t) {
    RunLevel** const rl = &prog_planes_->slices[t].run_levels[c];
    if (*rl != nullptr) prog_planes_->rl_memory->Free(*rl);
    *rl = nullptr;
  }
}

void Encoder::DeallocateProgPlanes() {
  if (prog_planes_ == nullptr) return;
  for (int c = 0; c < MAX_COMP; ++c) {
    Free(prog_planes_->coeffs[c]);
    if (prog_planes_->slices != nullptr) FreeProgRunLevels(c);
  }
  Free(prog_planes_->slices);
  FreePtr(&prog_planes_);
}

// Doubles the capacity of the slice's run_levels[] for component c, up to
// what its blocks can ever need. Old and new buffers coexist during the copy:
// with more than 32 run/levels per block on average, that's a transient 95 per
// block.
bool Encoder::GrowProgRunLevels(ProgPlanes::Slice* const s, int c) {
  const size_t new_capacity = std::min(2 * s->rl_capacity[c], s->rl_max[c]);
  RunLevel* const new_rl = AllocProgRunLevels(new_capacity);
  if (new_rl == nullptr) return false;
  memcpy(new_rl, s->run_levels[c], s->rl_used[c] * sizeof(RunLevel));
  prog_planes_->rl_memory->Free(s->run_levels[c]);
  s->run_levels[c] = new_rl;
  s->rl_capacity[c] = new_capacity;
  return true;
}

int Encoder::ProgPlaneIndex(int c, int mb_x, int mb_y, int i) const {
  const int h_samp = block_dims_[c] >> 4;
  const int v_samp = block_dims_[c] & 0xf;
  return mb_y * v_samp * prog_planes_->plane_w[c] + mb_x * h_samp +
         prog_planes_->sub_offset[c][i];
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
  const int v_samp = block_dims_[c] & 0xf;
  const int nb_slices = prog_planes_->nb_slices;
  RunLevel windowed[64];
  int eobrun = 0;  // persists across blocks
  // first block row of slice s + 1 (slices are never empty)
  const auto next_row = [&](int s) {
    return (s + 1 < nb_slices) ? (s + 1) * mb_h_ / nb_slices * v_samp : th;
  };
  int t = 0;  // slice holding the row's run/levels
  int t_end = next_row(t);
  for (int row = 0; row < th; ++row) {
    if (row == t_end) t_end = next_row(++t);
    const RunLevel* const run_levels = prog_planes_->slices[t].run_levels[c];
    for (int col = 0; col < tw; ++col) {
      if (!on_start()) return false;
      const int n = row * stride + col;
      const DCTCoeffs* const coeffs = &prog_planes_->coeffs[c][n];
      const RunLevel* const rl = run_levels + prog_planes_->rl_offset[c][n];
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

bool Encoder::CodeProgACScan(int c, int Ss, int Se, ProgACCoder* const ac,
                             BitWriter* const bw, size_t slab_size) const {
  // pass 1: gather the stats and build the table
  ac->Reset();
  WalkACBand(
      c, Ss, Se, [&](const RunLevel* rl, int n) { ac->AddWindowed(rl, n); },
      [&](int run) { ac->AddEOBRun(run); }, [] { return true; });
  ac->Compile();

  if (!bw->ReserveMore(kMaxMCUSize, slab_size)) return false;
  bool ok = true;
  HeaderWriter hw(bw, &ok);
  hw.WriteOneDHT(/*table_class=*/1, /*table_id=*/0, &ac->table);
  const HeaderWriter::ScanComponent sc = {c, /*dc=*/0, /*ac=*/0};
  hw.WriteSOS(&sc, 1, Ss, Se);
  if (!ok) return false;

  // pass 2: code the scan. Per-block buffer check: a whole row can exceed
  // the 2560-byte guarantee.
  if (!WalkACBand(
          c, Ss, Se,
          [&](const RunLevel* rl, int n) { ac->CodeWindowed(rl, n, bw); },
          [&](int run) { ac->CodeEOBRun(run, bw); },
          [&] { return bw->ReserveMore(kMaxMCUSize, slab_size); })) {
    return false;
  }
  bw->Flush();
  return true;
}

bool Encoder::EncodeProgressive() {
  const int nb_slices = GetNumSlices(mb_h_);
  const bool ok = AllocateProgPlanes(nb_slices) && EncodeProgScans();
  DeallocateProgPlanes();
  return ok;
}

bool Encoder::QuantizeProgSlice(int t, int y_start, int y_end,
                                int16_t* const scratch,
                                uint8_t* const rep_buf) {
  ProgPlanes::Slice* const s = &prog_planes_->slices[t];
  const QuantizeBlockFunc quantize_block = GetActiveQuantizeBlockFunc();
  for (int c = 0; c < nb_comps_; ++c) {
    const size_t nb_blocks = (size_t)(y_end - y_start) * mb_w_ * nb_blocks_[c];
    // Starting guess, like the baseline path. Grown by GrowProgRunLevels().
    s->rl_capacity[c] = nb_blocks * 8 + 64;
    s->rl_max[c] = nb_blocks * 63 + 64;
    s->run_levels[c] = AllocProgRunLevels(s->rl_capacity[c]);
    if (s->run_levels[c] == nullptr) return false;
  }
  s->stats.Reset();
  int DCs[MAX_COMP] = {0, 0, 0};  // re-chained across slices afterward
  // Same mb_y/mb_x/c/i nest as the DC-code pass -- keep them in sync.
  for (int mb_y = y_start; mb_y < y_end; ++mb_y) {
    for (int mb_x = 0; mb_x < mb_w_; ++mb_x) {
      const int16_t* in = GetMCUCoeffs(mb_x, mb_y, scratch, rep_buf);
      for (int c = 0; c < nb_comps_; ++c) {
        for (int i = 0; i < nb_blocks_[c]; ++i) {
          const int idx = ProgPlaneIndex(c, mb_x, mb_y, i);
          DCTCoeffs* const coeffs = &prog_planes_->coeffs[c][idx];
          // room for one block: 63 run/levels, plus the quantizers' slack
          const size_t rl_pos = s->rl_used[c];
          if (rl_pos + 64 > s->rl_capacity[c] && !GrowProgRunLevels(s, c)) {
            return false;
          }
          const int dc = quantize_block(in, c, quants_[quant_idx_[c]], coeffs,
                                        s->run_levels[c] + rl_pos);
          prog_planes_->rl_offset[c][idx] = (uint32_t)rl_pos;
          s->rl_used[c] = rl_pos + coeffs->nb_coeffs_;
          if (mb_x == 0 && mb_y == y_start && i == 0) s->first_dc[c] = dc;
          coeffs->dc_code_ = GenerateDCDiffCode(dc, &DCs[c]);
          s->stats.AddDC(quant_idx_[c], coeffs);
          in += 64;
        }
      }
    }
  }
  for (int c = 0; c < nb_comps_; ++c) s->last_dc[c] = DCs[c];
  return true;
}

bool Encoder::EncodeProgScans() {
  // 1. Quantize every block once, gathering DC entropy stats along the way.
  // trellis/rdo need the AC codes seeded, like the baseline path does
  InitCostCodes();
  const int nb_slices = prog_planes_->nb_slices;
  RunParallel(nb_slices, mb_h_, [&](int t, int y_start, int y_end) {
    alignas(16) int16_t scratch[64 * 6];
    uint8_t rep_buf[4 * 16 * 16];
    prog_planes_->slices[t].ok =
        QuantizeProgSlice(t, y_start, y_end, scratch, rep_buf);
  });
  DeallocateBlocks();  // we can free up some coeffs memory at this point
  stats_.Reset();
  for (int t = 0; t < nb_slices; ++t) {
    ProgPlanes::Slice* const s = &prog_planes_->slices[t];
    if (!s->ok) return SetError();
    if (t > 0) {  // re-code the slice's first DCs against the previous slice
      const int mb_y = t * mb_h_ / nb_slices;
      for (int c = 0; c < nb_comps_; ++c) {
        uint32_t* const dc_stats = s->stats.dc[quant_idx_[c]];
        DCTCoeffs* const coeffs =
            &prog_planes_->coeffs[c][ProgPlaneIndex(c, 0, mb_y, 0)];
        int predictor = prog_planes_->slices[t - 1].last_dc[c];
        --dc_stats[coeffs->dc_code_ & 0x0f];
        coeffs->dc_code_ = GenerateDCDiffCode(s->first_dc[c], &predictor);
        ++dc_stats[coeffs->dc_code_ & 0x0f];
      }
    }
    stats_.Add(s->stats, entropy_.nb_tables());
  }
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

  // 3. AC scans: luma, then chroma, one or two bands each. They're
  // independent of each other.
  struct Scan {
    int c, Ss, Se;
  } scans[2 * MAX_COMP];
  int nb_scans = 0;
  for (int c = 0; c < nb_comps_; ++c) {
    const int split = (c == 0) ? prog_luma_split_ : prog_chroma_split_;
    if (split < 63) {
      scans[nb_scans++] = {c, 1, split};
      scans[nb_scans++] = {c, split + 1, 63};
    } else {
      scans[nb_scans++] = {c, 1, 63};
    }
  }
  const size_t slab_size = SliceSlabSize(0, TotalRestartIntervals());
  const int nb_threads = GetNumSlices(nb_scans);
  if (nb_threads > 1) {
    // Each scan goes into its own chunk, handed out in order (luma first, the
    // largest), and the chunks are concatenated afterward.
    ChunkSet chunks(nullptr);
    if (!chunks.Reset(nb_scans)) return SetError();
    std::atomic<int> next_scan(0);
    RunParallel(nb_threads, nb_threads, [&](int, int, int) {
      ProgACCoder ac;
      for (int n; (n = next_scan++) < nb_scans;) {
        ThreadChunk& chunk = chunks[n];
        BitWriter bw(&chunk.data);
        chunk.ok = CodeProgACScan(scans[n].c, scans[n].Ss, scans[n].Se, &ac,
                                  &bw, slab_size) &&
                   bw.Finalize();
      }
    });
    ConcatenateChunks(chunks.data(), nb_scans);
    return ok_;
  }
  // Serial: straight into bw_, freeing each component right after its scans.
  ProgACCoder ac;
  for (int n = 0; n < nb_scans; ++n) {
    const int c = scans[n].c;
    if (!CodeProgACScan(c, scans[n].Ss, scans[n].Se, &ac, &bw_, slab_size)) {
      return SetError();
    }
    if (n + 1 == nb_scans || scans[n + 1].c != c) {
      FreePtr(&prog_planes_->coeffs[c]);
      FreeProgRunLevels(c);
    }
  }
  return ok_;
}

}  // namespace sjpeg

#endif  // !SJPEG_NO_PROGRESSIVE
