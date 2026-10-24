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
//  Fast and simple JPEG encoder
//
// Author: Skal (pascal.massimino@gmail.com)

#include <assert.h>
#include <float.h>  // for FLT_MAX
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>  // for memcpy / memset

#include <algorithm>
#include <mutex>  // NOLINT
#include <string>
#include <vector>

#define SJPEG_NEED_ASM_HEADERS
#include "sjpegi.h"

using namespace sjpeg;  // for the plain-C entry points at the end

namespace sjpeg {

// Some general default values. Declared in sjpegi.h: api.cc reads them too.
const float kDefaultQuality = 75.f;
const int kDefaultMethod = 4;
// Rounding bias for AC coefficients, as 8bit fixed point.
// A default value 0x78 leans toward filesize reduction.
const int32_t kDefaultBias = 0x78;
// for adaptive quantization:
const int kDefaultDeltaMaxLuma = 12;
const int kDefaultDeltaMaxChroma = 1;

////////////////////////////////////////////////////////////////////////////////
// Default memory manager (singleton)

static struct DefaultMemory : public MemoryManager {
 public:
  ~DefaultMemory() override = default;
  void* Alloc(size_t size) override { return malloc(size); }
  void Free(void* const ptr) override { free(ptr); }
} kDefaultMemory;

MemoryManager* GetDefaultMemoryManager() { return &kDefaultMemory; }

////////////////////////////////////////////////////////////////////////////////
// Encoder main class

Encoder::Encoder(SjpegYUVMode yuv_mode, int W, int H, ByteSink* const sink,
                 MemoryManager* const memory)
    : yuv_mode_(yuv_mode),
      W_(W),
      H_(H),
      bw_(sink),
      hw_(&bw_, &ok_),
      entropy_((yuv_mode == SJPEG_YUV_400) ? 1 : 2),
      qdelta_max_luma_(kDefaultDeltaMaxLuma),
      qdelta_max_chroma_(kDefaultDeltaMaxChroma),
      memory_hook_((memory == nullptr) ? &kDefaultMemory : memory) {
  InitComponents();
  ok_ = ok_ && (W_ > 0 && H_ > 0 && W_ <= kMaxDimension && H_ <= kMaxDimension);
  if (ok_) {
    mb_w_ = (W_ + (block_w_ - 1)) / block_w_;
    mb_h_ = (H_ + (block_h_ - 1)) / block_h_;
    mb_x_max_ = W_ / block_w_;
    mb_y_max_ = H_ / block_h_;
  }
  SetCompressionMethod(kDefaultMethod);
  SetQuality(kDefaultQuality);
  get_yuv_block_ = GetBlockFunc(yuv_mode_);
  SetQuantizationBias(kDefaultBias, false);
  SetDefaultMinQuantMatrices();
  InitializeStaticPointers();
  sink->Reset();
}

Encoder::~Encoder() {
  DeallocateBlocks();  // clean-up leftovers in case of we had an error
#if !defined(SJPEG_NO_PROGRESSIVE)
  DeallocateProgPlanes();
#endif
}

////////////////////////////////////////////////////////////////////////////////

void Encoder::SetQuality(float q) {
  q = GetQFactor(q);
  SetQuantMatrix(kDefaultMatrices[0], q, quants_[0].quant_);
  SetQuantMatrix(kDefaultMatrices[1], q, quants_[1].quant_);
}

void Encoder::SetQuantMatrices(const uint8_t m[2][64]) {
  SetQuantMatrix(m[0], 100, quants_[0].quant_);
  SetQuantMatrix(m[1], 100, quants_[1].quant_);
}

void Encoder::SetMinQuantMatrices(const uint8_t m[2][64], int tolerance) {
  SetMinQuantMatrix(m[0], quants_[0].min_quant_, tolerance);
  SetMinQuantMatrix(m[1], quants_[1].min_quant_, tolerance);
}

void Encoder::SetDefaultMinQuantMatrices() {
  SetDefaultMinQuantMatrix(quants_[0].min_quant_);
  SetDefaultMinQuantMatrix(quants_[1].min_quant_);
}

void Encoder::SetCompressionMethod(int method) {
  method = (method < 0) ? 0 : (method > 8) ? 8 : method;
  use_adaptive_quant_ = (method >= 3);
  optimize_size_ = (method != 0) && (method != 3);
  use_extra_memory_ = (method == 3) || (method == 4) || (method == 7);
  reuse_run_levels_ =
      (method == 1) || (method == 4) || (method == 5) || (method == 7);
  use_trellis_ = (method >= 7);
}

void Encoder::SetRDO(bool use_rdo) { use_rdo_ = use_rdo; }

void Encoder::SetTrellis(bool use_trellis) { use_trellis_ = use_trellis; }

void Encoder::SetProgressive(int luma_split, int chroma_split) {
#if !defined(SJPEG_NO_PROGRESSIVE)
  luma_split = (luma_split < 1) ? 1 : (luma_split > 64) ? 64 : luma_split;
  chroma_split = (chroma_split < 1)    ? 1
                 : (chroma_split > 64) ? 64
                                       : chroma_split;
  prog_luma_split_ = luma_split;
  prog_chroma_split_ = chroma_split;
  // use_extra_memory_/reuse_run_levels_ are baseline-only; EncodeProgressive()
  // has its own storage and doesn't need either of them.
#else
  (void)luma_split;
  (void)chroma_split;  // not compiled in: silently a no-op
#endif
}

void Encoder::SetMetadata(const std::string* data, MetadataType type) {
  switch (type) {
    case ICC:
      iccp_ = data;
      break;
    case EXIF:
      exif_ = data;
      break;
    case XMP:
      xmp_ = data;
      break;
    default:
    case MARKERS:
      app_markers_ = data;
      break;
  }
}

void Encoder::SetQuantizationBias(int bias, bool use_adaptive) {
  assert(bias >= 0 && bias <= 255);
  q_bias_ = bias;
  adaptive_bias_ = use_adaptive;
}

void Encoder::SetQuantizationDeltas(int qdelta_luma, int qdelta_chroma) {
  assert(qdelta_luma >= 0 && qdelta_luma <= 255);
  assert(qdelta_chroma >= 0 && qdelta_chroma <= 255);
  qdelta_max_luma_ = qdelta_luma;
  qdelta_max_chroma_ = qdelta_chroma;
}

////////////////////////////////////////////////////////////////////////////////
// CPU support

extern bool ForceSlowCImplementation;
bool ForceSlowCImplementation = false;  // undocumented! for tests.

bool SupportsSSE2() {
  if (ForceSlowCImplementation) return false;
#if defined(SJPEG_USE_SSE2)
  return true;
#endif
  return false;
}

bool SupportsNEON() {
  if (ForceSlowCImplementation) return false;
#if defined(SJPEG_USE_NEON)
  return true;
#endif
  return false;
}

// unlike SSE2/NEON above, AVX2 support is a real runtime question even on a
// binary built with AVX2 code paths compiled in (SJPEG_HAVE_AVX2): plenty of
// deployed x86-64 CPUs predate Haswell. __builtin_cpu_supports() is itself
// x86-only, so guard on the target too -- HAVE_AVX2=1 is meant for x86
// builds, but nothing stops it being set by mistake (or by a generic CI
// matrix) on another architecture, and this should degrade to "false"
// there, not fail to compile.
bool SupportsAVX2() {
  if (ForceSlowCImplementation) return false;
#if defined(SJPEG_HAVE_AVX2) && (defined(__i386__) || defined(__x86_64__))
#if defined(__clang__) && defined(_MSC_VER)
#if defined(__x86_64__)
#pragma comment(lib, "clang_rt.builtins-x86_64.lib")
#else
#pragma comment(lib, "clang_rt.builtins-i386.lib")
#endif
#endif
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") != 0;
#else
  return false;
#endif
}

////////////////////////////////////////////////////////////////////////////////
// static pointers to architecture-dependant implementation

Encoder::QuantizeErrorFunc Encoder::quantize_error_ = nullptr;
Encoder::QuantizeBlockFunc Encoder::quantize_block_ = nullptr;
void (*Encoder::fDCT_)(int16_t* in, int num_blocks) = nullptr;
Encoder::StoreHistoFunc Encoder::store_histo_ = nullptr;

void Encoder::InitializeStaticPointers() {
  static std::once_flag once;
  std::call_once(once, []() {
    store_histo_ = GetStoreHistoFunc();
    quantize_block_ = GetQuantizeBlockFunc();
    quantize_error_ = GetQuantizeErrorFunc();
    fDCT_ = GetFdct();
  });
  assert(store_histo_ != nullptr);
  assert(quantize_block_ != nullptr);
  assert(quantize_error_ != nullptr);
  assert(fDCT_ != nullptr);
}

////////////////////////////////////////////////////////////////////////////////
// memory and internal buffers management. We grow on demand.

bool Encoder::SetError() const {
  ok_ = false;
  return false;
}

size_t Encoder::SliceSlabSize(int first_interval, int end_interval) const {
  const int rows_per_interval =
      (restart_interval_rows_ > 0) ? restart_interval_rows_ : mb_h_;
  const int y_first = first_interval * rows_per_interval;
  const int y_end = std::min(mb_h_, end_interval * rows_per_interval);
  size_t slab = (y_first == 0 && y_end == mb_h_)
                    ? static_cast<size_t>(W_) * H_ / 4
                    : static_cast<size_t>(y_end - y_first) * block_h_ * mb_w_ *
                          block_w_ / 4;
  if (slab < 4096) slab = 4096;
  if (slab > (256 << 10)) slab = 256 << 10;
  return slab;
}

#if defined(SJPEG_NO_MULTITHREADING)
int Encoder::GetNumSlices(int /*cap*/, int /*grain*/) const { return 1; }
#endif  // SJPEG_NO_MULTITHREADING

bool Encoder::ReserveSlab() {
  // Worst-case macroblock is 24bits*64*6 coeffs = 1152 bytes, doubled by 0xff
  // stuffing, so kMaxMCUSize (2560) covers one MCU. Writer serves that out of a
  // larger slab and only reaches the sink when the slab runs out. Slab follows
  // the image rather than being fixed: at a flat 256k, a 64x64 thumbnail whose
  // JPEG is 3kB holds half a megabyte of capacity, and a string never gives it
  // back.
  ok_ = ok_ &&
        bw_.ReserveMore(kMaxMCUSize, SliceSlabSize(0, TotalRestartIntervals()));
  return ok_;
}

bool Encoder::AllocateBlocks(size_t num_blocks) {
  assert(in_blocks_ == nullptr);
  have_coeffs_ = false;
  size_t size = 0;
  if (!SafeMultiply<size_t>(num_blocks, 64 * sizeof(*in_blocks_), &size) ||
      !SafeAdd<size_t>(size, ALIGN_CST, &size)) {
    return false;
  }
  in_blocks_base_ = Alloc<uint8_t>(size);
  if (in_blocks_base_ == nullptr) return false;
  in_blocks_ = reinterpret_cast<int16_t*>(
      (ALIGN_CST + reinterpret_cast<uintptr_t>(in_blocks_base_)) & ~ALIGN_CST);
  return true;
}

void Encoder::DeallocateBlocks() {
  FreePtr(&in_blocks_base_);
  in_blocks_ = nullptr;  // safety
}

////////////////////////////////////////////////////////////////////////////////
// Perform YUV conversion and fDCT, and store the unquantized coeffs

void Encoder::TransformMCU(int mb_x, int mb_y, int16_t* const out,
                           uint8_t* rep_buf) {
  const bool yclip = (mb_y == mb_y_max_);
  GetSamples(mb_x, mb_y, yclip | (mb_x == mb_x_max_), out, rep_buf);
  fDCT_(out, mcu_blocks_);
}

void Encoder::TransformMCU(int mb_x, int mb_y, int16_t* const out) {
  TransformMCU(mb_x, mb_y, out, replicated_buffer_);
}

void Encoder::MaybeTransformMCU(int mb_x, int mb_y, int16_t** const in) {
  if (have_coeffs_) return;
  *in = in_blocks_;
  TransformMCU(mb_x, mb_y, *in);
}

const int16_t* Encoder::GetMCUCoeffs(int mb_x, int mb_y, int16_t* scratch,
                                     uint8_t* rep_buf) {
  if (have_coeffs_) {
    // CollectCoeffs() already filled in_blocks_, in MCU raster order.
    return in_blocks_ +
           static_cast<size_t>(mb_y * mb_w_ + mb_x) * 64 * mcu_blocks_;
  }
  TransformMCU(mb_x, mb_y, scratch, rep_buf);
  return scratch;
}

void Encoder::CollectCoeffsSlice(int y_start, int y_end, uint8_t* rep_buf) {
  assert(use_extra_memory_);
  int16_t* in =
      in_blocks_ + static_cast<size_t>(y_start) * mb_w_ * 64 * mcu_blocks_;
  for (int mb_y = y_start; mb_y < y_end; ++mb_y) {
    for (int mb_x = 0; mb_x < mb_w_; ++mb_x) {
      TransformMCU(mb_x, mb_y, in, rep_buf);
      in += 64 * mcu_blocks_;
    }
  }
}

void Encoder::CollectCoeffs() {
#if !defined(SJPEG_NO_MULTITHREADING)
  const int num_slices = GetNumSlices(mb_h_);
  if (num_slices > 1) {
    CollectCoeffsMultiThreaded(num_slices);
    return;
  }
#endif
  CollectCoeffsSlice(0, mb_h_, replicated_buffer_);
  have_coeffs_ = true;
}

////////////////////////////////////////////////////////////////////////////////
// Shared scan slice kernels (used by both single-threaded and multi-threaded)

bool Encoder::EmitRestartMarker(BitWriter* bw, int interval_idx,
                                int total_intervals, size_t slab_size) {
  if (interval_idx >= total_intervals - 1) return true;
  bw->Flush();  // pad to the byte boundary the marker must start on
  if (!bw->ReserveMore(2, slab_size)) return false;
  const uint8_t rst_marker[2] = {
      0xff, static_cast<uint8_t>(0xd0 + (interval_idx & 7))};
  bw->PutBytes(rst_marker, 2);
  return true;
}

bool Encoder::CodeScanSlice(int first_interval, int end_interval,
                            int total_intervals, BitWriter* bw,
                            size_t slab_size, int16_t* scratch,
                            uint8_t* rep_buf) {
  const int rows_per_interval =
      (restart_interval_rows_ > 0) ? restart_interval_rows_ : mb_h_;
  const QuantizeBlockFunc quantize_block = GetActiveQuantizeBlockFunc();
  RunLevel run_levels[64];

  for (int iv = first_interval; iv < end_interval; ++iv) {
    int DCs[3] = {0, 0, 0};  // this restart interval's DC predictors
    const int y_end = std::min(mb_h_, (iv + 1) * rows_per_interval);
    for (int mb_y = iv * rows_per_interval; mb_y < y_end; ++mb_y) {
      for (int mb_x = 0; mb_x < mb_w_; ++mb_x) {
        if (!bw->ReserveMore(kMaxMCUSize, slab_size)) return false;
        const int16_t* in = GetMCUCoeffs(mb_x, mb_y, scratch, rep_buf);
        for (int c = 0; c < nb_comps_; ++c) {
          DCTCoeffs coeffs;
          for (int i = 0; i < nb_blocks_[c]; ++i) {
            const int dc = quantize_block(in, c, quants_[quant_idx_[c]],
                                          &coeffs, run_levels);
            coeffs.dc_code_ = GenerateDCDiffCode(dc, &DCs[c]);
            CodeBlock(&coeffs, run_levels, bw);
            in += 64;
          }
        }
      }
    }
    if (!EmitRestartMarker(bw, iv, total_intervals, slab_size)) return false;
  }
  return true;
}

bool Encoder::QuantizeScanSlice(int first_interval, int end_interval,
                                SliceStore* store, EntropyStats* stats,
                                int16_t* scratch, uint8_t* rep_buf) {
  const int rows_per_interval =
      (restart_interval_rows_ > 0) ? restart_interval_rows_ : mb_h_;
  const QuantizeBlockFunc quantize_block = GetActiveQuantizeBlockFunc();
  const bool keep_run_levels = (store != nullptr);

  DCTCoeffs single_coeff;
  RunLevel fixed_rl[64];
  DCTCoeffs* coeffs = keep_run_levels ? store->coeffs.data() : nullptr;
  size_t nb_rl = 0;

  for (int iv = first_interval; iv < end_interval; ++iv) {
    int DCs[3] = {0, 0, 0};  // this restart interval's DC predictors
    const int y_end = std::min(mb_h_, (iv + 1) * rows_per_interval);
    for (int mb_y = iv * rows_per_interval; mb_y < y_end; ++mb_y) {
      for (int mb_x = 0; mb_x < mb_w_; ++mb_x) {
        if (keep_run_levels &&
            !store->run_levels.Reserve(nb_rl + 6 * 64, nb_rl)) {
          return false;
        }
        const int16_t* in = GetMCUCoeffs(mb_x, mb_y, scratch, rep_buf);
        for (int c = 0; c < nb_comps_; ++c) {
          const int q_idx = quant_idx_[c];
          for (int i = 0; i < nb_blocks_[c]; ++i) {
            DCTCoeffs* const cur_coeffs =
                keep_run_levels ? coeffs : &single_coeff;
            RunLevel* const rl =
                keep_run_levels ? store->run_levels.data() + nb_rl : fixed_rl;
            const int dc =
                quantize_block(in, c, quants_[q_idx], cur_coeffs, rl);
            cur_coeffs->dc_code_ = GenerateDCDiffCode(dc, &DCs[c]);
            if (stats != nullptr) stats->AddBlock(q_idx, cur_coeffs, rl);
            if (keep_run_levels) {
              nb_rl += cur_coeffs->nb_coeffs_;
              ++coeffs;
            }
            in += 64;
          }
        }
      }
    }
  }
  if (keep_run_levels) store->nb_run_levels = nb_rl;
  return true;
}

bool Encoder::ReplayScanSlice(int first_interval, int end_interval,
                              int total_intervals, size_t nb_blocks,
                              const DCTCoeffs* coeffs, const RunLevel* rl,
                              BitWriter* bw, size_t slab_size) {
  const size_t blocks_per_interval =
      (restart_interval_rows_ > 0)
          ? static_cast<size_t>(restart_interval_rows_) * mb_w_ * mcu_blocks_
          : nb_blocks;
  size_t n = 0;
  for (int iv = first_interval; iv < end_interval; ++iv) {
    // Non-final intervals have exactly blocks_per_interval blocks; capping at
    // nb_blocks handles only the image's final (potentially partial) interval.
    const size_t iv_end = std::min(nb_blocks, n + blocks_per_interval);
    for (; n < iv_end; ++n) {
      if (!bw->ReserveMore(kMaxMCUSize, slab_size)) return false;
      CodeBlock(&coeffs[n], rl, bw);
      rl += coeffs[n].nb_coeffs_;
    }
    if (!EmitRestartMarker(bw, iv, total_intervals, slab_size)) return false;
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////
// 1-pass Scan

void Encoder::SinglePassScan() {
  const int total_intervals = TotalRestartIntervals();
#if !defined(SJPEG_NO_MULTITHREADING)
  const int num_slices = GetNumSlices(total_intervals, have_coeffs_ ? 64 : 0);
  if (num_slices > 1) {
    SinglePassScanMultiThreaded(num_slices, total_intervals);
    return;
  }
#endif
  if (!CodeScanSlice(0, total_intervals, total_intervals, &bw_,
                     SliceSlabSize(0, total_intervals), in_blocks_,
                     replicated_buffer_)) {
    SetError();
  }
}

////////////////////////////////////////////////////////////////////////////////

void Encoder::SinglePassScanOptimized() {
  const int total_intervals = TotalRestartIntervals();
  const int num_slices = GetNumSlices(total_intervals, have_coeffs_ ? 64 : 0);
  ChunkSet chunks(memory_hook_);
  QuantizeSlices(num_slices, total_intervals, &chunks);
  if (!ok_) return;

  entropy_.Compile(stats_);
  WriteDHT();
  WriteSOS();

  if (!reuse_run_levels_) {
    SinglePassScan();  // redo everything, but with optimal tables now.
  } else {
    DeallocateBlocks();  // pass 2 replays run/levels; the coeffs are dead now
    ReplaySlices(num_slices, total_intervals, &chunks);
  }
}

// Quantization and replay helpers shared by the 2-pass optimized scan and the
// multi-pass dichotomy search. Re-quantizing in pass 2 would not merely waste
// work: WriteDHT() calls InitCodes(), which rebuilds the AC codes, and
// quants_[].codes_ points into them, so trellis and RDO quantization would
// silently re-decide against the optimized codes and stop matching the
// histogram those tables were built from.

void Encoder::QuantizeSlices(int num_slices, int total_intervals,
                             ChunkSet* chunks) {
  const int rows_per_interval =
      (restart_interval_rows_ > 0) ? restart_interval_rows_ : mb_h_;
  const bool collect_stats = optimize_size_;
  if (!chunks->Reset(num_slices)) {
    SetError();
    return;
  }

  InitCostCodes();

  ForEachSlice(
      num_slices, total_intervals,
      [&](int t, int first_interval, int end_interval) {
        ThreadChunk& chunk = (*chunks)[t];
        SliceStore* const store = reuse_run_levels_ ? &chunk.store : nullptr;
        if (collect_stats) chunk.store.stats.Reset();
        chunk.store.nb_blocks = 0;
        chunk.store.nb_run_levels = 0;
        if (reuse_run_levels_) {
          const int y_first = first_interval * rows_per_interval;
          const int y_end = std::min(mb_h_, end_interval * rows_per_interval);
          const size_t nb_blocks =
              (size_t)(y_end - y_first) * mb_w_ * mcu_blocks_;
          // Blocks code far fewer than 64 run/levels on average; the buffer
          // grows on demand.
          if (!chunk.store.coeffs.Reserve(nb_blocks) ||
              !chunk.store.run_levels.Reserve(nb_blocks * 8 + 6 * 64)) {
            chunk.ok = false;
            return;
          }
          chunk.store.nb_blocks = nb_blocks;
        }
        alignas(16) int16_t scratch[64 * 6];
        uint8_t rep_buf[4 * 16 * 16];
        chunk.ok = QuantizeScanSlice(
            first_interval, end_interval, store,
            collect_stats ? &chunk.store.stats : nullptr, scratch, rep_buf);
      });

  for (const ThreadChunk& chunk : *chunks) {
    if (!chunk.ok) SetError();
  }
  if (ok_ && collect_stats) MergeChunkStats(chunks->data(), num_slices);
}

void Encoder::MergeChunkStats(const ThreadChunk* chunks, int num_chunks) {
  stats_.Reset();
  for (int t = 0; t < num_chunks; ++t) {
    stats_.Add(chunks[t].store.stats, entropy_.nb_tables());
  }
}

void Encoder::ReplaySlices(int num_slices, int total_intervals,
                           ChunkSet* chunks) {
  if (num_slices == 1) {  // straight into the output
    const SliceStore& store = (*chunks)[0].store;
    if (!ReplayScanSlice(0, total_intervals, total_intervals, store.nb_blocks,
                         store.coeffs.data(), store.run_levels.data(), &bw_,
                         SliceSlabSize(0, total_intervals))) {
      SetError();
    }
    return;
  }
#if !defined(SJPEG_NO_MULTITHREADING)
  ForEachSlice(
      num_slices, total_intervals,
      [&](int t, int first_interval, int end_interval) {
        ThreadChunk& chunk = (*chunks)[t];
        BitWriter bw(&chunk.data);
        const size_t slab = SliceSlabSize(first_interval, end_interval);
        if (!ReplayScanSlice(first_interval, end_interval, total_intervals,
                             chunk.store.nb_blocks, chunk.store.coeffs.data(),
                             chunk.store.run_levels.data(), &bw, slab)) {
          chunk.ok = false;
          return;
        }
        bw.Flush();
        chunk.ok = bw.Finalize();
      });
  ConcatenateChunks(chunks->data(), num_slices);
#endif
}

float Encoder::EvaluateSize(int num_slices, int total_intervals,
                            ChunkSet* chunks) {
  QuantizeSlices(num_slices, total_intervals, chunks);
  if (!ok_) return 0.f;
  if (optimize_size_) {
    entropy_.Compile(stats_);
    InitCostCodes();
  }
  entropy_.InitCodes(false);
  if (optimize_size_) {
    // not counting the 0xff byte-stuffing that BlocksSize() tracks exactly
    // these depends on the bit sequence, not on symbol counts.
    // it's a little approximation we can live with.
    // Under-estimated ~1/256 = 0.39%: uniform bytes, 0xff produces 2 bytes.
    // Not worth correcting: below the search's own convergence precision.
    return ComputeSize(entropy_.Size(stats_));
  }
  BitCounter bc;
  for (const ThreadChunk& chunk : *chunks) {
    BlocksSize((int)chunk.store.nb_blocks, chunk.store.coeffs.data(),
               chunk.store.run_levels.data(), &bc);
  }
  return ComputeSize(bc.Size());
}

////////////////////////////////////////////////////////////////////////////////
// Marker writers: gather the frame state for the HeaderWriter (headers.cc)

void Encoder::WriteDQT() { hw_.WriteDQT(quants_, entropy_.nb_tables()); }

void Encoder::WriteSOF(bool progressive) {
  hw_.WriteSOF(W_, H_, nb_comps_, block_dims_, quant_idx_, progressive);
}

void Encoder::WriteDHT() {
  entropy_.InitCodes(false);
  hw_.WriteDHT(entropy_);
}

// DRI's interval field counts MCUs, whereas restart_interval_rows_ counts MCU
// rows. Encode() has already clamped the latter so the product fits 16 bits.
void Encoder::WriteDRI() { hw_.WriteDRI(restart_interval_rows_ * mb_w_); }

void Encoder::WriteSOS() { hw_.WriteSOS(nb_comps_, quant_idx_); }

////////////////////////////////////////////////////////////////////////////////
// main call

bool Encoder::Encode() {
  if (!ok_) return false;

  FinalizeQuantMatrix(&quants_[0], q_bias_, adaptive_bias_);
  FinalizeQuantMatrix(&quants_[1], q_bias_, adaptive_bias_);
  SetCostCodes(0);
  SetCostCodes(1);

  entropy_.SetDefaultTables();
  InitCostCodes();
  // DRI stores the interval as a 16-bit count of MCUs, so a row-based interval
  // can't exceed 0xffff MCUs. Clamp to the coarsest whole number of MCU rows
  // that fits: restarts become more frequent than asked for, never less, and
  // stay row-aligned as the scan loops and the DC resets require. Dimensions
  // are capped at kMaxDimension, so mb_w_ <= 8192 and this is always >= 7.
  if (restart_interval_rows_ > 0) {
    restart_interval_rows_ = std::min(restart_interval_rows_, 0xffff / mb_w_);
  }
  size_t nb_blocks = 1;
  if (use_extra_memory_) {
    if (!SafeMultiply<size_t>(mb_w_, mb_h_, &nb_blocks)) return false;
  }
  size_t total_blocks = 0;
  if (!SafeMultiply<size_t>(nb_blocks, mcu_blocks_, &total_blocks) ||
      !AllocateBlocks(total_blocks)) {
    return false;
  }

  hw_.WriteAPP0();
  if (!hw_.WriteAPPMarkers(*app_markers_) || !hw_.WriteEXIF(*exif_) ||
      !hw_.WriteICCP(*iccp_) || !hw_.WriteXMP(*xmp_, xmp_split_)) {
    return false;
  }

  if (passes_ > 1) {
    // Note: progressive mode is not supported together with the multi-pass
    // target-size/PSNR search; -prog is silently ignored in that case.
    LoopScan();
  } else {
    SinglePassEncode();
  }
  hw_.WriteEOI();
  ok_ = ok_ && bw_.Finalize();

  DeallocateBlocks();
  return ok_;
}

void Encoder::SinglePassEncode() {
  if (use_adaptive_quant_) {
    // Histogram analysis + derive optimal quant matrices.
    CollectHistograms();
    AnalyseHisto();
  }

  WriteDQT();

#if !defined(SJPEG_NO_PROGRESSIVE)
  // progressive path: must be after WriteDQT() and before WriteSOF()
  if (prog_luma_split_ < 64) {  // needs progressive coding?
    WriteSOF(/*progressive=*/true);
    if (!EncodeProgressive()) SetError();
    return;
  }
#endif

  // baseline coding
  WriteBaselineAndScan();
}

void Encoder::WriteBaselineAndScan() {
  WriteSOF();
  WriteDRI();

  if (optimize_size_) {
    SinglePassScanOptimized();
  } else {
    WriteDHT();
    WriteSOS();
    SinglePassScan();
  }
}

}  // namespace sjpeg

////////////////////////////////////////////////////////////////////////////////

bool SjpegCompress(const uint8_t* rgb, int width, int height, float quality,
                   std::string* output) {
  EncoderParam param;
  param.SetQuality(quality);
  if (!ValidDimensions(width, height)) return false;  // before '3 * width'
  return Encode(rgb, width, height, 3 * width, param, output);
}

bool SjpegDimensions(const std::string& jpeg_data, int* width, int* height,
                     int* is_yuv420) {
  return SjpegDimensions(reinterpret_cast<const uint8_t*>(jpeg_data.data()),
                         jpeg_data.size(), width, height, is_yuv420);
}

int SjpegFindQuantizer(const std::string& jpeg_data, uint8_t quant[2][64]) {
  return SjpegFindQuantizer(reinterpret_cast<const uint8_t*>(jpeg_data.data()),
                            jpeg_data.size(), quant);
}

////////////////////////////////////////////////////////////////////////////////
