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

#include <cstdlib>
#include <memory>
#include <mutex>  // NOLINT
#include <new>
#include <string>

#include "bit_writer.h"

#define SJPEG_NEED_ASM_HEADERS
#include "sjpegi.h"

using namespace sjpeg;   // for the plain-C entry points at the end

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
  ~DefaultMemory() override {}
  void* Alloc(size_t size) override { return malloc(size); }
  void Free(void* const ptr) override { free(ptr); }
} kDefaultMemory;

MemoryManager* GetDefaultMemoryManager() { return &kDefaultMemory; }

////////////////////////////////////////////////////////////////////////////////
// Encoder main class

Encoder::Encoder(SjpegYUVMode yuv_mode, int W, int H, ByteSink* const sink,
                 MemoryManager* const memory)
  : yuv_mode_(yuv_mode), W_(W), H_(H),
    ok_(true),
    bw_(sink),
    in_blocks_base_(nullptr),
    in_blocks_(nullptr),
    have_coeffs_(false),
    all_run_levels_(nullptr),
    nb_run_levels_(0),
    max_run_levels_(0),
    qdelta_max_luma_(kDefaultDeltaMaxLuma),
    qdelta_max_chroma_(kDefaultDeltaMaxChroma),
    passes_(1),
    search_hook_(nullptr),
    memory_hook_((memory == nullptr) ? &kDefaultMemory : memory) {
  SetCompressionMethod(kDefaultMethod);
  SetQuality(kDefaultQuality);
  get_yuv_block_ = GetBlockFunc(yuv_mode_);
  SetQuantizationBias(kDefaultBias, false);
  SetDefaultMinQuantMatrices();
  InitializeStaticPointers();
  memset(dc_codes_, 0, sizeof(dc_codes_));  // safety
  memset(ac_codes_, 0, sizeof(ac_codes_));
  sink->Reset();
}

Encoder::~Encoder() {
  Free(all_run_levels_);
  DeallocateBlocks();   // clean-up leftovers in case of we had an error
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
  reuse_run_levels_ = (method == 1) || (method == 4) || (method == 5)
                   || (method >= 7);
  use_trellis_ = (method >= 7);
}

void Encoder::SetMetadata(const std::string& data, MetadataType type) {
  switch (type) {
    case ICC: iccp_ = data; break;
    case EXIF: exif_ = data; break;
    case XMP: xmp_ = data; break;
    default:
    case MARKERS: app_markers_ = data; break;
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
bool ForceSlowCImplementation = false;   // undocumented! for tests.

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

bool Encoder::SetError() {
  ok_ = false;
  return false;
}

bool Encoder::CheckBuffers() {
  // Worst-case macroblock is 24bits*64*6 coeffs = 1152 bytes, doubled by 0xff
  // stuffing, so 2560 covers one MCU. Writer serves that out of a larger slab
  // and only reaches the sink when the slab runs out. Slab follows the image
  // rather than being fixed: at a flat 256k, a 64x64 thumbnail whose JPEG is
  // 3kB holds half a megabyte of capacity, and a string never gives it back.
  size_t chunk = (size_t)W_ * H_ / 4;
  if (chunk < 4096) chunk = 4096;
  if (chunk > (256 << 10)) chunk = 256 << 10;
  ok_ = ok_ && bw_.ReserveMore(2560, chunk);
  if (!ok_) return false;

  if (reuse_run_levels_) {
    if (nb_run_levels_ + 6*64 > max_run_levels_) {
      // need to grow storage for run/levels
      const size_t new_size = max_run_levels_ ? max_run_levels_ * 2 : 8192;
      RunLevel* const new_rl = Alloc<RunLevel>(new_size);
      if (new_rl == nullptr) return false;
      if (nb_run_levels_ > 0) {
        memcpy(new_rl, all_run_levels_,
               nb_run_levels_ * sizeof(new_rl[0]));
      }
      Free(all_run_levels_);
      all_run_levels_ = new_rl;
      max_run_levels_ = new_size;
      assert(nb_run_levels_ + 6 * 64 <= max_run_levels_);
    }
  }
  return true;
}

bool Encoder::AllocateBlocks(size_t num_blocks) {
  assert(in_blocks_ == nullptr);
  have_coeffs_ = false;
  const size_t size = num_blocks * 64 * sizeof(*in_blocks_);
  in_blocks_base_ = Alloc<uint8_t>(size + ALIGN_CST);
  if (in_blocks_base_ == nullptr) return false;
  in_blocks_ = reinterpret_cast<int16_t*>(
      (ALIGN_CST + reinterpret_cast<uintptr_t>(in_blocks_base_)) & ~ALIGN_CST);
  return true;
}

void Encoder::DeallocateBlocks() {
  Free(in_blocks_base_);
  in_blocks_base_ = nullptr;
  in_blocks_ = nullptr;          // sanity
}

////////////////////////////////////////////////////////////////////////////////
// Perform YUV conversion and fDCT, and store the unquantized coeffs

void Encoder::CollectCoeffs() {
  assert(use_extra_memory_);
  int16_t* in = in_blocks_;
  const int mb_x_max = W_ / block_w_;
  const int mb_y_max = H_ / block_h_;
  for (int mb_y = 0; mb_y < mb_h_; ++mb_y) {
    const bool yclip = (mb_y == mb_y_max);
    for (int mb_x = 0; mb_x < mb_w_; ++mb_x) {
      GetSamples(mb_x, mb_y, yclip | (mb_x == mb_x_max), in);
      fDCT_(in, mcu_blocks_);
      in += 64 * mcu_blocks_;
    }
  }
  have_coeffs_ = true;
}

////////////////////////////////////////////////////////////////////////////////
// 1-pass Scan

void Encoder::SinglePassScan() {
  ResetDCs();

  RunLevel base_run_levels[64];
  int16_t* in = in_blocks_;
  const int mb_x_max = W_ / block_w_;
  const int mb_y_max = H_ / block_h_;
  const QuantizeBlockFunc quantize_block = use_trellis_ ? TrellisQuantizeBlock
                                                        : quantize_block_;
  const bool have_coeffs = have_coeffs_;
  for (int mb_y = 0; mb_y < mb_h_; ++mb_y) {
    const bool yclip = (mb_y == mb_y_max);
    for (int mb_x = 0; mb_x < mb_w_; ++mb_x) {
      if (!CheckBuffers()) return;
      if (!have_coeffs) {
        in = in_blocks_;
        GetSamples(mb_x, mb_y, yclip | (mb_x == mb_x_max), in);
        fDCT_(in, mcu_blocks_);
      }
      for (int c = 0; c < nb_comps_; ++c) {
        DCTCoeffs base_coeffs;
        for (int i = 0; i < nb_blocks_[c]; ++i) {
          const int dc = quantize_block(in, c, &quants_[quant_idx_[c]],
                                        &base_coeffs, base_run_levels);
          base_coeffs.dc_code_ = GenerateDCDiffCode(dc, &DCs_[c]);
          CodeBlock(&base_coeffs, base_run_levels);
          in += 64;
        }
      }
    }
  }
}

void Encoder::FinalPassScan(size_t nb_mbs, const DCTCoeffs* coeffs) {
  DeallocateBlocks();     // we can free up some coeffs memory at this point
  if (!CheckBuffers()) return;  // call needed to finalize all_run_levels_
  assert(reuse_run_levels_);
  const RunLevel* run_levels = all_run_levels_;
  for (size_t n = 0; n < nb_mbs; ++n) {
    if (!CheckBuffers()) return;
    CodeBlock(&coeffs[n], run_levels);
    run_levels += coeffs[n].nb_coeffs_;
  }
}

////////////////////////////////////////////////////////////////////////////////

void Encoder::SinglePassScanOptimized() {
  const size_t nb_mbs = mb_w_ * mb_h_ * mcu_blocks_;
  DCTCoeffs* const base_coeffs =
      Alloc<DCTCoeffs>(reuse_run_levels_ ? nb_mbs : 1);
  if (base_coeffs == nullptr) return;
  DCTCoeffs* coeffs = base_coeffs;
  RunLevel base_run_levels[64];
  const QuantizeBlockFunc quantize_block = use_trellis_ ? TrellisQuantizeBlock
                                                        : quantize_block_;

  // We use the default Huffman tables as basis for bit-rate evaluation
  if (use_trellis_) InitCodes(true);

  ResetEntropyStats();
  ResetDCs();
  nb_run_levels_ = 0;
  int16_t* in = in_blocks_;
  const int mb_x_max = W_ / block_w_;
  const int mb_y_max = H_ / block_h_;
  const bool have_coeffs = have_coeffs_;
  const bool reuse_run_levels = reuse_run_levels_;
  for (int mb_y = 0; mb_y < mb_h_; ++mb_y) {
    const bool yclip = (mb_y == mb_y_max);
    for (int mb_x = 0; mb_x < mb_w_; ++mb_x) {
      if (!have_coeffs) {
        in = in_blocks_;
        GetSamples(mb_x, mb_y, yclip | (mb_x == mb_x_max), in);
        fDCT_(in, mcu_blocks_);
      }
      if (!CheckBuffers()) goto End;
      for (int c = 0; c < nb_comps_; ++c) {
        for (int i = 0; i < nb_blocks_[c]; ++i) {
          RunLevel* const run_levels =
              reuse_run_levels ? all_run_levels_ + nb_run_levels_
                               : base_run_levels;
          const int dc = quantize_block(in, c, &quants_[quant_idx_[c]],
                                        coeffs, run_levels);
          coeffs->dc_code_ = GenerateDCDiffCode(dc, &DCs_[c]);
          AddEntropyStats(coeffs, run_levels);
          if (reuse_run_levels) {
            nb_run_levels_ += coeffs->nb_coeffs_;
            ++coeffs;
            assert(coeffs <= &base_coeffs[nb_mbs]);
          }
          in += 64;
          assert(nb_run_levels_ <= max_run_levels_);
        }
      }
    }
  }

  CompileEntropyStats();
  WriteDHT();
  WriteSOS();

  if (!reuse_run_levels_) {
    SinglePassScan();   // redo everything, but with optimal tables now.
  } else {
    // Re-use the saved run/levels for fast 2nd-pass.
    FinalPassScan(nb_mbs, base_coeffs);
  }
 End:
  Free(base_coeffs);
}

////////////////////////////////////////////////////////////////////////////////
// main call

bool Encoder::Encode() {
  if (!ok_) return false;

  FinalizeQuantMatrix(&quants_[0], q_bias_);
  FinalizeQuantMatrix(&quants_[1], q_bias_);
  SetCostCodes(0);
  SetCostCodes(1);

  SetDefaultHuffmanTables();

  // colorspace init
  InitComponents();
  assert(nb_comps_ <= MAX_COMP);
  assert(mcu_blocks_ <= 6);
  // validate some input parameters
  if (W_ <= 0 || H_ <= 0 || W_ > kMaxDimension || H_ > kMaxDimension) {
    return SetError();
  }

  mb_w_ = (W_ + (block_w_ - 1)) / block_w_;
  mb_h_ = (H_ + (block_h_ - 1)) / block_h_;
  const size_t nb_blocks = use_extra_memory_ ? mb_w_ * mb_h_ : 1;
  if (!AllocateBlocks(nb_blocks * mcu_blocks_)) return false;

  WriteAPP0();

  // custom markers written 'as is'
  if (!WriteAPPMarkers(app_markers_)) return false;

  // metadata
  if (!WriteEXIF(exif_) || !WriteICCP(iccp_) || !WriteXMP(xmp_)) return false;

  if (passes_ > 1) {
    LoopScan();
  } else {
    if (use_adaptive_quant_) {
      // Histogram analysis + derive optimal quant matrices
      CollectHistograms();
      AnalyseHisto();
    }

    WriteDQT();
    WriteSOF();

    if (optimize_size_) {
      SinglePassScanOptimized();
    } else {
      WriteDHT();
      WriteSOS();
      SinglePassScan();
    }
  }
  WriteEOI();
  ok_ = ok_ && bw_.Finalize();

  DeallocateBlocks();
  return ok_;
}

}    // namespace sjpeg

////////////////////////////////////////////////////////////////////////////////

bool SjpegCompress(const uint8_t* rgb, int width, int height,
                   float quality, std::string* output) {
  EncoderParam param;
  param.SetQuality(quality);
  return Encode(rgb, width, height, 3 * width, param, output);
}

bool SjpegDimensions(const std::string& jpeg_data,
                     int* width, int* height, int* is_yuv420) {
  return SjpegDimensions(
      reinterpret_cast<const uint8_t*>(jpeg_data.data()),
      jpeg_data.size(), width, height, is_yuv420);
}

int SjpegFindQuantizer(const std::string& jpeg_data,
                       uint8_t quant[2][64]) {
  return SjpegFindQuantizer(
      reinterpret_cast<const uint8_t*>(jpeg_data.data()), jpeg_data.size(),
      quant);
}

////////////////////////////////////////////////////////////////////////////////
