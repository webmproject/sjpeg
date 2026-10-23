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
//  Fast & simple JPEG encoder: the Encoder class. Internal header.
//
// Author: Skal (pascal.massimino@gmail.com)

#ifndef SJPEG_JPEGI_H_
#define SJPEG_JPEGI_H_

#include <vector>

#if !defined(SJPEG_NO_MULTITHREADING)
#include <functional>
#include <memory>
#endif

// IWYU pragma: begin_exports
#include "bit_writer.h"
#include "blocks.h"
#include "common.h"
#include "dsp.h"
#include "entropy.h"
#include "headers.h"
#include "sjpeg.h"
#include "slices.h"
// IWYU pragma: end_exports

namespace sjpeg {

////////////////////////////////////////////////////////////////////////////////

class Encoder {
 public:
  // 'memory' can be null (default manager). It is passed at construction,
  // and not later, because sub-classes can allocate in their constructor.
  Encoder(SjpegYUVMode yuv_mode, int W, int H, ByteSink* sink,
          MemoryManager* memory);
  virtual ~Encoder();
  bool Ok() const { return ok_; }

  // setters
  void SetQuality(float q);
  void SetCompressionMethod(int method);
  void SetRDO(bool use_rdo = true);
  void SetTrellis(bool use_trellis = true);  // after SetCompressionMethod()
  // luma_split/chroma_split: see EncoderParam::progressive_luma_split in
  // sjpeg.h.
  void SetProgressive(int luma_split, int chroma_split);

  // all-in-one init from EncoderParam.
  bool InitFromParam(const EncoderParam& param);

  // Main call. Return false in case of parameter error (setting empty output).
  bool Encode();

  // return MCU samples at macroblock position (mb_x, mb_y)
  // clipped is true if the MCU is clipped and needs replication
  // rep_buf is 4 * 16 * 16 bytes of caller-owned scratch for replicating
  // clipped MCUs. It must not be shared between threads; the parallel scans
  // pass per-worker storage, everything else passes replicated_buffer_.
  virtual void GetSamples(int mb_x, int mb_y, bool clipped, int16_t* out_blocks,
                          uint8_t* rep_buf) = 0;

  int num_threads() const { return num_threads_; }
  void SetNumThreads(int num_threads) { num_threads_ = num_threads; }

#if !defined(SJPEG_NO_MULTITHREADING)
  static int HardwareConcurrency();
  void RunParallel(int num_threads, int total, FuncRef<int, int, int> fn) const;
#endif

 private:
  // setters
  void SetQuantMatrices(const uint8_t m[2][64]);
  void SetMinQuantMatrices(const uint8_t m[2][64], int tolerance);
  void SetDefaultMinQuantMatrices();

  void SetQuantizationBias(int bias, bool use_adaptive);
  void SetQuantizationDeltas(int qdelta_luma, int qdelta_chroma);

  typedef enum { ICC, EXIF, XMP, MARKERS } MetadataType;
  // '*data' is not copied: it must remain valid until Encode() returns.
  void SetMetadata(const std::string* data, MetadataType type);

 private:
  static constexpr size_t kMaxMCUSize = 2560;

  bool ReserveSlab();
  int TotalRestartIntervals() const {
    return (restart_interval_rows_ > 0)
               ? (mb_h_ + restart_interval_rows_ - 1) / restart_interval_rows_
               : 1;
  }
  size_t SliceSlabSize(int first_interval, int end_interval) const;

  // Slice count for a parallel pass over [0, cap): 1 if serial, else up to
  // num_threads_ bounded by 'cap' and a worthwhileness check on MCU count
  // (grain <= 0: linear kMinMCUsPerThread model; else ScaledThreadLimit()).
  int GetNumSlices(int cap, int grain = 64) const;

  // Marker writers: thin wrappers around hw_ that gather the frame state.
  void WriteDQT();
  void WriteSOF(bool progressive = false);
  void WriteDHT();
  void WriteDRI();
  void WriteSOS();

  void ResetDCs();
  bool EmitRestartMarker(sjpeg::BitWriter* bw, int interval_idx,
                         int total_intervals, size_t slab_size);

  // GetSamples() + fDCT_() for one MCU, into 'out'. Shared by the baseline
  // and progressive quantize loops.
  void TransformMCU(int mb_x, int mb_y, int16_t* out, uint8_t* rep_buf);
  void TransformMCU(int mb_x, int mb_y, int16_t* out);

  // TransformMCU() unless coefficients are already cached. Shared guard for
  // EncodeProgressive().
  void MaybeTransformMCU(int mb_x, int mb_y, int16_t** in);

  // The MCU's unquantized coefficients: the ones CollectCoeffs() cached if it
  // ran, otherwise computed into 'scratch'.
  const int16_t* GetMCUCoeffs(int mb_x, int mb_y, int16_t* scratch,
                              uint8_t* rep_buf);

  // Shared scan slice kernels used by both single-threaded and multi-threaded
  // scan passes.
  bool CodeScanSlice(int first_interval, int end_interval, int total_intervals,
                     sjpeg::BitWriter* bw, size_t slab_size, int16_t* scratch,
                     uint8_t* rep_buf);
  // 'store' (null if the run/levels aren't kept) and 'stats' (null if not
  // collected) receive the quantized blocks.
  bool QuantizeScanSlice(int first_interval, int end_interval,
                         SliceStore* store, EntropyStats* stats,
                         int16_t* scratch, uint8_t* rep_buf);
  bool ReplayScanSlice(int first_interval, int end_interval,
                       int total_intervals, size_t nb_blocks,
                       const DCTCoeffs* coeffs, const RunLevel* rl,
                       sjpeg::BitWriter* bw, size_t slab_size);

  // collect transformed coeffs (unquantized) only
  void CollectCoeffs();
  void CollectCoeffsSlice(int y_start, int y_end, uint8_t* rep_buf);

  // Seeds the AC codes the trellis / RDO quantizers read their costs from.
  void InitCostCodes() {
    if (use_trellis_ || use_rdo_) entropy_.InitCodes(/*only_ac=*/true);
  }

  void SinglePassScan();           // finalizing scan
  void SinglePassScanOptimized();  // optimize the Huffman table + finalize scan

  void SinglePassEncode();  // non-iterating encoding pass
  // WriteSOF()/WriteDRI() + the optimize_size_ scan dispatch. Shared by
  // SinglePassEncode()'s baseline path and LoopScan()'s PSNR-target finalize
  // step.
  void WriteBaselineAndScan();

#if !defined(SJPEG_NO_MULTITHREADING)
  static constexpr int kMinMCUsPerThread = 256;

  class ThreadPool;
  struct ThreadPoolDeleter {
    void operator()(ThreadPool* p) const;
  };
  mutable std::unique_ptr<ThreadPool, ThreadPoolDeleter> thread_pool_;

  // Returns the optimal thread count when serial post-processing overhead grows
  // linearly with thread count T, requiring O(T * grain) MCUs per thread.
  static int ScaledThreadLimit(int num_mcus, int grain = 64) {
    int threads = 1;
    while (threads < 16 && threads * (threads + 1) * grain <= num_mcus) {
      ++threads;
    }
    return threads;
  }

  // Parallel equivalents of the serial passes, emitting the same results.
  void CollectHistogramsMultiThreaded(int num_threads);
  void CollectCoeffsMultiThreaded(int num_threads);
  float ComputePSNRMultiThreaded(int num_threads) const;
  void SinglePassScanMultiThreaded(int num_threads, int total_intervals);
  // Appends the slices to bw_ in order, freeing each one as it goes.
  void ConcatenateChunks(ThreadChunk* chunks, int num_chunks);
#endif  // !SJPEG_NO_MULTITHREADING

  // Slice-based passes: a slice is a range of restart intervals. They run on
  // 'num_slices' threads (a single slice when serial), and emit the same
  // bitstream, byte for byte.
  // Runs fn(slice_idx, first, end) over a partition of [0, total).
  template <class F>
  void ForEachSlice(int num_slices, int total, const F& fn) const {
#if !defined(SJPEG_NO_MULTITHREADING)
    RunParallel(num_slices, total, fn);
#else
    (void)num_slices;
    fn(0, 0, total);
#endif
  }
  // Quantizes the slices into (*chunks)[], gathering the entropy statistics
  // into stats_ if optimizing the Huffman tables.
  void QuantizeSlices(int num_slices, int total_intervals, ChunkSet* chunks);
  // Codes the stored run/levels of the slices, in order, into bw_.
  void ReplaySlices(int num_slices, int total_intervals, ChunkSet* chunks);
  // Size, in bytes, of the stream coded from the quantized slices.
  float EvaluateSize(int num_slices, int total_intervals, ChunkSet* chunks);
  // Sums each slice's symbol frequencies into stats_.
  void MergeChunkStats(const ThreadChunk* chunks, int num_chunks);
  // dichotomy loop
  void LoopScan();

  // Histogram pass
  void CollectHistograms();
  void CollectHistogramsSlice(int y_start, int y_end, Histo histos[2],
                              int16_t* scratch, uint8_t* rep_buf);

  typedef int (*QuantizeBlockFunc)(const int16_t in[64], int idx,
                                   const Quantizer* const Q,
                                   DCTCoeffs* const out, RunLevel* const rl);
  static QuantizeBlockFunc quantize_block_;
  static QuantizeBlockFunc GetQuantizeBlockFunc();

  static int TrellisQuantizeBlock(const int16_t in[64], int idx,
                                  const Quantizer* Q, DCTCoeffs* out,
                                  RunLevel* rl);

  static int AdaptiveBiasQuantizeBlock(const int16_t in[64], int idx,
                                       const Quantizer* Q, DCTCoeffs* out,
                                       RunLevel* rl);

  static int RDOQuantizeBlock(const int16_t in[64], int idx, const Quantizer* Q,
                              DCTCoeffs* out, RunLevel* rl);

  static int RDOAdaptiveBiasQuantizeBlock(const int16_t in[64], int idx,
                                          const Quantizer* Q, DCTCoeffs* out,
                                          RunLevel* rl);

  // Picks quantize_block_ / TrellisQuantizeBlock / RDOQuantizeBlock /
  // RDOAdaptiveBiasQuantizeBlock / AdaptiveBiasQuantizeBlock.
  QuantizeBlockFunc GetActiveQuantizeBlockFunc() const;

  typedef uint32_t (*QuantizeErrorFunc)(const int16_t in[64],
                                        const Quantizer* const Q);
  static QuantizeErrorFunc quantize_error_;
  static QuantizeErrorFunc GetQuantizeErrorFunc();
  friend QuantizeErrorTestFunc GetQuantizeErrorFuncForTest();

  void CodeBlock(const DCTCoeffs* coeffs, const RunLevel* rl,
                 sjpeg::BitWriter* bw) const {
    entropy_.CodeBlock(coeffs, rl, quant_idx_[coeffs->idx_], bw);
  }
  // returns DC code (4bits for length, 12bits for suffix), updates DC_predictor
  static uint16_t GenerateDCDiffCode(int DC, int* DC_predictor);

  static void FinalizeQuantMatrix(Quantizer* q, int bias,
                                  bool adaptive = false);
  void SetCostCodes(int idx);

  size_t HeaderSize() const;
  void BlocksSize(int nb_mbs, const DCTCoeffs* coeffs, const RunLevel* rl,
                  sjpeg::BitCounter* bc) const;
  float ComputeSize(size_t entropy_bits) const;
  uint64_t ComputePSNRSlice(int y_start, int y_end) const;
  float ComputePSNR() const;
  static float GetPSNR(uint64_t err, uint64_t size);

 protected:
  bool SetError() const;  // sets ok_ to false, and returns false

  // format-specific parameters, set at construction by InitComponents()
  const SjpegYUVMode yuv_mode_;  // 444, 420 or 400 only
  enum { MAX_COMP = 3 };
  int nb_comps_ = 0;
  int quant_idx_[MAX_COMP];        // indices for quantization matrices
  int nb_blocks_[MAX_COMP];        // number of 8x8 blocks per components
  uint8_t block_dims_[MAX_COMP];   // component dimensions (8-pixels units)
  int block_w_ = 0, block_h_ = 0;  // maximum mcu width / height
  int mcu_blocks_ = 0;  // total blocks in mcu (= sum of nb_blocks_[])

  void InitComponents();

  // data accessible to sub-classes implementing alternate input format
  const int W_, H_;                  // width, height
  int mb_w_ = 0, mb_h_ = 0;          // width / height in units of mcu
  int mb_x_max_ = 0, mb_y_max_ = 0;  // clip boundary: last full MCU col/row

  // Replicate an RGB source sub_w x sub_h block, expanding it to w x h size.
  const uint8_t* GetReplicatedSamples(const uint8_t* rgb,    // block source
                                      int rgb_step,          // stride in source
                                      int sub_w, int sub_h,  // sub-block size
                                      int w, int h,          // size of mcu
                                      uint8_t* rep_buf);
  // Replicate a 16x16 sub-block similarly.
  const uint8_t* GetReplicatedYSamples(const uint8_t* in, int step, int sub_w,
                                       int sub_h, uint8_t* rep_buf);
  // set blocks that are totally outside of the picture to an average value
  void AverageExtraLuma(int sub_w, int sub_h, int16_t* out);
  // Replicate (if clipped) & convert a 16x16 luma block at MCU (mb_x,mb_y)
  // into out[0..3]. Shared by the NV12/NV21 and direct-YUV420 encoders.
  void GetLumaBlock16x16(const uint8_t* y, int y_step, int mb_x, int mb_y,
                         bool clipped, int16_t* out, uint8_t* rep_buf);
  uint8_t replicated_buffer_[4 * 16 * 16];  // tmp buffer for replication
  int pix_step_ = 3;  // bytes per input pixel (3=RGB, 4=BGRA/RGBA)

  sjpeg::RGBToYUVBlockFunc get_yuv_block_ = nullptr;  // set by GetBlockFunc()
  // if true, use per-block perceptual bias modulation
  bool adaptive_bias_ = false;

  // Memory management
  template <class T>
  T* Alloc(size_t num) {
    assert(memory_hook_ != nullptr);
    size_t bytes = 0;
    if (!SafeMultiply(num, sizeof(T), &bytes)) {
      SetError();
      return nullptr;
    }
    T* const ptr = reinterpret_cast<T*>(memory_hook_->Alloc(bytes));
    if (ptr == nullptr) SetError();
    return ptr;
  }
  template <class T>
  void Free(T* const ptr) {
    memory_hook_->Free(reinterpret_cast<void*>(ptr));
  }
  // Free(*ptr) followed by '*ptr = nullptr', as a single safe step.
  template <class T>
  void FreePtr(T** const ptr) {
    Free(*ptr);
    *ptr = nullptr;
  }

 protected:
  mutable bool ok_ = true;  // set to false if a new[] fails

 private:
  sjpeg::BitWriter bw_;  // output buffer
  HeaderWriter hw_;      // marker segments, written to bw_

  // Metadata: the caller's strings (see SetMetadata()), or no_metadata_.
  const std::string no_metadata_;
  const std::string* iccp_ = &no_metadata_;
  const std::string* xmp_ = &no_metadata_;
  const std::string* exif_ = &no_metadata_;
  const std::string* app_markers_ = &no_metadata_;
  uint16_t xmp_split_ = 0;  // user-supplied split point for extended metadata

  // compression tools, and the methods using them. See sjpeg.h for
  // description of methods.
  bool optimize_size_ = false;       // Huffman-optimize the codes  (0, 3)
  bool use_adaptive_quant_ = false;  // modulate the quant matrix   (3-8)
  bool use_extra_memory_ = false;    // save the unquantized coeffs (3, 4)
  bool reuse_run_levels_ = false;    // save quantized run/levels   (1, 4, 5)
  bool use_trellis_ = false;         // use trellis-quantization    (7, 8)
  bool use_rdo_ = false;             // use fast rate-distortion optimization
  int restart_interval_rows_ = 0;    // MCU rows per restart interval (0 = off)
  int num_threads_ = 1;              // total threads for parallel scan
                                     // (1 = single-threaded)

  int q_bias_ = 0;       // [0..255]: rounding bias for quant. of AC coeffs.
  Quantizer quants_[2];  // quant matrices
  int DCs_[3];           // DC predictors

  // DCT coefficients storage, aligned to 32 bytes
  static constexpr size_t ALIGN_CST = 31;
  uint8_t* in_blocks_base_ = nullptr;  // base memory for blocks
  int16_t* in_blocks_ = nullptr;       // aligned pointer to in_blocks_base_
  bool have_coeffs_ = false;  // true if the Fourier coefficients are stored
  bool AllocateBlocks(size_t num_blocks);  // returns false in case of error
  void DeallocateBlocks();

  EntropyCoder entropy_;  // Huffman tables and codes
  // histograms for dynamic codes. Could be temporaries.
  EntropyStats stats_;

  // Limits on how much we will decrease the bitrate in the luminance
  // and chrominance channels (respectively).
  int qdelta_max_luma_;
  int qdelta_max_chroma_;

  // Histogram handling

  // This function aggregates each 63 unquantized AC coefficients into an
  // histogram for further analysis.
  typedef void (*StoreHistoFunc)(const int16_t in[64], Histo* const histos,
                                 int nb_blocks);
  static StoreHistoFunc store_histo_;
  static StoreHistoFunc GetStoreHistoFunc();  // select between the above.

  // Provided the AC histograms have been stored with StoreHisto(), this
  // function will analyze impact of varying the quantization scales around
  // initial values, trading distortion for bit-rate in a controlled way.
  void AnalyseHisto();
  void ResetHisto();  // initialize histos_[]
  Histo histos_[2];

  // --- progressive (spectral-split-only) mode. See SetProgressive(). ---
  // prog_luma_split_ == 64 means progressive mode is off.
  int prog_luma_split_ = 64;  // Se of the low band, or >=63
  int prog_chroma_split_ = 8;

  // Per-component planes + scratch AC-Huffman state. Allocated by
  // AllocateProgPlanes(), freed by DeallocateProgPlanes(); nullptr otherwise
  // (always, if SJPEG_NO_PROGRESSIVE). Only forward-declared here so this
  // pointer can stay unconditional; see the full definition below.
  struct ProgPlanes;
  ProgPlanes* prog_planes_ = nullptr;

#if !defined(SJPEG_NO_PROGRESSIVE)
  bool EncodeProgressive();  // the whole progressive-mode encode path
  bool EncodeProgScans();    // ... once the planes are allocated
  struct ProgPlanes {
    // Non-MCU-interleaved storage (non-interleaved AC scans need true raster
    // order). The run/levels are packed back to back in quantization (MCU)
    // order, and grown on demand: rl_offset[] locates each block's ones.
    DCTCoeffs* coeffs[MAX_COMP];
    RunLevel* run_levels[MAX_COMP];
    // One per block, index in run_levels[]. 32 bits are enough: there are at
    // most 8192 x 8192 blocks of 63 run/levels. Stored right after coeffs[],
    // in the same allocation.
    uint32_t* rl_offset[MAX_COMP];
    size_t rl_used[MAX_COMP], rl_capacity[MAX_COMP];
    int plane_w[MAX_COMP], plane_h[MAX_COMP];  // MCU-padded stride
    // True (non-MCU-padded) per-component block dims per spec A.2.4; what
    // non-interleaved AC scans must actually visit.
    int true_w[MAX_COMP], true_h[MAX_COMP];
    // Scratch AC Huffman state, rebuilt per progressive scan (unlike
    // entropy_, which holds 2 permanent baseline tables).
    ProgACCoder ac;
  };
  bool AllocateProgPlanes();
  void DeallocateProgPlanes();
  bool GrowProgRunLevels(int c);  // returns false in case of error
  // index of sub-block 'i' of MCU (mb_x,mb_y) in component c's plane
  int ProgPlaneIndex(int c, int mb_x, int mb_y, int i) const;
  bool CheckProgBuffers() { return ReserveSlab(); }
  void EncodeProgAC(int c, int split);  // all AC scans for one component
  template <class BlockFn, class EOBFn, class StartFn>
  bool WalkACBand(int c, int Ss, int Se, BlockFn on_block, EOBFn on_eob,
                  StartFn on_start) const;
#endif  // !SJPEG_NO_PROGRESSIVE

  // multi-pass parameters
  int passes_ = 1;
  SearchHook default_hook_;
  SearchHook* search_hook_ = nullptr;

  // lower memory management
  MemoryManager* memory_hook_;

  static const float kHistoWeight[QSIZE];

  static void (*fDCT_)(int16_t* in, int num_blocks);
  static void InitializeStaticPointers();
};

////////////////////////////////////////////////////////////////////////////////
// Defined in encoders.cc, used by the public entry points in api.cc.

// Returns the Encoder sub-class matching 'yuv_mode' (resolving SJPEG_YUV_AUTO
// along the way), or null if it could not be constructed.
extern Encoder* EncoderFactory(const uint8_t* rgb, int W, int H, int stride,
                               SjpegYUVMode yuv_mode, ByteSink* sink,
                               PixelFormat fmt = kRGBInput,
                               MemoryManager* memory = nullptr,
                               int num_threads = 1);

// Same, for a single-channel (4:0:0) input.
extern Encoder* GrayEncoderFactory(const uint8_t* gray, int W, int H,
                                   int stride, ByteSink* sink,
                                   MemoryManager* memory = nullptr);

// Encodes with 'enc' and destroys it, tolerating a null 'enc'.
extern bool FinishEncoding(Encoder* enc, const EncoderParam* param);

////////////////////////////////////////////////////////////////////////////////

}  // namespace sjpeg

#endif  // SJPEG_JPEGI_H_
