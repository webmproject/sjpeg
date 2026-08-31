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
//  Fast & simple JPEG encoder. Internal header.
//
// Author: Skal (pascal.massimino@gmail.com)

#ifndef SJPEG_JPEGI_H_
#define SJPEG_JPEGI_H_

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

// IWYU pragma: begin_exports
#include "sjpeg.h"
#include "bit_writer.h"
// IWYU pragma: end_exports

#define SJPEG_STRINGIFY_HELPER(x) #x
#define SJPEG_STRINGIFY(x) SJPEG_STRINGIFY_HELPER(x)

#if defined(__clang__)
#define SJPEG_UNROLL(n) _Pragma(SJPEG_STRINGIFY(clang loop unroll_count(n)))
#elif defined(__GNUC__) && (__GNUC__ >= 8)
#define SJPEG_UNROLL(n) _Pragma(SJPEG_STRINGIFY(GCC unroll n))
#else
#define SJPEG_UNROLL(n)
#endif

// Progressive (spectral-split-only) encoding; on by default. Pass
// -DSJPEG_NO_PROGRESSIVE to strip the feature's code out entirely.

#if defined(__SSE2__)
#define SJPEG_USE_SSE2
#endif

#if defined(__SSSE3__)
#define SJPEG_USE_SSSE3
#endif

#if defined(__AVX2__)
#define SJPEG_USE_AVX2
#endif

// Experimental: gather-based AVX2 riskiness scoring (src/riskiness_avx2.cc).
// Off by default -- gather throughput is poor on first-gen AVX2 hardware
// (Haswell/Broadwell), needs real measurement before it's on by default.
// #define SJPEG_USE_AVX2_RISKINESS

#if defined(__ARM_NEON__) || defined(__aarch64__)
#define SJPEG_USE_NEON
#endif

#if defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)
#define SJPEG_AARCH64
#endif

#if defined(SJPEG_NEED_ASM_HEADERS)
#if defined(SJPEG_USE_AVX2)
#include <immintrin.h>
#elif defined(SJPEG_USE_SSSE3)
#include <tmmintrin.h>
#elif defined(SJPEG_USE_SSE2)
#include <emmintrin.h>
#endif

#if defined(SJPEG_USE_NEON)
#include <arm_neon.h>
#endif
#endif    // SJPEG_NEED_ASM_HEADERS

#include <assert.h>

////////////////////////////////////////////////////////////////////////////////

namespace sjpeg {

extern bool SupportsSSE2();
extern bool SupportsNEON();
extern bool SupportsAVX2();

// Constants below are marker codes defined in JPEG spec
// ISO/IEC 10918-1 : 1993(E) Table B.1
// See also: http://www.w3.org/Graphics/JPEG/itu-t81.pdf

#define M_SOF0  0xffc0
#define M_SOF1  0xffc1
#define M_DHT   0xffc4
#define M_SOI   0xffd8
#define M_EOI   0xffd9
#define M_SOS   0xffda
#define M_DQT   0xffdb

// Maximum picture dimension: SOF stores the width and height on 16 bits.
enum { kMaxDimension = 0xffff };

// Forward 8x8 Fourier transforms, in-place.
typedef void (*FdctFunc)(int16_t *coeffs, int num_blocks);
FdctFunc GetFdct();

// these are the default luma/chroma matrices (JPEG spec section K.1)
extern const uint8_t kDefaultMatrices[2][64];
extern const uint8_t kZigzag[64];
extern const uint8_t kInvZigzag[64];

// scoring tables in score_7.cc
extern const int kRGBSize;
extern const uint8_t kSharpnessScore[];

// internal riskiness scoring functions:
extern double DCTRiskinessScore(const int16_t yuv[3 * 64],
                                int16_t scores[8 * 8]);
extern double BlockRiskinessScore(const uint8_t* rgb, int stride,
                                  int16_t scores[8 * 8]);
extern int YUVToRiskIdx(int16_t y, int16_t u, int16_t v);

///////////////////////////////////////////////////////////////////////////////
// RGB->YUV conversion

// Internal pixel format for the input samples fed to the block functions.
enum PixelFormat { kRGBInput = 0, kBGRAInput = 1, kRGBAInput = 2 };

// convert 16x16 RGB block into YUV420, or 8x8 RGB block into YUV444 or YUV400
typedef void (*RGBToYUVBlockFunc)(const uint8_t* src, int src_stride,
                                  int16_t* blocks);
extern RGBToYUVBlockFunc GetBlockFunc(SjpegYUVMode mode,
                                      PixelFormat fmt = kRGBInput);

// convert a row of RGB samples to YUV444
typedef void (*RGBToIndexRowFunc)(const uint8_t* src, int width,
                                  uint16_t* dst);
extern RGBToIndexRowFunc GetRowFunc();

// Enhanced slower RGB->YUV conversion:
//  y_plane[] has dimension W x H, whereas u_plane[] and v_plane[] have
//  dimension (W + 1)/2 x (H + 1)/2.
bool ApplySharpYUVConversion(const uint8_t* const rgb,
                             int W, int H, int stride,
                             uint8_t* y_plane,
                             uint8_t* u_plane,
                             uint8_t* v_plane);

///////////////////////////////////////////////////////////////////////////////
// Generic sample-replication function. Replicate sub_w x sub_h area of 'src'
// into 'dst', assuming the individual samples are 'x_step' bytes each.
void Replicate8b(const uint8_t* src, int src_stride,
                 uint8_t* dst, int dst_stride,
                 int sub_w, int sub_h, int w, int h, int x_step);

// This variant will replicate src[] into a 16b output dst[], subtracting 128.
// This function operates on a 8x8 block only.
void Convert8To16bClipped(const uint8_t* src, int src_step, int16_t dst[8 * 8],
                          int sub_w, int sub_h);

// Convert a 8x8 block of unsigned-8b values to signed-16b, subtracting 128.
void Convert8To16b(const uint8_t* src, int src_step, int16_t dst[8 * 8]);

///////////////////////////////////////////////////////////////////////////////
// some useful helper functions around quant matrices

extern float GetQFactor(float q);   // convert quality factor -> scale factor
extern void CopyQuantMatrix(const uint8_t in[64], uint8_t out[64]);
extern void SetQuantMatrix(const uint8_t in[64], float q_factor,
                           uint8_t out[64]);
extern void SetMinQuantMatrix(const uint8_t* const m, uint8_t out[64],
                              int tolerance);
extern void SetDefaultMinQuantMatrix(uint8_t out[64]);

////////////////////////////////////////////////////////////////////////////////
// Shared by files enc.cc was split into. Everything else in those files is
// either a member of Encoder (already declared below) or file-local.

// Encoder defaults. Also read by EncoderParam, hence not file-local.
extern const float kDefaultQuality;
extern const int kDefaultMethod;
extern const int32_t kDefaultBias;         // rounding bias for AC coefficients
extern const int kDefaultDeltaMaxLuma;     // for adaptive quantization
extern const int kDefaultDeltaMaxChroma;

// Manager used when the caller supplies none.
extern MemoryManager* GetDefaultMemoryManager();

// Fixed-point precisions: FP_BITS for the reciprocal dividers used in place of
// a division, AC_BITS for the extra precision left over by the fdct's scaling.
enum { FP_BITS = 16, AC_BITS = 4 };

#if defined(__has_builtin)
#define SJPEG_HAS_BUILTIN(x) __has_builtin(x)
#else
#define SJPEG_HAS_BUILTIN(x) 0
#endif

#if SJPEG_HAS_BUILTIN(__builtin_clz) || \
    (defined(__GNUC__) && \
     ((__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || __GNUC__ >= 4))
#define SJPEG_HAVE_CLZ
#endif

#if SJPEG_HAS_BUILTIN(__builtin_ctzll) || \
    (defined(__GNUC__) && \
     ((__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || __GNUC__ >= 4))
#define SJPEG_HAVE_CTZ
#endif

// Number of bits needed to code 'v' > 0, i.e. 1 + floor(log2(v)).
static inline int CalcLog2(int v) {
#if defined(SJPEG_HAVE_CLZ)
  return 32 - __builtin_clz(v);
#else
  const int kLog2[16] = {
    0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4 };
  assert(v > 0 && v < (1 << 12));
  return (v & ~0xff) ? 8 + kLog2[v >> 8] :
         (v & ~0x0f) ? 4 + kLog2[v >> 4] :
                       0 + kLog2[v];
#endif
}

// Count trailing zeros of a non-zero 64-bit value.
// 'x' must be non-zero.
static inline int TrailingZeros64(uint64_t x) {
#if defined(SJPEG_HAVE_CTZ)
  return __builtin_ctzll(x);
#else
  int c = 63;
  x &= ~x + 1;
  if (x & 0x00000000FFFFFFFF) c -= 32;
  if (x & 0x0000FFFF0000FFFF) c -= 16;
  if (x & 0x00FF00FF00FF00FF) c -= 8;
  if (x & 0x0F0F0F0F0F0F0F0F) c -= 4;
  if (x & 0x3333333333333333) c -= 2;
  if (x & 0x5555555555555555) c -= 1;
  return c;
#endif
}

////////////////////////////////////////////////////////////////////////////////
// main structs

// Huffman tables
struct HuffmanTable {
  uint8_t bits_[16];     // number of symbols per bit count
  const uint8_t* syms_;  // symbol map, in increasing bit length
  int nb_syms_;          // cached value of sum(bits_[])
};

// quantizer matrices
struct Quantizer {
  uint8_t quant_[64];      // direct quantizer matrix
  uint8_t min_quant_[64];  // min quantizer value allowed
  uint16_t iquant_[64];    // precalc'd reciprocal for divisor
  uint16_t qthresh_[64];   // minimal absolute value that produce non-zero coeff
  uint16_t bias_[64];      // bias, for coring (default / mid tier)
  // Alternate survival thresholds for adaptive-bias mode.
  // * qthresh_busy_ (> qthresh_): kills more near-zero AC coeffs
  // * qthresh_flat_ (< qthresh_): protects them
  // See Encoder::AdaptiveBiasQuantizeBlock.
  uint16_t qthresh_flat_[64];
  uint16_t qthresh_busy_[64];
  const uint32_t* codes_;  // codes for bit-cost calculation
};

// compact Run/Level storage, separate from DCTCoeffs infos
// Run/Level Information is not yet entropy-coded, but just stored
struct RunLevel {
  int16_t run_;
  uint16_t level_;     // 4bits for length, 12bits for mantissa
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
enum { HSHIFT = 2,                       // size of bins is (1 << HSHIFT)
       HHALF = 1 << (HSHIFT - 1),
       MAX_HISTO_DCT_COEFF = (1 << 7),   // max coefficient, descaled by HSHIFT
       HLAMBDA = 0x80,
       // Limits on range of alternate quantizers explored around
       // the initial value.  (see details in AnalyseHisto())
       QDELTA_MIN = -12, QDELTA_MAX = 12,
       QSIZE = QDELTA_MAX + 1 - QDELTA_MIN,
};

struct Histo {
  // Reserve one extra entry for counting all coeffs greater than
  // MAX_HISTO_DCT_COEFF. Result isn't used, but it makes the loop easier.
  int counts_[64][MAX_HISTO_DCT_COEFF + 1];
};

////////////////////////////////////////////////////////////////////////////////

struct Encoder {
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
  // luma_split/chroma_split: see EncoderParam::progressive_luma_split in
  // sjpeg.h.
  void SetProgressive(int luma_split, int chroma_split);

  // all-in-one init from EncoderParam.
  bool InitFromParam(const EncoderParam& param);

  // Main call. Return false in case of parameter error (setting empty output).
  bool Encode();

  // return MCU samples at macroblock position (mb_x, mb_y)
  // clipped is true if the MCU is clipped and needs replication
  virtual void GetSamples(int mb_x, int mb_y, bool clipped,
                          int16_t* out_blocks) = 0;

 private:
  // setters
  void SetQuantMatrices(const uint8_t m[2][64]);
  void SetMinQuantMatrices(const uint8_t m[2][64], int tolerance);
  void SetDefaultMinQuantMatrices();

  void SetQuantizationBias(int bias, bool use_adaptive);
  void SetQuantizationDeltas(int qdelta_luma, int qdelta_chroma);

  typedef enum { ICC, EXIF, XMP, MARKERS } MetadataType;
  void SetMetadata(const std::string& data, MetadataType type);

 private:
  bool CheckBuffers();  // returns false in case of memory alloc error
  bool ReserveSlab();   // shared by CheckBuffers()/CheckProgBuffers()

  void Put16b(uint32_t size);
  void Put32b(uint32_t size);

  void WriteAPP0();
  bool WriteAPPMarkers(const std::string& data);
  bool WriteEXIF(const std::string& data);
  bool WriteICCP(const std::string& data);
  bool WriteXMP(const std::string& data);
  bool WriteXMPExtended(const std::string& data);
  void WriteDQT();
  void WriteSOF(bool progressive = false);
  void WriteDHT();
  void WriteSOS();
  void WriteEOI();

  // General per-scan marker writers; WriteDHT()/WriteSOS() above use these
  // too, in addition to the progressive path (see SetProgressive()).
  // Writes one Huffman table as its own DHT segment (table_class: 0=DC, 1=AC).
  void WriteOneDHT(int table_class, int table_id, const HuffmanTable* table);
  struct ScanComponent {
    int comp_idx;      // 0, 1 or 2
    int dc_table_id;    // Huffman DC table selector for this component
    int ac_table_id;    // Huffman AC table selector for this component
  };
  // Writes one SOS for the given component list and spectral range.
  void WriteProgSOS(const ScanComponent* comps, int nb_comps, int Ss, int Se);

#if !defined(SJPEG_NO_PROGRESSIVE)
  void AddEntropyStatsDC(const DCTCoeffs* const coeffs);
  void CodeBlockDC(const DCTCoeffs* const coeffs);
  void CompileEntropyStatsDC();  // builds dc_codes_[] from freq_dc_[] directly

  // Restricts a block's RunLevel list to window [Ss,Se]; see entropy.cc.
  static int WindowRunLevels(const DCTCoeffs* const coeffs,
                             const RunLevel* const rl, int Ss, int Se,
                             RunLevel* const out, bool* const out_has_eob);

  void ResetEntropyStatsAC();  // resets prog_planes_->freq_ac only
  // WindowRunLevels()'s explicit entries only; has_eob (EOBn) is separate.
  void AddEntropyStatsACWindowed(const RunLevel* const windowed,
                                 int nb_windowed);
  void CodeBlockACWindowed(const RunLevel* const windowed, int nb_windowed);
  // EOBn run-length coding across empty blocks; see entropy.cc.
  void AddEntropyStatsEOBRun(int run);
  void CodeEOBRun(int run);
  void CompileEntropyStatsAC();  // builds prog_planes_->opt_table_ac/ac_codes

  bool EncodeProgressive();  // the whole progressive-mode encode path
#endif  // !SJPEG_NO_PROGRESSIVE

  void ResetDCs();

  // GetSamples() + fDCT_() for one MCU, into 'out'. Shared by the baseline
  // and progressive quantize loops.
  void TransformMCU(int mb_x, int mb_y, int16_t* const out);

  // TransformMCU() unless coefficients are already cached. Shared guard for
  // SinglePassScan(), SinglePassScanOptimized(), EncodeProgressive().
  void MaybeTransformMCU(int mb_x, int mb_y, int16_t** const in);

  // collect transformed coeffs (unquantized) only
  void CollectCoeffs();

  // points Huffman_tables_[] at the standard tables of JPEG section K.3
  void SetDefaultHuffmanTables();

  // 2-pass Huffman optimizing scan
  void ResetEntropyStats();
  void AddEntropyStats(const DCTCoeffs* const coeffs,
                       const RunLevel* const run_levels);
  void CompileEntropyStats();
  size_t EntropySize() const;  // size, in bits, derived from freq_ac_/freq_dc_

  void SinglePassScan();           // finalizing scan
  void SinglePassScanOptimized();  // optimize the Huffman table + finalize scan

  void SinglePassEncode();         // non-iterating encoding pass

  // quantize and compute run/levels from already stored coeffs
  void StoreRunLevels(DCTCoeffs* coeffs);
  // just write already stored run_levels & coeffs:
  void FinalPassScan(size_t nb_mbs, const DCTCoeffs* coeffs);

  // dichotomy loop
  void LoopScan();

  // Histogram pass
  void CollectHistograms();

  typedef int (*QuantizeBlockFunc)(const int16_t in[64], int idx,
                                   const Quantizer* const Q,
                                   DCTCoeffs* const out, RunLevel* const rl);
  static QuantizeBlockFunc quantize_block_;
  static QuantizeBlockFunc GetQuantizeBlockFunc();

  static int TrellisQuantizeBlock(const int16_t in[64], int idx,
                                  const Quantizer* const Q,
                                  DCTCoeffs* const out,
                                  RunLevel* const rl);

  static int AdaptiveBiasQuantizeBlock(const int16_t in[64], int idx,
                                       const Quantizer* const Q,
                                       DCTCoeffs* const out,
                                       RunLevel* const rl);

  // Picks quantize_block_ / TrellisQuantizeBlock / AdaptiveBiasQuantizeBlock.
  QuantizeBlockFunc GetActiveQuantizeBlockFunc() const;

  typedef uint32_t (*QuantizeErrorFunc)(const int16_t in[64],
                                        const Quantizer* const Q);
  static QuantizeErrorFunc quantize_error_;
  static QuantizeErrorFunc GetQuantizeErrorFunc();

  void CodeBlock(const DCTCoeffs* const coeffs, const RunLevel* const rl);
  // returns DC code (4bits for length, 12bits for suffix), updates DC_predictor
  static uint16_t GenerateDCDiffCode(int DC, int* const DC_predictor);

  static void FinalizeQuantMatrix(Quantizer* const q, int bias, bool adaptive);
  void SetCostCodes(int idx);
  void InitCodes(bool only_ac);

  size_t HeaderSize() const;
  void BlocksSize(int nb_mbs, const DCTCoeffs* coeffs,
                  const RunLevel* rl, sjpeg::BitCounter* const bc) const;
  float ComputeSize(const DCTCoeffs* coeffs);
  float ComputePSNR() const;

 protected:
  bool SetError();   // sets ok_ to false, and returns false

  // format-specific parameters, set by virtual InitComponents()
  const SjpegYUVMode yuv_mode_;   // 444, 420 or 400 only
  enum { MAX_COMP = 3 };
  int nb_comps_;
  int quant_idx_[MAX_COMP];       // indices for quantization matrices
  int nb_blocks_[MAX_COMP];       // number of 8x8 blocks per components
  uint8_t block_dims_[MAX_COMP];  // component dimensions (8-pixels units)
  int block_w_, block_h_;         // maximum mcu width / height
  int mcu_blocks_;                // total blocks in mcu (= sum of nb_blocks_[])

  void InitComponents();

  // data accessible to sub-classes implementing alternate input format
  int W_, H_;           // width, height
  int mb_w_, mb_h_;     // width / height in units of mcu
  int mb_x_max_, mb_y_max_;   // clip boundary: last full MCU column/row

  // Replicate an RGB source sub_w x sub_h block, expanding it to w x h size.
  const uint8_t* GetReplicatedSamples(const uint8_t* rgb,    // block source
                                      int rgb_step,          // stride in source
                                      int sub_w, int sub_h,  // sub-block size
                                      int w, int h);         // size of mcu
  // Replicate a 16x16 sub-block similarly.
  const uint8_t* GetReplicatedYSamples(const uint8_t* in, int step,
                                       int sub_w, int sub_h);
  // set blocks that are totally outside of the picture to an average value
  void AverageExtraLuma(int sub_w, int sub_h, int16_t* out);
  uint8_t replicated_buffer_[4 * 16 * 16];  // tmp buffer for replication
  int pix_step_ = 3;  // bytes per input pixel (3=RGB, 4=BGRA/RGBA)

  sjpeg::RGBToYUVBlockFunc get_yuv_block_;  // set by GetBlockFunc()
  bool adaptive_bias_;   // if true, use per-block perceptual bias modulation

  // Memory management
  template<class T> T* Alloc(size_t num) {
    assert(memory_hook_ != nullptr);
    T* const ptr = reinterpret_cast<T*>(memory_hook_->Alloc(sizeof(T) * num));
    if (ptr == nullptr) SetError();
    return ptr;
  }
  template<class T> void Free(T* const ptr) {
    memory_hook_->Free(reinterpret_cast<void*>(ptr));
  }
  // Free(*ptr) followed by '*ptr = nullptr', as a single safe step.
  template<class T> void FreePtr(T** const ptr) {
    Free(*ptr);
    *ptr = nullptr;
  }

 protected:
  bool ok_;                // set to false if a new[] fails

 private:
  sjpeg::BitWriter bw_;    // output buffer

  std::string iccp_, xmp_, exif_, app_markers_;   // metadata
  uint16_t xmp_split_;     // user-supplied split point for extended metadata

  // compression tools. See sjpeg.h for description of methods.
  bool optimize_size_;        // Huffman-optimize the codes  (method 0, 3)
  bool use_adaptive_quant_;   // modulate the quant matrix   (method 3-8)
  bool use_extra_memory_;     // save the unquantized coeffs (method 3, 4)
  bool reuse_run_levels_;     // save quantized run/levels   (method 1, 4, 5)
  bool use_trellis_;          // use trellis-quantization    (method 7, 8)

  int q_bias_;           // [0..255]: rounding bias for quant. of AC coeffs.
  Quantizer quants_[2];  // quant matrices
  int DCs_[3];           // DC predictors

  // DCT coefficients storage, aligned
  static constexpr size_t ALIGN_CST = 15;
  uint8_t* in_blocks_base_;   // base memory for blocks
  int16_t* in_blocks_;        // aligned pointer to in_blocks_base_
  bool have_coeffs_;          // true if the Fourier coefficients are stored
  bool AllocateBlocks(size_t num_blocks);  // returns false in case of error
  void DeallocateBlocks();

  // this is the extra memory for compression method 1
  RunLevel* all_run_levels_;
  size_t nb_run_levels_, max_run_levels_;

  // Huffman_tables_ indices:
  //  0: luma dc, 1: chroma dc, 2: luma ac, 3: chroma ac
  const HuffmanTable *Huffman_tables_[4];
  uint32_t ac_codes_[2][256];
  uint32_t dc_codes_[2][12];

  // histograms for dynamic codes. Could be temporaries.
  uint32_t freq_ac_[2][256 + 1];  // frequency distribution for AC coeffs
  uint32_t freq_dc_[2][12 + 1];   // frequency distribution for DC coeffs
  uint8_t opt_syms_ac_[2][256];   // optimal table for AC symbols
  uint8_t opt_syms_dc_[2][12];    // optimal table for DC symbols
  HuffmanTable opt_tables_ac_[2];
  HuffmanTable opt_tables_dc_[2];

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
  int prog_luma_split_, prog_chroma_split_;   // Se of the low band, or >=63

  // Per-component planes + scratch AC-Huffman state. Allocated by
  // AllocateProgPlanes(), freed by DeallocateProgPlanes(); nullptr otherwise
  // (always, if SJPEG_NO_PROGRESSIVE). Only forward-declared here so this
  // pointer can stay unconditional; see the full definition below.
  struct ProgPlanes;
  ProgPlanes* prog_planes_;

#if !defined(SJPEG_NO_PROGRESSIVE)
  struct ProgPlanes {
    // Non-MCU-interleaved storage (non-interleaved AC scans need true raster
    // order). Fixed 63-entry RunLevel slot per block.
    DCTCoeffs* coeffs[MAX_COMP];
    RunLevel* run_levels[MAX_COMP];
    int plane_w[MAX_COMP], plane_h[MAX_COMP];  // MCU-padded stride
    // True (non-MCU-padded) per-component block dims per spec A.2.4; what
    // non-interleaved AC scans must actually visit.
    int true_w[MAX_COMP], true_h[MAX_COMP];
    // Scratch AC Huffman state, rebuilt per progressive scan (unlike
    // freq_ac_/ac_codes_, which hold 2 permanent baseline tables).
    uint32_t freq_ac[256 + 1];
    uint32_t ac_codes[256];
    uint8_t opt_syms_ac[256];
    HuffmanTable opt_table_ac;
  };
  bool AllocateProgPlanes();
  void DeallocateProgPlanes();
  // index of sub-block 'i' of MCU (mb_x,mb_y) in component c's plane
  int ProgPlaneIndex(int c, int mb_x, int mb_y, int i) const;
  bool CheckProgBuffers();  // like CheckBuffers(), but for the bw_ slab only
  void EncodeProgAC(int c, int split);  // all AC scans for one component
#endif  // !SJPEG_NO_PROGRESSIVE

  // multi-pass parameters
  int passes_;
  SearchHook default_hook_;
  SearchHook* search_hook_;

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
                               MemoryManager* memory = nullptr);

// Same, for a single-channel (4:0:0) input.
extern Encoder* GrayEncoderFactory(const uint8_t* gray, int W, int H,
                                   int stride, ByteSink* sink,
                                   MemoryManager* memory = nullptr);

// Encodes with 'enc' and destroys it, tolerating a null 'enc'.
extern bool FinishEncoding(Encoder* enc, const EncoderParam& param);

////////////////////////////////////////////////////////////////////////////////

}   // namespace sjpeg

#endif    // SJPEG_JPEGI_H_
