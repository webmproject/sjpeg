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
//  Fast and simple one-file JPEG encoder
//
// Author: Skal (pascal.massimino@gmail.com)

#include <stdlib.h>
#include <math.h>
#include <float.h>    // for FLT_MAX
#include <stdint.h>

#define SJPEG_NEED_ASM_HEADERS
#include "sjpegi.h"
#include "bit_writer.h"

// Some general default values:
static const int kDefaultQuality = 75;
static const int kDefaultMethod = 4;
// Rounding bias for AC coefficients, as 8bit fixed point.
// A default value 0x78 leans toward filesize reduction.
static const int32_t kDefaultBias = 0x78;
// for adaptive quantization:
static const int kDefaultDeltaMaxLuma = 12;
static const int kDefaultDeltaMaxChroma = 1;

// finer tuning of perceptual optimizations:

// Minimum average number of entries per bin required for performing histogram-
// -based optimization. Below this limit, the channel's histogram is declared
// under-populated and the corresponding optimization skipped.
static double kDensityThreshold = 0.5;
// Rejection limit on the correlation factor when extrapolating the distortion
// from histograms. If the least-square fit has a squared correlation factor
// less than this threshold, the corresponding quantization scale will be
// kept unchanged.
static double kCorrelationThreshold = 0.5;
// Bit-map of channels to omit during quantization matrix optimization.
// If the bit 'i + 8 * j' is set in this bit field, the matrix entry at
// position (i,j) will be kept unchanged during optimization.
// The default value is 0x103 = 1 + 2 + 256: the 3 entries in the top-left
// corner (with lowest-frequency) are not optimized, since it can lead to
// visual degradation of smooth gradients.
static const uint64_t kOmittedChannels = 0x0000000000000103ULL;

////////////////////////////////////////////////////////////////////////////////
namespace sjpeg {

const uint8_t kZigzag[64] = {
  0,   1,  8, 16,  9,  2,  3, 10,
  17, 24, 32, 25, 18, 11,  4,  5,
  12, 19, 26, 33, 40, 48, 41, 34,
  27, 20, 13,  6,  7, 14, 21, 28,
  35, 42, 49, 56, 57, 50, 43, 36,
  29, 22, 15, 23, 30, 37, 44, 51,
  58, 59, 52, 45, 38, 31, 39, 46,
  53, 60, 61, 54, 47, 55, 62, 63,
};

const uint8_t kDefaultMatrices[2][64] = {
  // these are the default luma/chroma matrices (JPEG spec section K.1)
  { 16,  11,  10,  16,  24,  40,  51,  61,
    12,  12,  14,  19,  26,  58,  60,  55,
    14,  13,  16,  24,  40,  57,  69,  56,
    14,  17,  22,  29,  51,  87,  80,  62,
    18,  22,  37,  56,  68, 109, 103,  77,
    24,  35,  55,  64,  81, 104, 113,  92,
    49,  64,  78,  87, 103, 121, 120, 101,
    72,  92,  95,  98, 112, 100, 103,  99 },
  { 17,  18,  24,  47,  99,  99,  99,  99,
    18,  21,  26,  66,  99,  99,  99,  99,
    24,  26,  56,  99,  99,  99,  99,  99,
    47,  66,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99 }
};

int GetQFactor(int q) {
  // we use the same mapping than jpeg-6b, for coherency
  return (q <= 0) ? 5000 : (q < 50) ? 5000 / q : (q < 100) ? 2 * (100 - q) : 0;
}

void SetQuantMatrix(const uint8_t in[64], int q_factor, uint8_t out[64]) {
  if (in == NULL || out == NULL) return;
  for (int i = 0; i < 64; ++i) {
    const int v = (in[i] * q_factor + 50) / 100;
    // clamp to prevent illegal quantizer values
    out[i] = (v < 1) ? 1 : (v > 255) ? 255u : v;
  }
}

void SetMinQuantMatrix(const uint8_t* const m, uint8_t out[64],
                       int tolerance) {
  assert(out != NULL);
  if (m != NULL) {
    for (int i = 0; i < 64; ++i) {
      const int v = (int)m[i] * (256 - tolerance) >> 8;
      out[i] = (v < 1) ? 1u : (v > 255) ? 255u : v;
    }
  } else {
    for (int i = 0; i < 64; ++i) {
      out[i] = 1u;
    }
  }
}

}    // namespace sjpeg

////////////////////////////////////////////////////////////////////////////////
// CPU support

namespace sjpeg {
bool SupportsSSE2() {
#if defined(SJPEG_USE_SSE2)
  return true;
#endif
  return false;
}

bool SupportsNEON() {
#if defined(SJPEG_USE_NEON)
  return true;
#endif
  return false;
}

}    // namespace sjpeg

namespace {
////////////////////////////////////////////////////////////////////////////////
// helper structures

// Huffman tables
struct HuffmanTable {
  uint8_t bits_[16];     // number of symbols per bit count
  const uint8_t* syms_;  // symbol map, in increasing bit length
  uint8_t nb_syms_;      // cached value of sum(bits_[])
};

// quantizer matrices
struct Quantizer {
  uint8_t quant_[64];      // direct quantizer matrice
  uint8_t min_quant_[64];  // min quantizer value allowed
  uint16_t iquant_[64];    // precalc'd reciprocal for divisor
  uint16_t qthresh_[64];   // minimal absolute value that produce non-zero coeff
  uint16_t bias_[64];      // bias, for coring
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
  int8_t bias_;        // perceptual bias
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
// main struct

class SjpegEncoder {
 public:
  SjpegEncoder(int W, int H, int step, const uint8_t* rgb);
  virtual ~SjpegEncoder();

  // setters
  void SetQuality(int q) {
    q = sjpeg::GetQFactor(q);
    sjpeg::SetQuantMatrix(sjpeg::kDefaultMatrices[0], q, quants_[0].quant_);
    sjpeg::SetQuantMatrix(sjpeg::kDefaultMatrices[1], q, quants_[1].quant_);
  }
  void SetQuantMatrices(const uint8_t m[2][64]) {
    sjpeg::SetQuantMatrix(m[0], 100, quants_[0].quant_);
    sjpeg::SetQuantMatrix(m[1], 100, quants_[1].quant_);
  }
  void SetMinQuantMatrices(const uint8_t* const m[2], int tolerance) {
    sjpeg::SetMinQuantMatrix(m[0], quants_[0].min_quant_, tolerance);
    sjpeg::SetMinQuantMatrix(m[1], quants_[1].min_quant_, tolerance);
  }

  void SetCompressionMethod(int method) {
    DCHECK(method >= 0 && method <= 8);
    use_adaptive_quant_ = (method >= 3);
    optimize_size_ = (method != 0) && (method != 3);
    use_extra_memory_ = (method == 3) || (method == 4) || (method == 7);
    reuse_run_levels_ = (method == 1) || (method == 4) || (method == 5)
                     || (method == 7) || (method == 8);
    use_trellis_ = (method >= 6);
  }

  typedef enum { ICC, EXIF, XMP, MARKERS } MetadataType;
  void SetMetadata(const std::string& data, MetadataType type) {
    switch (type) {
      case ICC: iccp_ = data; break;
      case EXIF: exif_ = data; break;
      case XMP: xmp_ = data; break;
      default:
      case MARKERS: app_markers_ = data; break;
    }
  }

  void SetQuantizationBias(int bias, bool use_adaptive) {
    DCHECK(bias >= 0 && bias <= 255);
    q_bias_ = bias;
    adaptive_bias_ = use_adaptive;
  }

  void SetQuantizationDeltas(int qdelta_luma, int qdelta_chroma) {
    DCHECK(qdelta_luma >= 0 && qdelta_luma <= 255);
    DCHECK(qdelta_chroma >= 0 && qdelta_chroma <= 255);
    qdelta_max_luma_ = qdelta_luma;
    qdelta_max_chroma_ = qdelta_chroma;
  }

  // getters
  int Size() const { return bw_.BytePos(); }
  uint8_t* Bits() const { return const_cast<uint8_t*>(bw_.Data()); }
  uint8_t* Grab(size_t *size) { return bw_.Grab(size); }

  // Main call. Return false in case of parameter error (setting empty output).
  bool Encode();

  // these are colorspace-dependant.
  virtual void InitComponents() = 0;
  // return MCU samples at macroblock position (mb_x, mb_y)
  // clipped is true if the MCU is clipped and needs replication
  virtual void GetSamples(int mb_x, int mb_y, bool clipped,
                          int16_t* out_blocks) = 0;

 private:
  void CheckBuffers();

  void WriteAPP0();
  bool WriteAPPMarkers(const std::string& data);
  bool WriteEXIF(const std::string& data);
  bool WriteICCP(const std::string& data);
  bool WriteXMP(const std::string& data);
  void WriteDQT();
  void WriteSOF();
  void WriteDHT();
  void WriteSOS();
  void WriteEOI();

  void ResetDCs();

  // 1-pass scan
  void Scan();

  // 2-pass Huffman optimizing scan
  void ResetEntropyStats();
  void AddEntropyStats(const DCTCoeffs* const coeffs,
                       const RunLevel* const run_levels);
  void CompileEntropyStats();
  void SinglePassScan();
  void MultiPassScan();

  // Histogram pass
  void CollectHistograms();

  void BuildHuffmanCodes(const HuffmanTable* const tab,
                         uint32_t* const codes);

  typedef int (*QuantizeBlockFunc)(const int16_t in[64], int idx,
                                   const Quantizer* const Q,
                                   DCTCoeffs* const out, RunLevel* const rl);
  static QuantizeBlockFunc quantize_block_;
  static QuantizeBlockFunc GetQuantizeBlockFunc();

  static int TrellisQuantizeBlock(const int16_t in[64], int idx,
                                  const Quantizer* const Q,
                                  DCTCoeffs* const out,
                                  RunLevel* const rl);

  void CodeBlock(const DCTCoeffs* const coeffs, const RunLevel* const rl);
  // returns DC code (4bits for length, 12bits for suffix), updates DC_predictor
  static uint16_t GenerateDCDiffCode(int DC, int* const DC_predictor);

  static void FinalizeQuantMatrix(Quantizer* const q, int bias);
  void SetCostCodes(int idx);
  void InitCodes(bool only_ac);

 protected:
  // format-specific parameters, set by virtual InitComponents()
  enum { MAX_COMP = 3 };
  int nb_comps_;
  int quant_idx_[MAX_COMP];       // indices for quantization matrices
  int nb_blocks_[MAX_COMP];       // number of 8x8 blocks per components
  uint8_t block_dims_[MAX_COMP];  // component dimensions (8-pixels units)
  int block_w_, block_h_;         // maximum mcu width / height
  int mcu_blocks_;                // total blocks in mcu (= sum of nb_blocks_[])

  // data accessible to sub-classes implementing alternate input format
  int W_, H_, step_;    // width, height, stride
  int mb_w_, mb_h_;     // width / height in units of mcu
  const uint8_t* const rgb_;   // samples

  // Replicate an RGB source sub_w x sub_h block, expanding it to w x h size.
  const uint8_t* GetReplicatedSamples(const uint8_t* rgb,    // block source
                                      int rgb_step,          // stride in source
                                      int sub_w, int sub_h,  // sub-block size
                                      int w, int h);         // size of mcu
  // Replicate an YUV sub-block similarly.
  const uint8_t* GetReplicatedYUVSamples(const uint8_t* in, int step,
                                         int sub_w, int sub_h, int w, int h);
  // set blocks that are totally outside of the picture to an average value
  void AverageExtraLuma(int sub_w, int sub_h, int16_t* out);
  uint8_t replicated_buffer_[3 * 16 * 16];   // tmp buffer for replication

  sjpeg::RGBToYUVBlockFunc get_yuv_block_;
  static sjpeg::RGBToYUVBlockFunc get_yuv444_block_;
  void SetYUVFormat(bool use_444) {
    get_yuv_block_ = sjpeg::GetBlockFunc(use_444);
  }
  bool adaptive_bias_;   // if true, use per-block perceptual bias modulation

 private:
  sjpeg::BitWriter bw_;    // output buffer

  std::string iccp_, xmp_, exif_, app_markers_;   // metadata
  const uint8_t* metadata_;

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
  enum { ALIGN_CST = 15 };
  uint8_t* in_blocks_base_;   // base memory for blocks
  int16_t* in_blocks_;        // aligned pointer to in_blocks_base_
  bool have_coeffs_;          // true if the Fourier coefficients are stored

  // these are for regular compression methods 0 or 2.
  RunLevel base_run_levels_[64];

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

  static const float kHistoWeight[QSIZE];

  static void (*fDCT_)(int16_t* in, int num_blocks);
  static void InitializeStaticPointers();
};

////////////////////////////////////////////////////////////////////////////////
// static pointers to architecture-dependant implementation

SjpegEncoder::QuantizeBlockFunc SjpegEncoder::quantize_block_ = NULL;
void (*SjpegEncoder::fDCT_)(int16_t* in, int num_blocks) = NULL;
SjpegEncoder::StoreHistoFunc SjpegEncoder::store_histo_ = NULL;
sjpeg::RGBToYUVBlockFunc SjpegEncoder::get_yuv444_block_ = NULL;

void SjpegEncoder::InitializeStaticPointers() {
  if (fDCT_ == NULL) {
    store_histo_ = GetStoreHistoFunc();
    quantize_block_ = GetQuantizeBlockFunc();
    fDCT_ = sjpeg::SjpegGetFdct();
    get_yuv444_block_ = sjpeg::GetBlockFunc(true);
  }
}

////////////////////////////////////////////////////////////////////////////////

SjpegEncoder::SjpegEncoder(int W, int H, int step, const uint8_t* const rgb)
  : W_(W), H_(H), step_(step),
    rgb_(rgb),
    bw_(W_ * H_ / 4),      // very reasonable guess about final size
    in_blocks_base_(NULL),
    in_blocks_(NULL),
    have_coeffs_(false),
    all_run_levels_(NULL),
    nb_run_levels_(0),
    max_run_levels_(0),
    qdelta_max_luma_(kDefaultDeltaMaxLuma),
    qdelta_max_chroma_(kDefaultDeltaMaxChroma) {
  SetCompressionMethod(kDefaultMethod);
  SetQuality(kDefaultQuality);
  SetYUVFormat(false);
  SetQuantizationBias(kDefaultBias, false);
  const uint8_t* tmp[2] = { NULL, NULL };
  SetMinQuantMatrices(tmp, 0);
  InitializeStaticPointers();
  memset(dc_codes_, 0, sizeof(dc_codes_));  // safety
  memset(ac_codes_, 0, sizeof(ac_codes_));
}

SjpegEncoder::~SjpegEncoder() {
  delete[] all_run_levels_;
  delete[] in_blocks_base_;
}

////////////////////////////////////////////////////////////////////////////////
// memory and internal buffers managment. We grow on demand.

void SjpegEncoder::CheckBuffers() {
  // maximum macroblock size, worst-case, is 24bits*64*6 coeffs = 1152bytes
  bw_.ReserveLarge(2048);

  if (reuse_run_levels_) {
    if (nb_run_levels_ + 6*64 > max_run_levels_) {
      // need to grow storage for run/levels
      const size_t new_size = max_run_levels_ ? max_run_levels_ * 2 : 8192;
      RunLevel* const new_rl = new RunLevel[new_size];
      if (nb_run_levels_ > 0) {
        memcpy(new_rl, all_run_levels_,
               nb_run_levels_ * sizeof(new_rl[0]));
      }
      delete[] all_run_levels_;
      all_run_levels_ = new_rl;
      max_run_levels_ = new_size;
      DCHECK(nb_run_levels_ + 6 * 64 <= max_run_levels_);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Headers
//
// NOTE(skal): all chunks start with a startcode '0xff??' (0xffd8 e.g),
// followed by the size of the payload *not counting the startcode*!
// That's why you often find these 'Reserve(data_size + 2)' below, the '+2'
// accounting for the 0xff?? startcode size.

void SjpegEncoder::WriteAPP0() {  // SOI + APP0
  const uint8_t kHeader0[] = {
    0xff, 0xd8,                     // SOI
    0xff, 0xe0, 0x00, 0x10,         // APP0
    0x4a, 0x46, 0x49, 0x46, 0x00,   // 'JFIF'
    0x01, 0x01,                     // v1.01
    0x00, 0x00, 0x01, 0x00, 0x01,   // aspect ratio = 1:1
    0x00, 0x00                      // thumbnail width/height
  };
  bw_.Reserve(sizeof(kHeader0));
  bw_.PutBytes(kHeader0, sizeof(kHeader0));
}

bool SjpegEncoder::WriteAPPMarkers(const std::string& data) {
  if (data.size() == 0) return true;
  const size_t data_size = data.size();
  bw_.Reserve(data_size);
  bw_.PutBytes(reinterpret_cast<const uint8_t*>(data.data()), data.size());
  return true;
}

bool SjpegEncoder::WriteEXIF(const std::string& data) {
  if (data.size() == 0) return true;
  const uint8_t kEXIF[] = "Exif\0";
  const size_t kEXIF_len = 6;  // includes the \0's
  const size_t data_size = data.size() + kEXIF_len + 2;
  if (data_size > 0xffff) return false;
  bw_.Reserve(data_size);
  bw_.PutByte(0xff);
  bw_.PutByte(0xe1);
  bw_.PutByte((data_size >> 8) & 0xff);
  bw_.PutByte((data_size >> 0) & 0xff);
  bw_.PutBytes(kEXIF, kEXIF_len);
  bw_.PutBytes(reinterpret_cast<const uint8_t*>(data.data()), data.size());
  return true;
}

bool SjpegEncoder::WriteICCP(const std::string& data) {
  if (data.size() == 0) return true;
  size_t data_size = data.size();
  const uint8_t* ptr = reinterpret_cast<const uint8_t*>(data.data());
  const uint8_t kICCP[] = "ICC_PROFILE";
  const size_t kICCP_len = 12;  // includes the \0
  const size_t chunk_size_max = 0xffff - kICCP_len - 4;
  size_t max_chunk = (data_size + chunk_size_max - 1) / chunk_size_max;
  if (max_chunk >= 256) return false;
  size_t seq = 1;
  while (data_size > 0) {
    size_t size = data_size;
    if (size > chunk_size_max) size = chunk_size_max;
    bw_.Reserve(size + kICCP_len + 4 + 2);
    bw_.PutByte(0xff);
    bw_.PutByte(0xe2);
    bw_.PutByte(((size + kICCP_len + 4) >> 8) & 0xff);
    bw_.PutByte(((size + kICCP_len + 4) >> 0) & 0xff);
    bw_.PutBytes(kICCP, kICCP_len);
    bw_.PutByte(seq & 0xff);
    bw_.PutByte(max_chunk & 0xff);
    bw_.PutBytes(ptr, size);
    ptr += size;
    data_size -= size;
    seq += 1;
  }
  return true;
}

bool SjpegEncoder::WriteXMP(const std::string& data) {
  if (data.size() == 0) return true;
  const uint8_t kXMP[] = "http://ns.adobe.com/xap/1.0/";
  const size_t kXMP_size = 29;
  const size_t data_size = 2 + data.size() + kXMP_size;
  if (data_size > 0xffff) return false;  // error
  bw_.Reserve(data_size);
  bw_.PutByte(0xff);
  bw_.PutByte(0xe1);
  bw_.PutByte((data_size >> 8) & 0xff);
  bw_.PutByte((data_size >> 0) & 0xff);
  bw_.PutBytes(kXMP, kXMP_size);
  bw_.PutBytes(reinterpret_cast<const uint8_t*>(data.data()), data.size());
  return true;
}

////////////////////////////////////////////////////////////////////////////////

#define FP_BITS 16    // fractional precision for fixed-point dividors
#define AC_BITS 4     // extra precision bits from fdct's scaling
#define BIAS_DC 0x80  // neutral bias for DC (mandatory!)

// divide-by-multiply helper macros
#define MAKE_INV_QUANT(Q) (((1u << FP_BITS) + (Q) / 2) / (Q))
#define DIV_BY_MULT(A, M) (((A) * (M)) >> FP_BITS)
#define QUANTIZE(A, M, B) (DIV_BY_MULT((A) + (B), (M)) >> AC_BITS)

void SjpegEncoder::FinalizeQuantMatrix(Quantizer* const q, int q_bias) {
  // Special case! for v=1 we can't represent the multiplier with 16b precision.
  // So, instead we max out the multiplier to 0xffffu, and twist the bias to the
  // value 0x80. The overall precision isn't affected: it's bit-exact the same
  // for our working range.
  // Note that quant=1 can start appearing at quality as low as 93.
  const uint16_t bias_1 = 0x80;
  const uint16_t iquant_1 = 0xffffu;
  for (int i = 0; i < 64; ++i) {
    const uint16_t v = q->quant_[i];
    const uint16_t iquant = (v == 1) ? iquant_1 : MAKE_INV_QUANT(v);
    const uint16_t bias = (v == 1) ? bias_1 : (i == 0) ? BIAS_DC : q_bias;
    const uint16_t ibias = (((bias * v) << AC_BITS) + 128) >> 8;
    const uint16_t qthresh =
        ((1 << (FP_BITS + AC_BITS)) + iquant - 1) / iquant - ibias;
    q->bias_[i] = ibias;
    q->iquant_[i] = iquant;
    q->qthresh_[i] = qthresh;
    DCHECK(QUANTIZE(qthresh, iquant, ibias) > 0);
    DCHECK(QUANTIZE(qthresh - 1, iquant, ibias) == 0);
  }
}

void SjpegEncoder::SetCostCodes(int idx) {
  quants_[idx].codes_ = ac_codes_[idx];
}

void SjpegEncoder::WriteDQT() {
  const int data_size = 2 * 65 + 2;
  const uint8_t kDQTHeader[] = { 0xff, 0xdb, 0x00, (uint8_t)data_size };
  bw_.Reserve(data_size + 2);
  bw_.PutBytes(kDQTHeader, sizeof(kDQTHeader));
  for (int n = 0; n <= 1; ++n) {
    bw_.PutByte(n);
    const uint8_t* quant = quants_[n].quant_;
    for (int i = 0; i < 64; ++i) {
      bw_.PutByte(quant[sjpeg::kZigzag[i]]);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

#define DATA_16b(X) ((uint8_t)((X) >> 8)), ((uint8_t)((X) & 0xff))

void SjpegEncoder::WriteSOF() {   // SOF
  const int data_size = 8 + 3 * nb_comps_;
  DCHECK(data_size <= 255);
  const uint8_t kHeader[] = {
    0xff, 0xc0, DATA_16b(data_size),         // SOF0 marker, size
    0x08,                                    // 8bits/components
    DATA_16b(H_), DATA_16b(W_),              // height, width
    (uint8_t)nb_comps_                       // number of components
  };
  bw_.Reserve(data_size + 2);
  bw_.PutBytes(kHeader, sizeof(kHeader));
  for (int c = 0; c < nb_comps_; ++c) {
    bw_.PutByte(c + 1);
    bw_.PutByte(block_dims_[c]);
    bw_.PutByte(quant_idx_[c]);
  }
}

////////////////////////////////////////////////////////////////////////////////
// standard Huffman  tables, as per JPEG standard section K.3.

static const uint8_t kDCSyms[12] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
static const uint8_t kACSyms[2][162] = {
  { 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
    0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
    0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
    0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
    0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
    0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
    0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
    0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
    0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa },
  { 0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
    0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
    0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
    0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
    0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
    0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
    0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
    0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
    0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
    0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
    0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
    0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa }
};

static const HuffmanTable kHuffmanTables[4] = {
  { { 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 }, kDCSyms, 12 },
  { { 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 }, kDCSyms, 12 },
  { { 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125 }, kACSyms[0], 162 },
  { { 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119 }, kACSyms[1], 162 }
};

////////////////////////////////////////////////////////////////////////////////
// This function generates a map from symbols to code + len stored in a packed
// way (lower 16bit is the lenth, upper 16bit is the VLC).
// The input is a JPEG-like description of the symbols:
// - bits[i] stores the number of codes having length i + 1.
// - symbols[] contain the symbols' map, in increasing bit-length order.
// There is no check performed on the validity symbols[]'s content.
// The values of tab[] not referring to an actual symbol will remain unchanged.
// Returns the number of symbols used (that is: sum{bits[i]})

static int BuildHuffmanTable(const uint8_t bits[16], const uint8_t* symbols,
                             uint32_t* const tab) {
  uint32_t code = 0;
  int nb = 0;
  for (int nb_bits = 1; nb_bits <= 16; ++nb_bits, code <<= 1) {
    int n = bits[nb_bits - 1];  // number of code for that given nb_bits
    nb += n;
    while (n-- > 0) {
      const int symbol = *symbols++;
      tab[symbol] = (code << 16) | nb_bits;
      ++code;
    }
  }
  return nb;
}

////////////////////////////////////////////////////////////////////////////////

void SjpegEncoder::InitCodes(bool only_ac) {
  const int nb_tables = (nb_comps_ == 1 ? 1 : 2);
  for (int c = 0; c < nb_tables; ++c) {   // luma, chroma
    for (int type = (only_ac ? 1 : 0); type <= 1; ++type) {
      const HuffmanTable* const h = Huffman_tables_[type * 2 + c];
      const int nb_syms = BuildHuffmanTable(h->bits_, h->syms_,
                                            type == 1 ? ac_codes_[c]
                                                      : dc_codes_[c]);
      DCHECK(nb_syms == h->nb_syms_);
      (void)nb_syms;
    }
  }
}

void SjpegEncoder::WriteDHT() {
  InitCodes(false);
  const int nb_tables = (nb_comps_ == 1 ? 1 : 2);
  for (int c = 0; c < nb_tables; ++c) {   // luma, chroma
    for (int type = 0; type <= 1; ++type) {               // dc, ac
      const HuffmanTable* const h = Huffman_tables_[type * 2 + c];
      const int data_size = 3 + 16 + h->nb_syms_;
      DCHECK(data_size <= 255);
      bw_.Reserve(data_size + 2);
      bw_.PutByte(0xff);
      bw_.PutByte(0xc4);
      bw_.PutByte(0x00 /*data_size >> 8*/);
      bw_.PutByte(data_size);
      bw_.PutByte((type << 4) | c);
      bw_.PutBytes(h->bits_, 16);
      bw_.PutBytes(h->syms_, h->nb_syms_);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

void SjpegEncoder::WriteSOS() {   // SOS
  const int data_size = 6 + nb_comps_ * 2;
  DCHECK(data_size <= 255);
  const uint8_t kHeader[] = {
      0xff, 0xda, DATA_16b(data_size), (uint8_t)nb_comps_
  };
  bw_.Reserve(data_size + 2);
  bw_.PutBytes(kHeader, sizeof(kHeader));
  for (int c = 0; c < nb_comps_; ++c) {
    bw_.PutByte(c + 1);
    bw_.PutByte(quant_idx_[c] * 0x11);
  }
  bw_.PutByte(0x00);        // Ss
  bw_.PutByte(0x3f);        // Se
  bw_.PutByte(0x00);        // Ah/Al
}

////////////////////////////////////////////////////////////////////////////////

void SjpegEncoder::WriteEOI() {   // EOI
  bw_.Flush();
  // append EOI
  bw_.Reserve(2);
  bw_.PutByte(0xff);
  bw_.PutByte(0xd9);
}

////////////////////////////////////////////////////////////////////////////////
// Quantize coefficients and pseudo-code coefficients

static int CalcLog2(int v) {
#if defined(__GNUC__) && \
    ((__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || __GNUC__ >= 4)
  return 32 - __builtin_clz(v);
#else
  const int kLog2[16] = {
    0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4 };
  DCHECK(v > 0 && v < (1 << 12));
  return (v & ~0xff) ? 8 + kLog2[v >> 8] :
         (v & ~0x0f) ? 4 + kLog2[v >> 4] :
                       0 + kLog2[v];
#endif
}

uint16_t SjpegEncoder::GenerateDCDiffCode(int DC, int* const DC_predictor) {
  const int diff = DC - *DC_predictor;
  *DC_predictor = DC;
  if (diff == 0) {
    return 0;
  }
  int suff, n;
  if (diff < 0) {
    n = CalcLog2(-diff);
    suff = (diff - 1) & ((1 << n) - 1);
  } else {
    n = CalcLog2(diff);
    suff = diff;
  }
  DCHECK((suff & 0xf000) == 0);
  DCHECK(n < 12);
  return n | (suff << 4);
}

////////////////////////////////////////////////////////////////////////////////
// various implementation of histogram collection

#if defined(SJPEG_USE_SSE2)
// Load eight 16b-words from *src.
#define LOAD_16(src) _mm_loadu_si128(reinterpret_cast<const __m128i*>(src))
// Store eight 16b-words into *dst
#define STORE_16(V, dst) _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), (V))

static int QuantizeBlockSSE2(const int16_t in[64], int idx,
                             const Quantizer* const Q,
                             DCTCoeffs* const out, RunLevel* const rl) {
  const uint16_t* const bias = Q->bias_;
  const uint16_t* const iquant = Q->iquant_;
  int prev = 1;
  int nb = 0;
  int16_t tmp[64], masked[64];
  for (int i = 0; i < 64; i += 8) {
    const __m128i m_bias = LOAD_16(bias + i);
    const __m128i m_mult = LOAD_16(iquant + i);
    const __m128i A = LOAD_16(in + i);                        // A = in[i]
    const __m128i B = _mm_srai_epi16(A, 15);                  // sign extract
    const __m128i C = _mm_sub_epi16(_mm_xor_si128(A, B), B);  // abs(A)
    const __m128i D = _mm_adds_epi16(C, m_bias);              // v' = v + bias
    const __m128i E = _mm_mulhi_epu16(D, m_mult);             // (v' * iq) >> 16
    const __m128i F = _mm_srli_epi16(E, AC_BITS);             // = QUANTIZE(...)
    const __m128i G = _mm_xor_si128(F, B);                    // v ^ mask
    STORE_16(F, tmp + i);
    STORE_16(G, masked + i);
  }
  for (int i = 1; i < 64; ++i) {
    const int j = sjpeg::kZigzag[i];
    const int v = tmp[j];
    if (v > 0) {
      const int n = CalcLog2(v);
      const uint16_t code = masked[j] & ((1 << n) - 1);
      rl[nb].level_ = (code << 4) | n;
      rl[nb].run_ = i - prev;
      prev = i + 1;
      ++nb;
    }
  }
  const int dc = (in[0] < 0) ? -tmp[0] : tmp[0];
  out->idx_ = idx;
  out->last_ = prev - 1;
  out->nb_coeffs_ = nb;
  return dc;
}
#undef LOAD_16
#undef STORE_16

#elif defined(SJPEG_USE_NEON)
static int QuantizeBlockNEON(const int16_t in[64], int idx,
                             const Quantizer* const Q,
                             DCTCoeffs* const out, RunLevel* const rl) {
  const uint16_t* const bias = Q->bias_;
  const uint16_t* const iquant = Q->iquant_;
  int prev = 1;
  int nb = 0;
  uint16_t tmp[64], masked[64];
  for (int i = 0; i < 64; i += 8) {
    const uint16x8_t m_bias = vld1q_u16(bias + i);
    const uint16x8_t m_mult = vld1q_u16(iquant + i);
    const int16x8_t A = vld1q_s16(in + i);                           // in[i]
    const uint16x8_t B = vreinterpretq_u16_s16(vabsq_s16(A));        // abs(in)
    const int16x8_t sign = vshrq_n_s16(A, 15);                       // sign
    const uint16x8_t C = vaddq_u16(B, m_bias);                       // + bias
    const uint32x4_t D0 = vmull_u16(vget_low_u16(C),
                                    vget_low_u16(m_mult));
    const uint32x4_t D1 = vmull_u16(vget_high_u16(C),
                                    vget_high_u16(m_mult));
    // collect hi-words of the 32b mult result using 'unzip'
    const uint16x8x2_t E = vuzpq_u16(vreinterpretq_u16_u32(D0),
                                     vreinterpretq_u16_u32(D1));
    const uint16x8_t F = vshrq_n_u16(E.val[1], AC_BITS);
    const uint16x8_t G = veorq_u16(F, vreinterpretq_u16_s16(sign));  // v ^ mask
    vst1q_u16(tmp + i, F);
    vst1q_u16(masked + i, G);
  }
  for (int i = 1; i < 64; ++i) {
    const int j = sjpeg::kZigzag[i];
    const int v = tmp[j];
    if (v > 0) {
      const int n = CalcLog2(v);
      const uint16_t code = masked[j] & ((1 << n) - 1);
      rl[nb].level_ = (code << 4) | n;
      rl[nb].run_ = i - prev;
      prev = i + 1;
      ++nb;
    }
  }
  const int dc = (in[0] < 0) ? -tmp[0] : tmp[0];
  out->idx_ = idx;
  out->last_ = prev - 1;
  out->nb_coeffs_ = nb;
  return dc;
}
#endif    // SJPEG_USE_NEON

static int QuantizeBlock(const int16_t in[64], int idx,
                         const Quantizer* const Q,
                         DCTCoeffs* const out, RunLevel* const rl) {
  const uint16_t* const bias = Q->bias_;
  const uint16_t* const iquant = Q->iquant_;
  int prev = 1;
  int nb = 0;
  // This function is speed-critical, so we're using some bit mask
  // to extract absolute values, instead of sign tests.
  const uint16_t* const qthresh = Q->qthresh_;
  for (int i = 1; i < 64; ++i) {
    const int j = sjpeg::kZigzag[i];
    int v = in[j];
    const int32_t mask = v >> 31;
    v = (v ^ mask) - mask;
    if (v >= qthresh[j]) {
      v = QUANTIZE(v, iquant[j], bias[j]);
      DCHECK(v > 0);
      const int n = CalcLog2(v);
      const uint16_t code = (v ^ mask) & ((1 << n) - 1);
      rl[nb].level_ = (code << 4) | n;
      rl[nb].run_ = i - prev;
      prev = i + 1;
      ++nb;
    }
  }
  const int dc = (in[0] < 0) ? -QUANTIZE(-in[0], iquant[0], bias[0])
                             : QUANTIZE(in[0], iquant[0], bias[0]);
  out->idx_ = idx;
  out->last_ = prev - 1;
  out->nb_coeffs_ = nb;
  return dc;
}

////////////////////////////////////////////////////////////////////////////////
// Trellis-based quantization

typedef uint32_t score_t;
static const score_t kMaxScore = 0xffffffffu;

struct TrellisNode {
  uint32_t code;
  int      nbits;
  score_t score;
  uint32_t disto;
  uint32_t bits;
  uint32_t run;
  const TrellisNode* best_prev;
  int pos;
  int rank;

  TrellisNode() : score(kMaxScore), best_prev(NULL) {}
  void InitSink() {
    score = 0u;
    disto = 0;
    pos = 0;
    rank = 0;
    nbits = 0;
    bits = 0;
  }
};

static bool SearchBestPrev(const TrellisNode* const nodes0, TrellisNode* node,
                           const uint32_t disto0[], const uint32_t codes[],
                           uint32_t lambda) {
  bool found = false;
  assert(codes[0xf0] != 0);
  const uint32_t base_disto = node->disto + disto0[node->pos - 1];
  for (const TrellisNode* cur = node - 1; cur >= nodes0; --cur) {
    const int run = node->pos - 1 - cur->pos;
    if (run < 0) continue;
    uint32_t bits = node->nbits;
    bits += (run >> 4) * (codes[0xf0] & 0xff);
    const uint32_t sym = ((run & 15) << 4) | node->nbits;
    assert(codes[sym] != 0);
    bits += codes[sym] & 0xff;
    const uint32_t disto = base_disto - disto0[cur->pos];
    const score_t score = disto + lambda * bits + cur->score;
    if (score < node->score) {
      node->score = score;
      node->disto = disto;
      node->bits = bits;
      node->best_prev = cur;
      node->rank = cur->rank + 1;
      node->run = run;
      found = true;
    }
  }
  return found;
}

// number of alternate levels to investigate
#define NUM_TRELLIS_NODES 2

int SjpegEncoder::TrellisQuantizeBlock(const int16_t in[64], int idx,
                                       const Quantizer* const Q,
                                       DCTCoeffs* const out,
                                       RunLevel* const rl) {
  const uint16_t* const bias = Q->bias_;
  const uint16_t* const iquant = Q->iquant_;
  TrellisNode nodes[1 + NUM_TRELLIS_NODES * 63];  // 1 sink + n channels
  nodes[0].InitSink();
  const uint32_t* const codes = Q->codes_;
  TrellisNode* cur_node = &nodes[1];
  uint32_t disto0[64];   // disto0[i] = sum of distortions up to i (inclusive)
  disto0[0] = 0;
  for (int i = 1; i < 64; ++i) {
    const int j = sjpeg::kZigzag[i];
    const uint32_t q = Q->quant_[j] << AC_BITS;
    const uint32_t lambda = q * q / 32u;
    int V = in[j];
    const int32_t mask = V >> 31;
    V = (V ^ mask) - mask;
    disto0[i] = V * V + disto0[i - 1];
    int v = QUANTIZE(V, iquant[j], bias[j]);
    if (v == 0) continue;
    int nbits = CalcLog2(v);
    for (int k = 0; k < NUM_TRELLIS_NODES; ++k) {
      const int err = V - v * q;
      cur_node->code = (v ^ mask) & ((1 << nbits) - 1);
      cur_node->pos = i;
      cur_node->disto = err * err;
      cur_node->nbits = nbits;
      cur_node->score = kMaxScore;
      if (SearchBestPrev(&nodes[0], cur_node, disto0, codes, lambda)) {
        ++cur_node;
      }
      --nbits;
      if (nbits <= 0) break;
      v = (1 << nbits) - 1;
    }
  }
  // search best entry point backward
  const TrellisNode* nz = &nodes[0];
  if (cur_node != nz) {
    score_t best_score = kMaxScore;
    while (cur_node-- != &nodes[0]) {
      const uint32_t disto = disto0[63] - disto0[cur_node->pos];
      // No need to incorporate EOB's bit cost (codes[0x00]), since
      // it's the same for all coeff except the last one #63.
      cur_node->disto += disto;
      cur_node->score += disto;
      if (cur_node->score < best_score) {
        nz = cur_node;
        best_score = cur_node->score;
      }
    }
  }
  int nb = nz->rank;
  out->idx_ = idx;
  out->last_ = nz->pos;
  out->nb_coeffs_ = nb;

  while (nb-- > 0) {
    const int32_t code = nz->code;
    const int n = nz->nbits;
    rl[nb].level_ = (code << 4) | n;
    rl[nb].run_ = nz->run;
    nz = nz->best_prev;
  }
  const int dc = (in[0] < 0) ? -QUANTIZE(-in[0], iquant[0], bias[0])
                             : QUANTIZE(in[0], iquant[0], bias[0]);
  return dc;
}

SjpegEncoder::QuantizeBlockFunc SjpegEncoder::GetQuantizeBlockFunc() {
#if defined(SJPEG_USE_SSE2)
  if (sjpeg::SupportsSSE2()) return QuantizeBlockSSE2;
#elif defined(SJPEG_USE_NEON)
  if (sjpeg::SupportsNEON()) return QuantizeBlockNEON;
#endif
  return QuantizeBlock;  // default
}

////////////////////////////////////////////////////////////////////////////////
// Code bitstream

void SjpegEncoder::ResetDCs() {
  for (int c = 0; c < nb_comps_; ++c) {
    DCs_[c] = 0;
  }
}

void SjpegEncoder::CodeBlock(const DCTCoeffs* const coeffs,
                             const RunLevel* const rl) {
  const int idx = coeffs->idx_;
  const int q_idx = quant_idx_[idx];

  // DC coefficient symbol
  const int dc_len = coeffs->dc_code_ & 0x0f;
  const uint32_t code = dc_codes_[q_idx][dc_len];
  bw_.PutPackedCode(code);
  if (dc_len > 0) {
    bw_.PutBits(coeffs->dc_code_ >> 4, dc_len);
  }

  // AC coeffs
  const uint32_t* const codes = ac_codes_[q_idx];
  for (int i = 0; i < coeffs->nb_coeffs_; ++i) {
    int run = rl[i].run_;
    while (run & ~15) {        // escapes
      bw_.PutPackedCode(codes[0xf0]);
      run -= 16;
    }
    const uint32_t suffix = rl[i].level_;
    const int n = suffix & 0x0f;
    const int sym = (run << 4) | n;
    bw_.PutPackedCode(codes[sym]);
    bw_.PutBits(suffix >> 4, n);
  }
  if (coeffs->last_ < 63) {     // EOB
    bw_.PutPackedCode(codes[0x00]);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Histogram

void SjpegEncoder::ResetHisto() {
  memset(histos_, 0, sizeof(histos_));
}

#if defined(SJPEG_USE_SSE2)
void StoreHistoSSE2(const int16_t in[64], Histo* const histos, int nb_blocks) {
  const __m128i kMaxHisto = _mm_set1_epi16(MAX_HISTO_DCT_COEFF);
  for (int n = 0; n < nb_blocks; ++n, in += 64) {
    uint16_t tmp[64];
    for (int i = 0; i < 64; i += 8) {
      const __m128i A =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + i));
      const __m128i B = _mm_srai_epi16(A, 15);                  // sign extract
      const __m128i C = _mm_sub_epi16(_mm_xor_si128(A, B), B);  // abs(A)
      const __m128i D = _mm_srli_epi16(C, HSHIFT);              // >>= HSHIFT
      const __m128i E = _mm_min_epi16(D, kMaxHisto);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(tmp + i), E);
    }
    for (int j = 0; j < 64; ++j) {
      const int k = tmp[j];
      ++histos->counts_[j][k];
    }
  }
}
#elif defined(SJPEG_USE_NEON)
void StoreHistoNEON(const int16_t in[64], Histo* const histos, int nb_blocks) {
  const uint16x8_t kMaxHisto = vdupq_n_u16(MAX_HISTO_DCT_COEFF);
  for (int n = 0; n < nb_blocks; ++n, in += 64) {
    uint16_t tmp[64];
    for (int i = 0; i < 64; i += 8) {
      const int16x8_t A = vld1q_s16(in + i);
      const int16x8_t B = vabsq_s16(A);               // abs(in)
      const uint16x8_t C = vreinterpretq_u16_s16(B);  // signed->unsigned
      const uint16x8_t D = vshrq_n_u16(C, HSHIFT);    // >>= HSHIFT
      const uint16x8_t E = vminq_u16(D, kMaxHisto);   // min(.,kMaxHisto)
      vst1q_u16(tmp + i, E);
    }
    for (int j = 0; j < 64; ++j) {
      const int k = tmp[j];
      ++histos->counts_[j][k];
    }
  }
}
#endif

// This C-version is does not produce the same counts_[] output than the
// assembly above. But the extra entry counts_[MAX_HISTO_DCT_COEFF] is
// not used for the final computation, and the global result is unchanged.
void StoreHisto(const int16_t in[64], Histo* const histos, int nb_blocks) {
  for (int n = 0; n < nb_blocks; ++n, in += 64) {
    for (int i = 0; i < 64; ++i) {
      const int k = (in[i] < 0 ? -in[i] : in[i]) >> HSHIFT;
      if (k < MAX_HISTO_DCT_COEFF) {
        ++histos->counts_[i][k];
      }
    }
  }
}

SjpegEncoder::StoreHistoFunc SjpegEncoder::GetStoreHistoFunc() {
#if defined(SJPEG_USE_SSE2)
  if (sjpeg::SupportsSSE2()) return StoreHistoSSE2;
#elif defined(SJPEG_USE_NEON)
  if (sjpeg::SupportsNEON()) return StoreHistoNEON;
#endif
  return StoreHisto;  // default
}

const float SjpegEncoder::kHistoWeight[QSIZE] = {
  // Gaussian with sigma ~= 3
  0, 0, 0, 0, 0,
  1,   5,  16,  43,  94, 164, 228, 255, 228, 164,  94,  43,  16,   5,   1,
  0, 0, 0, 0, 0
};

void SjpegEncoder::AnalyseHisto() {
  // A bit of theory and background: for each sub-band i in [0..63], we pick a
  // quantization scale New_Qi close to the initial one Qi. We evaluate a cost
  // function associated with F({New_Qi}) = distortion + lambda . rate,
  // where rate and distortion depend on the quantizers set in a complex non-
  // analytic way. Just, for well-behaved regular histograms, we expect the
  // rate to scale as -log(Q), and the distortion as Q^2.
  // We want the cost function to be stationnary around the initial {Qi} set,
  // in order to achieve the best transfer between distortion and rate when we
  // displace a little the Qi values. Mainly we want to use bits as efficiently
  // as possible, where every bit we use has maximal impact in lowering
  // distortion (and vice versa: if we spend an extra bit of coding, we want to
  // have the best bang for this buck. The optimization works up-hill too).
  //
  // Hence, lambda is picked to minimize F around {Qi}, as:
  //    lambda = -d(distortion) / d(rate)
  // where the derivates are evaluated using a double least-square fit on both
  // the clouds of {delta, distortion} and {delta, size} points.
  //
  // Note1: The least-square fitted slope of a {x,y} cloud is expressed as:
  //    slope = (<xy> - <x><y>) / (<xx> - <x><x>) = Cov(x,y) / Cov(x,x)
  // where <.> is our gaussian-averaging operator.
  // But since we are eventually computing a quotient of such slopes, we can
  // factor out the common (<xx> - <x><x>) denominator (which is strictly
  // positive).
  // Note2: we use a Gaussian-weighted average around the center value Qi
  // instead of averaging over the whole [QDELTA_MIN, QDELTA_MAX] range.
  // This rules out fringe samples on noisy cases (like: when the source is
  // already JPEG-compressed!).
  // Note3: We fall back to some sane value HLAMBDA in case of ill-condition.
  //
  // We use use the correlation coefficient
  //       r = Cov(x,y) / sqrt(Cov(x,x) * Cov(y,y))
  // to detect bad cases with poorly extrapolated distortion. In such
  // occurrence, we skip the channel. This is particularly important for
  // already-compressed JPEG sources that give treacherous comb-like
  // histograms.
  //
  // Once this particular lambda has been picked, we loop over each channel
  // and optimize them separately, locally picking the best New_Qi for each.
  // The choice of lambda ensure a good balancing between size and distortion,
  // and prevent being too aggressive on file-size reduction for instance.
  //
  const double r_limit = kCorrelationThreshold;
  for (int c = (nb_comps_ > 1 ? 1 : 0); c >= 0; --c) {
    const int idx = quant_idx_[c];
    const Histo* const histo = &histos_[idx];
    // For chrominance, it can be visually damageable to be too
    // agressive on the filesize. So with the default settings we
    // restrict the algorithm to mainly try to *increase* the bitrate
    // (and quality) by using a smaller qdelta_max_chroma_.
    // delta_max is only use during the second phase, but not during
    // the first phase of deriving an optimal lambda.
    DCHECK(QDELTA_MAX >= qdelta_max_luma_);
    DCHECK(QDELTA_MAX >=qdelta_max_chroma_);
    const int delta_max =
      ((idx == 0) ? qdelta_max_luma_ : qdelta_max_chroma_) - QDELTA_MIN;
    DCHECK(delta_max < QSIZE);
    float sizes[64][QSIZE];
    float distortions[64][QSIZE];
    double num = 0.;  // accumulate d(distortion) around delta_q = 0
    double den = 0.;  // accumulate d(size) around delta_q = 0
    uint64_t omit_channels = kOmittedChannels;
    for (int pos = 0; pos < 64; ++pos) {
      if (omit_channels & (1ULL << pos)) {
        continue;
      }
      const int dq0 = quants_[idx].quant_[pos];
      const int min_dq0 = quants_[idx].min_quant_[pos];
      // We should be using the exact bias:
      //    const int bias = quants_[idx].bias_[pos] << (FP_BITS - AC_BITS);
      // but this value is too precise considering the other approximations
      // we're using (namely: HSHIFT). So we better use the a mid value of 0.5
      // for the bias. This have the advantage of making it possible to
      // use pre-calculated look-up tables for every quantities in the loop.
      // This is still a TODO(skal) below, though. Not sure the gain is big.
      const int bias = 1 << FP_BITS >> 1;
      const int* const h = histo->counts_[pos];
      int total = 0;
      int last = 0;
      for (int i = 0; i < MAX_HISTO_DCT_COEFF; ++i) {
        total += h[i];
        if (h[i]) last = i + 1;
      }
      if (total < kDensityThreshold * last) {
        omit_channels |= 1ULL << pos;
        continue;
      }
      // accumulators for averaged values.
      double sw = 0., sx = 0.;
      double sxx = 0., syy1 = 0.;
      double sy1 = 0., sxy1 = 0.;   // accumulators for distortion cloud
      double sy2 = 0., sxy2 = 0.;   // accumulators for size cloud
      for (int delta = 0; delta < QSIZE; ++delta) {
        double bsum = 0., dsum = 0.;
        const int dq = dq0 + (delta + QDELTA_MIN);
        if (dq >= min_dq0 && dq <= 255) {
          // TODO(skal): pre-compute idq and use it in FinalizeQuantMatrix too
          const int idq = ((1 << FP_BITS) + dq - 1) / dq;
          for (int i = 0; i < last; ++i) {
            if (h[i]) {
              // v = current bin's centroid in the histogram
              // qv = quantized value for the bin's representant 'v'
              // dqv = dequantized qv, to be compared against v (=> 'error')
              // bits = approximate bit-cost of quantized representant
              // h[i] = this bin's weight
              const int v = (i << HSHIFT) + HHALF;
              const int qv = (v * idq + bias) >> FP_BITS;
              // TODO(skal): for a given 'last' value, we know the upper imit
              // on dq that will make *all* quantized 'qv' values be zero.
              // => We can restrict the loop on 'dq' using 'last'.
              if (qv) {
                const int bits = CalcLog2(qv);
                const int dqv = qv * dq;
                const int error = (v - dqv) * (v - dqv);
                bsum += h[i] * bits;
                dsum += h[i] * error;
              } else {
                dsum += h[i] * v * v;
              }
            }
          }   // end of 'i' loop
          distortions[pos][delta] = (float)dsum;
          sizes[pos][delta] = (float)bsum;
          const double w = kHistoWeight[delta];   // Gaussian weight
          if (w > 0.) {
            const double x = static_cast<double>(delta + QDELTA_MIN);
            sw   += w;
            sx   += w * x;
            sxx  += w * x * x;
            sy1  += w * dsum;
            syy1 += w * dsum * dsum;
            sy2  += w * bsum;
            sxy1 += w * dsum * x;
            sxy2 += w * bsum * x;
          }
        } else {  // the new quantizer is out-of-range.
          distortions[pos][delta] = FLT_MAX;
          sizes[pos][delta] = 0;
        }
      }
      // filter channels according to correlation factor.
      const double cov_xy1 = sw * sxy1 - sx * sy1;
      if (cov_xy1 * cov_xy1 < r_limit *
                              (sw * sxx - sx * sx) * (sw * syy1 - sy1 * sy1)) {
        omit_channels |= 1ULL << pos;
        continue;
      }
      // accumulate numerator and denominator for the derivate calculation
      num += cov_xy1;
      den += sw * sxy2 - sx * sy2;
    }

    // we evaluate lambda =~ -d(distortion)/d(size) at dq=0
    double lambda = HLAMBDA;
    // When increasing Q, size should significantly decrease and distortion
    // increase. If they don't, we are ill-conditionned and should fall back
    // to a safe value HLAMBDA.
    if (num > 1000. && den < -10.) {
      // This is our approximation of -d(Distortion) / d(Rate)
      // We limit it to 1. below, to avoid degenerated cases
      lambda = -num / den;
      if (lambda < 1.) {
        lambda = 1.;
      }
    }
    // now, optimize each channel using the optimal lambda selection
    for (int pos = 0; pos < 64; ++pos) {
      if (omit_channels & (1ULL << pos)) {
        continue;
      }
      double best_score = FLT_MAX;
      int best_dq = 0;
      for (int delta = 0; delta <= delta_max; ++delta) {
        if (distortions[pos][delta] < FLT_MAX) {
          const double score = distortions[pos][delta]
                             + lambda * sizes[pos][delta];
          if (score < best_score) {
            best_score = score;
            best_dq = delta + QDELTA_MIN;
          }
        }
      }
      quants_[idx].quant_[pos] += best_dq;
      DCHECK(quants_[idx].quant_[pos] >= 1);
    }
    FinalizeQuantMatrix(&quants_[idx], q_bias_);
    SetCostCodes(idx);
  }
}

void SjpegEncoder::CollectHistograms() {
  ResetHisto();
  int16_t* in = in_blocks_;
  const int mb_x_max = W_ / block_w_;
  const int mb_y_max = H_ / block_h_;
  for (int mb_y = 0; mb_y < mb_h_; ++mb_y) {
    const bool yclip = (mb_y == mb_y_max);
    for (int mb_x = 0; mb_x < mb_w_; ++mb_x) {
      if (!use_extra_memory_) {
        in = in_blocks_;
      }
      GetSamples(mb_x, mb_y, yclip | (mb_x == mb_x_max), in);
      fDCT_(in, mcu_blocks_);
      for (int c = 0; c < nb_comps_; ++c) {
        const int num_blocks = nb_blocks_[c];
        store_histo_(in, &histos_[quant_idx_[c]], num_blocks);
        in += 64 * num_blocks;
      }
    }
  }
  have_coeffs_ = use_extra_memory_;
}

////////////////////////////////////////////////////////////////////////////////
// 1-pass Scan

void SjpegEncoder::Scan() {
  ResetDCs();

  RunLevel run_levels[64];
  int16_t* in = in_blocks_;
  const int mb_x_max = W_ / block_w_;
  const int mb_y_max = H_ / block_h_;
  const QuantizeBlockFunc quantize_block = use_trellis_ ? TrellisQuantizeBlock
                                                        : quantize_block_;
  for (int mb_y = 0; mb_y < mb_h_; ++mb_y) {
    const bool yclip = (mb_y == mb_y_max);
    for (int mb_x = 0; mb_x < mb_w_; ++mb_x) {
      CheckBuffers();
      if (!have_coeffs_) {
        in = in_blocks_;
        GetSamples(mb_x, mb_y, yclip | (mb_x == mb_x_max), in);
        fDCT_(in, mcu_blocks_);
      }
      for (int c = 0; c < nb_comps_; ++c) {
        DCTCoeffs base_coeffs;
        for (int i = 0; i < nb_blocks_[c]; ++i) {
          const int dc = quantize_block(in, c, &quants_[quant_idx_[c]],
                                        &base_coeffs, run_levels);
          base_coeffs.dc_code_ = GenerateDCDiffCode(dc, &DCs_[c]);
          CodeBlock(&base_coeffs, run_levels);
          in += 64;
        }
      }
    }
  }
}

void SjpegEncoder::SinglePassScan() {
  WriteDHT();
  WriteSOS();
  Scan();
}

////////////////////////////////////////////////////////////////////////////////
// Multi-pass

void SjpegEncoder::ResetEntropyStats() {
  memset(freq_ac_, 0, sizeof(freq_ac_));
  memset(freq_dc_, 0, sizeof(freq_dc_));
}

void SjpegEncoder::AddEntropyStats(const DCTCoeffs* const coeffs,
                                   const RunLevel* const run_levels) {
  // freq_ac_[] and freq_dc_[] cannot overflow 32bits, since the maximum
  // resolution allowed is 65535 * 65535. The sum of all frequencies cannot
  // be greater than 32bits, either.
  const int idx = coeffs->idx_;
  const int q_idx = quant_idx_[idx];
  for (int i = 0; i < coeffs->nb_coeffs_; ++i) {
    const int run = run_levels[i].run_;
    freq_ac_[q_idx][0xf0] += (run >> 4);  // count escapes (all at once)
    const int suffix = run_levels[i].level_;
    const int sym = ((run & 0x0f) << 4) | (suffix & 0x0f);
    ++freq_ac_[q_idx][sym];
  }
  if (coeffs->last_ < 63) {     // EOB
    ++freq_ac_[q_idx][0x00];
  }
  ++freq_dc_[q_idx][coeffs->dc_code_ & 0x0f];
}

static int cmp(const void *pa, const void *pb) {
  const uint64_t a = *reinterpret_cast<const uint64_t*>(pa);
  const uint64_t b = *reinterpret_cast<const uint64_t*>(pb);
  DCHECK(a != b);  // tie-breaks can't happen
  return (a < b) ? 1 : -1;
}

static void BuildOptimalTable(HuffmanTable* const t,
                              const uint32_t* const freq, int size) {
  enum { MAX_BITS = 32, MAX_CODE_SIZE = 16 };
  DCHECK(size <= 256);
  DCHECK(t != NULL);

  // The celebrated merging algorithm from Huffman, with some restrictions:
  // * codes with all '1' are forbidden, to avoid trailing marker emulation
  // * code should be less than 16bits. So we're re-allocating them to shorter
  //   code, even if it means being suboptimal for extremely rare symbols that
  //   would eat a lot of bits.
  // This function will not touch the content of freq[].
  int codesizes[256 + 1];
  // chain[i] will hold the index of the next element in the subtree below
  // element 'i', or -1 if there's no sub-tree.
  // We use and maintain this list in order to efficiently increasing the
  // codesizes by one when merging two sub-trees into one.
  // To ease the merging (by avoiding 1 loop) we store the address of the last
  // element in the chain for each symbol. This makes the process being O(1).
  // It's probably better to keep the arrays separated instead of making
  // a struct, since we touch chain_end[] only once per merging, whereas
  // chain[] and codesizes[] are modified O(k) time per merging.
  int chain[256 + 1];
  int* chain_end[256 + 1];
  // sorted_freq[] remains sorted by decreasing frequencies along the process.
  uint64_t sorted_freq[256 + 1];

  // Counts and puts the symbols effectively used at the beginning of the table.
  int nb_syms = 0;
  for (int i = 0; i < size; ++i) {
    const uint64_t v = freq[i];
    if (v > 0) {
      // we pack the sorted key (32bits) and index (9bits) into a single uint64_t,
      // so we don't have to resort to structs (and we avoid tie-breaks, too)
      sorted_freq[nb_syms++] = (v << 9) | i;
    }
    codesizes[i] = 0;
    chain[i] = -1;
    chain_end[i] = &chain[i];
  }
  t->nb_syms_ = nb_syms;  // Record how many final symbols we'll have.

  // initial sort
  // TODO(skal): replace by counting-sort?? (merged with previous loop?)
  qsort(sorted_freq, nb_syms, sizeof(sorted_freq[0]), cmp);

  // fake last symbol, with lowest frequency: will be assigned to the forbidden
  // code '1111...1', but will eventually be discarded.
  sorted_freq[nb_syms++] = (1ULL << 9) | size;
  codesizes[size] = 0;
  chain[size] = -1;
  chain_end[size] = &chain[size];

  // Merging phase
  // Recursively merge the two symbols with lowest frequency. The resulting
  // super-symbol will be represented by a longer (by 1bit) code, since
  // it's the least frequent one.
  int nb = nb_syms;
  while (nb-- > 1) {
    // First, link the two sub-trees.
    const uint64_t s1 = sorted_freq[nb - 1];    // first symbol
    const uint64_t s2 = sorted_freq[nb];        // second symbol, appended
    // The 0x1ff masking is for taking only the symbol, discarding the
    // frequency that we stored in the upper bits for sorting.
    int i = s1 & 0x1ff;
    const int j = s2 & 0x1ff;
    assert(i <= size && j <= size);
    *chain_end[i] = j;
    chain_end[i] = chain_end[j];

    // Then, following the chain, increase the whole sub-tree's weight by 1bit.
    do {
      ++codesizes[i];
      i = chain[i];
    } while (i >= 0);

    // Create new symbol, with merged frequencies. Will take s1's spot.
    // We must use 64bit here to prevent overflow in the sum. Both s1 and
    // s2 are originally 32 + 9 bits wide.
    const uint64_t new_symbol = s1 + (s2 & ~0x1ff);
    // Perform insertion sort to find the new spot of the merged symbol.
    int k = nb - 1;
    while (k > 0) {
      if (sorted_freq[k - 1] < new_symbol) {
        sorted_freq[k] = sorted_freq[k - 1];
        --k;
      } else {
        break;
      }
    }
    sorted_freq[k] = new_symbol;
  }

  // Count bit distribution.
  uint8_t bits[MAX_BITS];
  memset(bits, 0, sizeof(bits));
  int max_bit_size = 0;
  for (int i = 0; i <= size; ++i) {
    int s = codesizes[i];
    DCHECK(s <= codesizes[size]);    // symbol #size is the biggest one.
    if (s > 0) {
      // This is slightly penalizing but only for ultra-rare symbol
      if (s > MAX_BITS) {
        s = MAX_BITS;
        codesizes[i] = MAX_BITS;    // clamp code-size
      }
      ++bits[s - 1];
      if (s > max_bit_size) {
        max_bit_size = s;
      }
    }
  }

  // We sort symbols by slices of increasing bitsizes, using counting sort.
  // This will generate a partition of symbols in the final syms_[] array.
  int start[MAX_BITS];     // start[i] is the first code with length i+1
  int position = 0;
  for (int i = 0; i < max_bit_size; ++i) {
    start[i] = position;
    position += bits[i];
  }
  DCHECK(position == nb_syms);

  // Now, we can ventilate the symbols directly to their final slice in the
  // partitioning, according to the their bit-length.
  // Note: we omit the last symbol, which is fake.
  uint8_t* const syms = const_cast<uint8_t*>(t->syms_);
  // Note that we loop til symbol = size-1, hence omitting the last fake symbol.
  for (int symbol = 0; symbol < size; ++symbol) {
    const int s = codesizes[symbol];
    if (s > 0) {
      DCHECK(s <= MAX_BITS);
      syms[start[s - 1]++] = symbol;
    }
  }
  DCHECK(start[max_bit_size - 1] == nb_syms - 1);

  // Fix codes with length greater than 16 bits. We move too long
  // codes up, and one short down, making the tree a little sub-optimal.
  for (int l = max_bit_size - 1; l >= MAX_CODE_SIZE; --l) {
    while (bits[l] > 0) {
      int k = l - 2;
      while (bits[k] == 0) {    // Search for a level with a leaf to split.
        --k;
      }
      /* Move up 2 symbols from bottom-most level l, and sink down one from
         level k, like this:
                    Before:                After:
                    /  ..                 /    ..
        k bits->   c     \               /\      \
                         /\             c  b     /\
                       .. /\                   ..  a
        l bits->         a  b
        Note that by the very construction of the optimal tree, the least
        probable symbols always come by pair with same bit-length.
        So there's always a pair of 'a' and 'b' to find.
      */
      bits[l    ] -= 2;     // remove 'a' and 'b'
      bits[l - 1] += 1;     // put 'a' one level up.
      bits[k    ] -= 1;     // remove 'c'
      bits[k + 1] += 2;     // put 'c' anb 'b' one level down.
    }
  }

  // remove last pseudo-symbol
  max_bit_size = MAX_CODE_SIZE;
  while (bits[--max_bit_size] == 0) {
    DCHECK(max_bit_size > 0);
  }
  --bits[max_bit_size];

  // update table with final book
  for (int i = 0; i < MAX_CODE_SIZE; ++i) {
    t->bits_[i] = bits[i];
  }
}

void SjpegEncoder::CompileEntropyStats() {
  // plug and build new tables
  for (int q_idx = 0; q_idx < (nb_comps_ == 1 ? 1 : 2); ++q_idx) {
    // DC tables
    Huffman_tables_[q_idx] = &opt_tables_dc_[q_idx];
    opt_tables_dc_[q_idx].syms_ = opt_syms_dc_[q_idx];
    BuildOptimalTable(&opt_tables_dc_[q_idx], freq_dc_[q_idx], 12);
    // AC tables
    Huffman_tables_[2 + q_idx] = &opt_tables_ac_[q_idx];
    opt_tables_ac_[q_idx].syms_ = opt_syms_ac_[q_idx];
    BuildOptimalTable(&opt_tables_ac_[q_idx], freq_ac_[q_idx], 256);
  }
}

////////////////////////////////////////////////////////////////////////////////

void SjpegEncoder::MultiPassScan() {
  const int nb_mbs = mb_w_ * mb_h_ * mcu_blocks_;
  DCTCoeffs* const base_coeffs =
      new DCTCoeffs[reuse_run_levels_ ? nb_mbs : 1];
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
  for (int mb_y = 0; mb_y < mb_h_; ++mb_y) {
    const bool yclip = (mb_y == mb_y_max);
    for (int mb_x = 0; mb_x < mb_w_; ++mb_x) {
      CheckBuffers();
      if (!have_coeffs_) {
        in = in_blocks_;
        GetSamples(mb_x, mb_y, yclip | (mb_x == mb_x_max), in);
        fDCT_(in, mcu_blocks_);
      }
      for (int c = 0; c < nb_comps_; ++c) {
        for (int i = 0; i < nb_blocks_[c]; ++i) {
          RunLevel* const run_levels =
              reuse_run_levels_ ? all_run_levels_ + nb_run_levels_
                                : base_run_levels;
          const int dc = quantize_block(in, c, &quants_[quant_idx_[c]],
                                        coeffs, run_levels);
          coeffs->dc_code_ = GenerateDCDiffCode(dc, &DCs_[c]);
          AddEntropyStats(coeffs, run_levels);
          if (reuse_run_levels_) {
            nb_run_levels_ += coeffs->nb_coeffs_;
            ++coeffs;
            DCHECK(coeffs <= &base_coeffs[nb_mbs]);
          }
          in += 64;
          DCHECK(nb_run_levels_ <= max_run_levels_);
        }
      }
    }
  }

  CompileEntropyStats();
  WriteDHT();
  WriteSOS();

  if (!reuse_run_levels_) {
    // redo everything, but with optimal tables now.
    Scan();
  } else {
    delete[] in_blocks_base_;
    in_blocks_base_ = NULL;     // we can free up memory for coeffs here
    in_blocks_ = NULL;          // sanity

    // Re-use the saved run/levels for fast 2nd-pass.
    coeffs = base_coeffs;
    CheckBuffers();   // this call is needed to finalize all_run_levels_.
    RunLevel* run_levels = all_run_levels_;
    for (int n = 0; n < nb_mbs; ++n) {
      CheckBuffers();
      CodeBlock(&coeffs[n], run_levels);
      run_levels += coeffs[n].nb_coeffs_;
    }
  }
  delete[] base_coeffs;
}

////////////////////////////////////////////////////////////////////////////////
// main call

bool SjpegEncoder::Encode() {
  FinalizeQuantMatrix(&quants_[0], q_bias_);
  FinalizeQuantMatrix(&quants_[1], q_bias_);
  SetCostCodes(0);
  SetCostCodes(1);

  // default tables
  for (int i = 0; i < 4; ++i) Huffman_tables_[i] = &kHuffmanTables[i];

  // colorspace init
  InitComponents();
  DCHECK(nb_comps_ <= MAX_COMP);
  DCHECK(mcu_blocks_ <= 6);
  // validate some input parameters
  if (W_ <= 0 || H_ <= 0 || rgb_ == NULL) {
    bw_.DeleteOutputBuffer();    // release output_ memory
    return false;
  }
  mb_w_ = (W_ + (block_w_ - 1)) / block_w_;
  mb_h_ = (H_ + (block_h_ - 1)) / block_h_;
  const int nb_blocks = use_extra_memory_ ? mb_w_ * mb_h_ : 1;
  in_blocks_base_ =
    new uint8_t[nb_blocks * mcu_blocks_ * 64 * sizeof(*in_blocks_) + ALIGN_CST];
  in_blocks_ = reinterpret_cast<int16_t*>(
      (ALIGN_CST + reinterpret_cast<uintptr_t>(in_blocks_base_)) & ~ALIGN_CST);
  have_coeffs_ = false;

  // Histogram analysis, deriving optimal quant matrices
  if (use_adaptive_quant_) {
    CollectHistograms();
    AnalyseHisto();
  }

  WriteAPP0();

  // custom markers written 'as is'
  if (!WriteAPPMarkers(app_markers_)) return false;

  // metadata
  if (!WriteEXIF(exif_) || !WriteICCP(iccp_) || !WriteXMP(xmp_)) return false;

  WriteDQT();
  WriteSOF();

  if (optimize_size_) {
    MultiPassScan();
  } else {
    SinglePassScan();
  }

  WriteEOI();

  delete[] in_blocks_base_;
  in_blocks_base_ = NULL;
  in_blocks_ = NULL;

  return true;
}

////////////////////////////////////////////////////////////////////////////////
// Edge replication

namespace {

int GetAverage(const int16_t* const out) {
  int DC = 0;
  for (int i = 0; i < 64; ++i) DC += out[i];
  return (DC + 32) >> 6;
}

void SetAverage(int DC, int16_t* const out) {
  for (int i = 0; i < 64; ++i) out[i] = DC;
}

}   // anonymous namespace

void SjpegEncoder::AverageExtraLuma(int sub_w, int sub_h, int16_t* out) {
  // out[] points to four 8x8 blocks. When one of this block is totally
  // outside of the frame, we set it flat to the average value of the previous
  // block ("DC"), in order to help compressibility.
  int DC = GetAverage(out);
  if (sub_w <= 8) {   // set block #1 to block #0's average value
    SetAverage(DC, out + 1 * 64);
  }
  if (sub_h <= 8) {   // Need to flatten block #2 and #3
    if (sub_w > 8) {  // block #1 was not flatten, so get its real DC
      DC = GetAverage(out + 1 * 64);
    }
    SetAverage(DC, out + 2 * 64);
    SetAverage(DC, out + 3 * 64);
  } else if (sub_w <= 8) {   // set block #3 to the block #2's average value
    DC = GetAverage(out + 2 * 64);
    SetAverage(DC, out + 3 * 64);
  }
}

const uint8_t* SjpegEncoder::GetReplicatedSamples(const uint8_t* rgb,
                                                  int rgb_step,
                                                  int sub_w, int sub_h,
                                                  int w, int h) {
  DCHECK(sub_w > 0 && sub_h > 0);
  if (sub_w > w) {
    sub_w = w;
  }
  if (sub_h > h) {
    sub_h = h;
  }
  uint8_t* dst = replicated_buffer_;
  for (int y = 0; y < sub_h; ++y) {
    memcpy(dst, rgb, 3 * sub_w);
    const uint8_t* const src0 = &dst[3 * (sub_w - 1)];
    for (int x = 3 * sub_w; x < 3 * w; x += 3) {
      memcpy(dst + x, src0, 3);
    }
    dst += 3 * w;
    rgb += rgb_step;
  }
  const uint8_t* dst0 = dst - 3 * w;
  for (int y = sub_h; y < h; ++y) {
    memcpy(dst, dst0, 3 * w);
    dst += 3 * w;
  }
  return replicated_buffer_;
}

// TODO(skal): merge with above function? Probably slower...
const uint8_t* SjpegEncoder::GetReplicatedYUVSamples(const uint8_t* in,
                                                     int step,
                                                     int sub_w, int sub_h,
                                                     int w, int h) {
  DCHECK(sub_w > 0 && sub_h > 0);
  if (sub_w > w) {
    sub_w = w;
  }
  if (sub_h > h) {
    sub_h = h;
  }
  uint8_t* out = replicated_buffer_;
  for (int y = 0; y < sub_h; ++y) {
    int x;
    for (x = 0; x < sub_w; ++x)
      out[x] = in[x];
    for (; x < w; ++x) {
      out[x] = out[sub_w - 1];
    }
    out += w;
    in += step;
  }
  const uint8_t* const out0 = out - w;
  for (int y = sub_h; y < h; ++y) {
    memcpy(out, out0, w);
    out += w;
  }
  return replicated_buffer_;
}

////////////////////////////////////////////////////////////////////////////////
// sub-class for YUV 4:2:0 version

class SjpegEncoder420 : public SjpegEncoder {
 public:
  SjpegEncoder420(int W, int H, int step, const uint8_t* const rgb)
    : SjpegEncoder(W, H, step, rgb) {}
  virtual ~SjpegEncoder420() {}
  virtual void InitComponents() {
    nb_comps_ = 3;

    quant_idx_[0] = 0;
    quant_idx_[1] = 1;
    quant_idx_[2] = 1;

    nb_blocks_[0] = 4;
    nb_blocks_[1] = 1;
    nb_blocks_[2] = 1;
    mcu_blocks_ = 6;

    block_w_ = 16;
    block_h_ = 16;
    block_dims_[0] = 0x22;
    block_dims_[1] = 0x11;
    block_dims_[2] = 0x11;
  }
  virtual void GetSamples(int mb_x, int mb_y, bool clipped,
                          int16_t* out_blocks) {
    const uint8_t* data = rgb_ + (3 * mb_x + mb_y * step_) * 16;
    int step = step_;
    if (clipped) {
      data = GetReplicatedSamples(data, step,
                                  W_ - mb_x * 16, H_ - mb_y * 16, 16, 16);
      step = 3 * 16;
    }
    get_yuv_block_(data, step, out_blocks);
    if (clipped) {
      AverageExtraLuma(W_ - mb_x * 16, H_ - mb_y * 16, out_blocks);
    }
  }
};

////////////////////////////////////////////////////////////////////////////////
// sub-class for YUV 4:4:4 version

class SjpegEncoder444 : public SjpegEncoder {
 public:
  SjpegEncoder444(int W, int H, int step, const uint8_t* const rgb)
      : SjpegEncoder(W, H, step, rgb) {
    SetYUVFormat(true);
  }
  virtual ~SjpegEncoder444() {}
  virtual void InitComponents() {
    nb_comps_ = 3;

    quant_idx_[0] = 0;
    quant_idx_[1] = 1;
    quant_idx_[2] = 1;

    nb_blocks_[0] = 1;
    nb_blocks_[1] = 1;
    nb_blocks_[2] = 1;
    mcu_blocks_ = 3;

    block_w_ = 8;
    block_h_ = 8;
    block_dims_[0] = 0x11;
    block_dims_[1] = 0x11;
    block_dims_[2] = 0x11;
  }
  virtual void GetSamples(int mb_x, int mb_y, bool clipped, int16_t* out) {
    const uint8_t* data = rgb_ + (3 * mb_x + mb_y * step_) * 8;
    int step = step_;
    if (clipped) {
      data = GetReplicatedSamples(data, step,
                                  W_ - mb_x * 8, H_ - mb_y * 8, 8, 8);
      step = 3 * 8;
    }
    get_yuv_block_(data, step, out);
  }
};

////////////////////////////////////////////////////////////////////////////////
// sub-class for the sharp YUV 4:2:0 version

class SjpegEncoderSharp420 : public SjpegEncoder420 {
 public:
  SjpegEncoderSharp420(int W, int H, int step, const uint8_t* const rgb)
      : SjpegEncoder420(W, H, step, rgb), yuv_memory_(NULL) {
    const int uv_w = (W + 1) >> 1;
    const int uv_h = (H + 1) >> 1;
    yuv_memory_ = new uint8_t[W * H + 2 * uv_w * uv_h];
    y_plane_ = yuv_memory_;
    y_step_ = W;
    u_plane_ = yuv_memory_ + W * H;
    v_plane_ = u_plane_ + uv_w * uv_h;
    uv_step_ = uv_w;
    sjpeg::ApplySharpYUVConversion(rgb, W, H, step,
                                   y_plane_, u_plane_, v_plane_);
  }
  virtual ~SjpegEncoderSharp420() { delete[] yuv_memory_; }
  virtual void GetSamples(int mb_x, int mb_y, bool clipped, int16_t* out);

 protected:
  void GetLumaSamples(int mb_x, int mb_y, bool clipped, int16_t* out) {
    int step = y_step_;
    const uint8_t* Y1 = y_plane_ + (mb_x + mb_y * step) * 16;
    if (clipped) {
      Y1 = GetReplicatedYUVSamples(Y1, step,
                                   W_ - mb_x * 16, H_ - mb_y * 16, 16, 16);
      step = 16;
    }
    const uint8_t* Y2 = Y1 + 8 * step;
    for (int y = 8, n = 0; y > 0; --y) {
      for (int x = 0; x < 8; ++x, ++n) {
        out[n + 0 * 64] = Y1[x] - 128;
        out[n + 1 * 64] = Y1[x + 8] - 128;
        out[n + 2 * 64] = Y2[x] - 128;
        out[n + 3 * 64] = Y2[x + 8] - 128;
      }
      Y1 += step;
      Y2 += step;
    }
    if (clipped) {
      AverageExtraLuma(W_ - mb_x * 16, H_ - mb_y * 16, out);
    }
  }

 private:
   uint8_t* y_plane_;
   int y_step_;
   uint8_t* u_plane_;
   uint8_t* v_plane_;
   int uv_step_;
   uint8_t* yuv_memory_;
};

void SjpegEncoderSharp420::GetSamples(int mb_x, int mb_y,
                                      bool clipped, int16_t* out) {
  GetLumaSamples(mb_x, mb_y, clipped, out);

  // Chroma
  const uint8_t* U = u_plane_ + (mb_x + mb_y * uv_step_) * 8;
  int step = uv_step_;
  if (clipped) {
    U = GetReplicatedYUVSamples(U, step,
                                ((W_ + 1) >> 1) - mb_x * 8,
                                ((H_ + 1) >> 1) - mb_y * 8, 8, 8);
    step = 8;
  }
  for (int y = 8, n = 0; y > 0; --y, U += step) {
    for (int x = 0; x < 8; ++x, ++n) {
      out[n + 4 * 64] = U[x] - 128;
    }
  }
  const uint8_t* V = v_plane_ + (mb_x + mb_y * uv_step_) * 8;
  step = uv_step_;
  if (clipped) {
    V = GetReplicatedYUVSamples(V, step,
                                ((W_ + 1) >> 1) - mb_x * 8,
                                ((H_ + 1) >> 1) - mb_y * 8, 8, 8);
    step = 8;
  }
  for (int y = 8, n = 0; y > 0; --y, V += step) {
    for (int x = 0; x < 8; ++x, ++n) {
      out[n + 5 * 64] = V[x] - 128;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// all-in-one factory to pickup the right encoder instance

SjpegEncoder* EncoderFactory(const uint8_t* rgb,
                             int W, int H, int stride, int yuv_mode) {
  if (yuv_mode <= 0) {
    yuv_mode = SjpegRiskiness(rgb, W, H, stride, NULL);
  }

  SjpegEncoder* enc = NULL;
  if (yuv_mode == 1) {
    enc = new SjpegEncoder420(W, H, stride, rgb);
  } else if (yuv_mode == 2) {
    enc = new SjpegEncoderSharp420(W, H, stride, rgb);
  } else {
    enc = new SjpegEncoder444(W, H, stride, rgb);
  }
  return enc;
}

}     // anonymous namespace

////////////////////////////////////////////////////////////////////////////////
// public plain-C functions

size_t SjpegEncode(const uint8_t* rgb, int W, int H, int stride,
                   uint8_t** out_data, int quality, int method, int yuv_mode) {
  if (rgb == NULL || out_data == NULL || W <= 0 || H <= 0 || stride < 3 * W) {
    return 0;
  }
  *out_data = NULL;  // safety

  SjpegEncoder* const enc = EncoderFactory(rgb, W, H, stride, yuv_mode);
  enc->SetQuality(quality);
  enc->SetCompressionMethod(method);
  size_t size = 0;
  if (enc->Encode()) {
    *out_data = enc->Grab(&size);
  } else {
    *out_data = NULL;
  }
  delete enc;
  return size;
}

////////////////////////////////////////////////////////////////////////////////

void SjpegFreeBuffer(uint8_t* buffer) {
  delete[] buffer;
}

////////////////////////////////////////////////////////////////////////////////

size_t SjpegCompress(const uint8_t* rgb, int W, int H, int quality,
                     uint8_t** out_data) {
  return SjpegEncode(rgb, W, H, 3 * W, out_data, quality, 4, 0);
}

////////////////////////////////////////////////////////////////////////////////

uint32_t SjpegVersion() {
  return SJPEG_VERSION;
}

////////////////////////////////////////////////////////////////////////////////
// Parametrized call

SjpegEncodeParam::SjpegEncodeParam() {
  Init(kDefaultQuality);
}

void SjpegEncodeParam::Init(int quality_factor) {
  Huffman_compress = true;
  adaptive_quantization = true;
  use_trellis = false;
  yuv_mode = 0;
  quantization_bias = kDefaultBias;
  qdelta_max_luma = kDefaultDeltaMaxLuma;
  qdelta_max_chroma = kDefaultDeltaMaxChroma;
  adaptive_bias = false;
  SetLimitQuantization(false);
  min_quant_tolerance_ = 0;
  SetQuality(quality_factor);
}

void SjpegEncodeParam::SetQuality(int quality_factor) {
  const int q = sjpeg::GetQFactor(quality_factor);
  sjpeg::SetQuantMatrix(sjpeg::kDefaultMatrices[0], q, quant_[0]);
  sjpeg::SetQuantMatrix(sjpeg::kDefaultMatrices[1], q, quant_[1]);
}

void SjpegEncodeParam::SetQuantMatrix(const uint8_t m[64], int idx,
                                      int reduction) {
  if (reduction <= 1) reduction = 1;
  if (m == NULL) return;
  for (int i = 0; i < 64; ++i) {
    const int v = m[i] * 100 / reduction;
    quant_[idx][i] = (v > 255) ? 255u : (v < 1) ? 1u : v;
  }
}

void SjpegEncodeParam::SetReduction(int reduction) {
  SetQuantMatrix(quant_[0], 0, reduction);
  SetQuantMatrix(quant_[1], 1, reduction);
}

void SjpegEncodeParam::SetLimitQuantization(bool limit_quantization,
                                            int tolerance) {
  if (!limit_quantization) {
    min_quant_[0] = NULL;
    min_quant_[1] = NULL;
  } else {
    min_quant_[0] = quant_[0];
    min_quant_[1] = quant_[1];
  }
  min_quant_tolerance_ = tolerance < 0 ? 0
                       : tolerance > 100 ? 100
                       : tolerance;
}

void SjpegEncodeParam::ResetMetadata() {
  iccp.clear();
  exif.clear();
  xmp.clear();
  app_markers.clear();
}

std::string SjpegEncode(const uint8_t* rgb, int W, int H, int stride,
                        const SjpegEncodeParam& param) {
  if (rgb == NULL || W <= 0 || H <= 0 || stride < 3 * W) return "";

  SjpegEncoder* const enc = EncoderFactory(rgb, W, H, stride, param.yuv_mode);
  enc->SetQuantMatrices(param.quant_);
  enc->SetMinQuantMatrices(param.min_quant_, param.min_quant_tolerance_);

  int method = param.Huffman_compress ? 1 : 0;
  if (param.adaptive_quantization) method += 3;
  if (param.use_trellis) {
    if (method == 4) method = 7;
    else if (method == 6) method = 8;
  }
  enc->SetCompressionMethod(method);
  enc->SetQuantizationBias(param.quantization_bias, param.adaptive_bias);
  enc->SetQuantizationDeltas(param.qdelta_max_luma, param.qdelta_max_chroma);

  enc->SetMetadata(param.iccp, SjpegEncoder::ICC);
  enc->SetMetadata(param.exif, SjpegEncoder::EXIF);
  enc->SetMetadata(param.xmp, SjpegEncoder::XMP);
  enc->SetMetadata(param.app_markers, SjpegEncoder::MARKERS);

  if (!enc->Encode()) {
    delete enc;
    return "";
  }
  size_t size;
  uint8_t* const buf = enc->Grab(&size);
  delete enc;

  std::string output;
  output.append(reinterpret_cast<const char*>(buf), size);
  delete[] buf;
  return output;
}

////////////////////////////////////////////////////////////////////////////////
// std::string variants

std::string SjpegCompress(const uint8_t* rgb, int W, int H, int quality) {
  std::string output;
  uint8_t* data = NULL;
  size_t size = SjpegCompress(rgb, W, H, quality, &data);
  if (size > 0) output.append(reinterpret_cast<const char*>(data), size);
  delete[] data;
  return output;
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
