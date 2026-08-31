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
//  Quantization: matrices, block quantization and trellis search
//
// Author: Skal (pascal.massimino@gmail.com)

#include <assert.h>
#include <stdint.h>
#include <string.h>

#include <cmath>

#define SJPEG_NEED_ASM_HEADERS
#include "sjpegi.h"

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

// Inverse of kZigzag: maps a natural coefficient index to its zig-zag scan
// position (kInvZigzag[kZigzag[i]] == i). Used by the SIMD run-length emitters
// to turn a natural-order non-zero bitmask into a zig-zag-ordered one.
// Not static: quantize_avx2.cc (a separate TU) needs it too.
const uint8_t kInvZigzag[64] = {
  0,   1,  5,  6, 14, 15, 27, 28,
  2,   4,  7, 13, 16, 26, 29, 42,
  3,   8, 12, 17, 25, 30, 41, 43,
  9,  11, 18, 24, 31, 40, 44, 53,
  10, 19, 23, 32, 39, 45, 52, 54,
  20, 22, 33, 38, 46, 51, 55, 60,
  21, 34, 37, 47, 50, 56, 59, 61,
  35, 36, 48, 49, 57, 58, 62, 63,
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

float GetQFactor(float q) {
  // we use the same mapping than jpeg-6b, for coherency
  q = (q <= 0) ? 5000 : (q < 50) ? 5000 / q : (q < 100) ? 2 * (100 - q) : 0;
  // We floor-round to integer here just to preserve compatibility with jpeg6b.
  return floorf(q);
}

void CopyQuantMatrix(const uint8_t in[64], uint8_t out[64]) {
  memcpy(out, in, 64 * sizeof(out[0]));
}

void SetQuantMatrix(const uint8_t in[64], float q_factor, uint8_t out[64]) {
  if (in == nullptr || out == nullptr) return;
  q_factor /= 100.f;
  for (size_t i = 0; i < 64; ++i) {
    const int v = static_cast<int>(in[i] * q_factor + .5f);
    // clamp to prevent illegal quantizer values
    out[i] = (v < 1) ? 1 : (v > 255) ? 255u : v;
  }
}

void SetMinQuantMatrix(const uint8_t m[64], uint8_t out[64], int tolerance) {
  assert(out != nullptr && m != nullptr);
  for (size_t i = 0; i < 64; ++i) {
    const int v = static_cast<int>(m[i] * (256 - tolerance) >> 8);
    out[i] = (v < 1) ? 1u : (v > 255) ? 255u : v;
  }
}

void SetDefaultMinQuantMatrix(uint8_t out[64]) {
  assert(out != nullptr);
  for (size_t i = 0; i < 64; ++i) out[i] = 1u;
}

////////////////////////////////////////////////////////////////////////////////

// FP_BITS (fractional precision of the fixed-point dividors) and AC_BITS
// (extra precision bits left by the fdct's scaling) live in sjpegi.h, because
// the histogram analysis needs them too.
#define BIAS_DC 0x80  // neutral bias for DC (mandatory!)

// divide-by-multiply helper macros
#define MAKE_INV_QUANT(Q) (((1u << FP_BITS) + (Q) / 2) / (Q))
#define DIV_BY_MULT(A, M) (((A) * (M)) >> FP_BITS)
#define QUANTIZE(A, M, B) (DIV_BY_MULT((A) + (B), (M)) >> AC_BITS)

// Fills bias_out[]/qthresh_out[] for a nominal 'bias' (DC and v==1 stay
// fixed regardless, see FinalizeQuantMatrix).
static void ComputeBiasTable(const Quantizer* const q, int bias,
                             uint16_t bias_out[64], uint16_t qthresh_out[64]) {
  const uint16_t bias_1 = 0x80;
  for (size_t i = 0; i < 64; ++i) {
    const uint16_t v = q->quant_[i];
    const uint16_t iquant = q->iquant_[i];
    const uint16_t b = (v == 1) ? bias_1 : (i == 0) ? BIAS_DC : bias;
    const uint16_t ibias = (((b * v) << AC_BITS) + 128) >> 8;
    const uint16_t qthresh =
        ((1 << (FP_BITS + AC_BITS)) + iquant - 1) / iquant - ibias;
    bias_out[i] = ibias;
    qthresh_out[i] = qthresh;
    assert(QUANTIZE(qthresh, iquant, ibias) > 0);
    assert(QUANTIZE(qthresh - 1, iquant, ibias) == 0);
  }
}

// Perceptual masking: bias offset between the mid and flat/busy tiers
// (0..255 scale). See ClassifyBlockActivity(). Tuned by eye, not PSNR/SSIM.
#define ADAPTIVE_BIAS_DELTA 32

void Encoder::FinalizeQuantMatrix(Quantizer* const q, int q_bias,
                                  bool adaptive) {
  // first, clamp the quant matrix:
  for (size_t i = 0; i < 64; ++i) {
    if (q->quant_[i] < q->min_quant_[i]) q->quant_[i] = q->min_quant_[i];
  }
  // Special case! for v=1 we can't represent the multiplier with 16b precision.
  // So, instead we max out the multiplier to 0xffffu, and twist the bias to the
  // value 0x80. The overall precision isn't affected: it's bit-exact the same
  // for our working range.
  // Note that quant=1 can start appearing at quality as low as 93.
  const uint16_t iquant_1 = 0xffffu;
  for (size_t i = 0; i < 64; ++i) {
    const uint16_t v = q->quant_[i];
    q->iquant_[i] = (v == 1) ? iquant_1 : MAKE_INV_QUANT(v);
  }
  ComputeBiasTable(q, q_bias, q->bias_, q->qthresh_);
  if (adaptive) {
    const int bias_flat = (q_bias + ADAPTIVE_BIAS_DELTA > 255)
                             ? 255 : q_bias + ADAPTIVE_BIAS_DELTA;
    const int bias_busy = (q_bias < ADAPTIVE_BIAS_DELTA)
                             ? 0 : q_bias - ADAPTIVE_BIAS_DELTA;
    uint16_t unused_bias[64];  // only the tiers' threshold is used
    ComputeBiasTable(q, bias_flat, unused_bias, q->qthresh_flat_);
    ComputeBiasTable(q, bias_busy, unused_bias, q->qthresh_busy_);
  }
}
#undef ADAPTIVE_BIAS_DELTA

void Encoder::SetCostCodes(int idx) {
  quants_[idx].codes_ = ac_codes_[idx];
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
  const __m128i zero = _mm_setzero_si128();
  uint64_t nzn = 0;  // natural-order non-zero mask: bit j set iff tmp[j] != 0.
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
    // Record which lanes are non-zero (F >= 0, so "> 0" == "!= 0"): pack the 8
    // per-lane 0xFFFF/0 compare results to one byte each and movemask to an
    // 8-bit chunk placed at bit offset i.
    const __m128i cmp = _mm_cmpgt_epi16(F, zero);
    const int m8 = _mm_movemask_epi8(_mm_packs_epi16(cmp, cmp)) & 0xff;
    nzn |= static_cast<uint64_t>(m8) << i;
  }
  // Emit run/level entries. Remap the non-zero AC set (drop DC = bit 0) from
  // natural to zig-zag order, then iterate set bits with 'ctz' so we touch only
  // the (few) non-zero coefficients: the classic zig-zag scan without the
  // data-dependent per-coefficient branch. Output is bit-identical.
  uint64_t zz = 0;
  for (uint64_t b = nzn & ~1ull; b != 0; b &= b - 1) {
    zz |= 1ull << kInvZigzag[TrailingZeros64(b)];
  }
  for (uint64_t b = zz; b != 0; b &= b - 1) {
    const int i = static_cast<int>(TrailingZeros64(b));
    const int j = kZigzag[i];
    const int n = CalcLog2(tmp[j]);
    const uint16_t code = masked[j] & ((1 << n) - 1);
    rl[nb].level_ = (code << 4) | n;
    rl[nb].run_ = i - prev;
    prev = i + 1;
    ++nb;
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
  uint64_t nzn = 0;  // natural-order non-zero mask: bit j set iff tmp[j] != 0.
  // Per-lane bit weights, used to turn a NEON compare result into a bitmask
  // (NEON has no movemask instruction).
  static const uint16_t kBitWeights[8] = {1, 2, 4, 8, 16, 32, 64, 128};
  const uint16x8_t weights = vld1q_u16(kBitWeights);
  for (int i = 0; i < 64; i += 8) {
    const uint16x8_t m_bias = vld1q_u16(bias + i);
    const uint16x8_t m_mult = vld1q_u16(iquant + i);
    const int16x8_t A = vld1q_s16(in + i);                           // in[i]
    const uint16x8_t B = vreinterpretq_u16_s16(vabsq_s16(A));        // abs(in)
    const int16x8_t sign = vshrq_n_s16(A, 15);                       // sign
    const uint16x8_t C = vaddq_u16(B, m_bias);                       // + bias
    const uint32x4_t D0 = vmull_u16(vget_low_u16(C), vget_low_u16(m_mult));
    const uint32x4_t D1 = vmull_u16(vget_high_u16(C), vget_high_u16(m_mult));
    // collect hi-words of the 32b mult result using 'unzip'
    const uint16x8x2_t E = vuzpq_u16(vreinterpretq_u16_u32(D0),
                                     vreinterpretq_u16_u32(D1));
    const uint16x8_t F = vshrq_n_u16(E.val[1], AC_BITS);
    const uint16x8_t G = veorq_u16(F, vreinterpretq_u16_s16(sign));  // v ^ mask
    vst1q_u16(tmp + i, F);
    vst1q_u16(masked + i, G);
    // Record which lanes are non-zero. vtstq(F, F) gives 0xFFFF where F != 0;
    // AND with the bit weights and horizontally add to an 8-bit chunk.
    const uint16x8_t nz = vandq_u16(vtstq_u16(F, F), weights);
#if defined(__aarch64__)
    const int m8 = vaddvq_u16(nz);
#else
    uint16x4_t s = vadd_u16(vget_low_u16(nz), vget_high_u16(nz));
    s = vpadd_u16(s, s);
    s = vpadd_u16(s, s);
    const int m8 = vget_lane_u16(s, 0);
#endif
    nzn |= static_cast<uint64_t>(m8) << i;
  }
  // Emit run/level entries. Remap the non-zero AC set (drop DC = bit 0) from
  // natural to zig-zag order, then iterate set bits with 'ctz' so we touch only
  // the (few) non-zero coefficients: the classic zig-zag scan without the
  // data-dependent per-coefficient branch. Output is bit-identical.
  uint64_t zz = 0;
  for (uint64_t b = nzn & ~1ull; b != 0; b &= b - 1) {
    zz |= 1ull << kInvZigzag[TrailingZeros64(b)];
  }
  for (uint64_t b = zz; b != 0; b &= b - 1) {
    const int i = static_cast<int>(TrailingZeros64(b));
    const int j = kZigzag[i];
    const int n = CalcLog2(tmp[j]);
    const uint16_t code = masked[j] & ((1 << n) - 1);
    rl[nb].level_ = (code << 4) | n;
    rl[nb].run_ = i - prev;
    prev = i + 1;
    ++nb;
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
    const int j = kZigzag[i];
    int v = in[j];
    const int32_t mask = v >> 31;
    v = (v ^ mask) - mask;
    if (v >= qthresh[j]) {
      v = QUANTIZE(v, iquant[j], bias[j]);
      assert(v > 0);
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
// Adaptive-bias quantization ("poor man's trellis").
//
// Two independent ideas, kept separate below:
// 1. Perceptual masking (ClassifyBlockActivity, ADAPTIVE_BIAS_DELTA in
//    FinalizeQuantMatrix): busy blocks mask error well, so quantize them
//    harder; flat blocks show banding easily, so protect them instead.
// 2. Bit-cost-correct rounding (AdaptiveBiasQuantizeBlock): JPEG's Huffman
//    code depends on (run, size), not the mantissa, so only the 0-vs-1
//    survival edge is worth re-rounding -- it's the only decision that
//    changes a coefficient's coded size. Every other coefficient is
//    bit-exact to plain QuantizeBlock().

// --- 1. perceptual masking: classify block activity into a tier ---

enum BlockActivityTier { kFlatBlock = -1, kNormalBlock = 0, kBusyBlock = 1 };

// Thresholds (AC_BITS-scaled) near the 10th/90th percentile of per-block AC
// activity on real photos, so only genuine outliers get reclassified.
// Heuristic, tune by eye (eval/ab_compare.py).
#define ACTIVITY_LO (28 << AC_BITS)
#define ACTIVITY_HI (360 << AC_BITS)

static BlockActivityTier ClassifyBlockActivity(const int16_t in[64]) {
  uint32_t activity = 0;
  for (int i = 1; i < 64; ++i) {
    const int v = in[i];
    activity += (v < 0) ? -v : v;
  }
  return (activity < ACTIVITY_LO) ? kFlatBlock
       : (activity > ACTIVITY_HI) ? kBusyBlock
                                   : kNormalBlock;
}
#undef ACTIVITY_LO
#undef ACTIVITY_HI

// --- 2. bit-cost-correct rounding, gated by the tier's survival test ---

int Encoder::AdaptiveBiasQuantizeBlock(const int16_t in[64], int idx,
                                       const Quantizer* const Q,
                                       DCTCoeffs* const out,
                                       RunLevel* const rl) {
  const BlockActivityTier tier = ClassifyBlockActivity(in);
  const uint16_t* const survive_thresh =
      (tier == kFlatBlock) ? Q->qthresh_flat_
    : (tier == kBusyBlock) ? Q->qthresh_busy_
                            : Q->qthresh_;
  const uint16_t* const bias = Q->bias_;      // always nearest-rounding
  const uint16_t* const qthresh = Q->qthresh_;  // mid survival test
  const uint16_t* const iquant = Q->iquant_;
  int prev = 1;
  int nb = 0;
  for (int i = 1; i < 64; ++i) {
    const int j = kZigzag[i];
    int v = in[j];
    const int32_t mask = v >> 31;
    v = (v ^ mask) - mask;
    if (v < survive_thresh[j]) continue;
    // At or above the mid threshold: standard nearest-rounding, same as
    // QuantizeBlock(). Below it (flat tier only, since busy's threshold is
    // stricter than mid's): floor to the smallest nonzero level instead of
    // letting it vanish.
    v = (v >= qthresh[j]) ? QUANTIZE(v, iquant[j], bias[j]) : 1;
    assert(v > 0);
    const int n = CalcLog2(v);
    const uint16_t code = (v ^ mask) & ((1 << n) - 1);
    rl[nb].level_ = (code << 4) | n;
    rl[nb].run_ = i - prev;
    prev = i + 1;
    ++nb;
  }
  const int dc = (in[0] < 0) ? -QUANTIZE(-in[0], iquant[0], bias[0])
                             : QUANTIZE(in[0], iquant[0], bias[0]);
  out->idx_ = idx;
  out->last_ = prev - 1;
  out->nb_coeffs_ = nb;
  return dc;
}

Encoder::QuantizeBlockFunc Encoder::GetActiveQuantizeBlockFunc() const {
  if (use_trellis_) return TrellisQuantizeBlock;
  if (adaptive_bias_) return AdaptiveBiasQuantizeBlock;
  return quantize_block_;
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

  TrellisNode() : score(kMaxScore), best_prev(nullptr) {}
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
  // Careful: loop overwrites node->disto, so compute this before it runs.
  const uint32_t base_disto = node->disto + disto0[node->pos - 1];
  for (const TrellisNode* cur = node - 1; cur >= nodes0; --cur) {
    const int run = node->pos - 1 - cur->pos;
    if (run < 0) continue;
    uint32_t bits = node->nbits;
    bits += (run >> 4) * (codes[0xf0] & 0xff);
    const uint32_t disto = base_disto - disto0[cur->pos];
    // Exact early-out: walking back towards the sink only grows the run, so both
    // disto and the ZRL part of bits are monotone here, and the two terms left
    // out -- symbol's code length and cur->score -- are non-negative. Once the
    // bound reaches the incumbent, nothing left can win.
    if (disto + lambda * bits >= node->score) break;
    const uint32_t sym = ((run & 15) << 4) | node->nbits;
    assert(codes[sym] != 0);
    bits += codes[sym] & 0xff;
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

int Encoder::TrellisQuantizeBlock(const int16_t in[64], int idx,
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
    const int j = kZigzag[i];
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

#if defined(SJPEG_HAVE_AVX2)
// defined in quantize_avx2.cc, built separately with -mavx2 (see Makefile)
// so this file itself doesn't need an AVX2 target.
extern int QuantizeBlockAVX2(const int16_t in[64], int idx,
                             const Quantizer* const Q, DCTCoeffs* const out,
                             RunLevel* const rl);
#endif

Encoder::QuantizeBlockFunc Encoder::GetQuantizeBlockFunc() {
#if defined(SJPEG_HAVE_AVX2)
  if (SupportsAVX2()) return QuantizeBlockAVX2;
#endif
#if defined(SJPEG_USE_SSE2)
  if (SupportsSSE2()) return QuantizeBlockSSE2;
#elif defined(SJPEG_USE_NEON)
  if (SupportsNEON()) return QuantizeBlockNEON;
#endif
  return QuantizeBlock;  // default
}

////////////////////////////////////////////////////////////////////////////////

#if defined(SJPEG_USE_SSE2)
// Load eight 16b-words from *src.
#define LOAD_16(src) _mm_loadu_si128((const __m128i*)(src))
#define LOAD_64(src) _mm_loadl_epi64((const __m128i*)(src))
// Store eight 16b-words into *dst
#define STORE_16(V, dst) _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), (V))

static uint32_t QuantizeErrorSSE2(const int16_t in[64],
                                  const Quantizer* const Q) {
  const uint16_t* const bias = Q->bias_;
  const uint16_t* const iquant = Q->iquant_;
  const uint8_t* const quant = Q->quant_;
  const __m128i zero = _mm_setzero_si128();
  uint32_t tmp[32];
  for (int i = 0; i < 64; i += 8) {
    const __m128i m_bias = LOAD_16(bias + i);
    const __m128i m_iquant = LOAD_16(iquant + i);
    const __m128i m_quant = _mm_unpacklo_epi8(LOAD_64(quant + i), zero);
    const __m128i A = LOAD_16(in + i);                        // v0 = in[i]
    const __m128i B = _mm_srai_epi16(A, 15);                  // sign extract
    const __m128i C = _mm_sub_epi16(_mm_xor_si128(A, B), B);  // abs(v0)
    const __m128i D = _mm_adds_epi16(C, m_bias);              // v' = v0 + bias
    const __m128i E = _mm_mulhi_epu16(D, m_iquant);           // (v' * iq) >> 16
    const __m128i F = _mm_srai_epi16(E, AC_BITS);
    const __m128i G = _mm_srai_epi16(C, AC_BITS);
    const __m128i H = _mm_mullo_epi16(F, m_quant);            // *= quant[j]
    const __m128i I = _mm_sub_epi16(G, H);
    const __m128i J = _mm_madd_epi16(I, I);                   // (v0-v) ^ 2
    STORE_16(J, tmp + i / 2);
  }
  uint32_t err = 0;
  for (int i = 0; i < 32; ++i) err += tmp[i];
  return err;
}
#undef LOAD_16
#undef LOAD_64
#undef STORE_16

#elif defined(SJPEG_USE_NEON)

static uint32_t QuantizeErrorNEON(const int16_t in[64],
                                  const Quantizer* const Q) {
  const uint16_t* const bias = Q->bias_;
  const uint16_t* const iquant = Q->iquant_;
  const uint8_t* const quant = Q->quant_;
  uint32x4_t sum1 = vdupq_n_u32(0);
  uint32x4_t sum2 = vdupq_n_u32(0);
  for (int i = 0; i < 64; i += 8) {
    const uint16x8_t m_bias = vld1q_u16(bias + i);
    const uint16x8_t m_mult = vld1q_u16(iquant + i);
    const uint16x8_t m_quant = vmovl_u8(vld1_u8(quant + i));
    const uint16x8_t A = vreinterpretq_u16_s16(vabsq_s16(vld1q_s16(in + i)));
    const uint16x8_t B = vaddq_u16(A, m_bias);
    const uint32x4_t C0 = vmull_u16(vget_low_u16(B), vget_low_u16(m_mult));
    const uint32x4_t C1 = vmull_u16(vget_high_u16(B), vget_high_u16(m_mult));
    // collect hi-words of the 32b mult result using 'unzip'
    const uint16x8x2_t D = vuzpq_u16(vreinterpretq_u16_u32(C0),
                                     vreinterpretq_u16_u32(C1));
    const uint16x8_t E = vshrq_n_u16(D.val[1], AC_BITS);
    const uint16x8_t F = vmulq_u16(E, m_quant);        // dequantized coeff
    const uint16x8_t G = vabdq_u16(F, vshrq_n_u16(A, AC_BITS));
    sum1 = vmlal_u16(sum1, vget_low_u16(G), vget_low_u16(G));
    sum2 = vmlal_u16(sum2, vget_high_u16(G), vget_high_u16(G));
  }
  const uint32x4_t sum3 = vaddq_u32(sum1, sum2);
#if defined(SJPEG_AARCH64)
  const uint32_t err = vaddvq_u32(sum3);
#else
  const uint64x2_t sum4 = vpaddlq_u32(sum3);
  const uint64_t sum5 = vgetq_lane_u64(sum4, 0) + vgetq_lane_u64(sum4, 1);
  const uint32_t err = (uint32_t)sum5;
#endif
  return err;
}

#endif    // SJPEG_USE_NEON

static uint32_t QuantizeError(const int16_t in[64], const Quantizer* const Q) {
  const uint16_t* const bias = Q->bias_;
  const uint16_t* const iquant = Q->iquant_;
  const uint8_t* const quant = Q->quant_;
  uint32_t err = 0;
  for (int j = 0; j < 64; ++j) {
    int32_t v0 = (in[j] < 0) ? -in[j] : in[j];
    const uint32_t v = quant[j] * QUANTIZE(v0, iquant[j], bias[j]);
    v0 >>= AC_BITS;
    err += (v0 - v) * (v0 - v);
  }
  return err;
}

Encoder::QuantizeErrorFunc Encoder::GetQuantizeErrorFunc() {
#if defined(SJPEG_USE_SSE2)
  if (SupportsSSE2()) return QuantizeErrorSSE2;
#elif defined(SJPEG_USE_NEON)
  if (SupportsNEON()) return QuantizeErrorNEON;
#endif
  return QuantizeError;  // default
}

}    // namespace sjpeg
