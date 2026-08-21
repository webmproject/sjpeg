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
//  Entropy coding: Huffman tables, and coefficient bitstream
//
// Author: Skal (pascal.massimino@gmail.com)

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sjpegi.h"

namespace sjpeg {

////////////////////////////////////////////////////////////////////////////////
// standard Huffman tables, as per JPEG standard section K.3.

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

void Encoder::SetDefaultHuffmanTables() {
  for (int i = 0; i < 4; ++i) Huffman_tables_[i] = &kHuffmanTables[i];
}

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

void Encoder::InitCodes(bool only_ac) {
  const int nb_tables = (nb_comps_ == 1 ? 1 : 2);
  for (int c = 0; c < nb_tables; ++c) {   // luma, chroma
    for (int type = (only_ac ? 1 : 0); type <= 1; ++type) {
      const HuffmanTable* const h = Huffman_tables_[type * 2 + c];
      const int nb_syms = BuildHuffmanTable(h->bits_, h->syms_,
                                            type == 1 ? ac_codes_[c]
                                                      : dc_codes_[c]);
      assert(nb_syms == h->nb_syms_);
      (void)nb_syms;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// DC coefficients

uint16_t Encoder::GenerateDCDiffCode(int DC, int* const DC_predictor) {
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
  assert((suff & 0xf000) == 0);
  assert(n < 12);
  return n | (suff << 4);
}

////////////////////////////////////////////////////////////////////////////////
// Code bitstream

void Encoder::ResetDCs() {
  for (int c = 0; c < nb_comps_; ++c) {
    DCs_[c] = 0;
  }
}

void Encoder::CodeBlock(const DCTCoeffs* const coeffs,
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
    // n is the magnitude category of a non-zero coefficient, so it is >= 1
    // here. The zero case is only reachable through the ZRL escape above.
    assert(n > 0);
#if defined(SJPEG_HAVE_64BIT)
    bw_.PutPackedCodeAndSuffix(codes[sym], suffix >> 4, n);
#else
    bw_.PutPackedCode(codes[sym]);
    bw_.PutBits(suffix >> 4, n);
#endif
  }
  if (coeffs->last_ < 63) {     // EOB
    bw_.PutPackedCode(codes[0x00]);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Huffman tables optimization

void Encoder::ResetEntropyStats() {
  memset(freq_ac_, 0, sizeof(freq_ac_));
  memset(freq_dc_, 0, sizeof(freq_dc_));
}

void Encoder::AddEntropyStats(const DCTCoeffs* const coeffs,
                              const RunLevel* const run_levels) {
  // freq_ac_[] and freq_dc_[] cannot overflow 32bits, since the maximum
  // resolution allowed is 65535 * 65535. The sum of all frequencies cannot
  // be greater than 32bits, either.
  const int idx = coeffs->idx_;
  const int q_idx = quant_idx_[idx];
  for (int i = 0; i < coeffs->nb_coeffs_; ++i) {
    const int run = run_levels[i].run_;
    const int tmp = (run >> 4);
    if (tmp) freq_ac_[q_idx][0xf0] += tmp;  // count escapes (all at once)
    const int suffix = run_levels[i].level_;
    const int sym = ((run & 0x0f) << 4) | (suffix & 0x0f);
    ++freq_ac_[q_idx][sym];
  }
  if (coeffs->last_ < 63) {     // EOB
    ++freq_ac_[q_idx][0x00];
  }
  ++freq_dc_[q_idx][coeffs->dc_code_ & 0x0f];
}

// Same total as BlocksSize(), minus uncounted 0xff byte-stuffing
size_t Encoder::EntropySize() const {
  size_t size = 0;
  const int nb_tables = (nb_comps_ == 1) ? 1 : 2;
  for (int q = 0; q < nb_tables; ++q) {
    for (int len = 0; len < 12; ++len) {
      const uint32_t freq = freq_dc_[q][len];
      if (freq > 0) size += freq * ((dc_codes_[q][len] & 0xff) + len);
    }
    const uint32_t* const codes = ac_codes_[q];
    for (int sym = 0; sym < 256; ++sym) {
      const uint32_t freq = freq_ac_[q][sym];
      if (freq > 0) size += freq * ((codes[sym] & 0xff) + (sym & 0x0f));
    }
  }
  return size;
}

static int cmp(const void *pa, const void *pb) {
  const uint64_t a = *reinterpret_cast<const uint64_t*>(pa);
  const uint64_t b = *reinterpret_cast<const uint64_t*>(pb);
  assert(a != b);  // tie-breaks can't happen
  return (a < b) ? 1 : -1;
}

static void BuildOptimalTable(HuffmanTable* const t,
                              const uint32_t* const freq, int size) {
  enum { MAX_BITS = 32, MAX_CODE_SIZE = 16 };
  assert(size <= 256);
  assert(t != nullptr);

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
      // we pack the sorted key (32bits) and index (9bits) into a single
      // uint64_t, so we don't have to resort to structs (and we avoid
      // tie-breaks, too)
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
    assert(s <= codesizes[size]);    // symbol #size is the biggest one.
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
  assert(position == nb_syms);

  // Now, we can ventilate the symbols directly to their final slice in the
  // partitioning, according to the their bit-length.
  // Note: we omit the last symbol, which is fake.
  uint8_t* const syms = const_cast<uint8_t*>(t->syms_);
  // Note that we loop til symbol = size-1, hence omitting the last fake symbol.
  for (int symbol = 0; symbol < size; ++symbol) {
    const int s = codesizes[symbol];
    if (s > 0) {
      assert(s <= MAX_BITS);
      syms[start[s - 1]++] = symbol;
    }
  }
  assert(start[max_bit_size - 1] == nb_syms - 1);

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
    assert(max_bit_size > 0);
  }
  --bits[max_bit_size];

  // update table with final book
  for (int i = 0; i < MAX_CODE_SIZE; ++i) {
    t->bits_[i] = bits[i];
  }
}

void Encoder::CompileEntropyStats() {
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

}    // namespace sjpeg
