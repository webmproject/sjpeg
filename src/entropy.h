// Copyright 2026 Google LLC
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
//  Entropy coding: symbol statistics, Huffman tables and coefficient coding.
//  Internal header: not installed, only included from sjpegi.h.
//
// Author: Skal (pascal.massimino@gmail.com)

#ifndef SJPEG_ENTROPY_H_
#define SJPEG_ENTROPY_H_

#include <stddef.h>
#include <stdint.h>

#include "bit_writer.h"

namespace sjpeg {

struct DCTCoeffs;
struct RunLevel;

// Huffman tables, as stored in a DHT segment.
struct HuffmanTable {
  uint8_t bits_[16];     // number of symbols per bit count
  const uint8_t* syms_;  // symbol map, in increasing bit length
  int nb_syms_;          // cached value of sum(bits_[])
};

// Symbol frequencies of the (up to) 2 quantization tables' worth of blocks:
// index 0 is luma, 1 is chroma.
// Symbol counts are 64-bit: in 4:4:4, a 65535x65535 picture's chroma table
// can see 2 x 67M blocks x 63 = 8.5G uses of a single symbol.
typedef uint64_t SymbolCount;

struct EntropyStats {
  SymbolCount ac[2][256 + 1];  // AC symbols
  SymbolCount dc[2][12 + 1];   // DC categories

  void Reset();
  // Sums 'other' into *this, for the first nb_tables tables.
  void Add(const EntropyStats& other, int nb_tables);
  // Tallies one block.
  void AddBlock(int q_idx, const DCTCoeffs* coeffs, const RunLevel* run_levels);
  void AddDC(int q_idx, const DCTCoeffs* coeffs);  // DC only
};

// Holds the Huffman tables in use (standard or optimized), and the codes
// derived from them.
class EntropyCoder {
 public:
  // Number of tables (luma, chroma) in use: 1 for grayscale, 2 otherwise.
  explicit EntropyCoder(int nb_tables);
  int nb_tables() const { return nb_tables_; }

  // Points the tables at the standard ones of JPEG section K.3.
  void SetDefaultTables();
  // Builds the codes from the current tables.
  void InitCodes(bool only_ac);
  // Replaces the tables by the optimal ones for the given statistics.
  // Call InitCodes() afterwards to get the codes.
  void Compile(const EntropyStats& stats);
  // Size, in bits, of the entropy-coded data described by 'stats'. Same total
  // as the sum of BlocksSize(), minus uncounted 0xff byte-stuffing.
  uint64_t Size(const EntropyStats& stats) const;

  // type: 0 = DC, 1 = AC. q_idx: 0 = luma, 1 = chroma.
  const HuffmanTable* table(int type, int q_idx) const {
    return tables_[type * 2 + q_idx];
  }
  const uint32_t* ac_codes(int q_idx) const { return ac_codes_[q_idx]; }
  const uint32_t* dc_codes(int q_idx) const { return dc_codes_[q_idx]; }

  void CodeBlock(const DCTCoeffs* coeffs, const RunLevel* rl, int q_idx,
                 BitWriter* bw) const;

#if !defined(SJPEG_NO_PROGRESSIVE)
  // Progressive mode: DC coding matches baseline, with optimal tables built
  // straight from the statistics (table and codes).
  void CompileDC(const EntropyStats& stats);
  const HuffmanTable* opt_table_dc(int q_idx) const {
    return &opt_tables_dc_[q_idx];
  }
  void CodeBlockDC(const DCTCoeffs* coeffs, int q_idx, BitWriter* bw) const;
#endif

 private:
  const int nb_tables_;
  const HuffmanTable* tables_[4];  // indices: dc0, dc1, ac0, ac1
  uint32_t ac_codes_[2][256];
  uint32_t dc_codes_[2][12];
  uint8_t opt_syms_ac_[2][256];  // optimal table for AC symbols
  uint8_t opt_syms_dc_[2][12];   // optimal table for DC symbols
  HuffmanTable opt_tables_ac_[2];
  HuffmanTable opt_tables_dc_[2];
};

#if !defined(SJPEG_NO_PROGRESSIVE)
// Progressive mode: AC needs its own per-scan table, since each spectral
// window has a different symbol alphabet.
struct ProgACCoder {
  SymbolCount freq[256 + 1];
  uint32_t codes[256];
  uint8_t opt_syms[256];
  HuffmanTable table;

  void Reset();
  // Stats gathering, for a block's run/levels in the window, or an EOB run.
  void AddWindowed(const RunLevel* windowed, int nb_windowed);
  void AddEOBRun(int run);
  // Builds the optimal table and its codes from the stats.
  void Compile();
  void CodeWindowed(const RunLevel* windowed, int nb_windowed,
                    BitWriter* bw) const;
  void CodeEOBRun(int run, BitWriter* bw) const;

  // Re-derives (run,level) for window [Ss,Se] from the block's full RunLevel
  // list. Returns the count, and whether the window ends with an EOB.
  static int WindowRunLevels(const DCTCoeffs* coeffs, const RunLevel* rl,
                             int Ss, int Se, RunLevel* out, bool* out_has_eob);
};
#endif  // !SJPEG_NO_PROGRESSIVE

}  // namespace sjpeg

#endif  // SJPEG_ENTROPY_H_
