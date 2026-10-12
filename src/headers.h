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
//  JPEG header / marker segments writer.
//  Internal header: not installed, only included from sjpegi.h.
//
// Author: Skal (pascal.massimino@gmail.com)

#ifndef SJPEG_HEADERS_H_
#define SJPEG_HEADERS_H_

#include <stddef.h>
#include <stdint.h>

#include <string>

#include "bit_writer.h"
#include "entropy.h"

namespace sjpeg {

struct Quantizer;

// Writes the marker segments to a BitWriter. Errors (allocation failure,
// oversized payload) are reported through the shared 'ok' flag, which is the
// owning encoder's own status.
class HeaderWriter {
 public:
  HeaderWriter(BitWriter* bw, bool* ok) : bw_(*bw), ok_(*ok) {}

  struct ScanComponent {
    int comp_idx;     // 0, 1 or 2
    int dc_table_id;  // Huffman DC table selector for this component
    int ac_table_id;  // Huffman AC table selector for this component
  };

  void WriteAPP0();  // SOI + APP0
  // Raw user APP markers, written as is.
  bool WriteAPPMarkers(const std::string& data);
  bool WriteEXIF(const std::string& data);
  bool WriteICCP(const std::string& data);
  // 'xmp_split' is the optional split point for extended XMP (0 = default).
  bool WriteXMP(const std::string& data, uint16_t xmp_split);
  void WriteDQT(const Quantizer* quants, int num_matrices);
  void WriteSOF(int W, int H, int nb_comps, const uint8_t* block_dims,
                const int* quant_idx, bool progressive);
  void WriteDRI(int interval_mcus);  // no-op if interval_mcus <= 0
  // Writes one Huffman table as its own DHT segment (table_class: 0=DC, 1=AC).
  void WriteOneDHT(int table_class, int table_id, const HuffmanTable* table);
  // Writes all the Huffman tables in use by 'coder' (luma, chroma; DC, AC).
  void WriteDHT(const EntropyCoder& coder);
  // Baseline SOS: all components, with the tables of their quantizer index.
  void WriteSOS(int nb_comps, const int* quant_idx);
  // Writes one SOS for the given component list and spectral range.
  void WriteSOS(const ScanComponent* comps, int nb_comps, int Ss, int Se);
  void WriteEOI();

 private:
  bool Reserve(size_t size);
  void Put16b(uint32_t size);
  void Put32b(uint32_t size);
  bool WriteXMPExtended(const std::string& data, uint16_t xmp_split);

  BitWriter& bw_;
  bool& ok_;
};

}  // namespace sjpeg

#endif  // SJPEG_HEADERS_H_
