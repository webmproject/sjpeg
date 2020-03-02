// Copyright 2018 Google Inc.
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
//  header and chunks writing
//
// Author: Skal (pascal.massimino@gmail.com)

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <algorithm>

#include "sjpegi.h"
#include "md5sum.h"

namespace sjpeg {

void Encoder::Put16b(uint32_t size) {
  bw_.PutByte((size >> 8) & 0xff);
  bw_.PutByte((size >> 0) & 0xff);
}

void Encoder::Put32b(uint32_t size) {
  Put16b(size >> 16);
  Put16b(size >>  0);
}

////////////////////////////////////////////////////////////////////////////////
// Headers
//
// NOTE(skal): all chunks start with a startcode '0xff??' (0xffd8 e.g),
// followed by the size of the payload *not counting the startcode*!
// That's why you often find these 'Reserve(data_size + 2)' below, the '+2'
// accounting for the 0xff?? startcode size.

static const uint8_t kHeaderAPP0[] = {
  0xff, 0xd8,                     // SOI
  0xff, 0xe0, 0x00, 0x10,         // APP0
  0x4a, 0x46, 0x49, 0x46, 0x00,   // 'JFIF'
  0x01, 0x01,                     // v1.01
  0x00, 0x00, 0x01, 0x00, 0x01,   // aspect ratio = 1:1
  0x00, 0x00                      // thumbnail width/height
};

void Encoder::WriteAPP0() {  // SOI + APP0
  ok_ = ok_ && bw_.Reserve(sizeof(kHeaderAPP0));
  if (!ok_) return;
  bw_.PutBytes(kHeaderAPP0, sizeof(kHeaderAPP0));
}

bool Encoder::WriteAPPMarkers(const std::string& data) {
  if (data.size() == 0) return true;
  const size_t data_size = data.size();
  ok_ = ok_ && bw_.Reserve(data_size);
  if (!ok_) return false;
  bw_.PutBytes(reinterpret_cast<const uint8_t*>(data.data()), data.size());
  return true;
}

bool Encoder::WriteEXIF(const std::string& data) {
  if (data.size() == 0) return true;
  const uint8_t kEXIF[] = "Exif\0";
  const size_t kEXIF_len = 6;  // includes the \0's
  const size_t data_size = data.size() + kEXIF_len + 2;
  if (data_size > 0xffff) return false;
  ok_ = ok_ && bw_.Reserve(data_size + 2);
  if (!ok_) return false;
  Put16b(0xffe1);
  Put16b(data_size);
  bw_.PutBytes(kEXIF, kEXIF_len);
  bw_.PutBytes(reinterpret_cast<const uint8_t*>(data.data()), data.size());
  return true;
}

bool Encoder::WriteICCP(const std::string& data) {
  if (data.size() == 0) return true;
  size_t data_size = data.size();
  const uint8_t* ptr = reinterpret_cast<const uint8_t*>(data.data());
  const uint8_t kICCP[] = "ICC_PROFILE";
  const size_t kICCP_len = 12;  // includes the \0
  const size_t kMaxChunkSize = 0xffff - kICCP_len - 4;
  size_t max_chunk = (data_size + kMaxChunkSize - 1) / kMaxChunkSize;
  if (max_chunk >= 256) return false;
  size_t seq = 1;
  while (data_size > 0) {
    const size_t size = std::min(kMaxChunkSize, data_size);
    const size_t total_size = size + kICCP_len + 4;
    ok_ = ok_ && bw_.Reserve(total_size + 2);
    if (!ok_) return false;
    Put16b(0xffe2);
    Put16b(total_size);
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

bool Encoder::WriteXMPExtended(const std::string& data) {
  const size_t kMainSize = 65503;
  if (data.size() < kMainSize) return true;  // too short! should be a main XMP
  if (data.size() > (1u << 31)) return false;   // too large
  size_t split = (xmp_split_ == 0) ? kMainSize : xmp_split_;
  split = std::min(split, data.size());
  // search for the extension tag
  const size_t note_pos = data.find(std::string("xmpNote:HasExtendedXMP=\""));
  if (note_pos == std::string::npos) return false;   // no extension!
  if (note_pos + 24 + 32 + 1 > split) return false;  // ill-formed
  if (data[note_pos + 24 + 32] != '\"') return false;
  // split in main / extension
  std::string main_data(&data[0], split);
  std::string ext_data(&data[split], data.size() - split);
  // compute GUID
  uint8_t* const guid = reinterpret_cast<uint8_t*>(&main_data[note_pos + 24]);
  MD5Digest(ext_data).Get(guid);

  // Main chunk.
  if (!WriteXMP(main_data)) return false;

  // Extended chunks.
  const uint8_t kXMPExt[] = "http://ns.adobe.com/xmp/extension/";
  const size_t kXMPExt_size = sizeof(kXMPExt);
  const size_t kBufSize = 65458;

  // 40 = 32 bytes for GUID + 4 bytes for size + 4 bytes for position
  const size_t kHeaderSize = kXMPExt_size + 40;
  const size_t num_chunks = ext_data.size() / kBufSize + 1;
  const size_t data_size = num_chunks * (kHeaderSize + 2 + 2) + ext_data.size();
  ok_ = ok_ && bw_.Reserve(data_size);
  if (!ok_) return false;
  size_t read_pos = 0, write_pos = 0;
  for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
    const uint32_t write_size = std::min(kBufSize, ext_data.size() - read_pos);
    Put16b(0xffe1);  // APP1
    Put16b(2 + kHeaderSize + write_size);
    bw_.PutBytes(kXMPExt, kXMPExt_size);
    bw_.PutBytes(guid, 32u);
    Put32b(ext_data.size());  // total size, not chunk size!
    Put32b(read_pos);
    bw_.PutBytes(reinterpret_cast<const uint8_t*>(&ext_data[read_pos]),
                 write_size);
    read_pos += write_size;
    write_pos += 2 + (2 + kHeaderSize + write_size);
  }
  return (write_pos == data_size);
}

bool Encoder::WriteXMP(const std::string& data) {
  if (data.size() == 0) return true;
  const uint8_t kXMP[] = "http://ns.adobe.com/xap/1.0/";
  const size_t kXMP_size = sizeof(kXMP);
  const size_t data_size = 2 + data.size() + kXMP_size;
  if (data_size <= 0xffff) {   // don't use extended XMP if small enough data
  ok_ = ok_ && bw_.Reserve(data_size + 2);
  if (!ok_) return false;
    Put16b(0xffe1);
    Put16b(data_size);
  bw_.PutBytes(kXMP, kXMP_size);
  bw_.PutBytes(reinterpret_cast<const uint8_t*>(data.data()), data.size());
  return true;
  }
  // need to split into main-chunk + extended sections
  return WriteXMPExtended(data);
}

void Encoder::WriteDQT() {
  const int num_matrices = (yuv_mode_ == SJPEG_YUV_400) ? 1 : 2;
  const size_t data_size = num_matrices * 65 + 2;
  const uint8_t kDQTHeader[] = { 0xff, 0xdb, 0x00, (uint8_t)data_size };
  ok_ = ok_ && bw_.Reserve(data_size + 2);
  if (!ok_) return;
  bw_.PutBytes(kDQTHeader, sizeof(kDQTHeader));
  for (int n = 0; n < num_matrices; ++n) {
    bw_.PutByte(n);
    const uint8_t* quant = quants_[n].quant_;
    for (int i = 0; i < 64; ++i) {
      bw_.PutByte(quant[kZigzag[i]]);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

#define DATA_16b(X) ((uint8_t)((X) >> 8)), ((uint8_t)((X) & 0xff))

void Encoder::WriteSOF() {   // SOF
  const size_t data_size = 3 * nb_comps_ + 8;
  assert(data_size <= 255);
  const uint8_t kHeader[] = {
    0xff, 0xc0, DATA_16b(data_size),         // SOF0 marker, size
    0x08,                                    // 8bits/components
    DATA_16b(H_), DATA_16b(W_),              // height, width
    (uint8_t)nb_comps_                       // number of components
  };
  ok_ = ok_ && bw_.Reserve(data_size + 2);
  if (!ok_) return;
  bw_.PutBytes(kHeader, sizeof(kHeader));
  for (int c = 0; c < nb_comps_; ++c) {
    bw_.PutByte(c + 1);
    bw_.PutByte(block_dims_[c]);
    bw_.PutByte(quant_idx_[c]);
  }
}

void Encoder::WriteDHT() {
  InitCodes(false);
  const int nb_tables = (nb_comps_ == 1 ? 1 : 2);
  for (int c = 0; c < nb_tables; ++c) {   // luma, chroma
    for (int type = 0; type <= 1; ++type) {               // dc, ac
      const HuffmanTable* const h = Huffman_tables_[type * 2 + c];
      const size_t data_size = 3 + 16 + h->nb_syms_;
      assert(data_size <= 255);
      ok_ = ok_ && bw_.Reserve(data_size + 2);
      if (!ok_) return;
      Put16b(0xffc4);
      Put16b(data_size);
      bw_.PutByte((type << 4) | c);
      bw_.PutBytes(h->bits_, 16);
      bw_.PutBytes(h->syms_, h->nb_syms_);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

void Encoder::WriteSOS() {   // SOS
  const size_t data_size = 3 + nb_comps_ * 2 + 3;
  assert(data_size <= 255);
  const uint8_t kHeader[] = {
      0xff, 0xda, DATA_16b(data_size), (uint8_t)nb_comps_
  };
  ok_ = ok_ && bw_.Reserve(data_size + 2);
  if (!ok_) return;
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

void Encoder::WriteEOI() {   // EOI
  if (ok_) bw_.Flush();
  ok_ = ok_ && bw_.Reserve(2);
  if (!ok_) return;
  // append EOI
  Put16b(0xffd9);
}

////////////////////////////////////////////////////////////////////////////////

#undef DATA_16b

}    // namespace sjpeg
