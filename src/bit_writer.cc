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
//  Utility for writing bits
//
// Author: Skal (pascal.massimino@gmail.com)

#include "bit_writer.h"

#include <string.h>

namespace sjpeg {

///////////////////////////////////////////////////////////////////////////////
// BitWriter

BitWriter::BitWriter() : buf_(NULL), max_pos_(0) {
  Reset(0);
}

BitWriter::BitWriter(size_t output_size_hint) : buf_(NULL), max_pos_(0) {
  Reset(0);
  Reserve(output_size_hint);
}

BitWriter::~BitWriter() {
  delete[] buf_;
}

void BitWriter::GrowBuffer(size_t max_size) {
  DCHECK(max_size > max_pos_);
  // TODO(skal): the x2 growth is probably over-shooting. Need to tune
  // depending on use-case (ie.: what is the expected average final size?)
  max_size += 256;
  if (max_size < 2 * max_pos_) {
    max_size = 2 * max_pos_;
  }
  uint8_t* const new_buf = new uint8_t[max_size];
  if (byte_pos_ > 0) {
    memcpy(new_buf, buf_, byte_pos_);
  }
  delete[] buf_;
  buf_ = new_buf;
  max_pos_ = max_size;
}

uint8_t* BitWriter::Grab(size_t *size) {
  DCHECK(size != NULL);
  uint8_t *buf = buf_;
  *size = byte_pos_;
  buf_ = NULL;
  max_pos_ = 0;
  Reset(0);
  return buf;
}

void BitWriter::DeleteOutputBuffer() {
  delete[] buf_;
  buf_ = NULL;
  max_pos_ = 0;
  Reset(0);
}

void BitWriter::Reset(size_t byte_pos) {
  nb_bits_ = 0;
  bits_ = 0x00000000U;
  byte_pos_ = byte_pos;
}

void BitWriter::Flush() {
  // align and pad the bitstream
  // nb_pad is the number of '1' bits we need to insert to reach a byte-aligned
  // position. So nb_pad is 1,2..,7 when nb_bits_= 7,...2,1
  const int nb_pad = (-nb_bits_) & 7;
  if (nb_pad) {
    PutBits((1 << nb_pad) - 1, nb_pad);
  }
  FlushBits();
}

///////////////////////////////////////////////////////////////////////////////

void BitCounter::AddBits(const uint32_t bits, size_t nbits) {
  size_ += nbits;
  bit_pos_ += nbits;
  bits_ |= bits << (32 - bit_pos_);
  while (bit_pos_ >= 8) {
    size_ += ((bits_ >> 24) == 0xff) ? 8 : 0;
    bits_ <<= 8;
    bit_pos_ -= 8;
  }
}

}   // namespace jpeg
