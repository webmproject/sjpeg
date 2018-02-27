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

#ifndef SJPEG_BIT_WRITER_H_
#define SJPEG_BIT_WRITER_H_

#include <assert.h>
#include <string.h>
#include <stdint.h>

namespace sjpeg {

///////////////////////////////////////////////////////////////////////////////
// BitWriter

class BitWriter {
 public:
  BitWriter();
  explicit BitWriter(size_t output_size_hint);
  ~BitWriter();

  // Restart writing to the beginning of the output buffer.
  void Reset() { Reset(0); }
  // Restart writing at a fixed byte position (leaving the beginning
  // of the buffer intact).
  void Reset(size_t byte_pos);

  // Verifies the that output buffer can store at least 'size' more bytes,
  // growing if needed. Returns the buffer position with 'size' writable bytes.
  // Prefer calling this function instead of accessing output_ directly!
  // The returned pointer is likely to change if Reserve() is called again.
  // Hence it should be used as quickly as possible.
  uint8_t* Reserve(size_t size) {
    if (byte_pos_ + size > max_pos_) {
      GrowBuffer(byte_pos_ + size);
    }
    return buf_ + byte_pos_;
  }
  // same as above, but with over-reservation in case we need to grow the
  // buffer.
  uint8_t* ReserveLarge(size_t size) {
    if (byte_pos_ + size > max_pos_) {
      const size_t requested_size = byte_pos_ + size;
      const size_t overgrown_size = max_pos_ * 3 / 2;
      GrowBuffer(requested_size > overgrown_size ? requested_size
                                                 : overgrown_size);
    }
    return buf_ + byte_pos_;
  }

  // Change the position of the end of the buffer. Doesn't allocate anything.
  // Note that 'size' can be negative. Not boundary check is performed.
  void Advance(size_t size) { byte_pos_ += size; }

  // Delete everything. Mainly useful in case of error.
  void DeleteOutputBuffer();

  // Make sure we can write 24 bits by flushing the past ones.
  // WARNING! There's no check for buffer overwrite. Use Reserve() before
  // calling this function.
  void FlushBits() {
    // worst case: 3 escaped codes = 6 bytes
    assert(byte_pos_ + 6 <= max_pos_);
    while (nb_bits_ >= 8) {
      buf_[byte_pos_++] = bits_ >> 24;
      if ((bits_ & 0xff000000U) == 0xff000000U) {   // escaping
        buf_[byte_pos_++] = 0x00;
      }
      bits_ <<= 8;
      nb_bits_ -= 8;
    }
  }
  // Writes the sequence 'bits' of length 'nb_bits' (less than 24).
  // WARNING! There's no check for buffer overwrite. Use Reserve() before
  // calling this function.
  void PutBits(uint32_t bits, int nb) {
    assert(nb <= 24 && nb > 0);
    assert((bits & ~((1 << nb) - 1)) == 0);
    FlushBits();    // make room for a least 24bits
    nb_bits_+= nb;
    bits_ |= bits << (32 - nb_bits_);
  }
  // Append one byte to buffer. FlushBits() must have been called before.
  // WARNING! There's no check for buffer overwrite. Use Reserve() before
  // calling this function.
  // Also: no 0xff escaping is performed by this function.
  void PutByte(uint8_t value) {
    assert(nb_bits_ == 0);
    buf_[byte_pos_++] = value;
    assert(byte_pos_ <= max_pos_);
  }
  // Same as multiply calling PutByte().
  void PutBytes(const uint8_t* buf, size_t size) {
    assert(nb_bits_ == 0);
    assert(byte_pos_ + size <= max_pos_);
    assert(buf != NULL);
    assert(size > 0);
    memcpy(buf_ + byte_pos_, buf, size);
    byte_pos_ += size;
  }

  // Handy helper to write a packed code in one call.
  void PutPackedCode(uint32_t code) { PutBits(code >> 16, code & 0xff); }

  // Write pending bits, and align bitstream with extra '1' bits.
  void Flush();

  // Returns pointer to the beginning of the output buffer.
  const uint8_t* Data() const { return buf_; }
  // Returns last written position, in bytes.
  size_t BytePos() const { return byte_pos_; }
  // Returns written position, in bits.
  size_t BitPos() const { return 8LL * byte_pos_ + nb_bits_; }
  // Returns total written size, in bytes.
  size_t ByteLength() const { return (BitPos() + 7LL) >> 3; }

  // Returns a pointer to the final buffer, and transfer ownership to the
  // caller, which is reponsible for later deallocating it using 'delete[]'.
  // The size of the returned buffer is stored in '*size'.
  // Upon return of this method, the object is empty as if just constructed.
  uint8_t* Grab(size_t* size);

 private:
  // Expand buffer size to contain at least 'size' bytes (probably more),
  // preserving previous content. Don't call directly, use Reserve() instead!
  void GrowBuffer(size_t size);

  int nb_bits_;      // number of unwritten bits
  uint32_t bits_;    // accumulator for unwritten bits
  size_t byte_pos_;  // write position, in bytes
  uint8_t* buf_;     // destination buffer (don't access directly!)
  size_t max_pos_;   // maximum write-position within output buffer
};

// Class for counting bits, including the 0xff escape
struct BitCounter {
  BitCounter() : bits_(0), bit_pos_(0), size_(0) {}

  void AddPackedCode(const uint32_t code) { AddBits(code >> 16, code & 0xff); }
  void AddBits(const uint32_t bits, size_t nbits);
  size_t Size() const { return size_; }

 private:
  uint32_t bits_;
  size_t bit_pos_;
  size_t size_;
};

}   // namespace sjpeg

#endif    // SJPEG_BIT_WRITER_H_
