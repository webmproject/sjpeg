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
#include <stdint.h>
#include <string.h>   // for memcpy

#include <string>
#include <vector>

#include "sjpeg.h"

namespace sjpeg {

///////////////////////////////////////////////////////////////////////////////
// Memory-Sink

class MemorySink : public ByteSink {
 public:
  explicit MemorySink(size_t expected_size);
  virtual ~MemorySink();
  virtual bool Commit(size_t used_size, size_t extra_size, uint8_t** data);
  virtual bool Finalize() { /* nothing to do */ return true; }
  virtual void Reset();
  void Release(uint8_t** buf_ptr, size_t* size_ptr);

 private:
  uint8_t* buf_;
  size_t pos_, max_pos_;
};

///////////////////////////////////////////////////////////////////////////////
// Sink for generic container
//   Container must supply .resize() and [], and be byte-based.

template<class T> class Sink : public ByteSink {
 public:
  explicit Sink(T* const output) : ptr_(output), pos_(0) {}
  virtual ~Sink() {}
  virtual bool Commit(size_t used_size, size_t extra_size, uint8_t** data) {
    pos_ += used_size;
    assert(pos_ <= ptr_->size());
    ptr_->resize(pos_ + extra_size);
    if (ptr_->size() != pos_ + extra_size) return false;
    *data = extra_size ? reinterpret_cast<uint8_t*>(&(*ptr_)[pos_]) : nullptr;
    return true;
  }
  virtual bool Finalize() { ptr_->resize(pos_); return true; }
  virtual void Reset() {
    ptr_->clear();
    pos_ = 0;
  }

 protected:
  T* const ptr_;
  size_t pos_;
};

typedef Sink<std::string> StringSink;
typedef Sink<std::vector<uint8_t> > VectorSink;

///////////////////////////////////////////////////////////////////////////////
// Performance toggles. Defined here rather than in sjpegi.h because this
// header is included on its own by bit_writer.cc. Set any of them to 0 on the
// command line to build the previous implementation instead, for A/B timing.
// All of them are bit-exact: same bytes out, either way.
//   SJPEG_USE_FAST_BITWRITER    64bit lazily-flushed BitWriter
// See doc/perf-plan.md for what each of them buys.

#if !defined(SJPEG_USE_FAST_BITWRITER)
#define SJPEG_USE_FAST_BITWRITER 1
#endif

#if SJPEG_USE_FAST_BITWRITER
// returns true if any byte of 'x' is 0xff
static inline bool HasFF(uint64_t x) {
  const uint64_t y = ~x;    // 0xff bytes are now zero bytes
  return ((y - 0x0101010101010101ull) & ~y & 0x8080808080808080ull) != 0;
}

static inline uint64_t BSwap64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_bswap64(x);
#else
  x = ((x & 0x00ff00ff00ff00ffull) << 8) | ((x >> 8) & 0x00ff00ff00ff00ffull);
  x = ((x & 0x0000ffff0000ffffull) << 16) | ((x >> 16) & 0x0000ffff0000ffffull);
  return (x << 32) | (x >> 32);
#endif
}

// The accumulator holds the bits to emit in its top bytes, so storing it as a
// whole means storing it big-endian. That is a byte swap on a little-endian
// host and nothing at all on a big-endian one -- and skipping the distinction
// would not fail loudly: the bytes would come out reversed in groups of eight,
// producing a corrupt JPEG rather than an error.
static inline uint64_t HToBE64(uint64_t x) {
#if (defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
     __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__) ||                  \
    defined(__BIG_ENDIAN__)
  return x;
#else
  return BSwap64(x);
#endif
}
#endif    // SJPEG_USE_FAST_BITWRITER

///////////////////////////////////////////////////////////////////////////////
// BitWriter

class BitWriter {
 public:
  explicit BitWriter(ByteSink* const sink);

  // Verifies the that output buffer can store at least 'size' more bytes.
  // Also flushes the previously written data.
  bool Reserve(size_t size) {
    if (!sink_->Commit(byte_pos_, size, &buf_)) return CommitFailed();
    byte_pos_ = 0;
    reserved_ = size;
    return true;
  }

#if SJPEG_USE_FAST_BITWRITER
  // Flush whole bytes out of the accumulator.
  // WARNING! There's no check for buffer overwrite. Use Reserve() before
  // calling this function. Note that the fast path below stores 8 bytes
  // whatever the number of pending ones, so up to 7 bytes past byte_pos_ get
  // clobbered; they are rewritten by the next flush. Callers reserve a whole
  // MCU's worst case ahead of each MCU (see Encoder::CheckBuffers), which
  // leaves room for both paths; the asserts below are what checks it.
  void FlushBits() {
    const int nb_bytes = nb_bits_ >> 3;
    if (nb_bytes == 0) return;
    // Tested on the accumulator, not on the stored bytes: the pending bytes are
    // the top ones of bits_ whatever the host's byte order.
    const uint64_t mask = (~0ull) << (64 - 8 * nb_bytes);
    if (!HasFF(bits_ & mask)) {            // common case: nothing to escape
      // Stores 8 bytes whatever the number pending, so it needs 8 in hand.
      assert(byte_pos_ + sizeof(uint64_t) <= reserved_);
      const uint64_t out = HToBE64(bits_);
      memcpy(buf_ + byte_pos_, &out, sizeof(out));
      byte_pos_ += nb_bytes;
    } else {
      // Two bytes per pending byte, worst case: every one of them is 0xff.
      assert(byte_pos_ + 2 * nb_bytes <= reserved_);
      uint64_t v = bits_;
      for (int i = 0; i < nb_bytes; ++i, v <<= 8) {
        const uint8_t tmp = static_cast<uint8_t>(v >> 56);
        buf_[byte_pos_++] = tmp;
        if (tmp == 0xff) buf_[byte_pos_++] = 0x00;   // escaping
      }
    }
    bits_ <<= 8 * nb_bytes;
    nb_bits_ -= 8 * nb_bytes;
  }
  // Writes the sequence 'bits' of length 'nb' (less or equal to 32).
  // WARNING! There's no check for buffer overwrite. Use Reserve() before
  // calling this function.
  void PutBits(uint32_t bits, int nb) {
    assert(nb <= 32 && nb > 0);
    assert((bits & ~(~0u >> (32 - nb))) == 0);
    // Only flush when the accumulator would not fit the new symbol. With 56
    // usable bits, about five symbols pile up per flush instead of one.
    if (nb_bits_ + nb > 56) FlushBits();
    nb_bits_ += nb;
    bits_ |= static_cast<uint64_t>(bits) << (64 - nb_bits_);
  }
#else
  // Make sure we can write 24 bits by flushing the past ones.
  // WARNING! There's no check for buffer overwrite. Use Reserve() before
  // calling this function.
  void FlushBits() {
    // worst case: 3 escaped codes = 6 bytes
    while (nb_bits_ >= 8) {
      const uint8_t tmp = bits_ >> 24;
      buf_[byte_pos_++] = tmp;
      if (tmp == 0xff) {   // escaping
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
#endif    // SJPEG_USE_FAST_BITWRITER
  // Append one byte to buffer. FlushBits() must have been called before.
  // WARNING! There's no check for buffer overwrite. Use Reserve() before
  // calling this function.
  // Also: no 0xff escaping is performed by this function.
  void PutByte(uint8_t value) {
    assert(nb_bits_ == 0);
    buf_[byte_pos_++] = value;
  }
  // Same as multiply calling PutByte().
  void PutBytes(const uint8_t* buf, size_t size) {
    assert(nb_bits_ == 0);
    assert(buf != nullptr);
    assert(size > 0);
    memcpy(buf_ + byte_pos_, buf, size);
    byte_pos_ += size;
  }

  // Handy helper to write a packed code in one call.
  void PutPackedCode(uint32_t code) { PutBits(code >> 16, code & 0xff); }

#if SJPEG_USE_FAST_BITWRITER
  // Write a packed code immediately followed by its 'n'-bit suffix. Both fit
  // in one PutBits() call: 16 bits of code and 11 of suffix, worst case.
  void PutPackedCodeAndSuffix(uint32_t code, uint32_t suffix, int n) {
    const int len = code & 0xff;
    PutBits(((code >> 16) << n) | suffix, len + n);
  }
#endif

  // Write pending bits, and align bitstream with extra '1' bits.
  void Flush();

  // To be called last.
  bool Finalize() { return Reserve(0) && sink_->Finalize(); }

 private:
  bool CommitFailed();

  ByteSink* sink_;

  int nb_bits_;      // number of unwritten bits
#if SJPEG_USE_FAST_BITWRITER
  uint64_t bits_;    // accumulator for unwritten bits
#else
  uint32_t bits_;    // accumulator for unwritten bits
#endif
  size_t byte_pos_;  // write position, in bytes
  size_t reserved_;  // bytes available in the slab starting at buf_
  uint8_t* buf_;     // destination buffer (don't access directly!)
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
