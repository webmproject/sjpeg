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
//  Storage for the slices of the multi-pass / parallel scans.
//  Internal header: not installed.
//
// Author: Skal (pascal.massimino@gmail.com)

#ifndef SJPEG_SLICES_H_
#define SJPEG_SLICES_H_

#include <stddef.h>
#include <stdint.h>

#include <new>

#include "bit_writer.h"
#include "blocks.h"
#include "common.h"
#include "entropy.h"

namespace sjpeg {

// Quantized blocks of a slice of the image (a range of restart intervals),
// plus the entropy statistics gathered while quantizing them.
struct SliceStore {
  explicit SliceStore(MemoryManager* memory)
      : coeffs(memory), run_levels(memory) {}
  Buffer<DCTCoeffs> coeffs;     // one per block
  Buffer<RunLevel> run_levels;  // their run/levels, back to back
  size_t nb_blocks = 0;         // entries of coeffs[] in use
  size_t nb_run_levels = 0;     // entries of run_levels[] in use
  EntropyStats stats;
};

// Per-slice output and scratch. Nothing in there throws on allocation
// failure: slices run on worker threads.
struct ThreadChunk {
  explicit ThreadChunk(MemoryManager* memory = nullptr)
      : data(0), store(memory) {}
  MemorySink data;  // this slice's entropy-coded bytes (parallel scans only)
  SliceStore store;
  bool ok = true;
  // The histogram pass hits the stats once per coefficient: keeps neighbouring
  // slices from sharing a cache line.
  uint8_t padding[64];
};

// The slices of a parallel pass. A lone slice runs on the calling thread, so
// its buffers can go through the user's memory manager (which isn't required
// to be thread-safe); with several, they use the C heap. Without
// multithreading, it is always a single slice.
class ChunkSet {
 public:
  explicit ChunkSet(MemoryManager* memory) : chunk_(memory) {}
  ChunkSet(const ChunkSet&) = delete;
  ChunkSet& operator=(const ChunkSet&) = delete;

#if !defined(SJPEG_NO_MULTITHREADING)
  ~ChunkSet() { delete[] chunks_; }
  // Sets up 'num_slices' slices, keeping them if the count is the same.
  // Returns false in case of allocation failure.
  bool Reset(int num_slices) {
    if (size_ == (size_t)num_slices) return true;
    delete[] chunks_;
    chunks_ = nullptr;
    size_ = 1;
    if (num_slices > 1) {
      chunk_.store.coeffs.Release();  // the lone slice is out of use
      chunk_.store.run_levels.Release();
      chunks_ = new (std::nothrow) ThreadChunk[num_slices];
      if (chunks_ == nullptr) return false;
      size_ = (size_t)num_slices;
    }
    return true;
  }
  ThreadChunk* data() { return (chunks_ != nullptr) ? chunks_ : &chunk_; }
  size_t size() const { return size_; }
#else
  bool Reset(int /*num_slices*/) { return true; }
  ThreadChunk* data() { return &chunk_; }
  size_t size() const { return 1; }
#endif
  ThreadChunk& operator[](int t) { return data()[t]; }
  const ThreadChunk* begin() const { return ((ChunkSet*)this)->data(); }
  const ThreadChunk* end() const { return begin() + size(); }

 private:
  ThreadChunk chunk_;  // the lone slice
#if !defined(SJPEG_NO_MULTITHREADING)
  ThreadChunk* chunks_ = nullptr;  // or several of them
  size_t size_ = 1;
#endif
};

}  // namespace sjpeg

#endif  // SJPEG_SLICES_H_
