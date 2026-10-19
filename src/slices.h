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

#include <string>
#if !defined(SJPEG_NO_MULTITHREADING)
#include <vector>
#endif

#include "blocks.h"
#include "common.h"
#include "entropy.h"
#include "sjpeg.h"

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

// Per-slice output and scratch. Cache-line aligned: the histogram pass hits
// the stats once per coefficient, and neighbouring slices would otherwise
// share lines.
struct alignas(64) ThreadChunk {
  explicit ThreadChunk(MemoryManager* memory) : store(memory) {}
  std::string data;  // this slice's entropy-coded bytes (parallel scans only)
  SliceStore store;
  bool ok = true;
};

// The slices of a parallel pass. A lone slice runs on the calling thread, so
// its buffers can go through the user's memory manager (which isn't required
// to be thread-safe); with several, they use the C heap. Without
// multithreading, it is always a single slice.
class ChunkSet {
 public:
  explicit ChunkSet(MemoryManager* memory)
#if !defined(SJPEG_NO_MULTITHREADING)
      : memory_(memory){}
#else
      : chunk_(memory) {
  }
#endif

        // Sets up 'num_slices' slices, keeping the buffers if the count is the
        // same.
        void Reset(int num_slices) {
#if !defined(SJPEG_NO_MULTITHREADING)
    if (chunks_.size() == (size_t)num_slices) return;
    chunks_.clear();
    chunks_.reserve(num_slices);
    for (int t = 0; t < num_slices; ++t) {
      chunks_.emplace_back((num_slices == 1) ? memory_ : nullptr);
    }
#else
    (void)num_slices;
#endif
  }
#if !defined(SJPEG_NO_MULTITHREADING)
  ThreadChunk* data() { return chunks_.data(); }
  size_t size() const { return chunks_.size(); }
#else
  ThreadChunk* data() { return &chunk_; }
  size_t size() const { return 1; }
#endif
  ThreadChunk& operator[](int t) { return data()[t]; }
  const ThreadChunk* begin() const { return ((ChunkSet*)this)->data(); }
  const ThreadChunk* end() const { return begin() + size(); }

 private:
#if !defined(SJPEG_NO_MULTITHREADING)
  MemoryManager* const memory_;
  std::vector<ThreadChunk> chunks_;
#else
  ThreadChunk chunk_;
#endif
};

}  // namespace sjpeg

#endif  // SJPEG_SLICES_H_
