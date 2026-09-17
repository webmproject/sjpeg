// Copyright 2026 Google Inc.
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
//  Multi-threaded scan engines. Restart intervals make the entropy-coded
//  segments independent -- DC predictors reset and the bitstream realigns to a
//  byte boundary -- so each worker can code a contiguous slice of intervals
//  into its own buffer, and the slices are concatenated in order afterwards.
//
//  These share the exact row/slice kernels (CodeScanSlice, QuantizeScanSlice,
//  ReplayScanSlice) with single-threaded scans and produce byte-identical
//  output to them for the same restart interval.
//
#include "sjpegi.h"  // IWYU pragma: keep

#if !defined(SJPEG_NO_MULTITHREADING)

#include <assert.h>
#include <stdint.h>
#include <string.h>  // for memset

#include <algorithm>
#include <condition_variable>  // NOLINT
#include <functional>
#include <memory>
#include <mutex>  // NOLINT
#include <new>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "bit_writer.h"

namespace sjpeg {

int Encoder::HardwareConcurrency() {
  const unsigned int hw = std::thread::hardware_concurrency();
  return std::max(1, static_cast<int>(hw));
}

int Encoder::GetNumSlices(int cap, int grain) const {
  if (num_threads_ <= 1 || cap <= 1) return 1;
  const int num_mcus = mb_w_ * mb_h_;
  const int worthwhile = (grain > 0)
                             ? ScaledThreadLimit(num_mcus, grain)
                             : std::max(1, num_mcus / kMinMCUsPerThread);
  return std::min({num_threads_, cap, worthwhile});
}

class Encoder::ThreadPool {
 public:
  explicit ThreadPool(int num_workers) {
    workers_.reserve(num_workers);
    for (int i = 0; i < num_workers; ++i) {
      workers_.emplace_back([this, i]() { WorkerLoop(i + 1); });
    }
  }
  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    cv_start_.notify_all();
    for (auto& w : workers_) w.join();
  }

  int num_workers() const { return static_cast<int>(workers_.size()); }

  void Run(int active_workers, const std::function<void(int)>& fn) {
    if (active_workers <= 0) {
      fn(0);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mu_);
      fn_ = &fn;
      active_workers_ = active_workers;
      remaining_ = active_workers;
      ++generation_;
    }
    cv_start_.notify_all();
    fn(0);
    {
      std::unique_lock<std::mutex> lock(mu_);
      cv_done_.wait(lock, [this]() { return remaining_ == 0; });
    }
  }

 private:
  void WorkerLoop(int thread_idx) {
    uint64_t last_gen = 0;
    while (true) {
      const std::function<void(int)>* fn = nullptr;
      int active = 0;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_start_.wait(lock, [this, last_gen]() {
          return stop_ || generation_ != last_gen;
        });
        if (stop_) return;
        last_gen = generation_;
        fn = fn_;
        active = active_workers_;
      }
      if (thread_idx <= active) {
        (*fn)(thread_idx);
        std::lock_guard<std::mutex> lock(mu_);
        if (--remaining_ == 0) {
          cv_done_.notify_one();
        }
      }
    }
  }

  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_start_;
  std::condition_variable cv_done_;
  const std::function<void(int)>* fn_ = nullptr;
  uint64_t generation_ = 0;
  int active_workers_ = 0;
  int remaining_ = 0;
  bool stop_ = false;
};

void Encoder::DeleteThreadPool() {
  delete thread_pool_;
  thread_pool_ = nullptr;
}

namespace {

// Contiguous, balanced partition of [0, total) into 'num_parts' ranges: part
// 'idx' spans [*start, *end). Sizes differ by at most one, so no part is left
// empty. (A ceil()-based split gives whole parts to the leading workers and
// can idle close to half of them: 9 intervals over 8 parts would use 5.)
void PartitionRange(int total, int num_parts, int idx, int* start, int* end) {
  *start = idx * total / num_parts;
  *end = (idx + 1) * total / num_parts;
}

}  // namespace

void Encoder::RunParallel(int num_threads, int total,
                          const std::function<void(int, int, int)>& fn) const {
  if (num_threads <= 1 || total <= 1) {
    fn(0, 0, total);
    return;
  }
  const int num_workers = num_threads - 1;
  assert(num_workers > 0);
  if (thread_pool_ == nullptr || thread_pool_->num_workers() < num_workers) {
    delete thread_pool_;
    thread_pool_ = new (std::nothrow) ThreadPool(num_workers);
    if (thread_pool_ == nullptr) {
      SetError();
      return;
    }
  }
  thread_pool_->Run(num_workers, [&](int t) {
    int start, end;
    PartitionRange(total, num_threads, t, &start, &end);
    if (start < end) fn(t, start, end);
  });
}

void Encoder::ConcatenateChunks(ThreadChunk* chunks, int num_chunks) {
  for (int t = 0; t < num_chunks; ++t) {
    if (!chunks[t].ok) {
      SetError();
      return;
    }
    if (chunks[t].data.empty()) continue;
    if (!bw_.Reserve(chunks[t].data.size())) {
      SetError();
      return;
    }
    bw_.PutBytes(reinterpret_cast<const uint8_t*>(chunks[t].data.data()),
                 chunks[t].data.size());
    std::string().swap(chunks[t].data);   // free as we go
  }
}

void Encoder::MergeChunkStats(const ThreadChunk* chunks, int num_chunks) {
  ResetEntropyStats();
  const int nb_tables = (nb_comps_ == 1) ? 1 : 2;
  for (int q = 0; q < nb_tables; ++q) {
    for (int t = 0; t < num_chunks; ++t) {
      for (int i = 0; i <= 256; ++i) freq_ac_[q][i] += chunks[t].freq_ac[q][i];
      for (int i = 0; i <= 12; ++i) freq_dc_[q][i] += chunks[t].freq_dc[q][i];
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Parallel histogram pass (CollectHistograms) and coefficient collection

void Encoder::CollectHistogramsMultiThreaded(int num_threads) {
  assert(num_threads > 1);
  struct alignas(64) HistoWorker {
    Histo histos[2];
  };
  std::unique_ptr<HistoWorker[]> workers(
      new (std::nothrow) HistoWorker[num_threads - 1]);
  if (workers == nullptr) {
    ResetHisto();
    CollectHistogramsSlice(0, mb_h_, histos_, in_blocks_, replicated_buffer_);
    have_coeffs_ = use_extra_memory_;
    return;
  }

  const int num_histos = (nb_comps_ > 1) ? 2 : 1;

  RunParallel(num_threads, mb_h_, [&](int t, int y_start, int y_end) {
    Histo* const dst_histos = (t == 0) ? histos_ : workers[t - 1].histos;
    if (t == 0) {
      ResetHisto();
    } else {
      memset(dst_histos, 0, sizeof(Histo) * num_histos);
    }
    alignas(32) int16_t mcu_scratch[6 * 64];
    uint8_t rep_buf[4 * 16 * 16];
    CollectHistogramsSlice(y_start, y_end, dst_histos, mcu_scratch, rep_buf);
  });

  for (int q = 0; q < num_histos; ++q) {
    int* const dst = &histos_[q].counts_[0][0];
    for (int w = 0; w < num_threads - 1; ++w) {
      const int* const src = &workers[w].histos[q].counts_[0][0];
      for (int i = 0; i < 64 * (MAX_HISTO_DCT_COEFF + 1); ++i) {
        dst[i] += src[i];
      }
    }
  }
  have_coeffs_ = use_extra_memory_;
}

void Encoder::CollectCoeffsMultiThreaded(int num_threads) {
  assert(use_extra_memory_);
  RunParallel(num_threads, mb_h_, [&](int /*t*/, int y_start, int y_end) {
    uint8_t rep_buf[4 * 16 * 16];
    CollectCoeffsSlice(y_start, y_end, rep_buf);
  });
  have_coeffs_ = true;
}

////////////////////////////////////////////////////////////////////////////////
// 1-pass parallel scan: quantize and code straight into the slice's buffer.

void Encoder::SinglePassScanMultiThreaded(int num_threads,
                                          int total_intervals) {
  std::vector<ThreadChunk> chunks(num_threads);
  RunParallel(num_threads, total_intervals,
              [&](int t, int first_interval, int end_interval) {
                ThreadChunk& chunk = chunks[t];
                StringSink sink(&chunk.data);
                BitWriter bw(&sink);
                const size_t slab = SliceSlabSize(first_interval, end_interval);
                int16_t scratch[64 * 6];
                uint8_t rep_buf[4 * 16 * 16];
                if (!CodeScanSlice(first_interval, end_interval,
                                   total_intervals, &bw, slab, scratch,
                                   rep_buf)) {
                  chunk.ok = false;
                  return;
                }
                bw.Flush();
                chunk.ok = bw.Finalize();
              });
  ConcatenateChunks(chunks.data(), num_threads);
}

////////////////////////////////////////////////////////////////////////////////
// Quantization and replay helpers shared by 2-pass optimized scan and multi-pass
// dichotomy search. Re-quantizing in pass 2 would not merely waste work:
// WriteDHT() calls InitCodes(), which rebuilds ac_codes_, and quants_[].codes_
// points into it, so trellis and RDO quantization would silently re-decide
// against the optimized codes and stop matching the histogram those tables
// were built from.

void Encoder::QuantizeSlicesMultiThreaded(int num_threads, int total_intervals,
                                          std::vector<ThreadChunk>* chunks) {
  const int rows_per_interval = restart_interval_rows_;
  chunks->resize(num_threads);
  const bool collect_stats = optimize_size_;

  if (use_trellis_ || use_rdo_) InitCodes(true);

  RunParallel(num_threads, total_intervals,
              [&](int t, int first_interval, int end_interval) {
                ThreadChunk& chunk = (*chunks)[t];
                if (collect_stats) {
                  memset(chunk.freq_ac, 0, sizeof(chunk.freq_ac));
                  memset(chunk.freq_dc, 0, sizeof(chunk.freq_dc));
                }

                const int y_first = first_interval * rows_per_interval;
                const int y_end =
                    std::min(mb_h_, end_interval * rows_per_interval);
                const size_t nb_blocks =
                    static_cast<size_t>(y_end - y_first) * mb_w_ * mcu_blocks_;
                chunk.nb_run_levels = 0;
                if (reuse_run_levels_) {
                  chunk.coeffs.resize(nb_blocks);
                  // Blocks code far fewer than 64 run/levels on average; grow on
                  // demand, like EnsureRunLevels() does for all_run_levels_.
                  if (chunk.run_levels.size() < nb_blocks * 8 + 6 * 64) {
                    chunk.run_levels.resize(nb_blocks * 8 + 6 * 64);
                  }
                }

                int16_t scratch[64 * 6];
                uint8_t rep_buf[4 * 16 * 16];
                chunk.ok = QuantizeScanSlice(
                    first_interval, end_interval,
                    reuse_run_levels_ ? chunk.coeffs.data() : nullptr,
                    reuse_run_levels_ ? &chunk.run_levels : nullptr,
                    &chunk.nb_run_levels,
                    collect_stats ? chunk.freq_ac : nullptr,
                    collect_stats ? chunk.freq_dc : nullptr, scratch, rep_buf);
              });

  if (collect_stats) {
    MergeChunkStats(chunks->data(), num_threads);
  }
}

void Encoder::ReplaySlicesMultiThreaded(int num_threads, int total_intervals,
                                        std::vector<ThreadChunk>* chunks) {
  RunParallel(num_threads, total_intervals,
              [&](int t, int first_interval, int end_interval) {
                ThreadChunk& chunk = (*chunks)[t];
                StringSink sink(&chunk.data);
                BitWriter bw(&sink);
                const size_t slab = SliceSlabSize(first_interval, end_interval);
                if (!ReplayScanSlice(first_interval, end_interval,
                                     total_intervals, chunk.coeffs.size(),
                                     chunk.coeffs.data(),
                                     chunk.run_levels.data(), &bw, slab)) {
                  chunk.ok = false;
                  return;
                }
                bw.Flush();
                chunk.ok = bw.Finalize();
                // Release the slice's scratch as soon as it has been coded.
                std::vector<DCTCoeffs>().swap(chunk.coeffs);
                std::vector<RunLevel>().swap(chunk.run_levels);
              });
  ConcatenateChunks(chunks->data(), num_threads);
}

void Encoder::SinglePassScanOptimizedMultiThreaded(int num_threads,
                                                   int total_intervals) {
  std::vector<ThreadChunk> chunks(num_threads);
  QuantizeSlicesMultiThreaded(num_threads, total_intervals, &chunks);
  CompileEntropyStats();
  WriteDHT();
  WriteSOS();
  if (!reuse_run_levels_) {
    SinglePassScan();
  } else {
    DeallocateBlocks();   // pass 2 replays run/levels; the coeffs are dead now
    ReplaySlicesMultiThreaded(num_threads, total_intervals, &chunks);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Multi-pass dichotomy search helpers

float Encoder::ComputePSNRMultiThreaded(int num_threads) const {
  std::vector<uint64_t> errors(num_threads, 0);
  RunParallel(num_threads, mb_h_,
              [&](int t, int y_start, int y_end) {
                errors[t] = ComputePSNRSlice(y_start, y_end);
              });
  uint64_t total_error = 0;
  for (int t = 0; t < num_threads; ++t) {
    total_error += errors[t];
  }
  const size_t nb_mbs = static_cast<size_t>(mb_w_) * mb_h_;
  return GetPSNR(total_error, 64ull * nb_mbs * mcu_blocks_);
}

float Encoder::EvaluateSizeMultiThreaded(int num_threads, int total_intervals,
                                         std::vector<ThreadChunk>* chunks) {
  QuantizeSlicesMultiThreaded(num_threads, total_intervals, chunks);
  if (optimize_size_) {
    CompileEntropyStats();
    if (use_trellis_ || use_rdo_) InitCodes(true);
    return ComputeSize(nullptr);
  }
  InitCodes(false);
  BitCounter bc;
  for (int t = 0; t < num_threads; ++t) {
    BlocksSize(static_cast<int>((*chunks)[t].coeffs.size()),
               (*chunks)[t].coeffs.data(), (*chunks)[t].run_levels.data(), &bc);
  }
  return ComputeSize(bc.Size());
}

}    // namespace sjpeg

#endif  // !SJPEG_NO_MULTITHREADING
