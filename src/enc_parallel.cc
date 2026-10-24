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
//  Thread pool behind Encoder::RunParallel(). The passes slice their work
//  with it: restart intervals make the entropy-coded segments independent --
//  DC predictors reset and the bitstream realigns to a byte boundary -- so
//  each slice can code a contiguous range of intervals into its own buffer,
//  concatenated in order afterwards. The output is the same, byte for byte,
//  for any number of slices.
//
//  Without multithreading support (SJPEG_NO_MULTITHREADING), RunParallel()
//  just runs the passes as one slice, on the calling thread (sjpegi.h).
//
#include "sjpegi.h"  // IWYU pragma: keep

#if !defined(SJPEG_NO_MULTITHREADING)

#include <assert.h>
#include <stdint.h>
#include <string.h>  // for memset

#include <algorithm>
#include <condition_variable>  // NOLINT
#include <memory>
#include <mutex>  // NOLINT
#include <new>
#include <thread>  // NOLINT
#if defined(_WIN32)
#if !defined(NOMINMAX)
#define NOMINMAX  // keeps std::min / std::max usable
#endif
#include <process.h>
#include <windows.h>
#else
#include <pthread.h>
#endif

#include "bit_writer.h"

namespace sjpeg {

int Encoder::NumThreads(int requested) {
  const int num_threads =
      (requested >= 0) ? requested : (int)std::thread::hardware_concurrency();
  return std::max(1, std::min(num_threads, (int)kMaxThreads));
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
  // The pool may end up with fewer workers than requested: check num_workers().
  explicit ThreadPool(int num_workers) : num_requested_(num_workers) {
    workers_ = new (std::nothrow) Worker[num_workers];
    if (workers_ == nullptr) return;
    for (int i = 0; i < num_workers; ++i) {
      workers_[i].pool = this;
      workers_[i].idx = i + 1;
      if (!StartThread(&workers_[i])) break;
      ++num_workers_;
    }
  }
  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    cv_start_.notify_all();
    for (int i = 0; i < num_workers_; ++i) JoinThread(&workers_[i]);
    delete[] workers_;
  }

  int num_workers() const { return num_workers_; }
  int num_requested() const { return num_requested_; }

  // Calls fn(0) on this thread and fn(1..active_workers) on the workers.
  void Run(int active_workers, FuncRef<int> fn) {
    assert(active_workers > 0 && active_workers <= num_workers_);
    {
      std::lock_guard<std::mutex> lock(mu_);
      fn_ = &fn;
      active_workers_ = active_workers;
      remaining_ = active_workers;
      ++generation_;
    }
    cv_start_.notify_all();
    fn(0);
    std::unique_lock<std::mutex> lock(mu_);
    cv_done_.wait(lock, [this]() { return remaining_ == 0; });
  }

 private:
  void WorkerLoop(int thread_idx) {
    uint64_t last_gen = 0;
    while (true) {
      const FuncRef<int>* fn = nullptr;
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

  // Native threads: unlike std::thread, their creation reports a failure
  // instead of throwing.
  struct Worker {
    ThreadPool* pool;
    int idx;
#if defined(_WIN32)
    HANDLE thread;
#else
    pthread_t thread;
#endif
  };
#if defined(_WIN32)
  static unsigned int __stdcall ThreadMain(void* arg) {
    Worker* const w = (Worker*)arg;
    w->pool->WorkerLoop(w->idx);
    return 0;
  }
  static bool StartThread(Worker* const w) {
    w->thread = (HANDLE)_beginthreadex(nullptr, 0, ThreadMain, w, 0, nullptr);
    return (w->thread != nullptr);
  }
  static void JoinThread(Worker* const w) {
    WaitForSingleObject(w->thread, INFINITE);
    CloseHandle(w->thread);
  }
#else
  static void* ThreadMain(void* arg) {
    Worker* const w = (Worker*)arg;
    w->pool->WorkerLoop(w->idx);
    return nullptr;
  }
  static bool StartThread(Worker* const w) {
    return (pthread_create(&w->thread, nullptr, ThreadMain, w) == 0);
  }
  static void JoinThread(Worker* const w) { pthread_join(w->thread, nullptr); }
#endif
  Worker* workers_ = nullptr;
  int num_workers_ = 0;
  const int num_requested_;
  std::mutex mu_;
  std::condition_variable cv_start_;
  std::condition_variable cv_done_;
  const FuncRef<int>* fn_ = nullptr;
  uint64_t generation_ = 0;
  int active_workers_ = 0;
  int remaining_ = 0;
  bool stop_ = false;
};

void Encoder::ThreadPoolDeleter::operator()(ThreadPool* p) const { delete p; }

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
                          FuncRef<int, int, int> fn) const {
  if (num_threads <= 1 || total <= 1) {
    fn(0, 0, total);
    return;
  }
  const int num_workers = num_threads - 1;
  assert(num_workers > 0);
  if (!thread_pool_ || thread_pool_->num_requested() < num_workers) {
    thread_pool_.reset(new (std::nothrow) ThreadPool(num_workers));
  }
  const auto run = [&](int t) {
    int start, end;
    PartitionRange(total, num_threads, t, &start, &end);
    if (start < end) fn(t, start, end);
  };
  // Threads that couldn't be created are made up for by the calling one: the
  // slices are the same, so is the output.
  const int available =
      thread_pool_ ? std::min(num_workers, thread_pool_->num_workers()) : 0;
  if (available > 0) {
    thread_pool_->Run(available, run);
  } else {
    run(0);
  }
  for (int t = available + 1; t < num_threads; ++t) run(t);
}

}  // namespace sjpeg

#else  // SJPEG_NO_MULTITHREADING: see the inlined stand-ins in sjpegi.h

namespace sjpeg {

void Encoder::ThreadPoolDeleter::operator()(ThreadPool* /*p*/) const {}

}  // namespace sjpeg

#endif  // !SJPEG_NO_MULTITHREADING
