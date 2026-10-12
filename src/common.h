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
//  Common definitions and helpers.
//  Internal header: not installed.
//
// Author: Skal (pascal.massimino@gmail.com)

#ifndef SJPEG_COMMON_H_
#define SJPEG_COMMON_H_

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <limits>
#include <new>
#include <type_traits>

#include "sjpeg.h"

#define SJPEG_STRINGIFY_HELPER(x) #x
#define SJPEG_STRINGIFY(x) SJPEG_STRINGIFY_HELPER(x)

#if defined(__clang__)
#define SJPEG_UNROLL(n) _Pragma(SJPEG_STRINGIFY(clang loop unroll_count(n)))
#elif defined(__GNUC__) && (__GNUC__ >= 8)
#define SJPEG_UNROLL(n) _Pragma(SJPEG_STRINGIFY(GCC unroll n))
#else
#define SJPEG_UNROLL(n)
#endif

// Progressive (spectral-split-only) encoding; on by default. Pass
// -DSJPEG_NO_PROGRESSIVE to strip the feature's code out entirely.

// Multi-threaded baseline encoding (EncoderParam::num_threads); on by default.
// Pass -DSJPEG_NO_MULTITHREADING to strip the feature's code out entirely,
// along with sjpeg's only use of <thread>. num_threads is then ignored and
// encoding always runs on the calling thread.

// SIMD feature-detection macros (SJPEG_USE_SSE2/SSSE3/AVX2/NEON,
// SJPEG_AARCH64) and the intrinsic headers they gate live in simd.h; only
// files that define SJPEG_NEED_ASM_HEADERS before including this one pull
// them in.
#if defined(SJPEG_NEED_ASM_HEADERS)
#include "simd.h"  // IWYU pragma: export
#endif             // SJPEG_NEED_ASM_HEADERS

////////////////////////////////////////////////////////////////////////////////

namespace sjpeg {

#if defined(__has_builtin)
#define SJPEG_HAS_BUILTIN(x) __has_builtin(x)
#else
#define SJPEG_HAS_BUILTIN(x) 0
#endif

////////////////////////////////////////////////////////////////////////////////
// Safe integer arithmetic helpers for buffer sizing and allocations.

#if (defined(__GNUC__) && __GNUC__ >= 5) ||                             \
    (defined(__clang__) && SJPEG_HAS_BUILTIN(__builtin_mul_overflow) && \
     SJPEG_HAS_BUILTIN(__builtin_add_overflow))
#define SJPEG_HAS_BUILTIN_OVERFLOW 1
#endif

// Returns true if a * b does not overflow type T, storing the result in *out.
// If overflow occurs, *out is set to 0 and false is returned.
template <typename T>
inline bool SafeMultiply(T a, T b, T* out) {
  if (out == nullptr) return false;
#if defined(SJPEG_HAS_BUILTIN_OVERFLOW)
  return !__builtin_mul_overflow(a, b, out);
#else
  static_assert(std::is_integral<T>::value, "Integral type required");
  if (a == 0 || b == 0) {
    *out = 0;
    return true;
  }
  if (a > 0 && b > 0) {
    if (b > std::numeric_limits<T>::max() / a) {
      *out = 0;
      return false;
    }
  } else if (a < 0 && b < 0) {
    if (a == std::numeric_limits<T>::min() ||
        b == std::numeric_limits<T>::min()) {
      *out = 0;
      return false;
    }
    if (-b > std::numeric_limits<T>::max() / -a) {
      *out = 0;
      return false;
    }
  } else if (a < 0) {
    if (a < std::numeric_limits<T>::min() / b) {
      *out = 0;
      return false;
    }
  } else {  // b < 0
    if (b < std::numeric_limits<T>::min() / a) {
      *out = 0;
      return false;
    }
  }
  *out = a * b;
  return true;
#endif
}

// Returns true if a + b does not overflow type T, storing the result in *out.
// If overflow occurs, *out is set to 0 and false is returned.
template <typename T>
inline bool SafeAdd(T a, T b, T* out) {
  if (out == nullptr) return false;
#if defined(SJPEG_HAS_BUILTIN_OVERFLOW)
  return !__builtin_add_overflow(a, b, out);
#else
  static_assert(std::is_integral<T>::value, "Integral type required");
  if (b > 0 && a > std::numeric_limits<T>::max() - b) {
    *out = 0;
    return false;
  }
  if (b < 0 && a < std::numeric_limits<T>::min() - b) {
    *out = 0;
    return false;
  }
  *out = a + b;
  return true;
#endif
}

extern bool SupportsSSE2();
extern bool SupportsNEON();
extern bool SupportsAVX2();

// Constants below are marker codes defined in JPEG spec
// ISO/IEC 10918-1 : 1993(E) Table B.1
// See also: http://www.w3.org/Graphics/JPEG/itu-t81.pdf

#define M_SOF0 0xffc0
#define M_SOF1 0xffc1
#define M_DHT 0xffc4
#define M_SOI 0xffd8
#define M_EOI 0xffd9
#define M_SOS 0xffda
#define M_DQT 0xffdb

// Maximum picture dimension: SOF stores the width and height on 16 bits.
enum { kMaxDimension = 0xffff };

////////////////////////////////////////////////////////////////////////////////
// Shared by files enc.cc was split into. Everything else in those files is
// either a member of Encoder (already declared below) or file-local.

// Encoder defaults. Also read by EncoderParam, hence not file-local.
extern const float kDefaultQuality;
extern const int kDefaultMethod;
extern const int32_t kDefaultBias;      // rounding bias for AC coefficients
extern const int kDefaultDeltaMaxLuma;  // for adaptive quantization
extern const int kDefaultDeltaMaxChroma;

// Manager used when the caller supplies none.
extern MemoryManager* GetDefaultMemoryManager();

// Fixed-point precisions: FP_BITS for the reciprocal dividers used in place of
// a division, AC_BITS for the extra precision left over by the fdct's scaling.
enum { FP_BITS = 16, AC_BITS = 4 };

#if SJPEG_HAS_BUILTIN(__builtin_clz) || \
    (defined(__GNUC__) &&               \
     ((__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || __GNUC__ >= 4))
#define SJPEG_HAVE_CLZ
#endif

#if SJPEG_HAS_BUILTIN(__builtin_ctzll) || \
    (defined(__GNUC__) &&                 \
     ((__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || __GNUC__ >= 4))
#define SJPEG_HAVE_CTZ
#endif

// Number of bits needed to code 'v' > 0, i.e. 1 + floor(log2(v)).
static inline int CalcLog2(int v) {
#if defined(SJPEG_HAVE_CLZ)
  return 32 - __builtin_clz(v);
#else
  const int kLog2[16] = {0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4};
  assert(v > 0 && v < (1 << 12));
  return (v & ~0xff)   ? 8 + kLog2[v >> 8]
         : (v & ~0x0f) ? 4 + kLog2[v >> 4]
                       : 0 + kLog2[v];
#endif
}

// Count trailing zeros of a non-zero 64-bit value.
// 'x' must be non-zero.
static inline int TrailingZeros64(uint64_t x) {
#if defined(SJPEG_HAVE_CTZ)
  return __builtin_ctzll(x);
#else
  int c = 63;
  x &= ~x + 1;
  if (x & 0x00000000FFFFFFFF) c -= 32;
  if (x & 0x0000FFFF0000FFFF) c -= 16;
  if (x & 0x00FF00FF00FF00FF) c -= 8;
  if (x & 0x0F0F0F0F0F0F0F0F) c -= 4;
  if (x & 0x3333333333333333) c -= 2;
  if (x & 0x5555555555555555) c -= 1;
  return c;
#endif
}

// Calls fn() and returns false if it threw std::bad_alloc, which is how the
// standard library reports its allocation failures. Anything else (thrown by
// a user's ByteSink, say) is not ours to handle.
// This is the library's only try/catch. Without exception support, it's a
// plain call.
template <class F>
bool NoThrow(F&& fn) {
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)
  try {
    fn();
  } catch (const std::bad_alloc&) {
    return false;
  }
#else
  fn();
#endif
  return true;
}

// Non-owning reference to a callable: what std::function does, without its
// heap allocation. The callable must outlive it.
template <class... Args>
class FuncRef {
 public:
  template <class F>
  FuncRef(const F& fn)  // NOLINT: implicit, on purpose
      : fn_(&fn),
        call_([](const void* f, Args... args) { (*(const F*)f)(args...); }) {}
  void operator()(Args... args) const { call_(fn_, args...); }

 private:
  const void* fn_;
  void (*call_)(const void*, Args...);
};

// Growable, uninitialized array of trivially-copyable T. Storage comes from
// 'memory' (the user's MemoryManager) or, if null, from the C heap (which is
// safe to use from worker threads).
template <class T>
class Buffer {
 public:
  explicit Buffer(MemoryManager* memory = nullptr) : memory_(memory) {}
  Buffer(Buffer&& other) noexcept
      : ptr_(other.ptr_), capacity_(other.capacity_), memory_(other.memory_) {
    other.ptr_ = nullptr;
    other.capacity_ = 0;
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ~Buffer() { Free(ptr_); }

  T* data() const { return ptr_; }
  void Release() {
    Free(ptr_);
    ptr_ = nullptr;
    capacity_ = 0;
  }

  // Makes room for at least 'n' elements, keeping the first 'used' ones.
  // Grows geometrically. Returns false on allocation failure.
  bool Reserve(size_t n, size_t used = 0) {
    if (n <= capacity_) return true;
    const size_t new_capacity = std::max(n, 2 * capacity_);
    size_t bytes = 0;
    if (!SafeMultiply(new_capacity, sizeof(T), &bytes)) return false;
    T* const ptr = (T*)Alloc(bytes);
    if (ptr == nullptr) return false;
    if (used > 0) memcpy(ptr, ptr_, used * sizeof(T));
    Free(ptr_);
    ptr_ = ptr;
    capacity_ = new_capacity;
    return true;
  }

 private:
  void* Alloc(size_t bytes) const {
    return (memory_ != nullptr) ? memory_->Alloc(bytes) : malloc(bytes);
  }
  void Free(void* ptr) const {
    if (ptr == nullptr) return;
    if (memory_ != nullptr) {
      memory_->Free(ptr);
    } else {
      free(ptr);
    }
  }

  T* ptr_ = nullptr;
  size_t capacity_ = 0;
  MemoryManager* memory_;
};

}  // namespace sjpeg

#endif  // SJPEG_COMMON_H_
