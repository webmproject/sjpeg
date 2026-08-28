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
//  Unit tests for the library's API. Usage:
//     ./unit_test [test-name]...

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "sjpeg.h"

namespace {

////////////////////////////////////////////////////////////////////////////////
// Minimal test harness

int g_num_checks = 0;
int g_num_failures = 0;
const char* g_test_name = "";

bool CheckImpl(bool cond, const char* expr, int line) {
  ++g_num_checks;
  if (!cond) {
    ++g_num_failures;
    printf("  FAILED %s (line %d): %s\n", g_test_name, line, expr);
  }
  return cond;
}
#define SJPEG_CHECK(expr) CheckImpl(!!(expr), #expr, __LINE__)
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))

struct TestCase { const char* name; void (*func)(); };

std::vector<TestCase>& Tests() {
  static std::vector<TestCase> tests;
  return tests;
}

struct TestRegistrar {
  TestRegistrar(const char* name, void (*func)()) {
    const TestCase test = { name, func };
    Tests().push_back(test);
  }
};

#define TEST(Name)                                          \
  void Test##Name();                                        \
  const TestRegistrar kRegister##Name(#Name, &Test##Name);  \
  void Test##Name()

////////////////////////////////////////////////////////////////////////////////
// Samples and shortcuts. The generators are deterministic, so that a failure
// is always reproducible.

const uint32_t kSeed = 7654321u;
uint32_t g_seed = kSeed;

uint8_t Random8b() {
  g_seed = 1103515245u * g_seed + 12345u;
  return static_cast<uint8_t>(g_seed >> 16);
}

// Noisy picture with some structure, hard to compress.
std::vector<uint8_t> MakeRGB(int width, int height) {
  g_seed = kSeed;
  std::vector<uint8_t> rgb(3 * static_cast<size_t>(width) * height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      uint8_t* const p = &rgb[3 * (x + static_cast<size_t>(y) * width)];
      p[0] = static_cast<uint8_t>(x * 5 + (Random8b() >> 3));
      p[1] = static_cast<uint8_t>(y * 3 + (Random8b() >> 4));
      p[2] = static_cast<uint8_t>(((x / 8) ^ (y / 8)) * 51);
    }
  }
  return rgb;
}

// Encodes a whole picture, with the packed stride.
template<class T> bool EncodeRGB(const std::vector<uint8_t>& rgb, int W, int H,
                                 const sjpeg::EncoderParam& param, T* out) {
  return sjpeg::Encode(rgb.data(), W, H, 3 * W, param, out);
}

// True if the bitstream announces the expected dimensions.
bool HasSize(const std::string& jpg, int W, int H) {
  int width = 0, height = 0;
  return SjpegDimensions(jpg, &width, &height, nullptr) &&
         width == W && height == H;
}

// Only used by TEST(Progressive) below.
#if !defined(SJPEG_NO_PROGRESSIVE)
// Walks a JPEG bitstream's marker structure (without decoding entropy data)
// and returns true if it's well-formed: starts with SOI, every marker's
// declared length stays in bounds, and it ends with a clean EOI right after
// the last scan's entropy data (no trailing garbage, no early truncation).
// Also reports the SOF marker byte seen (0xc0 or 0xc2) and the number of SOS
// (scan) markers found.
bool CheckMarkerStructure(const std::string& jpg, int* sof_marker,
                          int* num_sos) {
  const uint8_t* const data = reinterpret_cast<const uint8_t*>(jpg.data());
  const size_t size = jpg.size();
  *sof_marker = 0;
  *num_sos = 0;
  if (size < 4 || data[0] != 0xff || data[1] != 0xd8) return false;  // SOI
  size_t i = 2;
  while (i + 1 < size) {
    if (data[i] != 0xff) return false;
    const uint8_t marker = data[i + 1];
    if (marker == 0xd9) return (i + 2 == size);  // EOI must be the very last
    if (marker == 0x00 || marker == 0xff) return false;
    if (i + 3 >= size) return false;
    const int length = (data[i + 2] << 8) | data[i + 3];
    if (length < 2 || i + 2 + (size_t)length > size) return false;
    if (marker == 0xc0 || marker == 0xc2) *sof_marker = marker;
    i += 2 + length;
    if (marker == 0xda) {  // SOS: skip over its entropy-coded data too
      ++*num_sos;
      while (i + 1 < size) {
        if (data[i] == 0xff && data[i + 1] != 0x00 &&
            !(data[i + 1] >= 0xd0 && data[i + 1] <= 0xd7)) {
          break;
        }
        ++i;
      }
    }
  }
  return false;  // ran off the end without a clean EOI
}
#endif  // !SJPEG_NO_PROGRESSIVE

////////////////////////////////////////////////////////////////////////////////

// The sharp-YUV tables are built lazily: concurrent encoders must not race on
// them, nor see one half-filled. Runs first, since they are only initialized
// once per process and any earlier test would have done it already.
TEST(Threads) {
  const int W = 33, H = 21, kNumThreads = 8;
  const std::vector<uint8_t> rgb = MakeRGB(W, H);
  std::vector<std::string> out(kNumThreads);
  std::vector<std::thread> threads;
  for (int t = 0; t < kNumThreads; ++t) {
    threads.push_back(std::thread([&, t]() {
      sjpeg::EncoderParam param(72.f);
      param.yuv_mode = SJPEG_YUV_SHARP;
      sjpeg::Encode(rgb.data(), W, H, 3 * W, param, &out[t]);
    }));
  }
  for (size_t t = 0; t < threads.size(); ++t) threads[t].join();
  // same input, same parameters: the bitstreams must all be identical
  for (int t = 0; t < kNumThreads; ++t) {
    SJPEG_CHECK(!out[t].empty() && out[t] == out[0]);
  }
}

TEST(Compress) {
  const int kWidth = 61, kHeight = 37;
  const std::vector<uint8_t> rgb = MakeRGB(kWidth, kHeight);
  std::string out;
  SJPEG_CHECK(SjpegCompress(rgb.data(), kWidth, kHeight, 75.f, &out));
  SJPEG_CHECK(out.size() > 0);
  int width = 0, height = 0, is_yuv420 = -1;
  SJPEG_CHECK(SjpegDimensions(out, &width, &height, &is_yuv420));
  SJPEG_CHECK(width == kWidth && height == kHeight);
  SJPEG_CHECK(is_yuv420 == 0 || is_yuv420 == 1);
}

TEST(EncodeParams) {
  const int kWidth = 32, kHeight = 16;
  const std::vector<uint8_t> rgb = MakeRGB(kWidth, kHeight);
  const SjpegYUVMode kModes[] = { SJPEG_YUV_AUTO, SJPEG_YUV_420,
                                  SJPEG_YUV_SHARP, SJPEG_YUV_444,
                                  SJPEG_YUV_400 };
  for (size_t m = 0; m < ARRAY_SIZE(kModes); ++m) {
    sjpeg::EncoderParam param(80.f);
    param.yuv_mode = kModes[m];
    std::string out;
    SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &out));
    SJPEG_CHECK(HasSize(out, kWidth, kHeight));
  }
  // Higher quality must not compress better.
  std::string small, large;
  SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, sjpeg::EncoderParam(30.f),
                        &small));
  SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, sjpeg::EncoderParam(95.f),
                        &large));
  SJPEG_CHECK(small.size() < large.size());
}

TEST(InvalidArguments) {
  const int kWidth = 16, kHeight = 16;
  const std::vector<uint8_t> rgb = MakeRGB(kWidth, kHeight);
  uint8_t* data = nullptr;
  const auto enc = [](const uint8_t* src, int W, int H, int stride,
                      uint8_t** dst, SjpegYUVMode mode) {
    return SjpegEncode(src, W, H, stride, dst, 75.f, 4, mode);
  };
  SJPEG_CHECK(enc(nullptr, kWidth, kHeight, 3 * kWidth, &data, SJPEG_YUV_420)
              == 0);
  SJPEG_CHECK(enc(rgb.data(), kWidth, kHeight, 3 * kWidth, nullptr,
                  SJPEG_YUV_420) == 0);
  SJPEG_CHECK(enc(rgb.data(), 0, kHeight, 3 * kWidth, &data, SJPEG_YUV_420)
              == 0);
  SJPEG_CHECK(enc(rgb.data(), kWidth, -1, 3 * kWidth, &data, SJPEG_YUV_420)
              == 0);
  SJPEG_CHECK(enc(rgb.data(), kWidth, kHeight, 3 * kWidth - 1, &data,
                  SJPEG_YUV_420) == 0);

  // unknown yuv_mode: no encoder can be created for it. 7 is the largest
  // value the enum can hold without being out of range.
  SJPEG_CHECK(enc(rgb.data(), kWidth, kHeight, 3 * kWidth,
                  &data, static_cast<SjpegYUVMode>(7)) == 0);
  SJPEG_CHECK(data == nullptr);
  const sjpeg::EncoderParam param;
  std::string out;
  SJPEG_CHECK(!sjpeg::Encode(nullptr, kWidth, kHeight, 3 * kWidth, param,
                             &out));
  SJPEG_CHECK(!EncodeRGB(rgb, kWidth, kHeight, param,
                         static_cast<std::string*>(nullptr)));
  SJPEG_CHECK(!EncodeRGB(rgb, kWidth, 0, param, &out));
  SJPEG_CHECK(!sjpeg::EncodeGray(nullptr, kWidth, kHeight, kWidth, param,
                                 &out));
}

std::vector<uint8_t> MakePlane(int width, int height, int base) {
  g_seed = kSeed;
  std::vector<uint8_t> plane(static_cast<size_t>(width) * height);
  for (size_t i = 0; i < plane.size(); ++i) {
    plane[i] = static_cast<uint8_t>(base + (Random8b() >> 2));
  }
  return plane;
}

// Copies 'plane' into a buffer with the given stride, padding the extra bytes
// with a value that must never show up in the output.
std::vector<uint8_t> WithStride(const std::vector<uint8_t>& plane,
                                int width, int height, int stride) {
  std::vector<uint8_t> out(static_cast<size_t>(stride) * height, 0xd5);
  for (int y = 0; y < height; ++y) {
    memcpy(&out[static_cast<size_t>(y) * stride],
           &plane[static_cast<size_t>(y) * width], width);
  }
  return out;
}

typedef bool (*EncodeYUVFunc)(const uint8_t*, int, const uint8_t*, int,
                              const uint8_t*, int, int, int,
                              const sjpeg::EncoderParam&, sjpeg::ByteSink*);

// The padding bytes of the U/V planes must never reach the output, whatever
// the strides are. Dimensions are picked so that the last MCU row/column is
// clipped, since that's where the samples are replicated.
void CheckStrides(EncodeYUVFunc encode, int sub, int width, int height) {
  const int uv_w = (width + sub - 1) / sub, uv_h = (height + sub - 1) / sub;
  const std::vector<uint8_t> Y = MakePlane(width, height, 20);
  const std::vector<uint8_t> U = MakePlane(uv_w, uv_h, 60);
  const std::vector<uint8_t> V = MakePlane(uv_w, uv_h, 140);
  const sjpeg::EncoderParam param(80.f);
  std::string ref;
  for (int u_pad = 0; u_pad <= 7; ++u_pad) {
    for (int v_pad = 0; v_pad <= 7; v_pad += 7) {
      const int u_stride = uv_w + u_pad, v_stride = uv_w + v_pad;
      const std::vector<uint8_t> u = WithStride(U, uv_w, uv_h, u_stride);
      const std::vector<uint8_t> v = WithStride(V, uv_w, uv_h, v_stride);
      std::string out;
      SJPEG_CHECK(encode(Y.data(), width, u.data(), u_stride,
                         v.data(), v_stride, width, height, param,
                         sjpeg::MakeByteSink(&out).get()));
      if (ref.empty()) ref = out;
      SJPEG_CHECK(!out.empty() && out == ref);
    }
  }
}

// 17x13 is odd in both directions: the chroma planes have a half sample in
// the last row and column, on top of the clipped MCU.
TEST(EncodeYUV420Strides) {
  CheckStrides(&sjpeg::EncodeYUV420, 2, 20, 20);
  CheckStrides(&sjpeg::EncodeYUV420, 2, 17, 13);
}
TEST(EncodeYUV444Strides) {
  CheckStrides(&sjpeg::EncodeYUV444, 1, 20, 20);
  CheckStrides(&sjpeg::EncodeYUV444, 1, 17, 13);
}

TEST(EncodeNV) {
  const int kWidth = 18, kHeight = 14;
  const int uv_h = (kHeight + 1) / 2, uv_stride = 2 * ((kWidth + 1) / 2);
  const std::vector<uint8_t> Y = MakePlane(kWidth, kHeight, 30);
  const std::vector<uint8_t> UV = MakePlane(uv_stride, uv_h, 90);
  const sjpeg::EncoderParam param(75.f);
  std::string out12, out21;
  SJPEG_CHECK(sjpeg::EncodeNV12(Y.data(), kWidth, UV.data(), uv_stride,
                                kWidth, kHeight, param,
                                sjpeg::MakeByteSink(&out12).get()));
  SJPEG_CHECK(sjpeg::EncodeNV21(Y.data(), kWidth, UV.data(), uv_stride,
                                kWidth, kHeight, param,
                                sjpeg::MakeByteSink(&out21).get()));
  SJPEG_CHECK(HasSize(out12, kWidth, kHeight));
  SJPEG_CHECK(out12 != out21);   // U and V are swapped

  // one invalid argument at a time
  std::string out;
  const auto holder = sjpeg::MakeByteSink(&out);
  sjpeg::ByteSink* const sink = holder.get();
  const auto nv12 = [&](const uint8_t* y, int y_step, const uint8_t* uv,
                        int uv_step, int W, int H, sjpeg::ByteSink* s) {
    return sjpeg::EncodeNV12(y, y_step, uv, uv_step, W, H, param, s);
  };
  SJPEG_CHECK(!nv12(Y.data(), kWidth, UV.data(), uv_stride,
                    kWidth, kHeight, nullptr));
  SJPEG_CHECK(!nv12(nullptr, kWidth, UV.data(), uv_stride, kWidth, kHeight,
                    sink));
  SJPEG_CHECK(!nv12(Y.data(), kWidth, nullptr, uv_stride, kWidth, kHeight,
                    sink));
  SJPEG_CHECK(!nv12(Y.data(), kWidth, UV.data(), uv_stride, 0, kHeight, sink));
  SJPEG_CHECK(!nv12(Y.data(), kWidth - 1, UV.data(), uv_stride,
                    kWidth, kHeight, sink));
  SJPEG_CHECK(!nv12(Y.data(), kWidth, UV.data(), uv_stride - 1,
                    kWidth, kHeight, sink));
  SJPEG_CHECK(!sjpeg::EncodeNV21(Y.data(), kWidth, UV.data(), uv_stride,
                                 kWidth, kHeight, param, nullptr));
}

// Vertically flips 'height' rows of 'row_size' bytes.
std::vector<uint8_t> Flip(const std::vector<uint8_t>& src, int row_size,
                          int height) {
  std::vector<uint8_t> dst(src.size());
  for (int y = 0; y < height; ++y) {
    memcpy(&dst[static_cast<size_t>(y) * row_size],
           &src[static_cast<size_t>(height - 1 - y) * row_size], row_size);
  }
  return dst;
}

// Last row of 'p', to be paired with a negative stride.
const uint8_t* Last(const std::vector<uint8_t>& p, int row_size, int height) {
  return p.data() + static_cast<size_t>(height - 1) * row_size;
}

// Only |stride| is validated: a negative stride is legal, and describes a
// bottom-up buffer. It must encode exactly like the flipped picture does with
// a positive one. 17x13 is odd both ways, so the last MCU clips too.
TEST(NegativeStrides) {
  const int W = 17, H = 13, uv_w = (W + 1) / 2, uv_h = (H + 1) / 2;
  const int uv_stride = 2 * uv_w;
  const sjpeg::EncoderParam p(78.f);
  const std::vector<uint8_t> rgb = MakeRGB(W, H), Y = MakePlane(W, H, 20);
  const std::vector<uint8_t> U = MakePlane(uv_w, uv_h, 60);
  const std::vector<uint8_t> V = MakePlane(uv_w, uv_h, 140);
  const std::vector<uint8_t> UV = MakePlane(uv_stride, uv_h, 90);
  std::string a, b;
  const auto match = [&a, &b]() { SJPEG_CHECK(!a.empty() && a == b);
                                  a.clear();
                                  b.clear(); };
  SJPEG_CHECK(EncodeRGB(Flip(rgb, 3 * W, H), W, H, p, &a));
  SJPEG_CHECK(sjpeg::Encode(Last(rgb, 3 * W, H), W, H, -3 * W, p, &b));
  match();

  SJPEG_CHECK(sjpeg::EncodeYUV420(Flip(Y, W, H).data(), W,
                                  Flip(U, uv_w, uv_h).data(), uv_w,
                                  Flip(V, uv_w, uv_h).data(), uv_w, W, H, p,
                                  sjpeg::MakeByteSink(&a).get()));
  SJPEG_CHECK(sjpeg::EncodeYUV420(Last(Y, W, H), -W, Last(U, uv_w, uv_h), -uv_w,
                                  Last(V, uv_w, uv_h), -uv_w, W, H, p,
                                  sjpeg::MakeByteSink(&b).get()));
  match();

  SJPEG_CHECK(sjpeg::EncodeNV12(Flip(Y, W, H).data(), W,
                                Flip(UV, uv_stride, uv_h).data(), uv_stride,
                                W, H, p, sjpeg::MakeByteSink(&a).get()));
  SJPEG_CHECK(sjpeg::EncodeNV12(Last(Y, W, H), -W, Last(UV, uv_stride, uv_h),
                                -uv_stride, W, H, p,
                                sjpeg::MakeByteSink(&b).get()));
  match();
}

// Records every block it hands out, and refuses to release a pointer that
// doesn't come from it.
class TrackingMemory : public sjpeg::MemoryManager {
 public:
  virtual ~TrackingMemory() {}
  virtual void* Alloc(size_t size) {
    void* const ptr = malloc(size);
    if (ptr != nullptr) {
      ++num_allocs;
      live.push_back(ptr);
    }
    return ptr;
  }
  virtual void Free(void* const ptr) {
    if (ptr == nullptr) return;
    for (size_t i = 0; i < live.size(); ++i) {
      if (live[i] == ptr) {
        live.erase(live.begin() + i);
        free(ptr);
        return;
      }
    }
    ++num_foreign_frees;   // not ours: releasing it would corrupt the heap
  }
  int num_allocs = 0;
  int num_foreign_frees = 0;
  std::vector<void*> live;
};

TEST(MemoryManager) {
  const int kWidth = 40, kHeight = 24;
  const std::vector<uint8_t> rgb = MakeRGB(kWidth, kHeight);
  const SjpegYUVMode kModes[] = { SJPEG_YUV_420, SJPEG_YUV_SHARP,
                                  SJPEG_YUV_444, SJPEG_YUV_400 };
  for (size_t m = 0; m < ARRAY_SIZE(kModes); ++m) {
    TrackingMemory memory;
    sjpeg::EncoderParam param(75.f);
    param.yuv_mode = kModes[m];
    param.memory = &memory;
    std::string out;
    SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &out));
    SJPEG_CHECK(memory.num_allocs > 0);          // it was used at all
    SJPEG_CHECK(memory.num_foreign_frees == 0);  // and used for every free()
    SJPEG_CHECK(memory.live.empty());            // no leak
  }
}

// Dimensions are stored on 16 bits in the SOF marker. Anything larger must be
// refused rather than silently truncated.
TEST(LargeDimensions) {
  const int kMaxDim = 0xffff, kSmallDim = 2;
  const std::vector<uint8_t> rgb(
      3 * static_cast<size_t>(kMaxDim + 1) * kSmallDim, 0x80);
  const sjpeg::EncoderParam param(50.f);
  std::string out;
  SJPEG_CHECK(EncodeRGB(rgb, kMaxDim, kSmallDim, param, &out));
  SJPEG_CHECK(HasSize(out, kMaxDim, kSmallDim));
  SJPEG_CHECK(!EncodeRGB(rgb, kMaxDim + 1, kSmallDim, param, &out));
  SJPEG_CHECK(!EncodeRGB(rgb, kSmallDim, kMaxDim + 1, param, &out));
  SJPEG_CHECK(!sjpeg::EncodeGray(rgb.data(), kMaxDim + 1, kSmallDim,
                                 kMaxDim + 1, param, &out));
  uint8_t* data = nullptr;
  SJPEG_CHECK(SjpegEncode(rgb.data(), kMaxDim + 1, kSmallDim, 3 * (kMaxDim + 1),
                          &data, 50.f, 4, SJPEG_YUV_420) == 0);
  SJPEG_CHECK(data == nullptr);
}

// Refuses to allocate after the first 'num_ok' calls.
class FailingMemory : public TrackingMemory {
 public:
  explicit FailingMemory(int num_ok) : num_ok_(num_ok) {}
  virtual ~FailingMemory() {}
  virtual void* Alloc(size_t size) {
    if (num_ok_ <= 0) {
      ++num_refused;
      return nullptr;
    }
    --num_ok_;
    return TrackingMemory::Alloc(size);
  }
  int num_refused = 0;

 private:
  int num_ok_;
};

// An allocation failure must be reported, whatever the stage it occurs at,
// without crashing and without leaking.
TEST(AllocationFailure) {
  const int kWidth = 51, kHeight = 33;
  const std::vector<uint8_t> rgb = MakeRGB(kWidth, kHeight);
  const struct { sjpeg::EncoderParam::TargetMode mode; float value; } kTargets[]
      = { { sjpeg::EncoderParam::TARGET_NONE, 0.f },
          { sjpeg::EncoderParam::TARGET_SIZE, 2000.f },
          { sjpeg::EncoderParam::TARGET_PSNR, 38.f } };
  for (size_t t = 0; t < ARRAY_SIZE(kTargets); ++t) {
    for (int num_ok = 0; num_ok < 8; ++num_ok) {
      FailingMemory memory(num_ok);
      sjpeg::EncoderParam param(80.f);
      param.yuv_mode = SJPEG_YUV_420;
      param.memory = &memory;
      param.target_mode = kTargets[t].mode;
      param.target_value = kTargets[t].value;
      if (t > 0) param.passes = 5;
      std::string out;
      const bool ok = EncodeRGB(rgb, kWidth, kHeight, param, &out);
      SJPEG_CHECK(ok == (memory.num_refused == 0));
      SJPEG_CHECK(memory.live.empty());
    }
  }
}

TEST(Dimensions) {
  const int kWidth = 35, kHeight = 19;
  const std::vector<uint8_t> rgb = MakeRGB(kWidth, kHeight);
  std::string jpg;
  SJPEG_CHECK(SjpegCompress(rgb.data(), kWidth, kHeight, 60.f, &jpg));
  const uint8_t* const data = reinterpret_cast<const uint8_t*>(jpg.data());
  const size_t size = jpg.size();

  // all three pointers are optional
  int width = 0, height = 0, is_yuv420 = -1;
  SJPEG_CHECK(SjpegDimensions(jpg, &width, &height, &is_yuv420));
  SJPEG_CHECK(width == kWidth && height == kHeight);
  width = height = 0;
  SJPEG_CHECK(SjpegDimensions(data, size, &width, nullptr, nullptr));
  SJPEG_CHECK(width == kWidth);
  SJPEG_CHECK(SjpegDimensions(data, size, nullptr, &height, nullptr));
  SJPEG_CHECK(height == kHeight);
  SJPEG_CHECK(SjpegDimensions(data, size, nullptr, nullptr, &is_yuv420));
  SJPEG_CHECK(SjpegDimensions(data, size, nullptr, nullptr, nullptr));

  // truncated or invalid input is rejected, and never read out of bounds
  SJPEG_CHECK(!SjpegDimensions(nullptr, size, &width, &height, nullptr));
  for (size_t n = 0; n <= size; ++n) {
    int w = -1, h = -1, yuv420 = -1;
    if (SjpegDimensions(data, n, &w, &h, &yuv420)) {
      SJPEG_CHECK(w == kWidth && h == kHeight);   // never a partial answer
    }
  }
}

// Flat picture of the given color.
std::vector<uint8_t> MakeFlatRGB(int width, int height, int r, int g, int b) {
  const uint8_t color[3] = { static_cast<uint8_t>(r), static_cast<uint8_t>(g),
                             static_cast<uint8_t>(b) };
  std::vector<uint8_t> rgb(3 * static_cast<size_t>(width) * height);
  for (size_t i = 0; i < rgb.size(); ++i) rgb[i] = color[i % 3];
  return rgb;
}

TEST(Riskiness) {
  const int kSizes[] = { 8, 16, 32, 64, 128, 400 };
  for (size_t s = 0; s < ARRAY_SIZE(kSizes); ++s) {
    const int size = kSizes[s];
    // a gray picture is detected as such, whatever its dimensions
    const std::vector<uint8_t> gray = MakeFlatRGB(size, size, 128, 128, 128);
    float risk = -1.f;
    SJPEG_CHECK(SjpegRiskiness(gray.data(), size, size, 3 * size, &risk)
                == SJPEG_YUV_400);
    SJPEG_CHECK(risk >= 0.f && risk <= 100.f);

    // A flat but tinted picture is not gray either, even though its packed
    // y/u/v index sits close to the one of the gray level.
    const std::vector<uint8_t> tint = MakeFlatRGB(size, size, 140, 120, 90);
    SJPEG_CHECK(SjpegRiskiness(tint.data(), size, size, 3 * size, nullptr)
                != SJPEG_YUV_400);

    // and neither is a colored one
    std::vector<uint8_t> color(3 * static_cast<size_t>(size) * size);
    for (int y = 0; y < size; ++y) {
      for (int x = 0; x < size; ++x) {
        uint8_t* const p = &color[3 * (x + static_cast<size_t>(y) * size)];
        p[0] = ((x ^ y) & 8) ? 220 : 20;
        p[1] = 40;
        p[2] = ((x ^ y) & 8) ? 20 : 220;
      }
    }
    SJPEG_CHECK(SjpegRiskiness(color.data(), size, size, 3 * size, nullptr)
                != SJPEG_YUV_400);
  }
  // end to end: YUV_AUTO on a gray picture emits a single quantization matrix
  const int kWidth = 40, kHeight = 24;
  const std::vector<uint8_t> gray = MakeFlatRGB(kWidth, kHeight, 90, 90, 90);
  sjpeg::EncoderParam param(75.f);
  param.yuv_mode = SJPEG_YUV_AUTO;
  std::string out;
  SJPEG_CHECK(EncodeRGB(gray, kWidth, kHeight, param, &out));
  uint8_t quant[2][64];
  SJPEG_CHECK(SjpegFindQuantizer(out, quant) == 1);
}

// TARGET_SIZE converges by comparing ComputeSize(), which adds HeaderSize(),
// against the requested value. SJPEG_YUV_400 writes a single quantization
// matrix where the other modes write two: charging it for both (67 bytes)
// makes the search settle on the wrong quality. SJPEG_YUV_420 is the control.
TEST(TargetSize) {
  const int W = 96, H = 64;
  const std::vector<uint8_t> rgb = MakeRGB(W, H);
  const SjpegYUVMode kModes[] = { SJPEG_YUV_400, SJPEG_YUV_420 };
  for (size_t m = 0; m < ARRAY_SIZE(kModes); ++m) {
    // a plain encode gives a size that the search is sure to be able to reach
    sjpeg::EncoderParam param(60.f);
    param.yuv_mode = kModes[m];
    std::string out;
    SJPEG_CHECK(EncodeRGB(rgb, W, H, param, &out));
    const double target = out.size();

    param.target_mode = sjpeg::EncoderParam::TARGET_SIZE;
    param.target_value = static_cast<float>(target);
    param.tolerance = 1.f;   // percent
    param.passes = 12;
    SJPEG_CHECK(EncodeRGB(rgb, W, H, param, &out));
    // The search stops on |dq| rather than on the size, so it only lands
    // close by. One quantization matrix too many costs several percent.
    SJPEG_CHECK(fabs(out.size() - target) < 0.03 * target);
  }
}

// Behaves like a memory sink, but starts refusing to commit after a while.
class FailingSink : public sjpeg::ByteSink {
 public:
  explicit FailingSink(int num_ok) : num_ok_(num_ok) {}
  virtual ~FailingSink() {}
  virtual bool Commit(size_t used_size, size_t extra_size, uint8_t** data) {
    pos_ += used_size;
    if (num_ok_ <= 0) return false;
    --num_ok_;
    buf_.resize(pos_ + extra_size);
    *data = extra_size ? &buf_[pos_] : nullptr;
    return true;
  }
  virtual bool Finalize() { buf_.resize(pos_); return true; }
  virtual void Reset() { buf_.clear(); pos_ = 0; }

 private:
  std::vector<uint8_t> buf_;
  size_t pos_ = 0;
  int num_ok_;
};

// A sink that stops accepting data must be reported, not crash. This is what
// a file sink running out of space looks like.
TEST(SinkFailure) {
  const int kWidth = 32, kHeight = 32;
  const std::vector<uint8_t> rgb = MakeRGB(kWidth, kHeight);
  bool seen_failure = false, seen_success = false;
  for (int num_ok = 0; num_ok < 64; ++num_ok) {
    FailingSink sink(num_ok);
    sjpeg::EncoderParam param(90.f);
    param.yuv_mode = SJPEG_YUV_420;
    if (EncodeRGB(rgb, kWidth, kHeight, param, &sink)) {
      seen_success = true;
    } else {
      seen_failure = true;
    }
  }
  SJPEG_CHECK(seen_failure);   // the test is only meaningful if both happened
  SJPEG_CHECK(seen_success);
}

TEST(CompressionMethod) {
  const int kWidth = 48, kHeight = 32;
  const std::vector<uint8_t> rgb = MakeRGB(kWidth, kHeight);
  std::string out[11];
  for (int method = -1; method <= 9; ++method) {
    uint8_t* data = nullptr;
    const size_t size = SjpegEncode(rgb.data(), kWidth, kHeight, 3 * kWidth,
                                    &data, 76.f, method, SJPEG_YUV_420);
    SJPEG_CHECK(size > 0 && data != nullptr);
    if (data != nullptr) {
      out[method + 1].assign(reinterpret_cast<const char*>(data), size);
      SJPEG_CHECK(HasSize(out[method + 1], kWidth, kHeight));
    }
    SjpegFreeBuffer(data);
  }
  // methods outside of [0..8] are clamped to the nearest valid one
  SJPEG_CHECK(out[0] == out[1]);     // -1 -> 0
  SJPEG_CHECK(out[10] == out[9]);    //  9 -> 8
}

TEST(QuantMatrix) {
  for (int quality = 0; quality <= 100; quality += 5) {
    for (int chroma = 0; chroma <= 1; ++chroma) {
      uint8_t matrix[64];
      SjpegQuantMatrix(quality, chroma != 0, matrix);
      for (size_t i = 0; i < 64; ++i) SJPEG_CHECK(matrix[i] >= 1);
      const float estimate = SjpegEstimateQuality(matrix, chroma != 0);
      SJPEG_CHECK(estimate >= quality - 1.f && estimate <= quality + 1.f);
    }
  }
  // The matrices used for encoding must be recoverable from the bitstream.
  const int kWidth = 24, kHeight = 24;
  const std::vector<uint8_t> rgb = MakeRGB(kWidth, kHeight);
  sjpeg::EncoderParam param(70.f);
  param.adaptive_quantization = false;
  std::string out;
  SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &out));
  uint8_t quant[2][64];
  SJPEG_CHECK(SjpegFindQuantizer(out, quant) == 2);
  SJPEG_CHECK(memcmp(quant[0], param.GetQuantMatrix(0), 64) == 0);
  SJPEG_CHECK(memcmp(quant[1], param.GetQuantMatrix(1), 64) == 0);
}

#if !defined(SJPEG_NO_PROGRESSIVE)
TEST(Progressive) {
  const int kWidth = 40, kHeight = 24;
  const std::vector<uint8_t> rgb = MakeRGB(kWidth, kHeight);

  // Default (progressive off) must be a byte-for-byte no-op: explicitly
  // passing the "off" sentinel (64) must match not touching the field at all.
  {
    sjpeg::EncoderParam param1(75.f);
    sjpeg::EncoderParam param2(75.f);
    param2.progressive_luma_split = 64;
    std::string out1, out2;
    SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param1, &out1));
    SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param2, &out2));
    SJPEG_CHECK(!out1.empty() && out1 == out2);
    int sof = 0, num_sos = 0;
    SJPEG_CHECK(CheckMarkerStructure(out1, &sof, &num_sos));
    SJPEG_CHECK(sof == 0xc0);
    SJPEG_CHECK(num_sos == 1);
  }
  // Progressive on, default chroma split: DC + luma(2 bands) + Cb(2) + Cr(2).
  {
    sjpeg::EncoderParam param(75.f);
    param.progressive_luma_split = 20;
    std::string out;
    SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &out));
    int sof = 0, num_sos = 0;
    SJPEG_CHECK(CheckMarkerStructure(out, &sof, &num_sos));
    SJPEG_CHECK(sof == 0xc2);
    SJPEG_CHECK(num_sos == 7);
  }
  // Luma-only split: chroma_split == 64 disables the split for chroma only,
  // without needing a separate on/off flag.
  {
    sjpeg::EncoderParam param(75.f);
    param.progressive_luma_split = 20;
    param.progressive_chroma_split = 64;
    std::string out;
    SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &out));
    int sof = 0, num_sos = 0;
    SJPEG_CHECK(CheckMarkerStructure(out, &sof, &num_sos));
    SJPEG_CHECK(sof == 0xc2);
    SJPEG_CHECK(num_sos == 5);  // DC + luma(2) + Cb(1) + Cr(1)
  }
  // Extreme split points.
  for (const int split : {1, 62}) {
    sjpeg::EncoderParam param(75.f);
    param.progressive_luma_split = split;
    std::string out;
    SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &out));
    int sof = 0, num_sos = 0;
    SJPEG_CHECK(CheckMarkerStructure(out, &sof, &num_sos));
    SJPEG_CHECK(sof == 0xc2);
  }
  // Regression: a wide, dense row used to overflow the AC emit loop's
  // per-row output-buffer reservation (fixed: check per-block, not per-row).
  {
    const int kWideWidth = 2000, kWideHeight = 32;
    const std::vector<uint8_t> wide_rgb = MakeRGB(kWideWidth, kWideHeight);
    sjpeg::EncoderParam param(95.f);
    param.progressive_luma_split = 20;
    std::string out;
    SJPEG_CHECK(EncodeRGB(wide_rgb, kWideWidth, kWideHeight, param, &out));
    int sof = 0, num_sos = 0;
    SJPEG_CHECK(CheckMarkerStructure(out, &sof, &num_sos));
    SJPEG_CHECK(sof == 0xc2);
    SJPEG_CHECK(num_sos == 7);
  }
  // Same guarantee as AllocationFailure above, for progressive's own plane
  // storage (AllocateProgPlanes()/DeallocateProgPlanes()).
  for (int num_ok = 0; num_ok < 10; ++num_ok) {
    FailingMemory memory(num_ok);
    sjpeg::EncoderParam param(75.f);
    param.progressive_luma_split = 2;
    param.memory = &memory;
    std::string out;
    const bool ok = EncodeRGB(rgb, kWidth, kHeight, param, &out);
    SJPEG_CHECK(ok == (memory.num_refused == 0));
    SJPEG_CHECK(memory.live.empty());
  }
  // -prog under multi-pass search: see the LoopScan() note in Encode().
  {
    sjpeg::EncoderParam param(75.f);
    param.progressive_luma_split = 2;
    std::string out;
    SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &out));

    sjpeg::EncoderParam param2(75.f);
    param2.progressive_luma_split = 2;
    param2.target_mode = sjpeg::EncoderParam::TARGET_SIZE;
    param2.target_value = (float)out.size();
    param2.passes = 5;
    std::string out2;
    SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param2, &out2));
    int sof = 0, num_sos = 0;
    SJPEG_CHECK(CheckMarkerStructure(out2, &sof, &num_sos));
    SJPEG_CHECK(sof == 0xc0);
    SJPEG_CHECK(num_sos == 1);
  }
  // A flat image forces the EOBn run past kMaxEOBRun (32767), needing a
  // mid-scan flush.
  {
    const int kFlatWidth = 1024, kFlatHeight = 2048;
    const std::vector<uint8_t> flat_rgb(kFlatWidth * kFlatHeight * 3, 128);
    sjpeg::EncoderParam param(90.f);
    param.yuv_mode = SJPEG_YUV_420;
    param.progressive_luma_split = 20;
    std::string out;
    SJPEG_CHECK(EncodeRGB(flat_rgb, kFlatWidth, kFlatHeight, param, &out));
    int sof = 0, num_sos = 0;
    SJPEG_CHECK(CheckMarkerStructure(out, &sof, &num_sos));
    SJPEG_CHECK(sof == 0xc2);
    SJPEG_CHECK(num_sos == 7);
  }
  // Regression: trellis + progressive used ac_codes_[] uninitialized
  // (InitCodes(true) was missing), tripping an assert in SearchBestPrev().
  {
    sjpeg::EncoderParam param(75.f);
    param.use_trellis = true;
    param.progressive_luma_split = 2;
    std::string out;
    SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &out));
    int sof = 0, num_sos = 0;
    SJPEG_CHECK(CheckMarkerStructure(out, &sof, &num_sos));
    SJPEG_CHECK(sof == 0xc2);
  }
}
#endif  // !SJPEG_NO_PROGRESSIVE

}  // namespace

int main(int argc, char* argv[]) {
  const std::vector<TestCase>& tests = Tests();
  int num_run = 0;
  for (size_t n = 0; n < tests.size(); ++n) {
    // with arguments, only run the tests named on the command line
    bool selected = (argc <= 1);
    for (int c = 1; c < argc; ++c) selected |= !strcmp(argv[c], tests[n].name);
    if (!selected) continue;
    const int failures = g_num_failures;
    g_test_name = tests[n].name;
    tests[n].func();
    ++num_run;
    printf("%-4s %s\n", (g_num_failures == failures) ? "ok" : "FAIL",
           tests[n].name);
  }
  printf("--\n%d test(s), %d check(s), %d failure(s)\n",
         num_run, g_num_checks, g_num_failures);
  if (num_run == 0) {
    printf("no test was run!\n");
    return 1;
  }
  return (g_num_failures == 0) ? 0 : 1;
}
