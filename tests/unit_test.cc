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

#include <assert.h>
#include <climits>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <climits>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "sjpegi.h"
#include "sjpeg.h"

namespace sjpeg {
extern bool ForceSlowCImplementation;
}  // namespace sjpeg

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

#define SJPEG_TEST(Name)                                    \
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
  return (uint8_t)(g_seed >> 16);
}

// Noisy picture with some structure, hard to compress.
std::vector<uint8_t> MakeRGB(int width, int height) {
  g_seed = kSeed;
  std::vector<uint8_t> rgb(3 * (size_t)width * height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      uint8_t* const p = &rgb[3 * (x + (size_t)y * width)];
      p[0] = (uint8_t)(x * 5 + (Random8b() >> 3));
      p[1] = (uint8_t)(y * 3 + (Random8b() >> 4));
      p[2] = (uint8_t)(((x / 8) ^ (y / 8)) * 51);
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

// Only used by SJPEG_TEST(Progressive) below.
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
SJPEG_TEST(Threads) {
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

SJPEG_TEST(Compress) {
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

SJPEG_TEST(EncodeParams) {
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

SJPEG_TEST(InvalidArguments) {
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
  SJPEG_CHECK(enc(rgb.data(), kWidth, kHeight, 3 * kWidth, &data,
                  (SjpegYUVMode)7) == 0);
  SJPEG_CHECK(data == nullptr);
  const sjpeg::EncoderParam param;
  std::string out;
  SJPEG_CHECK(!sjpeg::Encode(nullptr, kWidth, kHeight, 3 * kWidth, param,
                             &out));
  SJPEG_CHECK(!EncodeRGB(rgb, kWidth, kHeight, param, (std::string*)nullptr));
  SJPEG_CHECK(!EncodeRGB(rgb, kWidth, 0, param, &out));
  SJPEG_CHECK(!sjpeg::EncodeGray(nullptr, kWidth, kHeight, kWidth, param,
                                 &out));

  uint8_t dummy_y = 0, dummy_u = 0, dummy_v = 0;
  SJPEG_CHECK(!sjpeg::ApplySharpYUVConversion(nullptr, kWidth, kHeight,
                                              3 * kWidth, &dummy_y, &dummy_u,
                                              &dummy_v));
  SJPEG_CHECK(!sjpeg::ApplySharpYUVConversion(rgb.data(), 0, kHeight,
                                              3 * kWidth, &dummy_y, &dummy_u,
                                              &dummy_v));
  SJPEG_CHECK(!sjpeg::ApplySharpYUVConversion(rgb.data(), kWidth, -1,
                                              3 * kWidth, &dummy_y, &dummy_u,
                                              &dummy_v));
  SJPEG_CHECK(!sjpeg::ApplySharpYUVConversion(rgb.data(), kWidth, kHeight,
                                              3 * kWidth - 1, &dummy_y,
                                              &dummy_u, &dummy_v));
  SJPEG_CHECK(!sjpeg::ApplySharpYUVConversion(rgb.data(), kWidth, kHeight,
                                              INT_MIN, &dummy_y, &dummy_u,
                                              &dummy_v));
  SJPEG_CHECK(!sjpeg::ApplySharpYUVConversion(rgb.data(), kWidth, kHeight,
                                              3 * kWidth, nullptr, &dummy_u,
                                              &dummy_v));
  SJPEG_CHECK(!sjpeg::ApplySharpYUVConversion(rgb.data(), 70000, 70000,
                                              3 * 70000, &dummy_y, &dummy_u,
                                              &dummy_v));
}

std::vector<uint8_t> MakePlane(int width, int height, int base) {
  g_seed = kSeed;
  std::vector<uint8_t> plane((size_t)width * height);
  for (size_t i = 0; i < plane.size(); ++i) {
    plane[i] = (uint8_t)(base + (Random8b() >> 2));
  }
  return plane;
}

// Copies 'plane' into a buffer with the given stride, padding the extra bytes
// with a value that must never show up in the output.
std::vector<uint8_t> WithStride(const std::vector<uint8_t>& plane,
                                int width, int height, int stride) {
  std::vector<uint8_t> out((size_t)stride * height, 0xd5);
  for (int y = 0; y < height; ++y) {
    memcpy(&out[(size_t)y * stride], &plane[(size_t)y * width], width);
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
SJPEG_TEST(EncodeYUV420Strides) {
  CheckStrides(&sjpeg::EncodeYUV420, 2, 20, 20);
  CheckStrides(&sjpeg::EncodeYUV420, 2, 17, 13);
}
SJPEG_TEST(EncodeYUV444Strides) {
  CheckStrides(&sjpeg::EncodeYUV444, 1, 20, 20);
  CheckStrides(&sjpeg::EncodeYUV444, 1, 17, 13);
}

SJPEG_TEST(EncodeNV) {
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
    memcpy(&dst[(size_t)y * row_size],
           &src[(size_t)(height - 1 - y) * row_size], row_size);
  }
  return dst;
}

// Last row of 'p', to be paired with a negative stride.
const uint8_t* Last(const std::vector<uint8_t>& p, int row_size, int height) {
  return p.data() + (size_t)(height - 1) * row_size;
}

// Only |stride| is validated: a negative stride is legal, and describes a
// bottom-up buffer. It must encode exactly like the flipped picture does with
// a positive one. 17x13 is odd both ways, so the last MCU clips too.
SJPEG_TEST(NegativeStrides) {
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
  for (SjpegYUVMode mode : {SJPEG_YUV_420, SJPEG_YUV_AUTO, SJPEG_YUV_SHARP}) {
    sjpeg::EncoderParam mode_p = p;
    mode_p.yuv_mode = mode;
    SJPEG_CHECK(EncodeRGB(Flip(rgb, 3 * W, H), W, H, mode_p, &a));
    SJPEG_CHECK(sjpeg::Encode(Last(rgb, 3 * W, H), W, H, -3 * W, mode_p, &b));
    match();
  }

  const std::vector<uint8_t> rgba = MakePlane(4 * W, H, 45);
  for (SjpegYUVMode mode : {SJPEG_YUV_420, SJPEG_YUV_AUTO, SJPEG_YUV_SHARP}) {
    sjpeg::EncoderParam mode_p = p;
    mode_p.yuv_mode = mode;
    SJPEG_CHECK(sjpeg::EncodeRGBA(Flip(rgba, 4 * W, H).data(), W, H, 4 * W,
                                  mode_p, sjpeg::MakeByteSink(&a).get()));
    SJPEG_CHECK(sjpeg::EncodeRGBA(Last(rgba, 4 * W, H), W, H, -4 * W, mode_p,
                                  sjpeg::MakeByteSink(&b).get()));
    match();
    SJPEG_CHECK(sjpeg::EncodeBGRA(Flip(rgba, 4 * W, H).data(), W, H, 4 * W,
                                  mode_p, sjpeg::MakeByteSink(&a).get()));
    SJPEG_CHECK(sjpeg::EncodeBGRA(Last(rgba, 4 * W, H), W, H, -4 * W, mode_p,
                                  sjpeg::MakeByteSink(&b).get()));
    match();
  }

  SJPEG_CHECK(sjpeg::EncodeYUV420(Flip(Y, W, H).data(), W,
                                  Flip(U, uv_w, uv_h).data(), uv_w,
                                  Flip(V, uv_w, uv_h).data(), uv_w, W, H, p,
                                  sjpeg::MakeByteSink(&a).get()));
  SJPEG_CHECK(sjpeg::EncodeYUV420(Last(Y, W, H), -W, Last(U, uv_w, uv_h), -uv_w,
                                  Last(V, uv_w, uv_h), -uv_w, W, H, p,
                                  sjpeg::MakeByteSink(&b).get()));
  match();

  SJPEG_CHECK(sjpeg::EncodeNV12(Flip(Y, W, H).data(), W,
                                Flip(UV, uv_stride, uv_h).data(), uv_stride, W,
                                H, p, sjpeg::MakeByteSink(&a).get()));
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

SJPEG_TEST(MemoryManager) {
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
SJPEG_TEST(LargeDimensions) {
  const int kMaxDim = 0xffff, kSmallDim = 2;
  const std::vector<uint8_t> rgb(3 * (size_t)(kMaxDim + 1) * kSmallDim, 0x80);
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
SJPEG_TEST(AllocationFailure) {
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

SJPEG_TEST(Dimensions) {
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
  const uint8_t color[3] = {(uint8_t)r, (uint8_t)g, (uint8_t)b};
  std::vector<uint8_t> rgb(3 * (size_t)width * height);
  for (size_t i = 0; i < rgb.size(); ++i) rgb[i] = color[i % 3];
  return rgb;
}

SJPEG_TEST(Riskiness) {
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
    std::vector<uint8_t> color(3 * (size_t)size * size);
    for (int y = 0; y < size; ++y) {
      for (int x = 0; x < size; ++x) {
        uint8_t* const p = &color[3 * (x + (size_t)y * size)];
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

SJPEG_TEST(RiskinessScoreRow) {
  const int kNoiseLevel = 4;
  const int kRGB3 = sjpeg::kRGBSize * sjpeg::kRGBSize * sjpeg::kRGBSize;

  const int kMaxWidth = 2048;
  std::vector<uint16_t> row1(kMaxWidth + 16), row2(kMaxWidth + 16);
  for (int i = 0; i < kMaxWidth + 16; ++i) {
    row1[i] = (uint16_t)((i * 17 + 23) % kRGB3);
    row2[i] = (uint16_t)((i * 31 + 47) % kRGB3);
  }

  const int kTestWidths[] = {
      1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
      127, 128, 255, 256, 333, 512, 1001, 1024, 1920, 2048};

  // Obtain SIMD and scalar C dispatch function pointers.
  sjpeg::ForceSlowCImplementation = false;
  const sjpeg::RiskinessScoreRowFunc score_simd =
      sjpeg::GetRiskinessScoreRowFunc();
  sjpeg::ForceSlowCImplementation = true;
  const sjpeg::RiskinessScoreRowFunc score_c =
      sjpeg::GetRiskinessScoreRowFunc();
  sjpeg::ForceSlowCImplementation = false;

  for (size_t w_idx = 0; w_idx < ARRAY_SIZE(kTestWidths); ++w_idx) {
    const int width = kTestWidths[w_idx];
    int64_t score_sum_simd = 0, score_num_simd = 0, gray_num_simd = 0;
    score_simd(row1.data(), row2.data(), width - 1, kNoiseLevel,
               &score_sum_simd, &score_num_simd, &gray_num_simd);

    int64_t score_sum_c = 0, score_num_c = 0, gray_num_c = 0;
    score_c(row1.data(), row2.data(), width - 1, kNoiseLevel,
            &score_sum_c, &score_num_c, &gray_num_c);

    SJPEG_CHECK(score_sum_simd == score_sum_c);
    SJPEG_CHECK(score_num_simd == score_num_c);
    SJPEG_CHECK(gray_num_simd == gray_num_c);
  }
}

SJPEG_TEST(RowToIndex) {
  const auto simd_func = sjpeg::GetRowFunc();
  sjpeg::ForceSlowCImplementation = true;
  const auto c_func = sjpeg::GetRowFunc();
  sjpeg::ForceSlowCImplementation = false;

  const int kEdgeValues[] = {0, 85, 128, 170, 255};
  for (int edge : kEdgeValues) {
    for (int w = 1; w <= 64; ++w) {
      std::vector<uint8_t> rgb(w * 3, edge);
      std::vector<uint16_t> dst_simd(w, 0);
      std::vector<uint16_t> dst_c(w, 0);
      simd_func(rgb.data(), w, dst_simd.data());
      c_func(rgb.data(), w, dst_c.data());
      for (int i = 0; i < w; ++i) {
        SJPEG_CHECK(dst_simd[i] == dst_c[i]);
      }
    }
  }
  for (int w = 1; w <= 64; ++w) {
    std::vector<uint8_t> rgb(w * 3);
    for (int i = 0; i < w * 3; ++i) {
      rgb[i] = (i * 29 + 85) % 256;
    }
    std::vector<uint16_t> dst_simd(w, 0);
    std::vector<uint16_t> dst_c(w, 0);
    simd_func(rgb.data(), w, dst_simd.data());
    c_func(rgb.data(), w, dst_c.data());
    for (int i = 0; i < w; ++i) {
      SJPEG_CHECK(dst_simd[i] == dst_c[i]);
    }
  }
}

// TARGET_SIZE converges by comparing ComputeSize(), which adds HeaderSize(),
// against the requested value. SJPEG_YUV_400 writes a single quantization
// matrix where the other modes write two: charging it for both (67 bytes)
// makes the search settle on the wrong quality. SJPEG_YUV_420 is the control.
SJPEG_TEST(TargetSize) {
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
    param.target_value = (float)target;
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
SJPEG_TEST(SinkFailure) {
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

SJPEG_TEST(CompressionMethod) {
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

SJPEG_TEST(QuantMatrix) {
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
SJPEG_TEST(Progressive) {
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

// Regression test: on bright/unshifted blocks in [0, 255], 16-bit intermediate
// addition could overflow and wrap around, producing a negative DC
// coefficient (-128) instead of positive (+32640).
SJPEG_TEST(FdctOverflow) {
  const sjpeg::FdctFunc fDCT = sjpeg::GetFdct();
  int16_t block[64];

  for (int i = 0; i < 64; ++i) block[i] = 255;
  fDCT(block, 1);
  assert(block[0] > 0);
  SJPEG_CHECK(block[0] > 0);
  SJPEG_CHECK(block[0] == 32640);

  for (int i = 0; i < 64; ++i) block[i] = 128;
  fDCT(block, 1);
  assert(block[0] > 0);
  SJPEG_CHECK(block[0] > 0);
  SJPEG_CHECK(block[0] == 16384);

  for (int i = 0; i < 64; ++i) block[i] = -128;
  fDCT(block, 1);
  assert(block[0] < 0);
  SJPEG_CHECK(block[0] < 0);
  SJPEG_CHECK(block[0] == -16384);

  for (int i = 0; i < 64; ++i) block[i] = 127;
  fDCT(block, 1);
  assert(block[0] > 0);
  SJPEG_CHECK(block[0] > 0);
  SJPEG_CHECK(block[0] == 16256);
}

#if defined(SJPEG_USE_NEON)
// QuantizeErrorNEON reduces its uint32x4_t sum via a signed reinterpret
// cast (see HorizontalSumS32 in simd.h); check that it still reports the
// right value once the per-block error wraps past 2^31 / 2^32.
SJPEG_TEST(QuantizeErrorNEONOverflow) {
  sjpeg::Quantizer Q = {};
  int16_t in[64];
  for (int j = 0; j < 64; ++j) {
    in[j] = 0;
    Q.quant_[j] = 255;
    Q.iquant_[j] = 65535;
    Q.bias_[j] = 65535;
  }

  // Ground truth: replicate QuantizeErrorNEON's per-lane arithmetic
  // (including its 16-bit truncation) in plain unsigned C++.
  uint32_t expected = 0;
  for (int j = 0; j < 64; ++j) {
    const uint16_t v0_raw = (uint16_t)((in[j] < 0) ? -in[j] : in[j]);
    const uint16_t sum_bias = (uint16_t)(v0_raw + Q.bias_[j]);
    const uint32_t prod = (uint32_t)sum_bias * Q.iquant_[j];
    const uint16_t e = (uint16_t)((uint16_t)(prod >> 16) >> sjpeg::AC_BITS);
    const uint16_t f = (uint16_t)(e * Q.quant_[j]);
    const uint16_t v0 = (uint16_t)(v0_raw >> sjpeg::AC_BITS);
    const uint16_t g = (f > v0) ? (f - v0) : (v0 - f);
    expected += (uint32_t)g * g;
  }
  SJPEG_CHECK(expected > (1u << 31));  // actually crosses the sign bit

  const sjpeg::QuantizeErrorTestFunc quantize_error =
      sjpeg::GetQuantizeErrorFuncForTest();
  SJPEG_CHECK(quantize_error(in, &Q) == expected);
}
#endif  // SJPEG_USE_NEON

SJPEG_TEST(RestartMarkers) {
  const int kWidth = 64, kHeight = 64;
  const std::vector<uint8_t> rgb = MakeRGB(kWidth, kHeight);

  const uint8_t kMarkerByteDRI = 0xdd;   // define restart interval
  const uint8_t kMarkerByteRST0 = 0xd0;  // first of the RST0..RST7 cycle

  auto HasMarker = [](const std::string& jpeg, uint8_t marker_byte) {
    const uint8_t* data = reinterpret_cast<const uint8_t*>(jpeg.data());
    const size_t size = jpeg.size();
    for (size_t i = 0; i + 1 < size; ++i) {
      if (data[i] == 0xff && data[i + 1] == marker_byte) return true;
    }
    return false;
  };

  auto ExtractDRI = [](const std::string& jpeg) -> uint16_t {
    const uint8_t* data = reinterpret_cast<const uint8_t*>(jpeg.data());
    const size_t size = jpeg.size();
    for (size_t i = 0; i + 5 < size; ++i) {
      if (data[i] == 0xff && data[i + 1] == kMarkerByteDRI &&
          data[i + 2] == 0x00 && data[i + 3] == 0x04) {
        return ((uint16_t)data[i + 4] << 8) | data[i + 5];
      }
    }
    return 0;
  };

  // Test 1: restart_interval_rows = 0 -> no DRI marker
  {
    sjpeg::EncoderParam param(85.0f);
    param.restart_interval_rows = 0;
    std::string out;
    SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &out));
    SJPEG_CHECK(!HasMarker(out, kMarkerByteDRI));
    SJPEG_CHECK(ExtractDRI(out) == 0);
    SJPEG_CHECK(HasSize(out, kWidth, kHeight));
  }

  // Test 2: restart_interval_rows = 1 in YUV420 -> DRI = 4 MCUs per row, RST0
  // present
  {
    sjpeg::EncoderParam param(85.0f);
    param.restart_interval_rows = 1;
    param.yuv_mode = SJPEG_YUV_420;
    std::string out;
    SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &out));
    SJPEG_CHECK(HasMarker(out, kMarkerByteDRI));
    SJPEG_CHECK(ExtractDRI(out) == 4);
    SJPEG_CHECK(HasMarker(out, kMarkerByteRST0));
    SJPEG_CHECK(HasSize(out, kWidth, kHeight));
  }

  // Test 3: restart_interval_rows = 2 in YUV420 -> DRI = 8 MCUs
  {
    sjpeg::EncoderParam param(85.0f);
    param.restart_interval_rows = 2;
    param.yuv_mode = SJPEG_YUV_420;
    std::string out;
    SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &out));
    SJPEG_CHECK(HasMarker(out, kMarkerByteDRI));
    SJPEG_CHECK(ExtractDRI(out) == 8);
    SJPEG_CHECK(HasSize(out, kWidth, kHeight));
  }

  // Number of MCUs per row, which is what DRI counts: 16x16 MCUs in 4:2:0,
  // 8x8 otherwise.
  auto MCUsPerRow = [](int w, SjpegYUVMode yuv_mode) {
    const int block_w = (yuv_mode == SJPEG_YUV_420) ? 16 : 8;
    return (w + block_w - 1) / block_w;
  };

  // Test 4: Decodability across resolutions, intervals, YUV modes, and Huffman
  // modes. DRI must hold the exact MCU count matching the requested rows.
  const int dimensions[][2] = {{16, 16}, {64, 64}, {127, 93}, {320, 240}};
  const int intervals[] = {0, 1, 2, 4};
  const SjpegYUVMode yuv_modes[] = {SJPEG_YUV_420, SJPEG_YUV_444,
                                    SJPEG_YUV_400};

  for (size_t d = 0; d < ARRAY_SIZE(dimensions); ++d) {
    const int w = dimensions[d][0];
    const int h = dimensions[d][1];
    const std::vector<uint8_t> test_rgb = MakeRGB(w, h);
    for (size_t i = 0; i < ARRAY_SIZE(intervals); ++i) {
      const int restart = intervals[i];
      for (size_t y = 0; y < ARRAY_SIZE(yuv_modes); ++y) {
        for (int opt = 0; opt <= 1; ++opt) {
          sjpeg::EncoderParam param(80.0f);
          param.yuv_mode = yuv_modes[y];
          param.restart_interval_rows = restart;
          param.Huffman_compress = (opt == 1);
          std::string out;
          SJPEG_CHECK(EncodeRGB(test_rgb, w, h, param, &out));
          SJPEG_CHECK(HasSize(out, w, h));
          if (restart > 0) {
            SJPEG_CHECK(HasMarker(out, kMarkerByteDRI));
            SJPEG_CHECK(ExtractDRI(out) == restart * MCUsPerRow(w, yuv_modes[y]));
          } else {
            SJPEG_CHECK(!HasMarker(out, kMarkerByteDRI));
          }
        }
      }
    }
  }

  // Test 5: the multi-pass target-size/PSNR search path writes its own
  // headers and stores DC deltas ahead of the final scan, so it needs DRI
  // and the interval DC resets wired up independently of the single-pass one.
  {
    const int w = 320, h = 240;
    const std::vector<uint8_t> test_rgb = MakeRGB(w, h);
    const struct {
      sjpeg::EncoderParam::TargetMode mode;
      float value;
    } kTargets[] = {{sjpeg::EncoderParam::TARGET_SIZE, 8000.f},
                    {sjpeg::EncoderParam::TARGET_PSNR, 38.f}};
    for (size_t t = 0; t < ARRAY_SIZE(kTargets); ++t) {
      for (size_t i = 0; i < ARRAY_SIZE(intervals); ++i) {
        const int restart = intervals[i];
        sjpeg::EncoderParam param(80.0f);
        param.yuv_mode = SJPEG_YUV_420;
        param.target_mode = kTargets[t].mode;
        param.target_value = kTargets[t].value;
        param.passes = 5;
        param.restart_interval_rows = restart;
        std::string out;
        SJPEG_CHECK(EncodeRGB(test_rgb, w, h, param, &out));
        SJPEG_CHECK(HasSize(out, w, h));
        SJPEG_CHECK(HasMarker(out, kMarkerByteDRI) == (restart > 0));
        if (restart > 0) {
          SJPEG_CHECK(ExtractDRI(out) == restart * MCUsPerRow(w, SJPEG_YUV_420));
        }
      }
    }
  }

  // Test 6: DRI's interval field is 16 bits, so a row-based interval that
  // would exceed 0xffff MCUs is clamped down to whole rows. It must never
  // wrap (which silently desynchronizes the decoder) nor reach zero (which
  // means 'no restarts' while markers are still being emitted). The image is
  // sized so that the clamped interval is still shorter than the picture,
  // hence markers really are emitted and the clamp has to reach the scan
  // loops, not merely the header.
  {
    const int w = 2048, h = 2048;  // 4:0:0 -> 256 MCUs/row, 256 MCU rows
    const int mcus_per_row = MCUsPerRow(w, SJPEG_YUV_400);
    const int max_rows = 0xffff / mcus_per_row;  // 255, i.e. under 256 rows
    const std::vector<uint8_t> test_rgb = MakeRGB(w, h);
    const int requested[] = {max_rows - 1, max_rows, max_rows + 1,
                             4 * max_rows};
    for (size_t i = 0; i < ARRAY_SIZE(requested); ++i) {
      sjpeg::EncoderParam param(80.0f);
      param.yuv_mode = SJPEG_YUV_400;
      param.restart_interval_rows = requested[i];
      std::string out;
      SJPEG_CHECK(EncodeRGB(test_rgb, w, h, param, &out));
      SJPEG_CHECK(HasSize(out, w, h));
      const int expected_rows =
          (requested[i] < max_rows) ? requested[i] : max_rows;
      const int dri = ExtractDRI(out);
      SJPEG_CHECK(dri == expected_rows * mcus_per_row);
      SJPEG_CHECK(dri > 0 && dri <= 0xffff);
      // The clamped interval really does restart, so the clamp must have
      // reached the scan loops and not just the header.
      SJPEG_CHECK(HasMarker(out, kMarkerByteRST0));
    }
  }

  // Test 7: trellis-based quantization (methods 7 and 8) substitutes its own
  // quantizer but reaches the very same scan loops, through either
  // SinglePassScan() or SinglePassScanOptimized()/FinalPassScan() depending on
  // Huffman_compress. Restart markers must survive both.
  {
    const int w = 160, h = 120;
    const std::vector<uint8_t> test_rgb = MakeRGB(w, h);
    for (size_t i = 0; i < ARRAY_SIZE(intervals); ++i) {
      const int restart = intervals[i];
      for (int adapt = 0; adapt <= 1; ++adapt) {
        for (int opt = 0; opt <= 1; ++opt) {
          sjpeg::EncoderParam param(80.0f);
          param.yuv_mode = SJPEG_YUV_420;
          param.use_trellis = true;
          param.adaptive_quantization = (adapt == 1);
          param.Huffman_compress = (opt == 1);
          param.restart_interval_rows = restart;
          std::string out;
          SJPEG_CHECK(EncodeRGB(test_rgb, w, h, param, &out));
          SJPEG_CHECK(HasSize(out, w, h));
          SJPEG_CHECK(HasMarker(out, kMarkerByteDRI) == (restart > 0));
          if (restart > 0) {
            SJPEG_CHECK(ExtractDRI(out) == restart * MCUsPerRow(w, SJPEG_YUV_420));
          }
        }
      }
    }
  }

  // Test 8: a negative interval is meaningless; every consumer guards on
  // '> 0', so it must behave exactly like 'disabled'.
  {
    const int w = 64, h = 64;
    const std::vector<uint8_t> test_rgb = MakeRGB(w, h);
    const int negatives[] = {-1, -7, -1000};
    for (size_t i = 0; i < ARRAY_SIZE(negatives); ++i) {
      sjpeg::EncoderParam param(80.0f);
      param.yuv_mode = SJPEG_YUV_420;
      param.restart_interval_rows = negatives[i];
      std::string out;
      SJPEG_CHECK(EncodeRGB(test_rgb, w, h, param, &out));
      SJPEG_CHECK(HasSize(out, w, h));
      SJPEG_CHECK(!HasMarker(out, kMarkerByteDRI));
      SJPEG_CHECK(ExtractDRI(out) == 0);
    }
  }

  // Test 9: the extremes of the format. kMaxDimension is 0xffff, so 65535 is
  // the largest side Encode() accepts.
  {
    // Count the RST markers. Unambiguous: 0xff in entropy-coded data is always
    // followed by a stuffed 0x00, and no header marker falls in 0xd0..0xd7.
    auto CountRestarts = [](const std::string& jpeg) {
      const uint8_t* data = reinterpret_cast<const uint8_t*>(jpeg.data());
      size_t count = 0;
      for (size_t i = 0; i + 1 < jpeg.size(); ++i) {
        if (data[i] == 0xff && (data[i + 1] & 0xf8) == 0xd0) ++count;
      }
      return count;
    };

    // Tallest picture, restarting on every MCU row: 8192 intervals, the most
    // the format can hold, cycling RST0..RST7 1024 times over.
    {
      const int w = 16, h = 65535;
      const std::vector<uint8_t> test_rgb = MakeRGB(w, h);
      sjpeg::EncoderParam param(80.0f);
      param.yuv_mode = SJPEG_YUV_400;
      param.restart_interval_rows = 1;
      std::string out;
      SJPEG_CHECK(EncodeRGB(test_rgb, w, h, param, &out));
      SJPEG_CHECK(HasSize(out, w, h));
      SJPEG_CHECK(ExtractDRI(out) == MCUsPerRow(w, SJPEG_YUV_400));
      // 8192 MCU rows, and a scan never ends on a restart marker.
      SJPEG_CHECK(CountRestarts(out) == 8191);
    }

    // Widest picture: 8192 MCUs per row, so 7 rows is the largest interval
    // that still fits DRI's 16 bits. 8 rows would need 65536 and clamps back
    // to 7, landing exactly on the boundary.
    {
      const int w = 65535, h = 64;
      const int mcus_per_row = MCUsPerRow(w, SJPEG_YUV_400);  // 8192
      const std::vector<uint8_t> test_rgb = MakeRGB(w, h);
      const int requested[] = {7, 8};
      for (size_t i = 0; i < ARRAY_SIZE(requested); ++i) {
        sjpeg::EncoderParam param(80.0f);
        param.yuv_mode = SJPEG_YUV_400;
        param.restart_interval_rows = requested[i];
        std::string out;
        SJPEG_CHECK(EncodeRGB(test_rgb, w, h, param, &out));
        SJPEG_CHECK(HasSize(out, w, h));
        const int dri = ExtractDRI(out);
        SJPEG_CHECK(dri == 7 * mcus_per_row);  // 57344
        SJPEG_CHECK(dri > 0 && dri <= 0xffff);
        // 8 MCU rows at an interval of 7 leaves exactly one marker.
        SJPEG_CHECK(CountRestarts(out) == 1);
      }
    }
  }
}

SJPEG_TEST(RateDistortionOptimization) {
  constexpr int kWidth = 64, kHeight = 64;
  const std::vector<uint8_t> rgb = MakeRGB(kWidth, kHeight);

  // 1. Basic encoding across quality factors with use_rdo.
  static constexpr float kQualities[] = {10.0f, 50.0f, 75.0f, 90.0f, 95.0f, 100.0f};
  for (const float q : kQualities) {
    sjpeg::EncoderParam param(q);
    param.use_rdo = true;
    std::string out;
    SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &out));
    SJPEG_CHECK(HasSize(out, kWidth, kHeight));
  }

  // 2. All YUV modes.
  static constexpr SjpegYUVMode kModes[] = {SJPEG_YUV_AUTO, SJPEG_YUV_420, SJPEG_YUV_SHARP,
                                            SJPEG_YUV_444, SJPEG_YUV_400};
  for (const SjpegYUVMode m : kModes) {
    sjpeg::EncoderParam param(80.0f);
    param.use_rdo = true;
    param.yuv_mode = m;
    std::string out;
    SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &out));
    SJPEG_CHECK(HasSize(out, kWidth, kHeight));
  }

  // 3. RDO produces smaller bitstream than default baseline.
  {
    sjpeg::EncoderParam param_default(75.0f);
    std::string out_default;
    SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param_default, &out_default));

    sjpeg::EncoderParam param_rdo(75.0f);
    param_rdo.use_rdo = true;
    std::string out_rdo;
    SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param_rdo, &out_rdo));

    SJPEG_CHECK(out_rdo.size() <= out_default.size());
  }

  // 4. Trellis takes precedence over RDO.
  {
    sjpeg::EncoderParam param_trellis(80.0f);
    param_trellis.use_trellis = true;
    param_trellis.use_rdo = false;
    std::string out_trellis;
    SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param_trellis, &out_trellis));

    sjpeg::EncoderParam param_both(80.0f);
    param_both.use_trellis = true;
    param_both.use_rdo = true;
    std::string out_both;
    SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param_both, &out_both));

    SJPEG_CHECK(out_both == out_trellis);
  }

#if !defined(SJPEG_NO_PROGRESSIVE)
  // 5. Progressive encoding with RDO actually applies RDO (differs from
  // progressive without RDO).
  {
    sjpeg::EncoderParam param(75.0f);
    param.progressive_luma_split = 2;
    param.progressive_chroma_split = 8;
    std::string out_default;
    SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &out_default));

    param.use_rdo = true;
    std::string out_rdo;
    SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &out_rdo));
    int sof = 0, num_sos = 0;
    SJPEG_CHECK(CheckMarkerStructure(out_rdo, &sof, &num_sos));
    SJPEG_CHECK(sof == 0xc2);
    SJPEG_CHECK(out_rdo != out_default);
  }
#endif

  // 6. -no_optim (Huffman_compress = false) path with RDO also differs.
  {
    sjpeg::EncoderParam param(75.0f);
    param.Huffman_compress = false;
    std::string out_default;
    SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &out_default));

    param.use_rdo = true;
    std::string out_rdo;
    SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &out_rdo));
    SJPEG_CHECK(out_rdo != out_default);
  }

  // 7. Multi-pass target-size (dichotomy) search with RDO also differs.
  {
    sjpeg::EncoderParam param(75.0f);
    param.target_mode = sjpeg::EncoderParam::TARGET_SIZE;
    param.target_value = kWidth * kHeight / 8.f;  // rough target
    param.passes = 5;
    std::string out_default;
    SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &out_default));

    param.use_rdo = true;
    std::string out_rdo;
    SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &out_rdo));
    SJPEG_CHECK(out_rdo != out_default);
  }

#if !defined(SJPEG_NO_MULTITHREADING)
  // 8. Multi-threaded encoding with RDO is bit-exact with serial encoding.
  {
    sjpeg::EncoderParam param(75.0f);
    param.use_rdo = true;
    param.restart_interval_rows = 1;
    param.num_threads = 1;
    std::string expected;
    SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &expected));
    for (int threads : {2, 4, 8}) {
      param.num_threads = threads;
      std::string out;
      SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &out));
      SJPEG_CHECK(out == expected);
    }
  }
#endif

  // 9. Directly calling enc->SetRDO(true/false) matches param.use_rdo.
  {
    sjpeg::EncoderParam param(75.0f);
    param.use_rdo = true;
    std::string expected_rdo;
    SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &expected_rdo));

    sjpeg::EncoderParam param_no_rdo(75.0f);
    param_no_rdo.use_rdo = false;
    std::string expected_no_rdo;
    SJPEG_CHECK(
        EncodeRGB(rgb, kWidth, kHeight, param_no_rdo, &expected_no_rdo));

    // Calling SetRDO(true) overrides param_no_rdo.use_rdo = false.
    std::string out;
    sjpeg::StringSink sink(&out);
    std::unique_ptr<sjpeg::Encoder> enc(sjpeg::EncoderFactory(
        rgb.data(), kWidth, kHeight, 3 * kWidth, SJPEG_YUV_AUTO, &sink));
    SJPEG_CHECK(enc != nullptr);
    SJPEG_CHECK(enc->InitFromParam(param_no_rdo));
    enc->SetRDO(true);
    SJPEG_CHECK(enc->Encode());
    SJPEG_CHECK(out == expected_rdo);

    // Calling SetRDO(false) overrides param.use_rdo = true.
    out.clear();
    std::unique_ptr<sjpeg::Encoder> enc2(sjpeg::EncoderFactory(
        rgb.data(), kWidth, kHeight, 3 * kWidth, SJPEG_YUV_AUTO, &sink));
    SJPEG_CHECK(enc2 != nullptr);
    SJPEG_CHECK(enc2->InitFromParam(param));
    enc2->SetRDO(false);
    SJPEG_CHECK(enc2->Encode());
    SJPEG_CHECK(out == expected_no_rdo);
  }
}

#if !defined(SJPEG_NO_MULTITHREADING)
// Restart intervals are independent by construction, so slicing the scan
// across threads must not change the bitstream at all.
SJPEG_TEST(MultiThreaded) {
  // Deliberately not a multiple of the MCU size, so the last row and column
  // are clipped and go through the sample-replication path.
  const int kWidth = 157, kHeight = 101;
  const std::vector<uint8_t> rgb = MakeRGB(kWidth, kHeight);

  enum QuantMode { kBaseline, kRDO, kTrellis };
  for (QuantMode quant : {kBaseline, kRDO, kTrellis}) {
    for (int huffman = 0; huffman <= 1; ++huffman) {
      for (int adaptive = 0; adaptive <= 1; ++adaptive) {
        for (SjpegYUVMode yuv :
             {SJPEG_YUV_420, SJPEG_YUV_444, SJPEG_YUV_400, SJPEG_YUV_SHARP}) {
          for (int rows : {1, 2, 3}) {
            sjpeg::EncoderParam param(80.0f);
            param.yuv_mode = yuv;
            param.Huffman_compress = (huffman == 1);
            param.adaptive_quantization = (adaptive == 1);
            param.use_trellis = (quant == kTrellis);
            param.use_rdo = (quant == kRDO);
            param.restart_interval_rows = rows;

            param.num_threads = 1;
            std::string expected;
            SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &expected));
            SJPEG_CHECK(HasSize(expected, kWidth, kHeight));

            for (int threads : {2, 3, 4, 8}) {
              param.num_threads = threads;
              std::string out;
              SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &out));
              // Byte for byte, not merely 'it decodes': every interval resets
              // the DC predictors, and the per-thread histograms sum to the
              // same totals in any order, so the slicing must be invisible.
              SJPEG_CHECK(out == expected);
            }
          }
        }
      }
    }
  }

  // Test all compression methods 0..8 (including methods 2, 6, 8 where
  // reuse_run_levels_ == false).
  const auto encode_method = [&](int method, int threads, std::string* out) {
    sjpeg::EncoderParam param(80.0f);
    param.restart_interval_rows = 1;
    param.num_threads = threads;
    sjpeg::StringSink sink(out);
    std::unique_ptr<sjpeg::Encoder> enc(sjpeg::EncoderFactory(
        rgb.data(), kWidth, kHeight, 3 * kWidth, SJPEG_YUV_420, &sink));
    if (enc == nullptr || !enc->InitFromParam(param)) return false;
    enc->SetCompressionMethod(method);
    return enc->Encode();
  };
  for (int method = 0; method <= 8; ++method) {
    std::string expected;
    SJPEG_CHECK(encode_method(method, 1, &expected));
    SJPEG_CHECK(HasSize(expected, kWidth, kHeight));
    for (int threads : {2, 4, 8}) {
      std::string out;
      SJPEG_CHECK(encode_method(method, threads, &out));
      SJPEG_CHECK(out == expected);
    }
  }

  // Also test an image large enough (95x88 = 8360 MCUs in 4:2:0) that
  // worthwhile >= 8, ensuring 8 worker threads are spawned.
  const int kLargeW = 1509, kLargeH = 1403;
  const std::vector<uint8_t> large_rgb = MakeRGB(kLargeW, kLargeH);
  for (int huffman : {0, 1}) {
    for (int adaptive : {0, 1}) {
      for (SjpegYUVMode yuv :
           {SJPEG_YUV_420, SJPEG_YUV_444, SJPEG_YUV_400, SJPEG_YUV_SHARP}) {
        sjpeg::EncoderParam param(80.0f);
        param.yuv_mode = yuv;
        param.Huffman_compress = (huffman == 1);
        param.adaptive_quantization = (adaptive == 1);
        param.restart_interval_rows = 1;
        param.num_threads = 1;
        std::string expected;
        SJPEG_CHECK(EncodeRGB(large_rgb, kLargeW, kLargeH, param, &expected));
        for (int threads : {2, 4, 8}) {
          param.num_threads = threads;
          std::string out;
          SJPEG_CHECK(EncodeRGB(large_rgb, kLargeW, kLargeH, param, &out));
          SJPEG_CHECK(out == expected);
        }
      }
    }
  }
}

SJPEG_TEST(MultiThreadedMultiPass) {
  // 52x52 = 2704 MCUs in 4:2:0 (103x103 = 10609 in 4:4:4), exceeding the
  // ScaledThreadLimit threshold for multi-threaded dichotomy search.
  const int kWidth = 821, kHeight = 819;
  const std::vector<uint8_t> rgb = MakeRGB(kWidth, kHeight);

  enum QuantMode { kBaseline, kRDO, kTrellis };
  for (sjpeg::EncoderParam::TargetMode mode :
       {sjpeg::EncoderParam::TARGET_SIZE, sjpeg::EncoderParam::TARGET_PSNR}) {
    for (QuantMode quant : {kBaseline, kRDO, kTrellis}) {
      for (int huffman : {0, 1}) {
        for (int adaptive : {0, 1}) {
          for (SjpegYUVMode yuv :
               {SJPEG_YUV_420, SJPEG_YUV_444, SJPEG_YUV_400}) {
            sjpeg::EncoderParam param(80.0f);
            param.yuv_mode = yuv;
            param.Huffman_compress = (huffman == 1);
            param.adaptive_quantization = (adaptive == 1);
            param.use_trellis = (quant == kTrellis);
            param.use_rdo = (quant == kRDO);
            param.restart_interval_rows = 1;
            param.passes = 3;
            param.target_mode = mode;
            param.target_value =
                (mode == sjpeg::EncoderParam::TARGET_SIZE) ? 80000.0f : 38.0f;

            param.num_threads = 1;
            std::string expected;
            SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &expected));
            SJPEG_CHECK(HasSize(expected, kWidth, kHeight));

            for (int threads : {2, 4, 8}) {
              param.num_threads = threads;
              std::string out;
              SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &out));
              SJPEG_CHECK(out == expected);
            }
          }
        }
      }
    }
  }

  // Explicitly exercise the !last_is_best path (where the final search pass is
  // further from target than an earlier pass) across Huffman and trellis modes.
  for (int huffman : {0, 1}) {
    for (QuantMode quant : {kBaseline, kRDO, kTrellis}) {
      sjpeg::EncoderParam base_param(80.0f);
      base_param.yuv_mode = SJPEG_YUV_420;
      base_param.Huffman_compress = (huffman == 1);
      base_param.use_trellis = (quant == kTrellis);
      base_param.use_rdo = (quant == kRDO);
      base_param.restart_interval_rows = 1;
      std::string pass0_jpg;
      SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, base_param, &pass0_jpg));

      sjpeg::EncoderParam param = base_param;
      param.passes = 2;
      param.target_mode = sjpeg::EncoderParam::TARGET_SIZE;
      // Target slightly above pass-0 size with zero tolerance so pass 0 is best
      // and pass 1 (at q=90) overshoots, triggering !last_is_best.
      param.target_value = (float)(pass0_jpg.size() + 10);
      param.tolerance = 0.0f;

      param.num_threads = 1;
      std::string expected;
      SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &expected));

      for (int threads : {2, 4}) {
        param.num_threads = threads;
        std::string out;
        SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &out));
        SJPEG_CHECK(out == expected);
      }
    }
  }

  // Test total_intervals == 1 (restart_interval_rows >= mb_h_) where row-based
  // stages (prologue and PSNR search) run multi-threaded while interval-based
  // scan emission falls back to 1 thread without spawning idle workers.
  for (sjpeg::EncoderParam::TargetMode mode :
       {sjpeg::EncoderParam::TARGET_SIZE, sjpeg::EncoderParam::TARGET_PSNR}) {
    for (int huffman : {0, 1}) {
      sjpeg::EncoderParam param(80.0f);
      param.yuv_mode = SJPEG_YUV_420;
      param.Huffman_compress = (huffman == 1);
      param.restart_interval_rows = 100;
      param.passes = 3;
      param.target_mode = mode;
      param.target_value =
          (mode == sjpeg::EncoderParam::TARGET_SIZE) ? 80000.0f : 38.0f;

      param.num_threads = 1;
      std::string expected;
      SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &expected));

      param.num_threads = 4;
      std::string out;
      SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &out));
      SJPEG_CHECK(out == expected);
    }
  }
}

SJPEG_TEST(MultiThreadedIntervalDefaults) {
  const int kWidth = 64, kHeight = 48;   // 4x3 MCUs in 4:2:0
  const std::vector<uint8_t> rgb = MakeRGB(kWidth, kHeight);

  // The reference: one MCU row per interval, coded serially.
  sjpeg::EncoderParam param(75.0f);
  param.yuv_mode = SJPEG_YUV_420;
  param.restart_interval_rows = 1;
  std::string expected;
  SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &expected));

  // Asking for threads without an interval picks the finest slicing (1 MCU row
  // per interval).
  param.restart_interval_rows = 0;
  param.num_threads = 4;
  std::string implied;
  SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &implied));
  SJPEG_CHECK(implied == expected);

  // Negative values for restart interval also default to 1 MCU row.
  param.restart_interval_rows = -1;
  std::string negative_interval;
  SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &negative_interval));
  SJPEG_CHECK(negative_interval == expected);

  // Zero thread count defaults to 1 (single-threaded).
  param.restart_interval_rows = 1;
  param.num_threads = 0;
  std::string zero_threads;
  SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &zero_threads));
  SJPEG_CHECK(zero_threads == expected);

  // Negative thread count (-1) uses all available cores.
  param.num_threads = -1;
  std::string auto_threads;
  SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &auto_threads));
  SJPEG_CHECK(auto_threads == expected);

  // More threads than there are intervals to hand out is harmless.
  param.num_threads = 64;
  std::string oversubscribed;
  SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &oversubscribed));
  SJPEG_CHECK(oversubscribed == expected);

  // An interval spanning the whole image (total_intervals == 1) with
  // num_threads > 1 bypasses parallel slice allocation and matches 1 thread.
  param.restart_interval_rows = 3;
  param.num_threads = 1;
  std::string single_interval_serial;
  SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &single_interval_serial));

  param.num_threads = 4;
  std::string single_interval_mt;
  SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &single_interval_mt));
  SJPEG_CHECK(single_interval_mt == single_interval_serial);
}

SJPEG_TEST(MultiThreadedSharpYUV) {
  const struct { int w, h; } kSizes[] = {
    {157, 101},  // odd prime dimensions
    {256, 256},  // power-of-two even dimensions
    {512, 384},  // standard aspect ratio
  };

  for (const auto& size : kSizes) {
    const int W = size.w;
    const int H = size.h;
    const int uv_w = (W + 1) >> 1;
    const int uv_h = (H + 1) >> 1;
    const std::vector<uint8_t> rgb = MakeRGB(W, H);

    // Reference conversion using 1 thread (encoder = nullptr)
    std::vector<uint8_t> y_ref(W * H), u_ref(uv_w * uv_h), v_ref(uv_w * uv_h);
    SJPEG_CHECK(sjpeg::ApplySharpYUVConversion(rgb.data(), W, H, 3 * W,
                                               y_ref.data(), u_ref.data(),
                                               v_ref.data(), nullptr));

    for (int threads : {2, 3, 4, 8}) {
      std::string dummy;
      sjpeg::StringSink sink(&dummy);
      std::unique_ptr<sjpeg::Encoder> enc(sjpeg::EncoderFactory(
          rgb.data(), W, H, 3 * W, SJPEG_YUV_420, &sink, sjpeg::kRGBInput,
          nullptr, threads));
      SJPEG_CHECK(enc != nullptr);

      std::vector<uint8_t> y(W * H), u(uv_w * uv_h), v(uv_w * uv_h);
      SJPEG_CHECK(sjpeg::ApplySharpYUVConversion(rgb.data(), W, H, 3 * W,
                                                 y.data(), u.data(),
                                                 v.data(), enc.get()));
      SJPEG_CHECK(y == y_ref);
      SJPEG_CHECK(u == u_ref);
      SJPEG_CHECK(v == v_ref);
    }

    // Also test through the full encoding pipeline.
    sjpeg::EncoderParam param(85.0f);
    param.yuv_mode = SJPEG_YUV_SHARP;
    param.restart_interval_rows = 1;
    param.num_threads = 1;
    std::string expected;
    SJPEG_CHECK(EncodeRGB(rgb, W, H, param, &expected));

    for (int threads : {2, 4, 8}) {
      param.num_threads = threads;
      std::string out;
      SJPEG_CHECK(EncodeRGB(rgb, W, H, param, &out));
      SJPEG_CHECK(out == expected);
    }
  }
}
#endif  // !SJPEG_NO_MULTITHREADING

SJPEG_TEST(SafeArithmetic) {
  using namespace sjpeg;

  // SafeMultiply with size_t
  size_t out_s = 0;
  SJPEG_CHECK(SafeMultiply<size_t>(10, 20, &out_s) && out_s == 200);
  SJPEG_CHECK(SafeMultiply<size_t>(0, 100, &out_s) && out_s == 0);
  SJPEG_CHECK(SafeMultiply<size_t>(100, 0, &out_s) && out_s == 0);
  SJPEG_CHECK(SafeMultiply<size_t>(0, SIZE_MAX, &out_s) && out_s == 0);
  SJPEG_CHECK(SafeMultiply<size_t>(SIZE_MAX / 2, 2, &out_s) &&
              out_s == (SIZE_MAX / 2) * 2);
  SJPEG_CHECK(!SafeMultiply<size_t>(SIZE_MAX, 2, &out_s));
  SJPEG_CHECK(!SafeMultiply<size_t>(SIZE_MAX / 2 + 1, 2, &out_s));
  SJPEG_CHECK(!SafeMultiply<size_t>(10, 20, nullptr));

  // SafeAdd with size_t
  SJPEG_CHECK(SafeAdd<size_t>(10, 20, &out_s) && out_s == 30);
  SJPEG_CHECK(SafeAdd<size_t>(0, 0, &out_s) && out_s == 0);
  SJPEG_CHECK(SafeAdd<size_t>(SIZE_MAX - 5, 5, &out_s) &&
              out_s == SIZE_MAX);
  SJPEG_CHECK(!SafeAdd<size_t>(SIZE_MAX, 1, &out_s));
  SJPEG_CHECK(!SafeAdd<size_t>(SIZE_MAX - 5, 6, &out_s));
  SJPEG_CHECK(!SafeAdd<size_t>(10, 20, nullptr));

  // Signed integers (int)
  int out_i = 0;
  SJPEG_CHECK(SafeMultiply<int>(10, 20, &out_i) && out_i == 200);
  SJPEG_CHECK(SafeMultiply<int>(-10, 20, &out_i) && out_i == -200);
  SJPEG_CHECK(SafeMultiply<int>(-10, -20, &out_i) && out_i == 200);
  SJPEG_CHECK(SafeMultiply<int>(0, -5, &out_i) && out_i == 0);
  SJPEG_CHECK(SafeMultiply<int>(INT_MAX, 1, &out_i) && out_i == INT_MAX);
  SJPEG_CHECK(SafeMultiply<int>(INT_MIN, 1, &out_i) && out_i == INT_MIN);
  SJPEG_CHECK(!SafeMultiply<int>(INT_MAX, 2, &out_i));
  SJPEG_CHECK(!SafeMultiply<int>(INT_MAX / 2 + 1, 2, &out_i));
  SJPEG_CHECK(!SafeMultiply<int>(INT_MIN, -1, &out_i));
  SJPEG_CHECK(!SafeMultiply<int>(-1, INT_MIN, &out_i));
  SJPEG_CHECK(!SafeMultiply<int>(INT_MIN, 2, &out_i));
  SJPEG_CHECK(!SafeMultiply<int>(10, 20, nullptr));

  // Signed add
  SJPEG_CHECK(SafeAdd<int>(10, 20, &out_i) && out_i == 30);
  SJPEG_CHECK(SafeAdd<int>(-10, 20, &out_i) && out_i == 10);
  SJPEG_CHECK(SafeAdd<int>(-10, -20, &out_i) && out_i == -30);
  SJPEG_CHECK(SafeAdd<int>(INT_MAX - 1, 1, &out_i) && out_i == INT_MAX);
  SJPEG_CHECK(SafeAdd<int>(INT_MIN + 1, -1, &out_i) && out_i == INT_MIN);
  SJPEG_CHECK(!SafeAdd<int>(INT_MAX, 1, &out_i));
  SJPEG_CHECK(!SafeAdd<int>(INT_MIN, -1, &out_i));
  SJPEG_CHECK(!SafeAdd<int>(10, 20, nullptr));

  // Automatic conversion of arguments when template parameter is explicit
  int w = 320, h = 240;
  SJPEG_CHECK(SafeMultiply<size_t>(w, h, &out_s) && out_s == 76800);
}

SJPEG_TEST(AdaptiveBias) {
  constexpr int kWidth = 64, kHeight = 64;
  const std::vector<uint8_t> rgb = MakeRGB(kWidth, kHeight);
  const auto encode = [&](const sjpeg::EncoderParam& p) {
    std::string out;
    SJPEG_CHECK(EncodeRGB(rgb, kWidth, kHeight, p, &out));
    return out;
  };

  // 1. Basic encoding across quality factors and YUV modes.
  for (const float q : {10.0f, 50.0f, 75.0f, 90.0f, 95.0f, 100.0f}) {
    sjpeg::EncoderParam p(q);
    p.adaptive_bias = true;
    SJPEG_CHECK(HasSize(encode(p), kWidth, kHeight));
  }
  for (const auto m : {SJPEG_YUV_AUTO, SJPEG_YUV_420, SJPEG_YUV_SHARP,
                       SJPEG_YUV_444, SJPEG_YUV_400}) {
    sjpeg::EncoderParam p(80.0f);
    p.adaptive_bias = true;
    p.yuv_mode = m;
    SJPEG_CHECK(HasSize(encode(p), kWidth, kHeight));
  }

  // 2. High quantization_bias must not underflow qthresh.
  const std::vector<uint8_t> flat = MakeFlatRGB(32, 32, 128, 128, 128);
  for (const int b : {128, 200, 240, 255}) {
    sjpeg::EncoderParam p(95.0f);
    p.adaptive_bias = true;
    p.quantization_bias = b;
    std::string out;
    SJPEG_CHECK(EncodeRGB(flat, 32, 32, p, &out));
    SJPEG_CHECK(HasSize(out, 32, 32) && out.size() < 600);
  }

  // 3. Interactions: distinct from default, combined with RDO, overridden by
  // trellis.
  sjpeg::EncoderParam p_bias(80.0f), p_rdo(80.0f), p_trellis(80.0f);
  p_bias.adaptive_bias = true;
  p_rdo.use_rdo = true;
  p_trellis.use_trellis = true;

  const std::string out_def = encode(sjpeg::EncoderParam(80.0f));
  const std::string out_bias = encode(p_bias);
  const std::string out_rdo = encode(p_rdo);
  const std::string out_trellis = encode(p_trellis);
  SJPEG_CHECK(out_bias != out_def);

  sjpeg::EncoderParam p_both = p_rdo;
  p_both.adaptive_bias = true;
  const std::string out_rdo_bias = encode(p_both);
  SJPEG_CHECK(out_rdo_bias != out_rdo && out_rdo_bias != out_bias);

  p_both = p_trellis;
  p_both.adaptive_bias = true;
  SJPEG_CHECK(encode(p_both) == out_trellis);

  // 4. Multi-pass search and progressive encoding.
  sjpeg::EncoderParam p_mp(75.0f);
  p_mp.adaptive_bias = true;
  p_mp.target_mode = sjpeg::EncoderParam::TARGET_SIZE;
  p_mp.target_value = 2000.0f;
  p_mp.passes = 5;
  SJPEG_CHECK(HasSize(encode(p_mp), kWidth, kHeight));

#if !defined(SJPEG_NO_PROGRESSIVE)
  sjpeg::EncoderParam p_prog(75.0f);
  p_prog.adaptive_bias = true;
  p_prog.progressive_luma_split = 2;
  p_prog.progressive_chroma_split = 8;
  int sof = 0, num_sos = 0;
  SJPEG_CHECK(CheckMarkerStructure(encode(p_prog), &sof, &num_sos) &&
              sof == 0xc2);
#endif

  // 5. Activity classification.
  const std::vector<uint8_t> flat8 = MakeFlatRGB(8, 8, 128, 128, 128);
  std::vector<uint8_t> busy8(8 * 8 * 3);
  for (int i = 0; i < 64; ++i) {
    const uint8_t v = ((i ^ (i >> 3)) & 1) ? 255 : 0;
    busy8[3 * i] = busy8[3 * i + 1] = busy8[3 * i + 2] = v;
  }
  uint32_t act_flat = 0, act_busy = 0;
  SJPEG_CHECK(sjpeg::BlockActivityScore(&flat8[0], 24, &act_flat) ==
              sjpeg::BlockActivityTier::kFlatBlock);
  SJPEG_CHECK(sjpeg::BlockActivityScore(&busy8[0], 24, &act_busy) ==
              sjpeg::BlockActivityTier::kBusyBlock);
  SJPEG_CHECK(act_flat < sjpeg::kActivityLo && act_busy > sjpeg::kActivityHi);
  SJPEG_CHECK(sjpeg::ClassifyActivity(act_flat) ==
              sjpeg::BlockActivityTier::kFlatBlock);
  SJPEG_CHECK(sjpeg::ClassifyActivity(act_busy) ==
              sjpeg::BlockActivityTier::kBusyBlock);
  SJPEG_CHECK(
      sjpeg::ClassifyActivity((sjpeg::kActivityLo + sjpeg::kActivityHi) / 2) ==
      sjpeg::BlockActivityTier::kNormalBlock);
}

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
