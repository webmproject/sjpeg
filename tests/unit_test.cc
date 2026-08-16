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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
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
#define CHECK(expr) CheckImpl(!!(expr), #expr, __LINE__)
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
  return sjpeg::Encode(&rgb[0], W, H, 3 * W, param, out);
}

// True if the bitstream announces the expected dimensions.
bool HasSize(const std::string& jpg, int W, int H) {
  int width = 0, height = 0;
  return SjpegDimensions(jpg, &width, &height, nullptr) &&
         width == W && height == H;
}

////////////////////////////////////////////////////////////////////////////////

TEST(Compress) {
  const int kWidth = 61, kHeight = 37;
  const std::vector<uint8_t> rgb = MakeRGB(kWidth, kHeight);
  std::string out;
  CHECK(SjpegCompress(&rgb[0], kWidth, kHeight, 75.f, &out));
  CHECK(out.size() > 0);
  int width = 0, height = 0, is_yuv420 = -1;
  CHECK(SjpegDimensions(out, &width, &height, &is_yuv420));
  CHECK(width == kWidth && height == kHeight);
  CHECK(is_yuv420 == 0 || is_yuv420 == 1);
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
    CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &out));
    CHECK(HasSize(out, kWidth, kHeight));
  }
  // Higher quality must not compress better.
  std::string small, large;
  CHECK(EncodeRGB(rgb, kWidth, kHeight, sjpeg::EncoderParam(30.f), &small));
  CHECK(EncodeRGB(rgb, kWidth, kHeight, sjpeg::EncoderParam(95.f), &large));
  CHECK(small.size() < large.size());
}

TEST(InvalidArguments) {
  const int kWidth = 16, kHeight = 16;
  const std::vector<uint8_t> rgb = MakeRGB(kWidth, kHeight);
  uint8_t* data = nullptr;
  const auto enc = [](const uint8_t* src, int W, int H, int stride,
                      uint8_t** dst, SjpegYUVMode mode) {
    return SjpegEncode(src, W, H, stride, dst, 75.f, 4, mode);
  };
  CHECK(enc(nullptr, kWidth, kHeight, 3 * kWidth, &data, SJPEG_YUV_420) == 0);
  CHECK(enc(&rgb[0], kWidth, kHeight, 3 * kWidth, nullptr, SJPEG_YUV_420) == 0);
  CHECK(enc(&rgb[0], 0, kHeight, 3 * kWidth, &data, SJPEG_YUV_420) == 0);
  CHECK(enc(&rgb[0], kWidth, -1, 3 * kWidth, &data, SJPEG_YUV_420) == 0);
  CHECK(enc(&rgb[0], kWidth, kHeight, 3 * kWidth - 1, &data,
            SJPEG_YUV_420) == 0);

  // unknown yuv_mode: no encoder can be created for it. 7 is the largest
  // value the enum can hold without being out of range.
  CHECK(enc(&rgb[0], kWidth, kHeight, 3 * kWidth,
            &data, static_cast<SjpegYUVMode>(7)) == 0);
  CHECK(data == nullptr);
  const sjpeg::EncoderParam param;
  std::string out;
  CHECK(!sjpeg::Encode(nullptr, kWidth, kHeight, 3 * kWidth, param, &out));
  CHECK(!EncodeRGB(rgb, kWidth, kHeight, param,
                   static_cast<std::string*>(nullptr)));
  CHECK(!EncodeRGB(rgb, kWidth, 0, param, &out));
  CHECK(!sjpeg::EncodeGray(nullptr, kWidth, kHeight, kWidth, param, &out));
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
void CheckStrides(EncodeYUVFunc encode, int sub) {
  const int kWidth = 20, kHeight = 20;
  const int uv_w = (kWidth + sub - 1) / sub, uv_h = (kHeight + sub - 1) / sub;
  const std::vector<uint8_t> Y = MakePlane(kWidth, kHeight, 20);
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
      CHECK(encode(&Y[0], kWidth, &u[0], u_stride, &v[0], v_stride,
                   kWidth, kHeight, param, sjpeg::MakeByteSink(&out).get()));
      if (ref.empty()) ref = out;
      CHECK(!out.empty() && out == ref);
    }
  }
}

TEST(EncodeYUV420Strides) { CheckStrides(&sjpeg::EncodeYUV420, 2); }
TEST(EncodeYUV444Strides) { CheckStrides(&sjpeg::EncodeYUV444, 1); }

TEST(EncodeNV) {
  const int kWidth = 18, kHeight = 14;
  const int uv_h = (kHeight + 1) / 2, uv_stride = 2 * ((kWidth + 1) / 2);
  const std::vector<uint8_t> Y = MakePlane(kWidth, kHeight, 30);
  const std::vector<uint8_t> UV = MakePlane(uv_stride, uv_h, 90);
  const sjpeg::EncoderParam param(75.f);
  std::string out12, out21;
  CHECK(sjpeg::EncodeNV12(&Y[0], kWidth, &UV[0], uv_stride, kWidth, kHeight,
                          param, sjpeg::MakeByteSink(&out12).get()));
  CHECK(sjpeg::EncodeNV21(&Y[0], kWidth, &UV[0], uv_stride, kWidth, kHeight,
                          param, sjpeg::MakeByteSink(&out21).get()));
  CHECK(HasSize(out12, kWidth, kHeight));
  CHECK(out12 != out21);   // U and V are swapped

  // one invalid argument at a time
  std::string out;
  const auto holder = sjpeg::MakeByteSink(&out);
  sjpeg::ByteSink* const sink = holder.get();
  const auto nv12 = [&](const uint8_t* y, int y_step, const uint8_t* uv,
                        int uv_step, int W, int H, sjpeg::ByteSink* s) {
    return sjpeg::EncodeNV12(y, y_step, uv, uv_step, W, H, param, s);
  };
  CHECK(!nv12(&Y[0], kWidth, &UV[0], uv_stride, kWidth, kHeight, nullptr));
  CHECK(!nv12(nullptr, kWidth, &UV[0], uv_stride, kWidth, kHeight, sink));
  CHECK(!nv12(&Y[0], kWidth, nullptr, uv_stride, kWidth, kHeight, sink));
  CHECK(!nv12(&Y[0], kWidth, &UV[0], uv_stride, 0, kHeight, sink));
  CHECK(!nv12(&Y[0], kWidth - 1, &UV[0], uv_stride, kWidth, kHeight, sink));
  CHECK(!nv12(&Y[0], kWidth, &UV[0], uv_stride - 1, kWidth, kHeight, sink));
  CHECK(!sjpeg::EncodeNV21(&Y[0], kWidth, &UV[0], uv_stride, kWidth, kHeight,
                           param, nullptr));
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
    CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &out));
    CHECK(memory.num_allocs > 0);          // it was used at all
    CHECK(memory.num_foreign_frees == 0);  // and used for every free()
    CHECK(memory.live.empty());            // no leak
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
  CHECK(EncodeRGB(rgb, kMaxDim, kSmallDim, param, &out));
  CHECK(HasSize(out, kMaxDim, kSmallDim));
  CHECK(!EncodeRGB(rgb, kMaxDim + 1, kSmallDim, param, &out));
  CHECK(!EncodeRGB(rgb, kSmallDim, kMaxDim + 1, param, &out));
  CHECK(!sjpeg::EncodeGray(&rgb[0], kMaxDim + 1, kSmallDim, kMaxDim + 1,
                           param, &out));
  uint8_t* data = nullptr;
  CHECK(SjpegEncode(&rgb[0], kMaxDim + 1, kSmallDim, 3 * (kMaxDim + 1), &data,
                    50.f, 4, SJPEG_YUV_420) == 0);
  CHECK(data == nullptr);
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
      CHECK(ok == (memory.num_refused == 0));
      CHECK(memory.live.empty());
    }
  }
}

TEST(QuantMatrix) {
  for (int quality = 0; quality <= 100; quality += 5) {
    for (int chroma = 0; chroma <= 1; ++chroma) {
      uint8_t matrix[64];
      SjpegQuantMatrix(quality, chroma != 0, matrix);
      for (size_t i = 0; i < 64; ++i) CHECK(matrix[i] >= 1);
      const float estimate = SjpegEstimateQuality(matrix, chroma != 0);
      CHECK(estimate >= quality - 1.f && estimate <= quality + 1.f);
    }
  }
  // The matrices used for encoding must be recoverable from the bitstream.
  const int kWidth = 24, kHeight = 24;
  const std::vector<uint8_t> rgb = MakeRGB(kWidth, kHeight);
  sjpeg::EncoderParam param(70.f);
  param.adaptive_quantization = false;
  std::string out;
  CHECK(EncodeRGB(rgb, kWidth, kHeight, param, &out));
  uint8_t quant[2][64];
  CHECK(SjpegFindQuantizer(out, quant) == 2);
  CHECK(memcmp(quant[0], param.GetQuantMatrix(0), 64) == 0);
  CHECK(memcmp(quant[1], param.GetQuantMatrix(1), 64) == 0);
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
