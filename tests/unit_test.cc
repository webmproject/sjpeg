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
  const sjpeg::EncoderParam param;
  std::string out;
  CHECK(!sjpeg::Encode(nullptr, kWidth, kHeight, 3 * kWidth, param, &out));
  CHECK(!EncodeRGB(rgb, kWidth, kHeight, param,
                   static_cast<std::string*>(nullptr)));
  CHECK(!EncodeRGB(rgb, kWidth, 0, param, &out));
  CHECK(!sjpeg::EncodeGray(nullptr, kWidth, kHeight, kWidth, param, &out));
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
