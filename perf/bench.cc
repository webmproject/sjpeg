// Copyright 2026 Google, Inc.
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
//  Encoding benchmark: read a source image once, then re-encode it in a loop
//  and report the best-of-N wall clock. Best rather than average, because
//  every perturbation here (scheduling, thermal, page faults) can only ever
//  add time.
//
//   usage: bench <image> [method] [quality] [tile] [iters] [passes] [yuv]
//
//     method   0, 1, 3, 4 or 7, expanded into the Huffman / adaptive-quant /
//              trellis flags the same way InitFromParam() does it.
//     tile     replicates the source tile x tile times, to reach a working
//              set that no longer fits in cache.
//     passes   > 1 selects the dichotomy search, targeting a fixed size.
//     yuv      SjpegYUVMode; omitted or negative leaves it at the default,
//              AUTO -- which is not free, as it runs SjpegRiskiness() over
//              the whole source before encoding anything.
//
//   env:
//     BENCH_OUT=<path>   write out the last encoded result, to compare two
//                        builds byte-for-byte (see verify.sh).
//     BENCH_SLOW_C=1     disable every SIMD implementation, to measure what
//                        the vector layer as a whole is worth.
//
// Author: Skal (pascal.massimino@gmail.com)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#include "sjpeg.h"
#include "utils.h"

// Defined in enc.cc. Not exposed in any header: it exists for tests, and this
// benchmark is the other legitimate user.
namespace sjpeg { extern bool ForceSlowCImplementation; }

int main(int argc, char* argv[]) {
  if (getenv("BENCH_SLOW_C")) sjpeg::ForceSlowCImplementation = true;
  if (argc < 2) {
    printf("usage: bench <image> [method] [q] [tile] [it] [passes] [yuv]\n");
    return 1;
  }
  const int method = (argc > 2) ? atoi(argv[2]) : 4;
  const float quality = (argc > 3) ? (float)atof(argv[3]) : 75.f;
  const int tile = (argc > 4) ? atoi(argv[4]) : 1;
  const int iters = (argc > 5) ? atoi(argv[5]) : 20;
  const int passes = (argc > 6) ? atoi(argv[6]) : 1;
  const int yuv = (argc > 7) ? atoi(argv[7]) : -1;   // -1 = AUTO

  // atoi() gives 0 for anything it cannot parse, and 0 tiles builds an empty
  // image that the encode then indexes; 0 iterations divides by a best-case
  // time that no run ever set.
  if (tile < 1 || iters < 1 || passes < 1) {
    printf("tile, iters and passes must all be >= 1\n");
    return 1;
  }

  const std::string in = ReadFile(argv[1]);
  int W = 0, H = 0;
  sjpeg::EncoderParam dummy;
  std::vector<uint8_t> rgb = ReadImage(in, &W, &H, &dummy);
  if (rgb.empty()) { printf("can't read %s\n", argv[1]); return 1; }

  // tile the image
  const int W2 = W * tile, H2 = H * tile;
  std::vector<uint8_t> big((size_t)W2 * H2 * 3);
  for (int y = 0; y < H2; ++y) {
    for (int t = 0; t < tile; ++t) {
      memcpy(&big[((size_t)y * W2 + (size_t)t * W) * 3],
             &rgb[(size_t)(y % H) * W * 3], (size_t)W * 3);
    }
  }

  sjpeg::EncoderParam p;
  p.SetQuality(quality);
  // method mapping, per InitFromParam():
  //   0: -,-      1: H,-     3: -,A     4: H,A     7: H,A,T
  p.Huffman_compress = (method == 1) || (method == 4) || (method == 7);
  p.adaptive_quantization = (method >= 3);
  p.use_trellis = (method >= 7);
  p.passes = passes;
  if (passes > 1) {
    p.target_mode = sjpeg::EncoderParam::TARGET_SIZE;
    p.target_value = (float)((size_t)W2 * H2 / 8);   // arbitrary target
    p.tolerance = 1.f;
  }

  if (yuv >= 0) p.yuv_mode = (SjpegYUVMode)yuv;

  // time the riskiness pass alone
  {
    double t_risk = 1e30;
    for (int i = 0; i < 5; ++i) {
      const double t0 = GetStopwatchTime();
      float risk;
      SjpegRiskiness(&big[0], W2, H2, W2 * 3, &risk);
      const double dt = GetStopwatchTime() - t0;
      if (dt < t_risk) t_risk = dt;
    }
    printf("  [SjpegRiskiness alone: %.1f ms]\n", t_risk * 1000.);
  }

  std::string out;
  double best = 1e30, total = 0;
  size_t size = 0;
  for (int i = 0; i < iters; ++i) {
    const double t0 = GetStopwatchTime();
    out.clear();
    if (!sjpeg::Encode(&big[0], W2, H2, W2 * 3, p, &out)) {
      printf("encoding error!\n"); return 1;
    }
    const double dt = GetStopwatchTime() - t0;
    total += dt;
    if (dt < best) best = dt;
    size = out.size();
  }
  const char* const out_file = getenv("BENCH_OUT");
  if (out_file != nullptr) {
    FILE* const f = fopen(out_file, "wb");
    if (f == nullptr) { printf("can't open %s\n", out_file); return 1; }
    if (fwrite(out.data(), 1, out.size(), f) != out.size()) {
      printf("error writing %s\n", out_file);
      fclose(f);
      return 1;
    }
    fclose(f);
  }
  const double mpix = (double)W2 * H2 / 1.e6;
  printf("%5dx%-5d (%5.2f MP) m=%d q=%3.0f p=%d: best %7.1f ms "
         "(%6.1f MP/s)  avg %7.1f ms  out=%zu\n",
         W2, H2, mpix, method, quality, passes, best * 1000., mpix / best,
         total / iters * 1000., size);
  return 0;
}
