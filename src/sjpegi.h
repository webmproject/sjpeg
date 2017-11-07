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
//  Fast & simple JPEG encoder. Internal header.
//
// Author: Skal (pascal.massimino@gmail.com)

#ifndef SJPEG_JPEGI_H_
#define SJPEG_JPEGI_H_

#include "sjpeg.h"

#ifndef NULL
#define NULL 0
#endif

#if defined(__SSE2__)
#define SJPEG_USE_SSE2
#endif

#if defined(__ANDROID__) && defined(__ARM_ARCH_7A__)
#define SJPEG_ANDROID_NEON  // Android targets that might support NEON
#endif

// The intrinsics currently cause compiler errors with arm-nacl-gcc and the
// inline assembly would need to be modified for use with Native Client.
#if (defined(__ARM_NEON__) || defined(SJPEG_ANDROID_NEON) || \
     defined(__aarch64__))
#define SJPEG_USE_NEON
#endif

#if defined(SJPEG_NEED_ASM_HEADERS)
#if defined(SJPEG_USE_SSE2)
#include <emmintrin.h>
#endif

#if defined(SJPEG_USE_NEON)
#include <arm_neon.h>
#endif
#endif    // SJPEG_NEED_ASM_HEADERS

#include <assert.h>
#define DCHECK(a) assert((a))

namespace sjpeg {

extern bool SupportsSSE2();
extern bool SupportsNEON();

// Constants below are marker codes defined in JPEG spec
// ISO/IEC 10918-1 : 1993(E) Table B.1
// See also: http://www.w3.org/Graphics/JPEG/itu-t81.pdf

#define M_SOF0  0xffc0
#define M_SOF1  0xffc1
#define M_DHT   0xffc4
#define M_SOI   0xffd8
#define M_EOI   0xffd9
#define M_SOS   0xffda
#define M_DQT   0xffdb

// Forward 8x8 Fourier transforms, in-place.
typedef void (*SjpegFdctFunc)(int16_t *coeffs, int num_blocks);
SjpegFdctFunc SjpegGetFdct();

// these are the default luma/chroma matrices (JPEG spec section K.1)
extern const uint8_t kDefaultMatrices[2][64];
extern const uint8_t kZigzag[64];

// scoring tables in score_7.cc
extern const int kRGBSize;
extern const uint8_t kSharpnessScore[];

// internal riskiness scoring functions:
extern double SjpegDCTRiskinessScore(const int16_t yuv[3 * 8],
                                     int16_t scores[8 * 8]);
extern double SjpegBlockRiskinessScore(const uint8_t* rgb, int stride,
                                       int16_t scores[8 * 8]);

///////////////////////////////////////////////////////////////////////////////
// RGB->YUV conversion

// convert 16x16 RGB block into YUV420, or 8x8 RGB block into YUV444
typedef void (*RGBToYUVBlockFunc)(const uint8_t* src, int src_stride,
                                  int16_t* blocks);
extern RGBToYUVBlockFunc GetBlockFunc(bool use_444);

// convert a row of RGB samples to YUV444
typedef void (*RGBToIndexRowFunc)(const uint8_t* src, int width,
                                  uint16_t* dst);
extern RGBToIndexRowFunc GetRowFunc();

// Enhanced slower RGB->YUV conversion:
//  y_plane[] has dimension W x H, whereas u_plane[] and v_plane[] have
//  dimension (W + 1)/2 x (H + 1)/2.
void ApplySharpYUVConversion(const uint8_t* const rgb,
                             int W, int H, int stride,
                             uint8_t* y_plane,
                             uint8_t* u_plane, uint8_t* v_plane);

}   // namespace sjpeg

#endif    // SJPEG_JPEGI_H_
