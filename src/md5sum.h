// Copyright 2020 Google Inc.
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
//  Simple MD5 Digest code
//
// Author: Skal (pascal.massimino@gmail.com)

#ifndef SJPEG_MD5SUM_H_
#define SJPEG_MD5SUM_H_

#include <cstdint>
#include <cstdio>
#include <string>

namespace sjpeg {

class MD5Digest {
 public:
  explicit MD5Digest(const std::string& data = "")
      : A(0x67452301), B(0xefcdab89), C(0x98badcfe), D(0x10325476) {
    uint32_t s = data.size();
    assert(data.size() < (1ull << 32));
    uint32_t i, j;
    for (i = 0; i + 64 <= s; i += 64) Add((const uint8_t*)&data[i]);
    uint8_t block[64 + 64];
    for (j = 0; i < s; ++i) block[j++] = data[i];
    block[j++] = 0x80;  // bit 1
    while ((j & 63) != 56) block[j++] = 0;  // pad
    for (i = 0, s *= 8; i < 8; ++i, s >>= 8) block[j++] = s & 0xff;
    Add(block);
    if (j > 64) Add(block + 64);
  }
  static uint32_t Rotate(uint32_t v, int n) {   // n != 0
    return (v << n) | (v >> (32 - n));
  }
  static uint32_t Get32(const uint8_t b[64], uint32_t i) {
    b += 4 * (i & 15);
    return ((uint32_t)b[0] <<  0) | ((uint32_t)b[1] <<  8) |
           ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
  }
  static void Put32(uint8_t* b, uint32_t v) {
    char tmp[3];
    for (uint32_t i = 0; i < 4; ++i, v >>= 8) {
      snprintf(tmp, sizeof(tmp), "%.2X", v & 0xff);
      *b++ = tmp[0];
      *b++ = tmp[1];
    }
  }

  void Add(const uint8_t block[64]) {
    const uint8_t Kr[64] = {
      7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,
      5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,
      4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,
      6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21
    };
    const uint32_t KK[64] = {  // (1ul << 32) * abs(std::sin(i + 1.))
      0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
      0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
      0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
      0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
      0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
      0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
      0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
      0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
      0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
      0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
      0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
      0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
      0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
      0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
      0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
      0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
    };
    uint32_t a = A, b = B, c = C, d = D;
    for (uint32_t i = 0; i < 64; ++i) {
      uint32_t e = a + KK[i];
      if (i < 16) {
        e += d ^ (b & (c ^ d));
        e += Get32(block, i);
      } else if (i < 32) {
        e += c ^ (d & (b ^ c));
        e += Get32(block, 5 * i + 1);
      } else if (i < 48) {
        e += b ^ c ^ d;
        e += Get32(block, 3 * i + 5);
      } else {
        e += c ^ (b | ~d);
        e += Get32(block, 7 * i);
      }
      a = d;
      d = c;
      c = b;
      b += Rotate(e, Kr[i]);
    }
    A += a;
    B += b;
    C += c;
    D += d;
  }

  std::string Get() const {  // returns the hex digest (upper case)
    uint8_t tmp[32];
    Get(tmp);
    return std::string((const char*)tmp, 32u);
  }
  void Get(uint8_t out[32]) const {
    Put32(out +  0, A);
    Put32(out +  8, B);
    Put32(out + 16, C);
    Put32(out + 24, D);
  }

 private:
  uint32_t A, B, C, D;
};

}  // namespace sjpeg

#endif  // SJPEG_MD5SUM_H_
