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
//  Fast & simple JPEG encoder.
//
// Author: Skal (pascal.massimino@gmail.com)

#ifndef SJPEG_JPEG_H_
#define SJPEG_JPEG_H_

#include <inttypes.h>
#include <string>

#define SJPEG_VERSION 0x000100   // 0.1.0

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

// Returns the library's version.
uint32_t SjpegVersion();

// Main function
// This is the simplest possible call. There is only one parameter (quality)
// and most decisions will be made automatically (YUV420/YUV444/etc...).
// Returns the compressed size, and fills *out_data with the bitstream.
// This returned buffer is allocated with 'new[]' operator. It must be
// deallocated using 'delete[]' call.
// Input data 'rgb' are the samples in sRGB format, in R/G/B memory order.
// Picture dimension is width x height.
// Retuns 0 in case of error.
size_t SjpegCompress(const uint8_t* rgb, int width, int height, int quality,
                     uint8_t** out_data);

// Encodes an RGB picture to JPEG.
//
//  the dimension of the picture pointed to by 'rgb', is W * H, with stride
//  'stride' (must be greater or equal to 3*W). The dimensions must be strictly
//  positive.
//
// The compressed bytes are made available in *out_data, which is a buffer
// allocated with new []. This buffer must be disallocated using 'delete []'.
//
// Return parameter -if positive- is the size of the JPEG string,
// or 0 if an error occured.
//
// Parameter 'quality' correspond to the usual quality factor in JPEG:
//     0=very bad, 100=very good.
// Parameter 'compression_method' refer to the efforts and resources spent
//  trying to compress better. Default (fastest) method should be 0. Method 1
//  will optimize the size at the expense of using more RAM. Method 2 does
//  the same as method #1, but but without any additional RAM (but using twice
//  more CPU). Methods 3, 4, 5, and 6 behave like methods 0, 1, and 2, except
//  that the quantization matrices are fine-tuned to the source's content using
//  histogram. This requires an additional pass, and is hence slower, but can
//  give substantial filesize reduction, especially for hi-quality settings.
//  Method 5 will try to not use extra RAM to store the Fourier-transformed
//  coefficients, at the expense of being ~15% slower, but will still use some
//  memory for the Huffman size-optimization. Eventually, method 6 will use
//  a minimal amount of RAM, but will be must slower.
//  To recap:
//     method                     | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
//     ---------------------------+---+---+---+---+---+---+---|
//     Huffman size-optimization  |   | x | x |   | x | x | x |
//     Adaptive quantization      |   |   |   | x | x | x | x |
//     Extra RAM for Huffman pass |   | x |   |   | x | x |   |
//     Extra RAM for histogram    |   |   |   | x | x |   |   |
//
//  Methods sorted by decreasing speed: 0 > 1 > 2 > 3 > 4 > 5 > 6
//  Sorted by increasing efficiency: 0 < [1|2] < 3 < [4|5|6]
//
//  If you don't have any strict requirements on CPU and memory, you should
//  probably use method #4.
//
// Parameter 'yuv_mode': decides which colorspace to use. Possible values:
//   * 0: automated decision between YUV 4:2:0 / sharp-YUV 4:2:0 and YUV 4:4:4
//   * 1: YUV 4:2:0
//   * 2: YUV 4:2:0 with 'sharp' conversion
//   * 3: YUV 4:4:4
size_t SjpegEncode(const uint8_t* rgb,
                   int width, int height, int stride,
                   uint8_t** out_data,
                   int quality,
                   int compression_method,
                   int yuv_mode);

////////////////////////////////////////////////////////////////////////////////
// JPEG-parsing tools

// Decode the dimensions of a JPEG bitstream, doing as few read operations as
// possible. Return false if an error occured (invalid bitstream, invalid
// parameter...).
// The pointers 'width', 'height', 'is_yuv420' can be passed NULL.
bool SjpegDimensions(const uint8_t* data, size_t size,
                     int* width, int* height, int* is_yuv420);

// Finds the location of the first two quantization matrices within a JPEG
// 'data' bitstream. Matrices are 64 coefficients stored as uint8_t.
// Note that the input can be truncated to include the headers only, but still
// must start as a valid JPEG with an 0xffd8 marker.
// Returns the number of matrices detected.
// Returns 0 in case of bitstream error, or if the DQT chunk is missing.
int SjpegFindQuantizer(const uint8_t* data, size_t size,
                       uint8_t quant[2][64]);

// Returns an estimation of the quality factor that would best approximate
// the quantization coefficients in matrix[].
int SjpegEstimateQuality(const uint8_t matrix[64], bool for_chroma);

// Generate a default quantization matrix for the given quality factor,
// in a libjpeg-6b fashion.
void SjpegQuantMatrix(int quality, bool for_chroma, uint8_t matrix[64]);

// Returns the favored conversion mode to use (YUV420 / sharp-YUV420 / YUV444)
// Return values:
//   * 1: YUV 4:2:0
//   * 2: sharp-YUV 4:2:0
//   * 3: YUV 4:4:4
// If risk is not NULL, the riskiness score (between 0 and 100) is returned.
int SjpegRiskiness(const uint8_t* rgb, int width, int height, int step,
                   float* risk);

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif

////////////////////////////////////////////////////////////////////////////////
// The following handy variants are equivalent to the functions defined above,
// except that they are taking string argument for input/output data.
// There is also a generic call using SjpegEncodeParam to pass encoding
// parameters.
// These are C++ only.

// Structure for holding encoding parameter, to be passed to the unique
// call to SjpegEncode() below. For a more detailed description of some fields,
// see SjpegEncode()'s doc above.
struct SjpegEncodeParam {
  SjpegEncodeParam();
  explicit SjpegEncodeParam(int quality_factor) {
    Init(quality_factor);
  }
  // Sets the compression factor. 0 = bad quality, 100 = best quality.
  // The call will actually initialize quant[][].
  void SetQuality(int quality_factor);

  // Reduce the output size by a factor 'reduction'
  //  reduction ~= 100 -> small size reduction
  //  reduction ~=   1 -> large size reduction
  // Note: 'reduction' can be larger than 100.
  // This function is incompatible with SetQuality()
  void SetQuantMatrix(const uint8_t m[64], int idx, int reduction = 100);

  // This function is reduce the size
  //   reduction ~= 100: small reduction
  //   reduction ~= 1: large reduction
  // This function will affect the content of 'quant[][]' and hence must
  // be called after SetQuality() or SetQuantMatrix() in order to be effective.
  void SetReduction(int reduction);

  // Limit the quantization by setting up some minimal quantization matrices based
  // on the current content of quant[][] matrices.
  // Hence, this function must be called after SetQuality() or SetQuantMatrix().
  void SetLimitQuantization(bool limit_quantization = true, int tolerance = 0);

  int yuv_mode;                 // YUV-420/444 decisions
  bool Huffman_compress;        // if true, use optimized Huffman tables.
  bool adaptive_quantization;   // if true, use optimized quantizer matrices.
  bool adaptive_bias;           // if true, use perceptual bias adaptation
  // Fine-grained control over compression parameters
  int quantization_bias;    // [0..255] Rounding bias for quantization.
  int qdelta_max_luma;      // [0..12] How much to hurt luma in adaptive quant
  int qdelta_max_chroma;    // [0..12] How much to hurt chroma in adaptive quant
                            // A higher value might be useful for images
                            // encoded without chroma subsampling.

  // metadata: extra EXIF/XMP/ICCP data that will be embedded in
  // APP1 or APP2 markers. They should contain only the raw payload and not
  // the prefixes ("Exif\0", "ICC_PROFILE", etc...). These will be added
  // automatically during encoding.
  // Conversely, the content of app_markers is written as is, right after APP0.
  std::string exif;
  std::string xmp;
  std::string iccp;
  std::string app_markers;
  void ResetMetadata();      // clears the above

 protected:
  void Init(int quality_factor);

  uint8_t quant_[2][64];         // quantization matrices to use
  const uint8_t* min_quant_[2];  // if limit_quantization is true, these pointers
                                 // should point luma / chroma minimum allowed
                                 // quantizer values.
  int min_quant_tolerance_;      // Tolerance going over min_quant_ ([0..100])

  friend std::string SjpegEncode(const uint8_t* rgb,
                                 int width, int height, int stride,
                                 const SjpegEncodeParam& param);
};

// Same as the first version of SjpegEncode(), except encoding parameters are
// delivered in a SjpegEncodeParam. Upon failure (memory allocation or
// invalid parameter), the function returns an empty string.
std::string SjpegEncode(const uint8_t* rgb,
                        int width, int height, int stride,
                        const SjpegEncodeParam& param);

////////////////////////////////////////////////////////////////////////////////
// Variant of the function above, but using std::string as interface.

std::string SjpegCompress(const uint8_t* rgb,
                          int width, int height, int quality);

bool SjpegDimensions(const std::string& jpeg_data,
                     int* width, int* height, int* is_yuv420);

int SjpegFindQuantizer(const std::string& jpeg_data, uint8_t quant[2][64]);

#endif    // SJPEG_JPEG_H_
