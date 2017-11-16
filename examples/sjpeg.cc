// Copyright 2017 Google, Inc.
//
//  simple JPEG compressor or re-compressor
//
// usage:
//   sjpeg input.{jpg,png} [-o output.jpg] [-q quality]
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
// Author: Skal (pascal.massimino@gmail.com)
//

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <vector>
using std::vector;

#include "sjpeg.h"
#include "./utils.h"

///////////////////////////////////////////////////////////////////////////////

void PrintMatrix(const char name[], const uint8_t m[64], bool for_chroma) {
  printf(" %s quantization matrix (estimated quality: %d)\n",
         name, SjpegEstimateQuality(m, for_chroma));
  for (int j = 0; j < 8; ++j) {
    for (int i = 0; i < 8; ++i) printf("%3d ", m[i + j * 8]);
    printf("\n");
  }
  printf("------\n");
}

///////////////////////////////////////////////////////////////////////////////

static const char* kYUVModeNames[4] = {
  "automatic", "YUV420", "SharpYUV420", "YUV444"
};
static const char* kNoYes[2] = { "no", "yes" };

int main(int argc, char * argv[]) {
  const char* input_file = NULL;
  const char* output_file = NULL;
  SjpegEncodeParam param;
  int reduction = 100;
  int quality = 75;
  bool use_reduction = true;  // until '-q' is used...
  bool no_metadata = false;
  bool estimate = false;
  bool limit_quantization = true;
  int info = 0;
  bool quiet = false;
  bool short_output = false;
  float riskiness = 0;
  int yuv_mode_rec = 0;
  const char* const usage =
    "sjpeg: Commandline utility to recompress or compress pictures to JPEG.\n"
    "Usage:  sjpeg infile [-o outfile.jpg] [-q quality] ...\n"
    "  -q quality ...... Quality factor in [0..100] range.\n"
    "                    Value of 100 gives the best quality, largest file.\n"
    "                    Default value is 75.\n"
    "  -r reduction .... Reduction factor in [0..100] range.\n"
    "                    Default value is 100. Lower value will reduce the \n"
    "                    file size.\n"
    "  -o filename ..... specifies the output file name.\n"
    "  -estimate ....... Just estimate and print the JPEG source quality.\n"
    "  -i .............. Just print some information about the input file.\n"
    "  -version ........ Print the version and exit.\n"
    "  -quiet .......... Quiet mode. Just save the file.\n"
    "  -short .......... Print shorter 1-line info.\n"
    "\n"
    "Advanced options:\n"
    "  -yuv_mode ....... YUV mode to use:\n"
    "                    0: automatic decision (default)\n"
    "                    1: use YUV 4:2:0\n"
    "                    2: use 'Sharp' YUV 4:2:0 conversion\n"
    "                    3: use YUV 4:4:4 (full resolution for U/V planes)\n"
    "  -no_limit ....... If true, allow the quality factor to be larger\n"
    "                    than the original (JPEG input only).\n"
    "  -no_optim ....... Don't use Huffman optimization (=faster)\n"
    "  -no_adapt ....... Don't use adaptive quantization (=faster)\n"
    "  -no_metadata .... Ignore metadata from the source.\n"
    "\n"
    "\n"
    "If the input format is JPEG, the recompression will not go beyond the\n"
    "original quality, *except* if '-no_limit' option is used."
    "\n"
  ;

  // parse command line
  if (argc <= 1) {
    fprintf(stderr, usage);
    return 0;
  }
  for (int c = 1; c < argc; ++c) {
    if (!strcmp(argv[c], "-h") || !strcmp(argv[c], "--help")) {
      fprintf(stdout, usage);
      return 0;
    } else if (!strcmp(argv[c], "-o") && c + 1 < argc) {
      output_file = argv[++c];
    } else if (!strcmp(argv[c], "-q") && c + 1 < argc) {
      quality = atoi(argv[++c]);
      use_reduction = false;
      if (quality < 0 || quality > 100) {
        printf("Error: invalid range for option '%s': %s\n",
               argv[c - 1], argv[c]);
        return 1;
      }
    } else if (!strcmp(argv[c], "-r") && c + 1 < argc) {
      reduction = atoi(argv[++c]);
      use_reduction = true;
      if (reduction <= 0 || reduction > 100) {
        printf("Error: invalid range for option '%s': %s\n",
               argv[c - 1], argv[c]);
        return 1;
      }
    } else if (!strcmp(argv[c], "-estimate")) {
      estimate = true;
    } else if (!strcmp(argv[c], "-no_limit")) {
      limit_quantization = false;
    } else if (!strcmp(argv[c], "-no_adapt")) {
      param.adaptive_quantization = false;
    } else if (!strcmp(argv[c], "-no_optim")) {
      param.Huffman_compress = false;
    } else if (!strcmp(argv[c], "-adapt_bias")) {
      param.adaptive_bias = true;
    } else if (!strcmp(argv[c], "-no_metadata")) {
      no_metadata = true;
    } else if (!strcmp(argv[c], "-yuv_mode") && c + 1 < argc) {
      param.yuv_mode = atoi(argv[++c]);
      if (param.yuv_mode < 0 || param.yuv_mode > 3) {
        printf("Error: invalid range for option '%s': %s\n",
               argv[c - 1], argv[c]);
        return 1;
      }
    } else if (!strcmp(argv[c], "-i") || !strcmp(argv[c], "-info")) {
      info = true;
    } else if (!strcmp(argv[c], "-quiet")) {
      quiet = true;
    } else if (!strcmp(argv[c], "-short")) {
      short_output = true;
    } else if (!strcmp(argv[c], "-version")) {
      const uint32_t version = SjpegVersion();
      printf("%d.%d.%d\n",
             (version >> 16) & 0xff,
             (version >>  8) & 0xff,
             (version >>  0) & 0xff);
      return 0;
    } else {
      input_file = argv[c];
    }
  }
  if (input_file == NULL) {
    fprintf(stderr, "Missing input file.\n");
    if (!quiet) fprintf(stderr, usage);
    return -1;
  }

  // Read input file into the buffer in_bytes[]
  std::string input = ReadFile(input_file);
  if (input.size() == 0) return 1;

  const ImageType input_type = GuessImageType(input);
  uint8_t quant_matrices[2][64];
  const int nb_matrices =
    (input_type == SJPEG_JPEG) ? SjpegFindQuantizer(input, quant_matrices)
                               : 0;
  const bool is_jpeg = (input_type == SJPEG_JPEG) && (nb_matrices > 0);
  if (is_jpeg && use_reduction) {   // use 'reduction' factor for JPEG source
    param.SetQuantMatrix(quant_matrices[0], 0, reduction);
    param.SetQuantMatrix(quant_matrices[1], 1, reduction);
    param.SetLimitQuantization(true);
  } else {    // the '-q' option has been used.
    param.SetQuality(quality);
    param.SetLimitQuantization(false);
  }

  if (estimate) {
    const int q = is_jpeg ? SjpegEstimateQuality(quant_matrices[0], 0) : 100;
    printf("%d\n", q);
    return 0;
  }
  int W, H;
  vector<uint8_t> in_bytes = ReadImage(input, &W, &H, &param);
  if (in_bytes.size() == 0) return 1;

  if (!short_output && !quiet) {
    fprintf(stdout, "Input [%s]: %s (%ld bytes, %d x %d)\n",
            ImageTypeName(input_type), input_file,
            static_cast<long>(input.size()),
            W, H);
    if (info) {
      yuv_mode_rec = SjpegRiskiness(&in_bytes[0], W, H, 3 * W, &riskiness);
      fprintf(stdout, "Riskiness:   %.1f (recommended yuv_mode: %s)\n",
              riskiness, kYUVModeNames[yuv_mode_rec]);

      if (is_jpeg) {
        printf("Input is JPEG w/ %d matrices:\n", nb_matrices);
        if (nb_matrices > 0) {
          PrintMatrix("Luma", quant_matrices[0], false);
        }
        if (nb_matrices > 1) {
          PrintMatrix("UV-chroma", quant_matrices[1], true);
        }
      }
    }
  }
  if (info) return 0;   // done
  
  // finish setting up the quantization matrices
  if (limit_quantization == false) param.SetLimitQuantization(false);

  if (no_metadata) param.ResetMetadata();

  const double start = GetStopwatchTime();
  const std::string out = SjpegEncode(&in_bytes[0], W, H, 3 * W, param);
  const double encode_time = GetStopwatchTime() - start;

  if (out.size() == 0) {
    fprintf(stderr, "ERROR: call to SjpegEncode() failed");
    return -1;
  }

  if (!short_output && !quiet) {
    yuv_mode_rec = SjpegRiskiness(&in_bytes[0], W, H, 3 * W, &riskiness);
    fprintf(stdout, "new size:   %ld bytes (%.2lf%% of original)\n"
                    "reduction:  r=%d (adaptive: %s, Huffman: %s)\n"
                    "yuv mode:   %s (riskiness: %.1lf%%)\n"
                    "elapsed:    %d ms\n",
                    (long)out.size(), 100. * out.size() / input.size(),
                    reduction,
                    kNoYes[param.adaptive_quantization],
                    kNoYes[param.Huffman_compress],
                    kYUVModeNames[yuv_mode_rec], riskiness,
                    (int)(1000. * encode_time));
    if (param.iccp.size()) {
      fprintf(stdout, "ICCP:       %ld bytes\n", (long)param.iccp.size());
    }
    if (param.exif.size()) {
      fprintf(stdout, "EXIF:       %ld bytes\n", (long)param.exif.size());
    }
    if (param.xmp.size()) {
      fprintf(stdout, "XMP:        %ld bytes\n", (long)param.xmp.size());
    }
  } else if (!quiet) {
    fprintf(stdout, "%ld %ld %.2lf %%\n",
           static_cast<long>(input.size()), static_cast<long>(out.size()),
           100. * out.size() / input.size());
  }

  // Save the result.
  if (!SaveFile(output_file, out, quiet)) return 1;

  return 0;     // ok.
}

///////////////////////////////////////////////////////////////////////////////
