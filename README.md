```
                     ____ __   __ __  ____ ____  ____
                    /  __\__) /  \  \/  _ \   / /  _ \
                   _\_   \  (/      /   __/  /_/   __/
                   \_____/___/__//_/\__/_/_____/_____/
                          (__)_/  _ \/  _ \/ .__\
                          _)  \   __/   __/  /  \
                         /____/__/  \_____/_____/v0.1.3
```

*   **sjpeg**: 's' stands for 'simple'.
*   sjpeg is a simple, fast encoding library for baseline and progressive JPEG
    files. It also provides command-line tools to compress PNG, JPEG, or PPM
    images into JPEG.
*   Why 'simple'? The encoder started as a single, compact source file. It has
    grown and split for clarity and performance (SIMD, progressive, trellis),
    but the philosophy remains lightweight and direct.

# The sjpeg Library

The primary header is `src/sjpeg.h`.

## C API

### One-liner compression

```c
size_t SjpegCompress(const uint8_t* rgb, int width, int height,
                     float quality, uint8_t** out_data);
```

Makes automatic decisions (YUV 4:2:0 vs 4:4:4, quantization, Huffman tables)
based on input content. Returns compressed byte count, or 0 on error. `out_data`
must be freed with `SjpegFreeBuffer()` (or `delete[]`).

A C++ overload returning `bool` into `std::string*` is also available.

### Fine-tuned compression

```c
size_t SjpegEncode(const uint8_t* rgb,
                   int width, int height, int stride,
                   uint8_t** out_data,
                   float quality,
                   int compression_method,
                   SjpegYUVMode yuv_mode);
```

*   `stride`: scanline stride in bytes ($\ge 3 \times \text{width}$).
*   `compression_method`: effort/RAM trade-off (0..8).
    *   `0`: fastest baseline (default quantizer, fixed Huffman tables).
    *   `1`, `2`: optimized Huffman tables (method 2 uses less RAM, more CPU).
    *   `3`..`6`: content-adaptive quantization matrix modulation via
        histograms.
    *   `7`, `8`: Trellis-based rate-distortion quantization.
*   `yuv_mode`:
    *   `SJPEG_YUV_AUTO` (0): automated selection via riskiness metric.
    *   `SJPEG_YUV_420` (1): standard YUV 4:2:0.
    *   `SJPEG_YUV_SHARP` (2): sharp edge-directed YUV 4:2:0 downsampling.
    *   `SJPEG_YUV_444` (3): full-resolution chroma YUV 4:4:4.
    *   `SJPEG_YUV_400` (4): grayscale (luma only).

### Buffer management

```c
void SjpegFreeBuffer(const uint8_t* buffer);
```

Releases memory allocated by `SjpegCompress()` and `SjpegEncode()`.

### Helper functions

*   **Bitstream inspection**:
    *   `SjpegDimensions(data, size, &w, &h, &is_yuv420)`: parses picture
        dimensions and subsampling from a JPEG bitstream without decoding
        pixels.
    *   `SjpegFindQuantizer(data, size, quant)`: extracts the DQT quantization
        matrices (in natural order).
*   **Quantization utilities**:
    *   `SjpegEstimateQuality(matrix, for_chroma)`: estimates the quality factor
        best matching a quantization matrix.
    *   `SjpegQuantMatrix(quality, for_chroma, matrix)`: generates standard
        libjpeg-6b quantization matrices for a given quality factor.
*   **Subsampling decision (riskiness)**:
    *   `SjpegRiskiness(rgb, width, height, stride, &risk)`: analyzes raw RGB
        pixels, computes the riskiness score (0..100), and returns the
        recommended `SjpegYUVMode` (`_420`, `_SHARP`, or `_444`).

--------------------------------------------------------------------------------

## C++ API (`sjpeg::EncoderParam`)

For full control over the encoder, use `sjpeg::EncoderParam` and
`sjpeg::Encode()`:

```cpp
#include "sjpeg.h"

sjpeg::EncoderParam param;
param.SetQuality(80.0f);
param.yuv_mode = SJPEG_YUV_AUTO;
param.Huffman_compress = true;
param.adaptive_quantization = true;

std::string output;
if (!sjpeg::Encode(rgb, width, height, stride, param, &output)) {
  // handle error
}
```

### Key `EncoderParam` features

*   **Quality / Matrix setting**:
    *   `param.SetQuality(q)`: standard quality factor in `[0.0, 100.0]`.
    *   `param.SetQuantization(m, reduction)`: sets custom matrices (e.g.
        extracted from a source JPEG via `SjpegFindQuantizer`) scaled by a
        reduction percentage.
    *   `param.SetLimitQuantization(true, tolerance)`: when recompressing,
        limits loss by clamping matrices to not exceed the source's coarseness.
*   **Compression flags**:
    *   `Huffman_compress`: enables multi-pass Huffman code optimization.
    *   `adaptive_quantization`: content-based spatial variance quantization.
    *   `adaptive_bias`: perceptual rounding-bias adaptation.
    *   `use_trellis`: rate-distortion Trellis quantization optimization.
*   **Progressive JPEG**:
    *   `progressive_luma_split`: AC spectral split index (1..63, default 64 =
        disabled; `2` is a common choice).
    *   `progressive_chroma_split`: AC spectral split index for chroma (default
        `8`).
*   **Restart markers**:
    *   `restart_interval_rows`: emits a DRI header and restart markers
        (`RST0`..`RST7`) every $N$ MCU rows (default: 0 = disabled). Resets DC
        predictors at regular intervals to limit error propagation and allow
        independent slice decoding (ignored for progressive encoding).
*   **Target size / PSNR search**:
    *   `target_mode`: `TARGET_SIZE` (bytes) or `TARGET_PSNR` (dB).
    *   `target_value`: desired byte size or target PSNR.
    *   `passes`: max iterations for binary search convergence (default 10).
    *   `qmin`, `qmax`, `tolerance`: search bounds and stopping tolerance.
*   **Metadata**:
    *   `param.exif`: raw EXIF payload (embedded in APP1).
    *   `param.iccp`: raw ICC profile (embedded in APP2).
    *   `param.xmp`: raw XMP payload. Automatically split across Extended XMP
        APP1 markers with MD5 digests if $> 65504$ bytes.
    *   `param.app_markers`: arbitrary raw APP markers emitted directly after
        APP0.
*   **Custom output**:
    *   Pass a `sjpeg::ByteSink*` to `sjpeg::Encode()` for streaming output
        directly to files or custom sinks without intermediate buffering.

--------------------------------------------------------------------------------

# Command-Line Tools

## `sjpeg`

Compresses an input image (PNG, JPEG, or PPM) to JPEG.

```bash
# Basic encoding
sjpeg in.png -o out.jpg -q 75

# Recompress an existing JPEG with quality limit
sjpeg in.jpg -o out.jpg -r 90

# Target file size search (e.g. 50 KB)
sjpeg in.png -o out.jpg -size 50000

# High-efficiency encoding with Trellis and Progressive scans
sjpeg in.png -o out.jpg -q 85 -trellis -progressive

# Emit restart markers every 4 MCU rows (DRI / RST0..RST7)
sjpeg in.png -o out.jpg -q 80 -restart 4

# Sharp YUV 4:2:0 chroma downsampling
sjpeg in.png -o out.jpg -q 80 -sharp

# Pass metadata
sjpeg in.png -o out.jpg -exif photo.exif -icc color.icc
```

Run `sjpeg -h` for the complete list of flags (`-restart`, `-444`, `-420`,
`-gray`, `-psnr`, `-no_adapt`, `-no_optim`, `-no_limit`, `-crc`, `-md5`).

## `vjpeg`

![vjpeg UI](https://github.com/webmproject/sjpeg/blob/main/examples/vjpeg_sample.jpg "the vjpeg interface")

An interactive OpenGL/GLUT visualizer for inspecting compression artifacts in
real time. Allows adjusting quality, subsampling, matrices, and quantization
parameters dynamically via keyboard shortcuts.

--------------------------------------------------------------------------------

# Building

## Using CMake (Recommended)

```bash
cmake -B build
cmake --build build -j4
```

### Standard CMake Options

| Option | Default | Description |
|---|---|---|
| `SJPEG_ENABLE_SIMD` | `ON` | Enables SIMD vectorization (SSE2, NEON). |
| `SJPEG_HAVE_AVX2` | `OFF` | Compiles AVX2 kernel variants (x86-64). Dispatched at runtime via `SupportsAVX2()`. |
| `SJPEG_ENABLE_PROGRESSIVE` | `ON` | Enables progressive JPEG encoding support. |
| `SJPEG_BUILD_EXAMPLES` | `ON` | Builds `examples/sjpeg` and `examples/vjpeg`. |
| `SJPEG_BUILD_TESTS` | `ON` | Builds `tests/unit_test`. |

To run unit tests:

```bash
./build/unit_test
```

## Using GNU Make

```bash
make -j4
```

To enable AVX2 with GNU Make:

```bash
make HAVE_AVX2=1 -j4
```

## Compiling with Android NDK

```bash
cmake -B build \
      -DCMAKE_TOOLCHAIN_FILE=../cmake/android.cmake \
      -DSJPEG_ANDROID_NDK_PATH=${NDK_ROOT} \
      -DANDROID_ABI=arm64-v8a
cmake --build build -j4
```

Supported `ANDROID_ABI` values: `arm64-v8a` (default), `armeabi-v7a`,
`armeabi-v7a with NEON`, `x86`, `x86_64`.

--------------------------------------------------------------------------------

# Optional Compile-Time Tuning Flags

When building from source, additional fine-grained optimizations and code paths
can be enabled or disabled via compiler definitions (`-D...`):

| Macro / Flag | Default | Description |
|---|---|---|
| `SJPEG_HAVE_AVX2` | Off (`CMake`) | Compiles optimized AVX2 kernels for FDCT, quantization, histogram, and color conversion on x86-64. Safely falls back to SSE2 on older CPUs at runtime. |
| `SJPEG_USE_AVX2_RISKINESS` | Off | Experimental gather-based AVX2 implementation of `SjpegRiskiness()` (`src/riskiness_avx2.cc`). Disabled by default due to gather latency variability on early AVX2 hardware (Haswell/Excavator). |
| `SJPEG_USE_AVX2_YUV_GATHER` | On (with AVX2) | Enables gather-based AVX2 gamma-table lookups for Sharp RGB $\to$ YUV (`src/yuv_convert_avx2.cc`). Bit-exact with C and $\sim 1.15\times$ faster. |
| `SJPEG_USE_PEXT` | Off | In `src/quantize_avx2.cc`, uses BMI2 `_pext_u32` to compact the natural-order non-zero coefficient mask (2 ops vs 5). Off by default because `pext` is microcoded with multi-cycle latency on AMD Zen 1/Zen+. Requires `-mbmi2`. |
| `SJPEG_NO_PROGRESSIVE` | Off | Completely strips progressive encoding code, structures, and buffers, reducing binary footprint for baseline-only deployments. (Set automatically by `-DSJPEG_ENABLE_PROGRESSIVE=OFF`). |
| `SJPEG_FORCE_32BIT` | Off | Forces 32-bit bit-writer paths on 64-bit platforms for cross-architecture verification (`src/bit_writer.h`). |
| `SJPEG_USE_COUNTFF_LUT` | On | Uses a 64-bit lookup table to accelerate 0xFF escape byte counting in the bit-writer (`src/bit_writer.h`). |

---

# Discussion & Bug Reports

*   Mailing list / forum:
    [https://groups.google.com/forum/#!forum/sjpeg](https://groups.google.com/forum/#!forum/sjpeg)
*   Bug tracker:
    [https://github.com/webmproject/sjpeg/issues](https://github.com/webmproject/sjpeg/issues)
