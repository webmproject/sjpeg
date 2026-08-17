# sjpeg: split enc.cc, then optimize

**Status:** planned, not started.
**Precondition:** PR #150 (`skal/misc-improvements`) is landed. Nothing here
should start before then — phase 1 touches every line of `enc.cc` and would make
#150 unreviewable.

Baseline for every number below: `skal/misc-improvements` @ `7601185`, Apple M5
Pro (NEON), clang `-O3 -DNDEBUG`, `tests/testdata/source3.jpg` (1600x1200),
best-of-30 wall clock. Cross-checked on x86_64/SSE2.

Full analysis, with profiles and reasoning:
<https://claude.ai/code/artifact/b1d4778a-5501-482e-a289-a6ae0fad4a68>

---

## Phase 0 — measurement harness

Land this first, or at least keep it out-of-tree and stable, so every later
number is comparable. The harness lives out-of-tree in `build/perf/`:

> **`build/` is a scratch directory — treat everything in it as disposable.**
> `perf/` is meant to be promoted to a top-level `perf/` when it earns its
> keep. That promotion is a plain `git mv build/perf perf`: `verify.sh` walks
> up from its own location to find `tests/testdata`, and `bench.cc` only needs
> `-I src -I examples`. What the move would additionally want: a
> CMake/Makefile target for `bench`, and a `.gitignore` line for the binaries
> it drops next to the sources.

- `perf/bench.cc` — links the library directly, decodes once, tiles the image to
  a chosen size, re-encodes in a loop, reports best-of-N. Env vars:
  `BENCH_OUT=<path>` dumps the encoded result (for bit-exactness checks),
  `BENCH_SLOW_C=1` sets `sjpeg::ForceSlowCImplementation` to A/B the whole SIMD
  layer.
- `perf/verify.sh` — bit-exactness sweep across 4 images x 5 methods x
  3 qualities x {1, 6} passes = 120 configurations. It takes two `bench`
  binaries (paths relative to wherever you run it) and prints `N/120 identical`, exiting non-zero
  on any difference. Both directions are checked: it reports 120/120 for the
  phase-2 prototype, and correctly isolates `source2.jpg` as the only image
  whose output moves under the phase-3.2 riskiness subsampling.

Build:

```sh
S=$PWD/..            # from build/
c++ -O3 -DNDEBUG -std=c++11 -g -DSJPEG_HAVE_JPEG -DSJPEG_HAVE_PNG \
    -I$S/src -I$S/examples -I/opt/homebrew/include \
    perf/bench.cc $S/src/*.cc $S/examples/utils.cc \
    /opt/homebrew/lib/libpng.dylib /opt/homebrew/lib/libjpeg.dylib -lz \
    -o perf/bench
```

Profile (macOS):

```sh
./perf/bench ../tests/testdata/source3.jpg 4 75 1 4000 1 1 >/dev/null &
sleep 1.5; sample $! 8 -f prof.txt; kill $!
awk '/Sort by top of stack/,/Binary Images/' prof.txt
```

x86_64 cross-build for the SSE2 paths (libpng/libjpeg are arm64-only under
Homebrew, so drop the codec defines and use the `.ppm` input):

```sh
c++ -arch x86_64 -O3 -DNDEBUG -std=c++11 -msse2 -I$S/src -I$S/examples \
    perf/bench.cc $S/src/*.cc $S/examples/utils.cc -o perf/bench_x86
```

Rosetta runs it; treat the timings as relative only, but the **bit-exactness
result is authoritative** — it is the only way this machine exercises
`QuantizeBlockSSE2`, `QuantizeErrorSSE2`, `StoreHistoSSE2`, `FdctSSE2`.

### Reference profile to regress against

| configuration | total | top entries |
|---|---:|---|
| m4 q75 AUTO (default) | 10.3 ms | CodeBlock 32%, **SjpegRiskiness 26%**, QuantizeBlock 15%, StoreHisto 8%, fDCT 5% |
| m4 q75 forced 420 | 7.5 ms | CodeBlock 42%, QuantizeBlock 20%, StoreHisto 12%, AddEntropyStats 7%, fDCT 6% |
| m4 q75 8 passes | 34.3 ms | **BitCounter::AddBits 49%**, QuantizeBlock 20%, StoreOptimalHuffmanTables 8%, BlocksSize 7% |
| m7 q75 (trellis) | 34.3 ms | **TrellisQuantizeBlock 81%**, CodeBlock 10% |

Two facts worth not re-deriving: throughput is flat at 255–265 MP/s from 1.9 MP
to 17.3 MP, so this is **not** memory-bound (the access pattern is sequential and
prefetches perfectly); and the whole SIMD layer is worth only **1.74x**
(13.2 ms -> 7.6 ms), which bounds how much "better vectors" can ever return.

---

## Phase 1 — split enc.cc (mechanical, zero behavior change)

`enc.cc` is 2573 lines doing six jobs. Every `////` banner in it already falls on
a clean function boundary, and the cut lines below land exactly on those banners
— no function body is touched.

| new file | lines | ranges in enc.cc | job |
|---|---:|---|---|
| `quantize.cc` | 535 | 64–146, 341–382, 544–953 | quant matrix setup/finalization, `QuantizeBlock` x3, trellis, `QuantizeError` x3 |
| `encoders.cc` | 545 | 1725–2269 | `InitComponents`, edge replication, `Encoder420/444/400/Sharp`, NV12/NV21 + direct-YUV adapters, `EncoderFactory` |
| `entropy.cc` | 442 | 383–543, 954–994, 1355–1594 | standard + optimal Huffman tables, `GenerateDCDiffCode`, `CodeBlock`, entropy statistics |
| `enc.cc` (kept) | 478 | 1–63, 147–340, 1288–1354, 1595–1724, 2550– | Encoder lifecycle, setters, buffers, CPU dispatch, scan drivers, `Encode()` |
| `histogram.cc` | 293 | 995–1287 | `StoreHisto` x3, `AnalyseHisto`, `CollectHistograms` |
| `api.cc` | 280 | 2270–2549 | `SjpegEncode`/`SjpegCompress`, `EncoderParam`, `Encode()` overloads |

Sums to 2573 exactly.

### Why this is cheap

`Encoder` is already fully declared in `sjpegi.h` and every helper is already a
private static member, so the definitions relocate to another translation unit
with **no class edit** — no new friend, no visibility change, no forward
declaration.

A scan of every file-local symbol against these boundaries turns up only six
things needing promotion to `sjpegi.h`:

```cpp
enum { AC_BITS = 4, FP_BITS = 16 };        // AnalyseHisto needs FP_BITS, quantize.cc both
inline int CalcLog2(int v);                 // used by quantize + entropy + histogram
inline int TrailingZeros64(uint64_t x);     // travels with CalcLog2
extern const float kDefaultQuality;         // + kDefaultBias, kDefaultDeltaMax{Luma,Chroma}
MemoryManager* GetDefaultMemoryManager();   // one ctor use + one assert in InitFromParam
Encoder* EncoderFactory(...);
bool FinishEncoding(Encoder*, const EncoderParam&);
```

Everything else resolves itself:

- `kDensityThreshold`, `kCorrelationThreshold`, `kOmittedChannels` are read only
  by `AnalyseHisto` — relocate them into `histogram.cc` instead of exporting.
  Good moment to make the first two `const` (already flagged in the #150
  backlog).
- `kHuffmanTables` crosses only because `Encode()` inlines the four-line default
  table fill; move that loop into `entropy.cc` and the symbol never leaves.
- `QUANTIZE` / `DIV_BY_MULT` end up entirely inside `quantize.cc`. Today those
  macros are visible across 2500 lines.
- The two `cmp` symbols (qsort comparator at 1384, an `__m128i` local at 578)
  land in different TUs, removing a shadowing hazard.

### Chores

- Sources are listed explicitly in **three** build files: `CMakeLists.txt`,
  `Makefile`, `Android.mk`. Five new files = five entries in each.
- Precedent exists: `dichotomy.cc` is already the multi-pass scan carved out of
  `enc.cc`.

### Optional further step

The scan drivers (`CollectCoeffs`, `SinglePassScan`, `FinalPassScan`,
`SinglePassScanOptimized`, ~200 lines) would sit naturally next to `LoopScan` in
a `scan.cc`, leaving `enc.cc` as pure lifecycle and config at ~320 lines. Costs a
seventh file. Decide after living with the six for a while.

### Verification

One commit, no behavior change:

```sh
./perf/verify.sh perf/bench_before perf/bench_after   # expect 120/120 identical
./perf/verify.sh perf/bench_before_x86 perf/bench_after_x86
```

---

## Phase 2 — bit-exact optimizations

These four are independent of each other, all verified bit-identical over the
120-config sweep on both NEON and SSE2, and each lands in one file after the
split. Land them in this order.

### 2.1 — 64-bit lazy BitWriter (`bit_writer.h`, `entropy.cc`) — measured **-25%**

`BitWriter` holds 32 bits with only 24 usable, so `PutBits()` must call
`FlushBits()` every time — and `CodeBlock` calls it twice per non-zero
coefficient (Huffman code, then suffix). `FlushBits()` then drains one byte at a
time through a mispredicting `if (tmp == 0xff)`. 76% of `CodeBlock` sits in those
two calls.

Three changes:

1. Widen `bits_` to `uint64_t`, flush **lazily** — only when the next symbol
   would not fit under 56 bits. About five symbols accumulate per flush instead
   of one. This is where most of the gain is; widening alone without the lazy
   flush was only worth 10%.
2. Emit whole bytes with one byte-swapped 8-byte store, taking the byte loop only
   when a SWAR test finds a 0xff (~0.4% of the time).
3. Add `PutPackedCodeAndSuffix()` and use it from `CodeBlock` — 16 + 11 bits
   worst case, comfortably inside the wider accumulator.

```cpp
void FlushBits() {
  const int nb_bytes = nb_bits_ >> 3;
  if (nb_bytes == 0) return;
  const uint64_t out  = BSwap64(bits_);          // first byte to emit = lowest byte
  const uint64_t mask = (~0ull) >> (64 - 8 * nb_bytes);
  if (!HasFF(out & mask)) {                      // common case: no escaping
    memcpy(buf_ + byte_pos_, &out, sizeof(out));  // writes up to 7 extra bytes
    byte_pos_ += nb_bytes;
  } else {
    uint64_t v = bits_;
    for (int i = 0; i < nb_bytes; ++i, v <<= 8) {
      const uint8_t tmp = static_cast<uint8_t>(v >> 56);
      buf_[byte_pos_++] = tmp;
      if (tmp == 0xff) buf_[byte_pos_++] = 0x00;
    }
  }
  bits_ <<= 8 * nb_bytes;
  nb_bits_ -= 8 * nb_bytes;
}

void PutBits(uint32_t bits, int nb) {
  assert(nb <= 32 && nb > 0);
  if (nb_bits_ + nb > 56) FlushBits();
  nb_bits_ += nb;
  bits_ |= static_cast<uint64_t>(bits) << (64 - nb_bits_);
}

void PutPackedCodeAndSuffix(uint32_t code, uint32_t suffix, int n) {
  const int len = code & 0xff;
  PutBits(((code >> 16) << n) | suffix, len + n);
}
```

**Safety invariant to document and assert.** The wide store can run up to 7 bytes
past the logical write position. That is safe as written — `CheckBuffers()`
reserves 2048 bytes against a 1152-byte worst-case MCU, and `WriteEOI()` calls
`Flush()` *before* its own `Reserve(2)` — but it is currently implicit and must
not stay that way.

### 2.2 — 64-bit lazy BitCounter (`bit_writer.{h,cc}`, `dichotomy.cc`) — measured **-37%** on the size search

`BitCounter::AddBits` exists only to answer "how many bits would this be?", and
answers by simulating the byte stream to count 0xff stuffing. At 49% it is the
single largest entry in the encoder for `passes > 1`. Same treatment: 64-bit lazy
accumulator, one merged call per coefficient (`AddPackedCodeAndSuffix`), escape
count by popcount.

The ordinary `haszero` idiom is **not** safe for counting — a zero byte borrows
into its neighbour and produces false positives. Mask to 0x7f before the add:

```cpp
static int CountFF(uint64_t v, int nb) {   // 0xff bytes among the top 'nb'
  const uint64_t x = ~v;                                    // look for zero bytes
  uint64_t z = ((x & 0x7f7f7f7f7f7f7f7full) + 0x7f7f7f7f7f7f7f7full) | x;
  z = ~z & 0x8080808080808080ull;                           // 0x80 per zero byte
  return __builtin_popcountll(z >> (64 - 8 * nb));
}
```

`Size()` must flush before returning, so it can no longer be `const`.

### 2.3 — chunked output commit (`bit_writer.h`, `enc.cc`) — measured **-4.5%** on large images

`CheckBuffers()` calls `bw_.Reserve(2048)` unconditionally per MCU; for the
`std::string` / `std::vector` sinks each is a virtual call into `resize()`, which
value-initializes. Track the remaining slack in `BitWriter` and only commit when
it runs out:

```cpp
bool ReserveMore(size_t size, size_t chunk) {
  if (byte_pos_ + size <= reserved_) return true;
  return Reserve(chunk);   // Reserve() now records reserved_ = size
}
// enc.cc: ok_ = ok_ && bw_.ReserveMore(2048, 256 << 10);
```

Nothing on small images, nothing for `MemorySink`; this only helps the
container-backed API. Note it still guarantees >= 2048 bytes of headroom, so
2.1's overrun invariant holds.

### 2.4 — fold entropy stats into `StoreRunLevels` (`dichotomy.cc`, `entropy.cc`) — est. **-8%** on the search

Each dichotomy pass walks the coefficients three times: `StoreRunLevels`
(quantize + materialize), then `StoreOptimalHuffmanTables` re-walks the same
run/levels only to accumulate frequencies, then `BlocksSize` walks them again.
Step 2 is pure duplication — `StoreRunLevels` already has the coefficients in
registers, and `SinglePassScanOptimized` proves the fusion works by doing exactly
that.

### Measured phase-2 result (2.1 + 2.2 + 2.3)

| configuration | before | after | gain |
|---|---:|---:|---:|
| m0 q75 420 | 5.8 ms | 4.2 ms | 27.6% |
| m1 q75 420 | 6.7 ms | 4.7 ms | 29.9% |
| m4 q75 420 | 7.5 ms | 5.6 ms | 25.3% |
| m4 q75 AUTO | 10.2 ms | 7.7 ms | 24.5% |
| m4 q75 6 passes | 31.7 ms | 20.2 ms | 36.3% |
| m4 q75 17.3 MP | 51.9 ms | 49.6 ms | 4.5% (isolates 2.3) |
| m7 q75 (trellis) | 34.3 ms | 32.6 ms | 5.0% |

Trellis gains little because it is dominated by its own search — see 3.4.

---

## Phase 3 — SIMD, and the changes that shift output

### 3.1 — zig-zag permutation in the vector stage (`quantize.cc`) — est. -10..15%, bit-exact

The most useful line-level result in the whole analysis: **`QuantizeBlockNEON` is
89% scalar.** The vector stage (abs, bias, reciprocal multiply, shift, non-zero
mask) is ~11%; the rest is the two scalar loops after it — the natural -> zig-zag
bit remap, and the run/level emission, which is a long serial dependency per
coefficient (`ctz` -> index -> `kZigzag[]` load -> `tmp[j]` load -> `clz` ->
mask -> two 16-bit stores, with `prev` carried across iterations). Same shape in
the SSE2 version.

Attack it directly:

- **Permute into zig-zag order in the vector stage** so `zz` is just `nzn`, the
  remap loop disappears, and the emission loop reads sequentially.
  `vqtbl4q_u8` on aarch64, `pshufb` with SSSE3; SSE2-only keeps the current path.
- **Store one array instead of two.** `masked[]` is just `tmp ^ sign`; keeping
  `tmp[]` plus a 64-bit sign mask halves the permutation and store traffic, and
  the per-coefficient sign is one shift-and-mask.
- **One 32-bit store per run/level.** `rl[nb].run_` and `rl[nb].level_` are
  adjacent 16-bit fields of a 4-byte struct.
- **Drop the horizontal reduction in the NEON mask build.** The
  weights-multiply-then-`vaddvq_u16` idiom has long cross-lane latency and is
  serialized into `nzn` eight times per block. `vshrn_n_u16(cmp, 4)` gives a
  64-bit word with one nibble per lane in a single instruction; iterate those
  nibbles with `ctz` and the reduction is gone.
- Minor: `vuzpq_u16` computes both halves when only `val[1]` is used — use
  `vuzp2q_u16` on aarch64. On SSE2, packing two compare vectors into one
  `movemask` halves that sequence.

The fDCT (6%) and colour conversion (3.4%) are respectable as they stand; the
only move left there is processing two blocks per call to interleave dependency
chains, worth a couple of percent at most. Not where the next hour goes.

### 3.2 — subsample `SjpegRiskiness` (`jpeg_tools.cc`) — est. -15% **| changes decisions**

`yuv_mode` defaults to `SJPEG_YUV_AUTO`, so every encode runs a full extra pass
over the source RGB — a row conversion plus 3 lookups/pixel into a 114 KB table —
to produce **one enum**. Forcing the mode takes the default encode from 10.3 ms
to 7.5 ms.

Stepping the loop by 2 in both directions costs 1.2 ms instead of 2.7 ms and
takes the default encode to 7.1 ms — but it **flipped the recommendation on
`source2.jpg`**, so the factor is a quality call, not a free win. Lower-risk
variants: skip only rows (the row conversion is a third of the cost), or
terminate early once the running estimate is unambiguously clear of all three
thresholds.

Not worth doing: converting the `double` accumulators in that loop to integers.
Tried it, changed nothing — the loop is bound purely by its table lookups.

### 3.3 — subsample the histogram pass (`histogram.cc`) — est. -6% **| shifts quant matrices**

`StoreHisto` is 12% of the default encode and already near the store-throughput
limit (one increment per coefficient into `counts_[64][129]`). Little to win by
making the loop faster; the win is doing less of it. `AnalyseHisto` consumes a
global distribution, and sampling every other MCU row would leave it
statistically indistinguishable.

If the counts must stay exact, the remaining lever is locality: batching 8–16
blocks and updating `counts_` row by row touches one 516-byte row at a time
instead of 64 scattered rows per block. Matters more on x86, where 66 KB of
histogram does not fit in L1, than on the M-series parts used here.

### 3.4 — prune the trellis node scan (`quantize.cc`) — est. -30% on m7 **| shifts output**

Method 7 is 81% `TrellisQuantizeBlock` and 4.5x slower than m4.
`SearchBestPrev()` scans every previously created node for every new one —
O(n^2), up to 127 nodes. Since every added term is non-negative, any candidate
whose `cur->score` alone already exceeds the incumbent `node->score` cannot win,
yet still pays a multiply, two table reads and a compare. A one-line skip on that
condition, plus a bound on how far back the scan reaches (each 16 of run costs an
extra ZRL code, so distant predecessors are rarely optimal), prunes most of it.
Nothing here is SIMD-shaped; it is a search-pruning problem.

### 3.5 — size from the frequency table (`dichotomy.cc`) — est. -40% on the search **| shifts chosen q**

The exact bit total is `sum_s freq[s] * (len[s] + (s & 0xf))` over the 256 AC
symbols plus the 12 DC ones — the frequency table `CompileEntropyStats()` already
built. That is 268 terms per pass instead of one traversal of every non-zero
coefficient in the image. The only part not recoverable from the histogram is the
0xff stuffing (a few tenths of a percent of the stream); approximating it moves
the chosen quality slightly.

Cheaper variant that keeps exactness where it matters: evaluate the exact counter
on a fraction of the macroblocks for the early passes, where the dichotomy only
needs to bracket *q*, and go exact for the last one or two.

---

## Bigger, separate conversation

The encoder is single-threaded, and the histogram/quantize passes are
embarrassingly parallel per MCU row. The only serial dependencies are the DC
predictor chain and the bit stream itself, both of which resolve with a cheap
per-stripe fixup. That is a much larger multiple than anything above, at a much
larger design cost, and it does not belong in the same effort.

---

## Summary table

| # | change | effect | risk | file (post-split) |
|---|---|---|---|---|
| 2.1 | 64-bit lazy BitWriter + merged code/suffix | **-25%** | bit-exact | `bit_writer.h`, `entropy.cc` |
| 2.2 | 64-bit lazy BitCounter + popcount escapes | **-37%** (search) | bit-exact | `bit_writer.*`, `dichotomy.cc` |
| 2.3 | chunked output commit | **-4.5%** | bit-exact | `bit_writer.h`, `enc.cc` |
| 2.4 | fold entropy stats into `StoreRunLevels` | -8% (search) | bit-exact | `dichotomy.cc` |
| 3.1 | zig-zag permute in the vector stage | -10..15% | bit-exact | `quantize.cc` |
| 3.2 | subsample `SjpegRiskiness` | -15% | flips modes | `jpeg_tools.cc` |
| 3.3 | subsample the histogram pass | -6% | shifts quant | `histogram.cc` |
| 3.4 | prune the trellis node scan | -30% (m7) | shifts output | `quantize.cc` |
| 3.5 | size from the frequency table | -40% (search) | shifts *q* | `dichotomy.cc` |

Rows 2.1–2.3 are measured; the rest are estimates from their share of the
profile. Phase 2 is a pure win and should land as four independent commits.
Phase 3.1 is also bit-exact. Everything from 3.2 down is a quality decision as
much as a performance one and needs a separate call.
