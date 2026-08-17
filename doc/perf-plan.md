# sjpeg: split enc.cc, then optimize

**Status:** phases 1 and 2 are in, one commit each, all bit-exact. 3.4 is in
too, and turned out to be bit-exact after all -- the plan had assumed it must
change the output. 3.1 was implemented, measured on three machines, and
**withdrawn**: a win on Apple silicon, a loss on Cortex-A76. 3.2, 3.3 and 3.5
are open, and each of them does change the output.

The measurement scaffolding is gone. Every optimization was built behind a
`SJPEG_DISABLE_*` switch so it could be compiled out and timed on its own;
those switches produced the per-optimization tables below, on three machines,
and were then removed. What is left of them is one gate that is not scaffolding
at all -- `SJPEG_HAVE_64BIT`, which picks the bit writer by the target's
register width. See "How this was measured" below.

Baseline for every number below: `7601185`, Apple M5 Pro (NEON), clang
`-O3 -DNDEBUG`, `tests/testdata/source3.jpg` (1600x1200), best of three
interleaved A,B pairs of best-of-30. Two more machines, and what to make of the
differences between them, are in "Cross-platform results" just below.

| configuration | before | after | |
|---|---:|---:|---:|
| m0 q75 420 | 5.8 ms | 4.0 ms | **-31.0%** |
| m1 q75 420 | 6.7 ms | 4.5 ms | **-32.8%** |
| m4 q75 420 | 7.7 ms | 5.5 ms | **-28.6%** |
| m4 q75 AUTO (default) | 10.5 ms | 8.3 ms | **-21.0%** |
| m4 q95 420 | 10.6 ms | 7.2 ms | **-32.1%** |
| m4 q75 6 passes | 32.5 ms | 19.4 ms | **-40.3%** |
| m7 q75 420 (trellis) | 35.5 ms | 18.4 ms | **-48.2%** |
| m4 q75 420, 17.3 MP | 67.7 ms | 48.8 ms | **-27.9%** |

Method 7 used to move least -- it is dominated by its own search, and nothing
in phase 2 touched it. 3.4 is that search, and it is now the largest single
number in the table.

### Cross-platform results

Three machines, same tree, `./perf/ab.sh` each time. Every one of them reports
**1320/1320 byte-identical** (the 120-config sweep plus the 1200-config quality
ladder), across two compilers and both vector paths, so the bit-exactness claim
is not an artefact of one toolchain.

| configuration | M5 Pro, clang 21 | RPi 5 (A76), gcc 14 | x86_64, gcc 15 |
|---|---:|---:|---:|
| m0 q75 420 | -31.0% | -18.4% | -18.9% |
| m1 q75 420 | -32.8% | -17.1% | -12.3% |
| m4 q75 420 | -28.6% | -14.8% | -9.1% |
| m4 q75 AUTO | -21.0% | -11.5% | -9.2% |
| m4 q95 420 | -32.1% | -15.4% | -12.3% |
| m4 q75 6 passes | -40.3% | -27.2% | -25.6% |
| m7 q75 (trellis) | -48.2% | -49.2% | -47.8% |
| m4 q75 17.3 MP | -27.9% | -16.2% | -12.5% |

The trellis row is the one that behaves the same everywhere, to within a point
and a half across three microarchitectures and two compilers. It removes work
rather than exploiting a machine: an exact early-out that ends a quadratic scan
once nothing left in it can win. Nothing else here transfers that cleanly.

### What each optimization is worth, on three machines

Each row is the full build minus that one change, on the configuration where
that change acts -- quoting any of them on the default row would report ~0% for
two real optimizations. The columns do not add up to the totals above, and are
not meant to: two optimizations on the same code path each cover for the
other's absence when it is removed alone.

| optimization | measured on | M5 | RPi 5 | x86_64 |
|---|---|---:|---:|---:|
| 64bit `BitWriter` | m0 q75 420 | -31.0% | -18.4% | -16.1% |
| 64bit `BitCounter` | m4 q75 6 passes | -36.6% | -21.9% | -24.8% |
| chunked commit | m4 q75 420 | -8.3% | -3.2% | -0.9% |
| fused entropy stats | m4 q75 6 passes | -5.3% | -1.7% | -3.9% |
| trellis prune | m7 q75 420 | -43.5% | -47.5% | -46.4% |
| *noise floor* | *control, worst row* | *1.8%* | *0.6%* | *3.1%* |

Read the noise floor first. On the Haswell nothing under about 3% is a result,
even with `COMPARE_REPEAT=3`, which is why the chunked-commit cell there is
reported as noise rather than as -0.9%: the same run has that optimization
reading +17.8% on a row it cannot touch.

The two halves of the bit writer split exactly as their mechanism says they
should, on all three machines: the writer owns the single-pass rows and does
almost nothing for the size search, the counter owns the six-pass row and does
nothing anywhere else. That is the strongest evidence in this document that the
columns are measuring what they claim to.

The percentages differ a lot; the milliseconds much less. What was optimized is
the writing of the coefficient bitstream, and m0, m1 and m4 write bitstreams of
similar size -- 606.0 kB, 599.0 kB and 572.4 kB for `source3.jpg` at q75, so m0
writes 5.9% more than m4 -- while the analysis around that write differs by
much more. The absolute saving should therefore be roughly constant down that
column, drifting down by a few percent:

| configuration | M5 | RPi 5 | x86 |
|---|---:|---:|---:|
| m0 q75 420 | 1.9 ms | 3.7 ms | 6.9 ms |
| m1 q75 420 | 2.2 ms | 4.1 ms | 4.2 ms |
| m4 q75 420 | 2.1 ms | 5.1 ms | 3.5 ms |
| m4 q75 AUTO | 2.1 ms | 5.3 ms | 3.0 ms |

(An earlier version of this section claimed the three methods write *the same*
bitstream and that the column should be exactly flat. They do not: Huffman
optimization changes the tables and adaptive quantization changes the matrices,
so both the coefficients and the byte count move. The expectation is "roughly
constant, drifting down about 6%", not "identical".)

The M5 is flat to within 0.3 ms. The Pi drifts the wrong way, up by 1.6 ms from
m0 to m4, and x86 drifts down by twice what the byte counts justify -- both
inside, or close to, floors that the next two sections measure at 2-5% (Pi) and
up to 13% (x86) on these rows. Neither is worth a mechanism.

What the table does say, robustly, is that the percentage measures how much of an
encode *that machine* spends writing bits, not how good the change is. The M5
does the rest of a m4 encode in 5.4 ms where the Pi needs 30.2 ms, a 5.6x gap on
work that is mostly SIMD, which is why a comparable saving reads as -28% on one
and -14% on the other. That also says where the next work is on a small core,
and it is not the bit writer: it is `StoreHisto`, `QuantizeBlock` and the fdct,
i.e. 3.3.

### Where the time goes now, on the three machines

From `perf/breakdown.sh`, which is not an A/B: every number is a difference
between two runs of the same binary, so it needs no second build and says
nothing about any optimization. It describes the encoder as it stands, which is
what decides where to work next.

| | M5 | RPi 5 | x86_64 |
|---|---:|---:|---:|
| base encode (m0), share of default | 47.6% | 44.0% | 41.5% |
| Huffman optimization (m1 - m0) | 6.0% | 10.7% | **22.3%** |
| adaptive quantization (m3 - m0) | 12.2% | 17.3% | 12.8% |
| `SjpegRiskiness` (AUTO - forced) | 32.5% | 25.5% | 13.0% |
| trellis (m7 - m4), method 7 only | 156% | 110% | 114% |
| what the vector layer is worth | 1.74x | 1.52x | 1.46x |
| 6 passes, vs 1 pass | 3.49x | 2.70x | 2.46x |

Three things to take from this. `SjpegRiskiness` is the largest single item in
the default encode on two of the three, and all it decides is one enum -- that
is 3.2, and it changes output. Huffman optimization costs the Haswell nearly
four times the share it costs the M5, so whoever optimizes that code will
conclude different things depending on which machine they sit at. And the
vector layer is worth much the same everywhere, 1.5x to 1.7x, which is an
argument against more SIMD work being the next move on any of them.

One caveat the harness surfaces itself: on x86 the two independent readings of
`SjpegRiskiness` disagree -- 13.0% by the AUTO/forced difference against 19.7%
timed directly -- where the M5 and the Pi agree to half a point. On a machine
with a 3% floor the stage split is a difference of two noisy numbers, and
should not be quoted to the decimal. The direct reading is printed underneath
the table for exactly this reason.

### Reading the x86 numbers: that machine has a wide noise floor

Three runs of the headline table on the Haswell, same two binaries, gave three
answers -- m0 -31.1% / -21.3% / -31.5%, m4/AUTO -8.4% / **+3.6%** / -7.2%. Four
independent ways of asking how much of that is the machine agree closely:

| row | control: two builds of one program | the zig-zag column, before the toggle went away | six timings of one `bench_after` | headline, three runs |
|---|---:|---:|---:|---:|
| m0 q75 420 | **+12.8%** | +10.3% | 14.8 - 18.2 ms (23%) | 20% |
| m1 q75 420 | -4.7% | -1.3% | 20.1 - 24.3 ms (21%) | 13% |
| m4 q75 420 | -0.6% | +0.3% | 27.9 - 32.5 ms (16%) | 9% |
| m4 q75 AUTO | -1.3% | +5.8% | 34.9 - 40.0 ms (15%) | 8% |
| m4 q95 420 | +2.7% | +0.5% | 38.9 - 42.9 ms (10%) | 2% |
| m4 q75 6 passes | **-3.3%** | **-3.3%** | 73.9 - 79.6 ms (8%) | 0.6% |
| m7 q75 (trellis) | -0.5% | -2.1% | 144.2 - 147.5 ms (2.3%) | 1% |
| m4 q75 3x3 tiled | +1.4% | -0.5% | 272.8 - 283.7 ms (4%) | 0.03% |

The first column is the control `ab.sh` now prints before anything else, and the
second is the accident that prompted it: `SJPEG_USE_ZIGZAG_PERMUTE` was gated on
`SJPEG_USE_NEON && SJPEG_AARCH64`, so on x86 `-DSJPEG_DISABLE_ZIGZAG_PERMUTE`
changed nothing -- preprocessing every `src/*.cc` for `x86_64` with and without
it gives byte-identical output -- and that leave-one-out timed one program
against itself. The deliberate control reproduces it (+12.8% against +10.3% on
m0, -3.3% against -3.3% on 6 passes), which is the harness checking its own work.

So on that box nothing under about 10% is a measurement on the short rows. The
same control on the M5 reads 0.0% to 0.3% on every row, which is why its numbers
were flat and believable in the first place. The floor scales with how long a row
runs, so the x86 rows worth reading are 6-pass, m7 and tiled.

`COMPARE_REPEAT=3` shrinks the floor when it comes out wide, by interleaving the
A,B pair three times and keeping the best of each -- best-of-30 *inside* one
invocation does nothing about drift *between* invocations, which is what this is.

### What each optimization is worth on x86 (Haswell, gcc 15)

Measured at `9191ef1`, after 3.1 was withdrawn. Read every column against the
control; the long rows carry the weight.

| row | control | -BITWRITER | -BITCOUNTER | -CHUNKED | -FUSED |
|---|---:|---:|---:|---:|---:|
| m0 q75 420 | +12.8% | -17.7% | -8.8% | -2.6% | -1.3% |
| m1 q75 420 | -4.7% | -10.5% | -5.0% | -0.4% | -0.4% |
| m4 q75 420 | -0.6% | -7.1% | +6.4% | -0.3% | 0.0% |
| m4 q75 AUTO | -1.3% | -5.6% | +0.3% | +0.5% | +3.7% |
| m4 q95 420 | +2.7% | -4.0% | -0.7% | -1.4% | +2.4% |
| m4 q75 6 passes | -3.3% | +1.7% | **-22.5%** | +0.3% | +1.3% |
| m7 q75 (trellis) | -0.5% | **-4.3%** | +1.1% | +0.2% | +1.6% |
| m4 q75 3x3 tiled | +1.4% | **-5.8%** | +2.8% | +2.9% | -1.2% |

- **`FAST_BITWRITER` carries x86**, and the two rows with the tightest floors
  are the two that prove it: -4.3% on m7 against a 0.5% control, -5.8% on tiled
  against 1.4%.
- **`FAST_BITCOUNTER` pays only on multi-pass** -- -22.5% on 6 passes, nothing
  outside the floor anywhere else. `BitCounter` is constructed in
  `Encoder::ComputeSize`, reached only from `LoopScan`, i.e. `passes_ > 1`, so
  its seven single-pass rows are a *third* control, and they read -8.8% to +6.4%.
- **The two are mirror images**, and this run makes it unambiguous: `BITWRITER`
  reads +1.7% on 6 passes (floor 3.3%) where `BITCOUNTER` reads -22.5%. A
  multi-pass encode measures six times and writes once. Same split on the Pi
  (-0.8% and -19.0%), from a much quieter machine.
- **`CHUNKED_COMMIT` and `FUSED_STATS` are both under this machine's floor.**
  Chunked commit read -19.6% to +0.3% on the previous run and -2.6% to +2.9% on
  this one; fused stats never leaves +-4%. Both are real and both are small --
  the Pi resolves them at about -2% and -2.3% (6 passes) respectively -- and a
  2% effect is simply not measurable against a 1.4-3.3% floor. Recorded as
  "confirmed elsewhere, unresolvable here" rather than averaged into a number.

The withdrawal of 3.1 is a no-op here, as it must be: x86 never compiled the
permutation. The headline reproduces the two earlier runs within the floor
(6 passes -26.6% / -24.7% / -26.3%), which is the check that the revert touched
nothing it should not have.

`perf record` on that machine failed with `perf_event_paranoid = 4`; it wants
`sudo sysctl -w kernel.perf_event_paranoid=1` before it will open the counters.

### What each optimization is worth on the RPi 5 (A76, gcc 14)

The Pi is the quiet machine of the three, so this is the best per-toggle data we
have. Measured at `9191ef1`.

| row | control | -BITWRITER | -BITCOUNTER | -CHUNKED | -FUSED |
|---|---:|---:|---:|---:|---:|
| m0 q75 420 | 0.0% | -18.6% | +0.6% | +0.6% | 0.0% |
| m1 q75 420 | -1.0% | -13.6% | +1.4% | -1.4% | +1.9% |
| m4 q75 420 | +3.8% | -10.9% | -0.3% | -1.3% | +0.7% |
| m4 q75 AUTO | -1.8% | -9.7% | -0.5% | -0.5% | +0.5% |
| m4 q95 420 | +0.5% | -14.4% | +0.5% | -1.3% | -0.3% |
| m4 q75 6 passes | +9.0% (see below) | -3.3% | **-20.5%** | +1.0% | -1.2% |
| m7 q75 (trellis) | -0.1% | **-2.4%** | -0.7% | -1.1% | +1.0% |
| m4 q75 3x3 tiled | -4.3% | -10.0% | -3.4% | -1.7% | -2.9% |

- **`FAST_BITWRITER` carries every row but one**, 8/8 negative, and it clears the
  floor comfortably where the floor is tightest: -2.4% on m7 against 0.1%.
- **`FAST_BITCOUNTER` is its mirror**: -20.5% on 6 passes, and nothing outside
  the floor anywhere else, because a multi-pass encode measures six times and
  writes once. `BitCounter` is constructed only in `Encoder::ComputeSize`, under
  `LoopScan`, so its seven single-pass rows are again a control and read -3.4% to
  +1.4%. x86 gives the same pair from a noisier machine: -22.5% and +1.7%.
- **`CHUNKED_COMMIT`**: 6/8 negative, all about 1%. Across both Pi runs the sign
  is stable and the magnitude is 1-2%. Real, small, and only this machine can
  see it.
- **`FUSED_STATS`**: -1.2% on 6 passes here, -2.3% on the previous run, ~0
  everywhere else -- which is the only place it does any work. Same verdict.

**A single control sample can itself be unlucky.** The 6-pass control reads
+9.0%, which would put the floor on that row above `FUSED_STATS` and much of
`CHUNKED_COMMIT`. It is one bad draw: that column's B run took 84.5 ms, while
the five other timings of *that same binary* in this log sit between 76.6 and
78.2 ms. Taken across all six, the row's real spread is about 2%. So read the
control as evidence that a floor exists and roughly how big, not as a threshold
to test against -- it is one sample drawn the same way as every other column,
which is exactly what makes it comparable, and exactly why it is not tight.
`COMPARE_REPEAT=3` tightens every column including the control, and is the right
answer when a decision hangs on a 2% effect.

Floors on this machine, taken properly from the six timings of `bench_after`:
0.6% on m0, 1.9% on m1, 2.0% on m4, 2.1% on AUTO, 3.7% on q95, 2.1% on 6 passes
(excluding the outlier), 0.9% on m7, 5.2% on tiled.

### The withdrawal of 3.1, measured

The Pi is where the revert had to show up, and it did. Same all-off baseline
either side of it -- 35.5 ms then 35.3 ms on m4 q75, so the machine had not
moved -- against these:

| row | with 3.1 | without | shipped binary |
|---|---:|---:|---:|
| m0 q75 420 | -16.8% | **-18.6%** | 16.8 -> 16.2 ms |
| m1 q75 420 | -12.7% | **-16.4%** | 22.6 -> 20.9 ms |
| m4 q75 420 | -10.4% | **-14.4%** | 31.8 -> 30.2 ms |
| m4 q75 AUTO | -11.7% | -12.0% | 40.8 -> 38.9 ms |
| m4 q95 420 | -16.4% | -15.6% | 39.9 -> 37.9 ms |
| m4 q75 6 passes | -18.2% | **-24.6%** | 88.3 -> 77.6 ms |
| m7 q75 (trellis) | -3.7% | -4.0% | 134.9 -> 131.3 ms |
| m4 q75 3x3 tiled | -15.1% | -14.1% | 290.4 -> 276.7 ms |

Every absolute time dropped. 6 passes was the predicted row -- the leave-one-out
had said +11.4%, forecasting about -26%, and it came in at -24.6%. The
single-pass rows came in 2-4 points better than the leave-one-out predicted,
which is more than it forecast and in a couple of places more than the floor
explains; a leave-one-out column is a single sample per row, and that is its
limit.

On x86 the revert is a no-op, as it must be, since that machine never compiled
the permutation. Its headline reproduces across three runs: 6 passes at -26.6%,
-24.7%, -26.3%.

### 3.1 does not pay on Cortex-A76

The zig-zag permutation is the one change that does not transfer, which is what
the toggle was for. Same toggle, the two aarch64 machines:

| row | M5 Pro (best of 3 pairs) | RPi 5 |
|---|---:|---:|
| m0 q75 420 | -7.7% | +1.8% |
| m1 q75 420 | -6.8% | +2.8% |
| m4 q75 420 | -5.6% | -3.1% |
| m4 q75 AUTO | -3.7% | +2.0% |
| m4 q95 420 | **-9.9%** | +0.7% |
| m4 q75 6 passes | -1.1% | **+11.4%** |
| m7 q75 (trellis) | +2.2% | -1.5% |
| m4 q75 3x3 tiled | -6.3% | -0.6% |

The cost of the permute is fixed: eight vector loads to build the two tables,
then 16 `vqtbl4q_u8` per block. The saving is proportional to the number of
non-zero coefficients, since what it deletes is the natural -> zig-zag remap
loop, which iterates once per non-zero. So its value rises with quality and
collapses on sparse blocks -- and both machines agree on that shape: q95 is the
M5's best row (-9.9%) and 6 passes is its worst (-1.1%), because a dichotomy
spends most of its passes at low q where the blocks are nearly empty.

What differs is the break-even. A four-register `TBL` is several times more
expensive per issue slot on an A76 than on an Apple core, which moves break-even
past any coefficient count that occurs in practice there. So on the Pi the
permute is a wash single-pass (+2% at most, mostly inside the floor) and costs
11.4% on the row where blocks are sparsest. Undoing it would take that machine's
6-pass result from -18.2% to -26.3%, and leave every other row unchanged.

`SJPEG_USE_NEON && SJPEG_AARCH64` is therefore the wrong gate: it says "aarch64"
where the real predicate is "TBL is cheap here", which we have verified on Apple
silicon and refuted on the most common aarch64 core in the world. There is no
clean compile-time test for the real predicate.

**Decision: withdrawn**, and the outcome is measured in "The withdrawal of 3.1,
measured" above. 3.1 was reverted exactly -- `quantize.cc` and `sjpegi.h` are
byte-identical to their state before it landed. That gives back
~150 lines of `#if`, the `kZigzagBytes` table, 256 bytes of `Quantizer`, and the
regression it was costing method 7, which on the M5 goes from -3.9% to **-8.0%**
now that the trellis sees the smaller struct again. What it costs is 5-7 points
on the M5's single-pass rows. Every change that remains is a win on all three
machines, which is the claim the series is making.

Runtime dispatch is the real answer and is out of scope here: the permuted and
natural paths produce their coefficients in different orders, so the choice is
baked into the run/level emission, not just the quantizer tables. If it comes
back, it comes back that way, and `perf/` will measure it on all three machines
before it is on by default anywhere.

### How this was measured, and why the switches are gone

Every optimization was built behind a `SJPEG_DISABLE_*` switch that compiled
the implementation it replaced. That is what made `perf/ab.sh` able to build an
honest "before" column without checking out an older commit and comparing
across compilers, and what made the per-optimization table above possible at
all: leave one out, rebuild, re-measure.

Three things that apparatus taught, worth keeping even though the switches are
not:

**A control row.** Two builds of the *same* sources, compared by the same
protocol as everything else, using a macro that does not exist. Whatever it
reports is the machine talking. On the Haswell it read up to 12% before
`COMPARE_REPEAT=3` and about 3% after -- so half the columns measured there
were noise, and nothing said so until the control row was there to say it.

**A check that a switch changes anything.** A toggle gating code behind an
architecture test compiles to nothing on a machine without that path, and its
leave-one-out then builds the same program twice and prints a column that looks
exactly like a result. The zig-zag column on x86 read -3.3%..+10.3% across
eight rows, every one of them from a binary byte-for-byte identical to the one
it was compared against. `ab.sh` now preprocesses the sources with and without
each switch and says so on the row.

**A gate that can fail.** For most of this work the bit-exactness gate could
not: `sweep.sh A B | tail -8 || exit 1` tests the exit status of `tail`. A
build that broke the output would have been reported as a clean A/B with
timings attached. A gate whose only evidence is that it has never fired is not
evidence.

The switches themselves were scaffolding, and are now removed: chunked commit,
fused statistics and the trellis prune are unconditional, because each does
less work for the same bytes out and there is no machine where that loses.

One remains, under a name that says what decides it. The 64bit accumulator and
its eight-byte flush want a 64bit register; on a 32bit target the compiler
emulates the accumulator in a register pair and the single store becomes two.
Nobody here has measured that -- every machine this was developed on is 64bit
-- so a 32bit target keeps the implementation written for it, selected by
`SJPEG_HAVE_64BIT` from the target rather than by hand. `SJPEG_FORCE_32BIT`
selects it on a 64bit host too, so the path a 32bit target takes can be built
and compared anywhere; without that it would be a branch no test ever compiles,
which is the kind that goes stale unnoticed. `ab.sh` builds both on every run
and checks they agree.

On the M5 the wider accumulator is worth -32.8% on m0 and -37.0% on the size
search. On a genuinely 32bit board that column would be the measurement this
choice was made without -- if you have one, that run is the interesting one.

```sh
./perf/build.sh /tmp/bench_64
./perf/build.sh /tmp/bench_32 -DSJPEG_FORCE_32BIT
./perf/compare.sh /tmp/bench_32 /tmp/bench_64
./perf/verify.sh  /tmp/bench_32 /tmp/bench_64    # expect 120/120
./perf/sweep.sh   /tmp/bench_32 /tmp/bench_64    # 1200 configs over q0..q100
```

`perf/build.sh` cleans before building on purpose: the Makefile has no
dependency on the flags, so flipping one and rebuilding otherwise links a
mixture of both -- an ODR violation, since these change a struct layout, not
just a code path.

Full analysis, with profiles and reasoning:
<https://claude.ai/code/artifact/b1d4778a-5501-482e-a289-a6ae0fad4a68>

---

## Phase 0 — measurement harness

This is in already, in `perf/`, so that every number below is reproducible and
comparable. All paths and commands here are relative to the top of the tree.

- `perf/bench.cc` — links the library directly, reads the source once, tiles it
  to a chosen size, re-encodes in a loop, reports best-of-N. Env vars:
  `BENCH_OUT=<path>` dumps the encoded result (for bit-exactness checks),
  `BENCH_SLOW_C=1` sets `sjpeg::ForceSlowCImplementation` to A/B the whole SIMD
  layer.
- `perf/verify.sh` — bit-exactness sweep across 4 images x 5 methods x
  3 qualities x {1, 6} passes = 120 configurations. It takes two `bench`
  binaries and prints `N/120 identical`, exiting non-zero on any difference.
  It finds `tests/testdata` by walking up from its own location, so it does not
  care where it is run from. Both directions are checked: it reports 120/120
  for the phase-2 prototype, and correctly isolates `source2.jpg` as the only
  image whose output moves under the phase-3.2 riskiness subsampling.

Build — either build system knows about it. It is not built by `make all`,
being a development tool, and it is never installed:

```sh
make bench                                  # -> perf/bench
cmake --build <builddir> --target bench     # or -DSJPEG_BUILD_PERF=OFF to skip
```

Profile (macOS):

```sh
./perf/bench tests/testdata/source3.jpg 4 75 1 4000 1 1 >/dev/null &
sleep 1.5; sample $! 8 -f prof.txt; kill $!
awk '/Sort by top of stack/,/Binary Images/' prof.txt
```

x86_64 cross-build for the SSE2 paths. This one stays a hand-rolled command
line: libpng/libjpeg are arm64-only under Homebrew, so the codec defines have
to go and the input has to be the `.ppm`:

```sh
c++ -arch x86_64 -O3 -DNDEBUG -std=c++11 -msse2 -Isrc -Iexamples \
    perf/bench.cc src/*.cc examples/utils.cc -o perf/bench_x86
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

## Phase 1 — split enc.cc (mechanical, zero behavior change) — DONE

`enc.cc` was 2573 lines doing six jobs. Every `////` banner in it already fell
on a clean function boundary, and the cuts landed exactly on those banners — no
function body was touched. Line counts below are the files as they ended up,
which is a little above the extracted ranges: each carries its own licence
header and include block.

| new file | lines | ranges in the old enc.cc | job |
|---|---:|---|---|
| `quantize.cc` | 564 | 64–146, 341–382, 544–953 | quant matrix setup/finalization, `QuantizeBlock` x3, trellis, `QuantizeError` x3 |
| `encoders.cc` | 575 | 1725–2269 | `InitComponents`, edge replication, `Encoder420/444/400/Sharp`, NV12/NV21 + direct-YUV adapters, `EncoderFactory` |
| `entropy.cc` | 433 | 383–543, 954–994, 1355–1594 | standard + optimal Huffman tables, `GenerateDCDiffCode`, `CodeBlock`, entropy statistics |
| `enc.cc` (kept) | 466 | 1–63, 147–340, 1288–1354, 1595–1724, 2550– | Encoder lifecycle, setters, buffers, CPU dispatch, scan drivers, `Encode()` |
| `histogram.cc` | 343 | 995–1287 | `StoreHisto` x3, `AnalyseHisto`, `CollectHistograms` |
| `api.cc` | 307 | 2270–2549 | `SjpegEncode`/`SjpegCompress`, `EncoderParam`, `Encode()` overloads |

The ranges sum to 2573 exactly.

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

Two more turned up while compiling, both of the same kind: `SetDefaultMinQuantMatrix()`
(defined in `quantize.cc`, called from the `enc.cc` setters) and a
`GrayEncoderFactory()`, added so that `Encoder400G` need not leave
`encoders.cc` just because `EncodeGray()` constructs one.

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

One commit, no behavior change: 120/120 identical against `7601185`, on NEON
and on the x86_64/SSE2 cross-build, and no measurable difference in speed.

---

## Phase 2 — bit-exact optimizations — DONE

These four are independent of each other, all verified bit-identical over the
120-config sweep on both NEON and SSE2, and each lands in one file after the
split. Land them in this order.

### 2.1 — 64-bit lazy BitWriter (`bit_writer.h`, `entropy.cc`) — DONE, **-20 to -28%**

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

### 2.2 — 64-bit lazy BitCounter (`bit_writer.{h,cc}`, `dichotomy.cc`) — DONE, **-34.9%** on the size search

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

### 2.3 — chunked output commit (`bit_writer.h`, `enc.cc`) — DONE, **-4.8%** on large images

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

### 2.4 — fold entropy stats into `StoreRunLevels` (`dichotomy.cc`, `entropy.cc`) — DONE, **-5.0%** on the search

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

### 3.1 — zig-zag permutation in the vector stage (`quantize.cc`) — implemented, then **WITHDRAWN**

Kept here because the analysis below is still the right reading of where
`QuantizeBlockNEON` spends its time, and 3.4 will want it. Only the permutation
itself was reverted, and only because it does not pay on Cortex-A76 — see "3.1
does not pay on Cortex-A76" above for the numbers and the reasoning.

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

### 3.4 — prune the trellis node scan (`quantize.cc`) — DONE, **-43 to -47% on m7**, bit-exact

Method 7 is 81% `TrellisQuantizeBlock` and 4.5x slower than m4.
`SearchBestPrev()` scans every previously created node for every new one —
O(n^2), up to 127 nodes. This was filed under "shifts output", on the
assumption that pruning a search means accepting a different answer. It does
not: the prune is exact, and the output is byte-identical.

The scan walks backwards towards the sink, so the run between candidate and
node only grows as it goes. The distortion grows with it, because `disto0[]` is
non-decreasing, and so does the ZRL part of the bit count. Both terms of
`disto + lambda * bits` are therefore monotone *in the direction the loop
travels*, and the two terms left out of that bound — the symbol's own code
length and the candidate's own score — are non-negative. Once the bound reaches
the incumbent score, nothing left in the scan can beat it. It is a `break`, and
it discards only candidates that provably lose.

Worth, on method 7: -15% at q40, -44% at q75, -70% at q95, -77% at q100. The
node count grows with quality, so the quadratic term is worst exactly where the
prune bites hardest.

The first attempt was the one described in the original plan — skip a candidate
whose `cur->score` already exceeds the incumbent — and it came out **56%
slower**. Node scores accumulate with position, so a backwards scan meets the
expensive candidates first, before any incumbent exists to compare them
against; the test almost never fired and the branch was unpredictable. The
difference between the two is not the arithmetic, it is that the surviving
bound is monotone in the direction of travel, which is what turns a `continue`
into a `break`.

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

| # | change | measured | status | file |
|---|---|---|---|---|
| 1 | split `enc.cc` into six files | 0% | done | — |
| 2.1 | 64-bit lazy BitWriter + merged code/suffix | **-20 to -28%** | done | `bit_writer.h`, `entropy.cc` |
| 2.2 | 64-bit lazy BitCounter + popcount escapes | **-34.9%** (search) | done | `bit_writer.*`, `dichotomy.cc` |
| 2.3 | chunked output commit | **-4.8%** (large) | done | `bit_writer.h`, `enc.cc` |
| 2.4 | fold entropy stats into `StoreRunLevels` | **-5.0%** (search) | done | `dichotomy.cc` |
| — | force-inline the per-coefficient puts | **-7%** | done | `bit_writer.h` |
| 3.1 | zig-zag permute in the vector stage | -6 to -10% on Apple, **+11% on A76** | withdrawn | `quantize.cc` |
| 3.4 | prune the trellis node scan | **-43 to -47%** (m7) | done, bit-exact | `quantize.cc` |
| 3.2 | subsample `SjpegRiskiness` | est. -15% | open, flips modes | `jpeg_tools.cc` |
| 3.3 | subsample the histogram pass | est. -6% | open, shifts quant | `histogram.cc` |
| 3.5 | size from the frequency table | est. -40% (search) | open, shifts *q* | `dichotomy.cc` |

Everything marked done is bit-exact, verified over 1320 configurations on three
machines. 3.2, 3.3 and 3.5 change the output, and are quality decisions as much
as performance ones. 3.4 was on that list and came off it: the assumption that
pruning a search must change its answer was simply wrong, and worth checking
before accepting a quality cost for speed.

The force-inline row was not in the original plan and is the cheapest thing
here: clang was leaving the merged code+suffix put out of line, so the encoder
paid a call per non-zero coefficient. It only became visible in a profile once
2.1 had made the surrounding work cheap enough for the call to stand out —
which is the argument for re-profiling after each step rather than working
through a list.

Two things the estimates did not anticipate. The split costs nothing in speed
(measured at 0% across every configuration, so the cross-TU boundaries do not
block anything the compiler was doing). And 3.1, the only change here that was
architecture-specific, turned out to be the only one that did not survive
contact with a second aarch64 machine — the toggle it shipped behind is what
made that measurable, and then what made backing it out a one-line experiment
rather than an argument.
