#!/bin/bash
# What this machine makes of the encoder, in one command.
#
#   usage: ab.sh [work-dir]        (default: a fresh temporary directory)
#
#   AB_NO_BREAKDOWN=1   skip the profile-free breakdown at the end
#   COMPARE_REPEAT=3    best-of-3 for every timing, for a machine with a wide
#                       noise floor -- the control row below says whether this
#                       machine is one
#   BD_ITERS=15         fewer repetitions inside the breakdown (small board)
#   JOBS=2              fewer parallel compiles (board short on memory)
#
# Three things, in order.
#
# A noise floor: two builds of the same sources, compared by the same protocol
# as everything else. Whatever that row reports is the machine talking, and no
# number anywhere else means anything unless it is larger.
#
# The two bit writer implementations against each other. Which one a build gets
# is decided by the target's register width, so on any 64bit machine the 32bit
# one is never compiled and never tested -- SJPEG_FORCE_32BIT selects it anyway.
# They must agree byte for byte, and the timing says what the wider accumulator
# is worth here. On a 32bit board that column is the measurement nobody making
# this choice had: it was made on 64bit machines only.
#
# Then perf/breakdown.sh, which is not an A/B at all: it describes the encoder
# as it stands -- where the time goes, what the vector layer is worth, how the
# trellis and the multi-pass search scale -- from differences between runs of
# the same binary. That is the part that says where to work next, and it needs
# no profiler, which matters because perf is blocked by perf_event_paranoid on
# at least one of the machines this gets run on.
#
# To measure a change rather than the machine, build both sides with build.sh
# and hand them to verify.sh, sweep.sh and compare.sh directly -- all three take
# two binaries and have no opinion about where they came from.
#
# NOTE: use bash, not zsh (see verify.sh).

set -u
cd "$(dirname "$0")/.."

W=${1:-$(mktemp -d)}
mkdir -p "$W"

# Every build below is a clean build with its own flags, so whichever one ran
# last leaves src/*.o and src/libsjpeg.a compiled with that build's flags. The
# Makefile has no dependency on the flags (see build.sh), so a plain 'make'
# afterwards would relink those objects and hand back a library that is not the
# default one. Leave the tree with no objects at all rather than misleading ones.
cleanup() {
  find src examples perf tests -name '*.o' -delete 2>/dev/null
  rm -f src/libsjpeg.a examples/libutils.a perf/bench
}
trap cleanup EXIT

echo "=============================================================="
echo " host    : $(uname -sm)"
echo " compiler: $(${CXX:-g++} --version 2>/dev/null | head -1)"
echo " commit  : $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo " work dir: $W"
# What this build actually compiles in. Worth printing rather than assuming: a
# 32-bit userland on a 64-bit ARM board defines neither SJPEG_AARCH64 nor,
# without -mfpu=neon, SJPEG_USE_NEON, so the vector kernels quietly disappear
# and the numbers below come out smaller for a reason that has nothing to do
# with the code. SJPEG_HAVE_64BIT decides which bit writer it gets.
echo ' platform:' $(echo '#include "sjpegi.h"' |
                    ${CXX:-g++} -Isrc -dM -E -x c++ - 2>/dev/null |
                    awk '/^#define SJPEG_(USE|HAVE|AARCH64)/ { print $2 }' |
                    sort | tr '\n' ' ')
echo "=============================================================="
echo

echo "## building (three clean builds, this takes a minute)"
./perf/build.sh "$W/bench" || exit 1
./perf/build.sh "$W/bench_control" "-DSJPEG_AB_NOISE_CONTROL" || exit 1
./perf/build.sh "$W/bench_32" "-DSJPEG_FORCE_32BIT" || exit 1
echo

echo "## the noise floor: two builds of the SAME code"
./perf/compare.sh "$W/bench_control" "$W/bench" || exit 1
echo

echo "## the 32bit and 64bit bit writers must agree, byte for byte"
./perf/verify.sh "$W/bench_32" "$W/bench" || exit 1
# Not piped into tail. '|| exit 1' after a pipe tests the exit status of tail,
# which is 0 whatever the sweep found, so the gate would never fire -- and tail
# would throw away the DIFFER lines on the way past.
if ! ./perf/sweep.sh "$W/bench_32" "$W/bench" > "$W/sweep.log"; then
  echo
  echo "BIT-EXACTNESS FAILED. The two implementations disagree:"
  grep -E 'DIFFER|FAILED' "$W/sweep.log" | head -20
  echo "  (full log: $W/sweep.log)"
  exit 1
fi
tail -8 "$W/sweep.log"
echo

echo "## what the 64bit accumulator is worth here (A = 32bit, B = 64bit)"
./perf/compare.sh "$W/bench_32" "$W/bench" || exit 1

if [ -z "${AB_NO_BREAKDOWN:-}" ]; then
  echo
  echo "## where this machine actually spends the time (no A/B, no profiler)"
  echo
  ./perf/breakdown.sh "$W/bench"
fi
