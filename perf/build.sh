#!/bin/bash
# Clean-build the benchmark and copy it aside, for A/B comparisons.
#
#   usage: build.sh <output-path> [extra CPPFLAGS...]
#
# The clean is not optional. The Makefile has no dependency on the flags, so
# 'make bench CPPFLAGS=-DSOMETHING' after a normal build recompiles nothing and
# silently links a mixture of the two -- which, for a class whose layout the
# flag changes, is an ODR violation, not just a bad measurement.
#
#   ./perf/build.sh /tmp/bench_on
#   ./perf/build.sh /tmp/bench_off -DSJPEG_DISABLE_FAST_BITWRITER
#
# NOTE: use bash, not zsh (see verify.sh).

set -eu
OUT=${1:?usage: build.sh <output-path> [extra CPPFLAGS...]}
shift
cd "$(dirname "$0")/.."

find src examples perf tests -name '*.o' -delete
rm -f src/libsjpeg.a examples/libutils.a perf/bench
make bench -j8 CPPFLAGS="$*" >/dev/null
cp perf/bench "$OUT"
echo "built $OUT ${*:+with $*}"
