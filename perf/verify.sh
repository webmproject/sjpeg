#!/bin/bash
# Bit-exactness sweep: encode the test corpus with two builds of the encoder and
# compare the output byte-for-byte.
#
#   usage: verify.sh <bench-A> <bench-B> [testdata-dir]
#
# Both binaries must be built from perf/bench.cc against the two trees being
# compared. Any difference is a regression: every change in phase 1 and phase 2
# of the plan is required to be bit-exact.
#
# NOTE: use bash, not zsh. zsh does not word-split unquoted variables by
# default, which silently turns "4 75 1" into a single argv entry and makes the
# whole sweep test one configuration over and over.

set -u
A=${1:?usage: verify.sh <bench-A> <bench-B> [testdata-dir]}
B=${2:?usage: verify.sh <bench-A> <bench-B> [testdata-dir]}
# Walk up from this script to find tests/testdata, so the sweep works whether
# perf/ sits under the (temporary) build dir or at the top level of the tree.
find_testdata() {
  local d
  d=$(cd "$(dirname "$0")" && pwd)
  while [ "$d" != "/" ]; do
    [ -d "$d/tests/testdata" ] && { echo "$d/tests/testdata"; return 0; }
    d=$(dirname "$d")
  done
  echo "can't locate tests/testdata above $(dirname "$0")" >&2
  return 1
}
D=${3:-$(find_testdata)} || exit 1

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

fail=0
total=0

run() {   # image method quality passes
  local img=$1 m=$2 q=$3 p=$4
  # Clear both outputs first, and insist they come back. bench writes BENCH_OUT
  # only when the encode succeeds, so without this a failed run leaves the
  # previous configuration's file in place -- and two stale files compare equal,
  # which scores a crash as a pass for every row after the first.
  rm -f "$TMP/a.jpg" "$TMP/b.jpg"
  BENCH_OUT="$TMP/a.jpg" "$A" "$img" "$m" "$q" 1 1 "$p" >/dev/null 2>&1
  BENCH_OUT="$TMP/b.jpg" "$B" "$img" "$m" "$q" 1 1 "$p" >/dev/null 2>&1
  total=$((total + 1))
  if [ ! -s "$TMP/a.jpg" ] || [ ! -s "$TMP/b.jpg" ]; then
    echo "FAILED  $(basename "$img")  m=$m q=$q passes=$p  (no output written)"
    fail=$((fail + 1))
  elif ! cmp -s "$TMP/a.jpg" "$TMP/b.jpg"; then
    echo "DIFFER  $(basename "$img")  m=$m q=$q passes=$p"
    fail=$((fail + 1))
  fi
}

for img in "$D"/source1.png "$D"/source2.jpg "$D"/source3.jpg "$D"/source4.ppm; do
  [ -f "$img" ] || { echo "missing $img"; exit 1; }
  for m in 0 1 3 4 7; do
    for q in 40 75 95; do
      for p in 1 6; do
        run "$img" "$m" "$q" "$p"
      done
    done
  done
done

echo "$((total - fail))/$total identical"
[ "$fail" -eq 0 ] || exit 1
