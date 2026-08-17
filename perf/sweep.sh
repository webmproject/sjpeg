#!/bin/bash
# Quality-ladder sweep: compare two bench binaries over the whole quality
# range, byte-for-byte and for speed.
#
#   usage: sweep.sh <bench-A> <bench-B> [image]
#
# The ladder is deliberately non-uniform. Quality does not map linearly onto
# anything the encoder cares about: the low end is a few coarse matrices that
# all behave alike, while the top few points are where the interesting things
# happen -- the quantizer reaches 1 (around q93, where FinalizeQuantMatrix has
# to special-case the reciprocal), the blocks stop being sparse, and 0xff
# escapes in the bitstream go from rare to common. So the ladder is coarse
# below q70 and dense above q90.
#
# NOTE: use bash, not zsh (see verify.sh).

set -u
A=${1:?usage: sweep.sh <bench-A> <bench-B> [image]}
B=${2:?usage: sweep.sh <bench-A> <bench-B> [image]}

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
D=$(find_testdata) || exit 1
IMG=${3:-$D/source3.jpg}

QUALITIES="0 1 2 3 5 8 12 17 23 30 38 46 54 62 70 76 81 85 88 90 92 93 94 95 96 97 98 99 99.5 100"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

run() {   # binary quality method iters label passes -> "ms size"
  # Clear the output first: bench writes it only on success, and a stale file
  # from the previous configuration compares equal to its twin, which would
  # score a failed encode as a bit-exact pass. See the check at each call site.
  rm -f "$TMP/$5.jpg"
  BENCH_OUT="$TMP/$5.jpg" "$1" "$IMG" "$3" "$2" 1 "$4" "$6" 2>/dev/null |
    sed -n 's/.*best *\([0-9.]*\) ms.*out=\([0-9]*\).*/\1 \2/p'
}

set -- $QUALITIES
echo "image: $(basename "$IMG"), $# points from q0 to q100"
echo
echo "== speed and size vs quality, m4 420, best of 20 =="
printf "%6s %10s %9s %9s %8s  %s\n" "q" "bytes" "A" "B" "change" ""
ladder_fail=0
for q in $QUALITIES; do
  a=($(run "$A" "$q" 4 20 a 1))
  b=($(run "$B" "$q" 4 20 b 1))
  [ ${#a[@]} -eq 2 ] && [ ${#b[@]} -eq 2 ] ||
    { echo "q=$q FAILED to run"; ladder_fail=$((ladder_fail + 1)); continue; }
  if cmp -s "$TMP/a.jpg" "$TMP/b.jpg"; then ok=""
  else ok="  <-- DIFFER"; ladder_fail=$((ladder_fail + 1)); fi
  printf "%6s %10s %8.2f %8.2f %7.1f%%%s\n" "$q" "${a[1]}" "${a[0]}" "${b[0]}" \
         "$(echo "${a[0]} ${b[0]}" | awk '{print ($2 - $1) / $1 * 100}')" "$ok"
done

echo
echo "== bit-exactness over the ladder: 4 images x 5 methods x {1,6} passes =="
# Counted apart from the quality ladder above: one 'identical' ratio that mixed
# the two would divide this section's failures by that section's total.
fail=0
total=0
for img in "$D"/source1.png "$D"/source2.jpg "$D"/source3.jpg "$D"/source4.ppm; do
  [ -f "$img" ] || { echo "missing $img"; exit 1; }
  IMG_SAVED=$IMG
  IMG=$img
  n=0
  for m in 0 1 3 4 7; do
    for q in $QUALITIES; do
      for p in 1 6; do
        run "$A" "$q" "$m" 1 a "$p" >/dev/null
        run "$B" "$q" "$m" 1 b "$p" >/dev/null
        n=$((n + 1)); total=$((total + 1))
        if [ ! -s "$TMP/a.jpg" ] || [ ! -s "$TMP/b.jpg" ]; then
          echo "FAILED  $(basename "$img")  m=$m q=$q passes=$p  (no output written)"
          fail=$((fail + 1))
        elif ! cmp -s "$TMP/a.jpg" "$TMP/b.jpg"; then
          echo "DIFFER  $(basename "$img")  m=$m q=$q passes=$p"
          fail=$((fail + 1))
        fi
      done
    done
  done
  echo "  $(basename "$img"): $n configurations"
  IMG=$IMG_SAVED
done

echo
echo "$((total - fail))/$total identical"
[ "$ladder_fail" -eq 0 ] ||
  echo "$ladder_fail of the quality-ladder points above differ as well"
[ "$((fail + ladder_fail))" -eq 0 ] || exit 1
