#!/bin/bash
# Where the encoder's time goes, and what the vector layer is worth -- both
# without a profiler, since perf is routinely unavailable on the machines worth
# measuring (perf_event_paranoid, containers, no root).
#
#   usage: breakdown.sh <bench> [image]
#          BD_ITERS=30 breakdown.sh ...      # fewer/more repetitions
#
# Everything here is a difference between two runs of the *same* binary, so it
# needs no second build and says nothing about any optimization. It describes
# the encoder as it currently stands, which is what decides where to work next.
#
# The decomposition rides the method ladder, which switches features on one at
# a time -- m0 is plain, m1 adds Huffman optimization, m3 adds adaptive
# quantization, m4 has both, m7 adds the trellis -- so a difference between two
# rows is the cost of one stage. The AUTO/forced pair isolates SjpegRiskiness,
# which runs over the whole source, before any encoding, to produce one enum.
#
# NOTE: use bash, not zsh (see verify.sh).

set -u
BIN=${1:?usage: breakdown.sh <bench> [image]}

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
IMG=${2:-$D/source3.jpg}

ITERS=${BD_ITERS:-30}
# The trellis is far slower than everything else; do not run it as many times.
ITERS7=$(( ITERS / 4 )); [ "$ITERS7" -lt 3 ] && ITERS7=3

enc() {   # method quality passes yuv [iters] -> best ms
  "$BIN" "$IMG" "$1" "$2" 1 "${5:-$ITERS}" "$3" "$4" 2>/dev/null |
    sed -n 's/.*best *\([0-9.]*\) ms.*/\1/p'
}

sub() { echo "$1 $2" | awk '{printf "%.2f", $1 - $2}'; }
pct() { echo "$1 $2" | awk '{printf "%.1f%%", ($2 > 0) ? $1 / $2 * 100 : 0}'; }

echo "image: $(basename "$IMG"), best of $ITERS ($ITERS7 for the trellis)"
echo

##############################################################################
echo "== 1. where the time goes, q75 =="
echo "   (each stage is the difference between two runs of this same binary)"
echo

m0=$(enc 0 75 1 1)
m1=$(enc 1 75 1 1)
m3=$(enc 3 75 1 1)
m4=$(enc 4 75 1 1)
m7=$(enc 7 75 1 1 "$ITERS7")
auto=$(enc 4 75 1 -1)
# The bench times SjpegRiskiness on its own before it starts encoding, which is
# a direct reading rather than a difference. Worth printing both: they measure
# the same thing two ways and should agree.
risk_alone=$("$BIN" "$IMG" 4 75 1 1 1 -1 2>/dev/null |
             sed -n 's/.*SjpegRiskiness alone: *\([0-9.]*\) ms.*/\1/p')

printf "%-42s %8s %9s\n" "stage" "ms" "of default"
printf "%-42s %8s %9s\n" "  base encode (m0, YUV forced)" "$m0" "$(pct "$m0" "$auto")"
printf "%-42s %8s %9s\n" "+ Huffman optimization (m1 - m0)" \
       "$(sub "$m1" "$m0")" "$(pct "$(sub "$m1" "$m0")" "$auto")"
printf "%-42s %8s %9s\n" "+ adaptive quantization (m3 - m0)" \
       "$(sub "$m3" "$m0")" "$(pct "$(sub "$m3" "$m0")" "$auto")"
printf "%-42s %8s %9s\n" "+ SjpegRiskiness (AUTO - forced)" \
       "$(sub "$auto" "$m4")" "$(pct "$(sub "$auto" "$m4")" "$auto")"
printf "%-42s %8s %9s\n" "  ...measured directly, for comparison" "$risk_alone" \
       "$(pct "$risk_alone" "$auto")"
echo "  ----"
printf "%-42s %8s %9s\n" "  default encode (m4, AUTO)" "$auto" "100.0%"
printf "%-42s %8s %9s\n" "+ trellis (m7 - m4), method 7 only" \
       "$(sub "$m7" "$m4")" "$(pct "$(sub "$m7" "$m4")" "$auto")"
echo
# Derived, not asserted: this script's whole claim is to describe the machine it
# is running on, so it must not narrate another one's numbers over the top.
echo "  SjpegRiskiness and its colour conversion cost" \
     "$(pct "$(sub "$auto" "$m4")" "$auto") of the default encode here,"
echo "  and all they decide is one enum."
echo

##############################################################################
echo "== 2. what the vector layer is worth (BENCH_SLOW_C=1 disables all SIMD) =="
echo

printf "%-28s %9s %9s %9s\n" "configuration" "C only" "SIMD" "speedup"
ratios=""
for cfg in "m0 q75 420|0 75 1 1|$ITERS" \
           "m1 q75 420|1 75 1 1|$ITERS" \
           "m4 q75 420|4 75 1 1|$ITERS" \
           "m4 q75 AUTO|4 75 1 -1|$ITERS" \
           "m4 q95 420|4 95 1 1|$ITERS" \
           "m7 q75 420|7 75 1 1|$ITERS7"; do
  label=${cfg%%|*}; rest=${cfg#*|}; args=${rest%%|*}; it=${rest##*|}
  # shellcheck disable=SC2086
  s=$(enc $args "$it")
  # shellcheck disable=SC2086
  c=$(BENCH_SLOW_C=1 enc $args "$it")
  [ -n "$s" ] && [ -n "$c" ] || { echo "$label: FAILED"; continue; }
  r=$(echo "$c $s" | awk '{printf "%.4f", $1 / $2}')
  ratios="$ratios $r"
  printf "%-28s %9s %9s %8.2fx\n" "$label" "$c" "$s" "$r"
done
echo
# Averaged from the rows above rather than quoted from another machine: a core
# with narrower vector units and a slower scalar side gives a different number,
# and this ratio is what decides whether more SIMD work is worth doing here.
# shellcheck disable=SC2086
echo "  Measured at $(echo $ratios |
        awk '{s = 0; for (i = 1; i <= NF; i++) s += $i;
              printf "%.2f", NF ? s / NF : 0}')x on average over these rows."
echo

##############################################################################
echo "== 3. method 7 across quality =="
echo "   (the trellis scan is O(n^2) in the node count, which grows with q,"
echo "    so anything that prunes it pays more at the top of the range)"
echo

printf "%8s %10s %10s\n" "quality" "m4 ms" "m7 ms"
for q in 40 75 90 95 98 100; do
  a=$(enc 4 "$q" 1 1)
  b=$(enc 7 "$q" 1 1 "$ITERS7")
  [ -n "$a" ] && [ -n "$b" ] || { echo "q=$q FAILED"; continue; }
  printf "%8s %10s %10s\n" "$q" "$a" "$b"
done
echo

##############################################################################
echo "== 4. the multi-pass search =="
echo "   (ComputeSize() re-walks every non-zero coefficient once per pass; the"
echo "    frequency table it already builds could give the same total in 268"
echo "    terms, which is the open item 3.5)"
echo

printf "%8s %10s %14s\n" "passes" "ms" "vs 1 pass"
one=""
for p in 1 2 4 6 8; do
  t=$(enc 4 75 "$p" 1)
  [ -n "$t" ] || { echo "passes=$p FAILED"; continue; }
  # The p=1 row IS the baseline. Timing it a second time made the first ratio
  # print as 0.98x or 1.02x, which reads as a measurement rather than a unit.
  [ -n "$one" ] || one=$t
  printf "%8s %10s %13sx\n" "$p" "$t" \
         "$(echo "$t $one" | awk '{printf "%.2f", $1 / $2}')"
done
