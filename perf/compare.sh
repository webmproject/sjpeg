#!/bin/bash
# Timing comparison between two bench binaries, over the configurations that
# exercise different parts of the encoder.
#
#   usage: compare.sh <bench-A> <bench-B> [image]
#
# Reports the best-of-N wall clock of each, and the change. Each row is run
# A,B,A,B... so that a thermal drift shows up as noise in both columns rather
# than as a fake win for whichever ran second.
#
# NOTE: use bash, not zsh (see verify.sh).

set -u
A=${1:?usage: compare.sh <bench-A> <bench-B> [image]}
B=${2:?usage: compare.sh <bench-A> <bench-B> [image]}

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
IMG=${3:-$(find_testdata)/source3.jpg} || exit 1

# label                     method quality tile iters passes yuv
CONFIGS=(
  "m0 q75 420                    0      75    1    30      1   1"
  "m1 q75 420                    1      75    1    30      1   1"
  "m4 q75 420                    4      75    1    30      1   1"
  "m4 q75 AUTO (default)         4      75    1    30      1  -1"
  "m4 q95 420                    4      95    1    30      1   1"
  "m4 q75 6 passes               4      75    1    10      6   1"
  "m7 q75 420 (trellis)          7      75    1    10      1   1"
  "m4 q75 420, 3x3 tiled         4      75    3     6      1   1"
)

ms() {   # binary + args -> best-of-N in ms
  "$@" 2>/dev/null | sed -n 's/.*best *\([0-9.]*\) ms.*/\1/p'
}

printf "%-24s %9s %9s %8s\n" "configuration" "A" "B" "change"
for cfg in "${CONFIGS[@]}"; do
  read -r -a f <<< "$cfg"
  n=${#f[@]}
  args=("${f[@]:n-6}")
  label="${f[*]:0:n-6}"
  a=$(ms "$A" "$IMG" "${args[@]}")
  b=$(ms "$B" "$IMG" "${args[@]}")
  [ -n "$a" ] && [ -n "$b" ] || { echo "$label: FAILED"; continue; }
  printf "%-24s %8.1f %8.1f %7.1f%%\n" "$label" "$a" "$b" \
         "$(echo "$a $b" | awk '{print ($2 - $1) / $1 * 100}')"
done
