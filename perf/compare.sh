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
# Best-of-N inside one invocation does not protect against drift *between*
# invocations, which on some machines is the larger effect: a Haswell has been
# measured swinging 23% across six timings of the same binary, on the same row
# where its longest configuration held to 2%. COMPARE_REPEAT=3 runs the whole
# A,B pair that many times and keeps the best of each, which does attack that.
# Worth turning up whenever ab.sh's control row comes out wide.
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

REPEAT=${COMPARE_REPEAT:-1}

ms() {   # binary + args -> best-of-N in ms
  "$@" 2>/dev/null | sed -n 's/.*best *\([0-9.]*\) ms.*/\1/p'
}

best() {   # best of two, either of which may be empty
  if [ -z "$1" ]; then echo "$2"
  elif [ -z "$2" ]; then echo "$1"
  else echo "$1 $2" | awk '{print ($1 < $2) ? $1 : $2}'
  fi
}

printf "%-24s %9s %9s %8s\n" "configuration" "A" "B" "change"
for cfg in "${CONFIGS[@]}"; do
  read -r -a f <<< "$cfg"
  n=${#f[@]}
  args=("${f[@]:n-6}")
  label="${f[*]:0:n-6}"
  a=""; b=""; r=0
  while [ "$r" -lt "$REPEAT" ]; do
    a=$(best "$a" "$(ms "$A" "$IMG" "${args[@]}")")
    b=$(best "$b" "$(ms "$B" "$IMG" "${args[@]}")")
    r=$((r + 1))
  done
  [ -n "$a" ] && [ -n "$b" ] || { echo "$label: FAILED"; continue; }
  printf "%-24s %8.1f %8.1f %7.1f%%\n" "$label" "$a" "$b" \
         "$(echo "$a $b" | awk '{print ($2 - $1) / $1 * 100}')"
done
