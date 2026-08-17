#!/bin/bash
# One-shot A/B of the whole optimization series, on whatever machine you run it.
#
#   usage: ab.sh [work-dir]            (default: a temporary directory)
#          AB_PER_TOGGLE=1 ab.sh       adds a leave-one-out per optimization
#
# Builds the tree twice from the same sources -- once as shipped, once with
# every optimization disabled -- then checks the two agree byte-for-byte and
# reports what the difference is worth here. The disabled build is what the
# encoder was before the series, so it is the honest "before" column without
# needing to check out an older commit.
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

# Every optimization in the series, each switched off by -DSJPEG_DISABLE_<name>.
# The list is spelled out so a reader can see it, and checked against the source
# right below so it cannot go stale: a toggle added to the encoder without a
# line here would never be measured, and nothing would say so.
TOGGLES="FAST_BITWRITER FAST_BITCOUNTER CHUNKED_COMMIT FUSED_STATS ZIGZAG_PERMUTE"

in_source=$(grep -rhoE 'SJPEG_DISABLE_[A-Z_][A-Z_]*' src/ |
            sed 's/^SJPEG_DISABLE_//' | sort -u | tr '\n' ' ')
listed=$(echo $TOGGLES | tr ' ' '\n' | sort | tr '\n' ' ')
if [ "$in_source" != "$listed" ]; then
  echo "ab.sh: the toggle list is stale." >&2
  echo "  src/ switches on: $in_source" >&2
  echo "  ab.sh measures  : $listed" >&2
  exit 1
fi

ALL_OFF=""
for t in $TOGGLES; do ALL_OFF="$ALL_OFF -DSJPEG_DISABLE_$t"; done

echo "=============================================================="
echo " host    : $(uname -sm)"
echo " compiler: $(${CXX:-g++} --version 2>/dev/null | head -1)"
echo " commit  : $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo " work dir: $W"
echo "=============================================================="
echo

echo "## building (two clean builds, this takes a minute)"
./perf/build.sh "$W/bench_before" $ALL_OFF || exit 1
./perf/build.sh "$W/bench_after" || exit 1
echo

echo "## correctness: the optimizations must not change one byte"
./perf/verify.sh "$W/bench_before" "$W/bench_after" || exit 1
# Not piped into tail. '|| exit 1' after a pipe tests the exit status of tail,
# which is 0 whatever the sweep found, so the gate that everything here rests on
# would never fire -- and tail would throw away the DIFFER lines on the way past.
if ! ./perf/sweep.sh "$W/bench_before" "$W/bench_after" > "$W/sweep.log"; then
  echo
  echo "BIT-EXACTNESS FAILED. The optimizations changed the output:"
  grep -E 'DIFFER|FAILED' "$W/sweep.log" | head -20
  echo "  (full log: $W/sweep.log)"
  exit 1
fi
tail -8 "$W/sweep.log"
echo

echo "## speed: before (all disabled) vs after (as shipped)"
./perf/compare.sh "$W/bench_before" "$W/bench_after" || exit 1

if [ -n "${AB_PER_TOGGLE:-}" ]; then
  echo
  echo "## leave-one-out: what each optimization is worth on its own"
  echo "   (each column is the full build minus that one change. They do not"
  echo "    add up to the total above, and are not meant to: two optimizations"
  echo "    on the same code path each cover for the other's absence.)"
  for t in $TOGGLES; do
    ./perf/build.sh "$W/bench_no_$t" "-DSJPEG_DISABLE_$t" >/dev/null || exit 1
    echo
    echo "-- without $t:"
    ./perf/compare.sh "$W/bench_no_$t" "$W/bench_after" | tail -n +2
  done
fi
