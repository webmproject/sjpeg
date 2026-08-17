#!/bin/bash
# One-shot A/B of the whole optimization series, on whatever machine you run it.
#
#   usage: ab.sh [work-dir]        (default: a fresh temporary directory)
#
#   AB_PER_TOGGLE=1     add a leave-one-out column for each optimization
#   COMPARE_REPEAT=3    best-of-3 for every timing, for a machine with a wide
#                       noise floor -- the control row below says whether this
#                       machine is one
#   JOBS=2              fewer parallel compiles (board short on memory)
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
# default one -- with the last leave-one-out's optimization silently disabled.
# Leave the tree with no objects at all rather than with misleading ones.
cleanup() {
  find src examples perf tests -name '*.o' -delete 2>/dev/null
  rm -f src/libsjpeg.a examples/libutils.a perf/bench
}
trap cleanup EXIT

# Every optimization in the series, each switched off by -DSJPEG_DISABLE_<name>.
# The list is spelled out so a reader can see it, and checked against the source
# right below so it cannot go stale: a toggle added to the encoder without a
# line here would never be measured, and nothing would say so.
TOGGLES="FAST_BITWRITER FAST_BITCOUNTER CHUNKED_COMMIT FUSED_STATS"

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

# Does disabling $1 change anything on this machine? A toggle that gates code
# behind an architecture or SIMD test compiles to nothing on a machine without
# that path, and its leave-one-out then builds the same program twice. Worth
# knowing before reading the column as a result: the zig-zag permutation, since
# withdrawn, once read -3.3%..+10.3% across the eight rows on x86, all of it
# from a binary that was byte-for-byte the one it was compared against.
#
# Only a source that mentions the macro can change. If a header mentions it that
# is every source, but then the first one already differs and the loop stops.
toggle_has_effect() {   # toggle-name -> 0 if the macro changes the source here
  local f files
  if grep -qE "SJPEG_(DISABLE|USE)_$1" src/*.h; then
    files=$(echo src/*.cc)
  else
    files=$(grep -lE "SJPEG_(DISABLE|USE)_$1" src/*.cc)
  fi
  for f in $files; do
    ${CXX:-g++} -Isrc -E "$f" > "$W/probe_a.i" 2>/dev/null || return 0
    ${CXX:-g++} -Isrc -E "-DSJPEG_DISABLE_$1" "$f" > "$W/probe_b.i" 2>/dev/null || return 0
    cmp -s "$W/probe_a.i" "$W/probe_b.i" || return 0
  done
  return 1
}

INACTIVE=""
STATE=""
for t in $TOGGLES; do
  if toggle_has_effect "$t"; then
    STATE="$STATE $t"
  else
    STATE="$STATE $t(inactive)"
    INACTIVE="$INACTIVE $t"
  fi
done

echo "=============================================================="
echo " host    : $(uname -sm)"
echo " compiler: $(${CXX:-g++} --version 2>/dev/null | head -1)"
echo " commit  : $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo " work dir: $W"
# What this build actually compiles in. Worth printing rather than assuming: a
# 32-bit userland on a 64-bit ARM board defines neither SJPEG_AARCH64 nor,
# without -mfpu=neon, SJPEG_USE_NEON, so the vector kernels quietly disappear
# and the numbers below come out smaller for a reason that has nothing to do
# with the code.
echo ' platform:' $(echo '#include "sjpegi.h"' |
                    ${CXX:-g++} -Isrc -dM -E -x c++ - 2>/dev/null |
                    awk '/^#define SJPEG_(USE|HAVE|AARCH64)/ { print $2 }' |
                    grep -Ev "SJPEG_USE_($(echo $TOGGLES | tr ' ' '|'))" |
                    sort | tr '\n' ' ')
# The optimizations, and whether each is more than a no-op here. This line
# cannot be read off sjpegi.h the way the one above it is: two of the five are
# switched inside a .cc and never define a SJPEG_USE_ macro at all.
echo " toggles :$STATE"
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

  # Read this row first. Both binaries are the same program -- the macro is
  # unused -- so whatever it reports is what the machine contributes, and no
  # number below it means anything unless it is larger. It costs one build.
  ./perf/build.sh "$W/bench_control" "-DSJPEG_AB_NOISE_CONTROL" >/dev/null || exit 1
  echo
  echo "-- control (two builds of the SAME code: this is the noise floor):"
  ./perf/compare.sh "$W/bench_control" "$W/bench_after" | tail -n +2

  for t in $TOGGLES; do
    case " $INACTIVE " in
      *" $t "*) note="   [inactive here: another noise reading, not a result]" ;;
      *)        note="" ;;
    esac
    ./perf/build.sh "$W/bench_no_$t" "-DSJPEG_DISABLE_$t" >/dev/null || exit 1
    echo
    echo "-- without $t:$note"
    ./perf/compare.sh "$W/bench_no_$t" "$W/bench_after" | tail -n +2
  done
fi
