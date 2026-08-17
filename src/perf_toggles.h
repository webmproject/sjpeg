// Copyright 2026 Google, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//  Build-time switches for the encoder performance work: the one place where
//  they are declared.
//
// Author: Skal (pascal.massimino@gmail.com)

#ifndef SJPEG_PERF_TOGGLES_H_
#define SJPEG_PERF_TOGGLES_H_

// Each optimization below is compiled in unless the matching SJPEG_DISABLE_* is
// defined on the command line, which builds the implementation that preceded it
// instead. perf/ab.sh uses that to produce an honest "before" column, and to
// leave each one out in turn, without checking out an older commit.
//
// All of them are bit-exact: the same bytes come out either way, which
// perf/verify.sh and perf/sweep.sh check across the corpus. doc/perf-plan.md
// records what each is worth, and on which machine -- they differ by a lot.
//
//   SJPEG_USE_FAST_BITWRITER    64bit lazily-flushed BitWriter
//   SJPEG_USE_FAST_BITCOUNTER   the same, for the size-estimating counter
//   SJPEG_USE_CHUNKED_COMMIT    commit the output in slabs, not once per MCU
//   SJPEG_USE_FUSED_STATS       gather the entropy statistics while storing the
//                               run/levels, rather than re-walking them after
//   SJPEG_USE_TRELLIS_PRUNE     end the trellis' predecessor scan once nothing
//                               left in it can win
//
// Declaring them here, rather than next to each use, is what lets perf/ab.sh
// check its own list against the source: a toggle it does not know about would
// silently stay enabled in the "before" build and never be measured.

#if !defined(SJPEG_DISABLE_FAST_BITWRITER)
#define SJPEG_USE_FAST_BITWRITER
#endif

#if !defined(SJPEG_DISABLE_FAST_BITCOUNTER)
#define SJPEG_USE_FAST_BITCOUNTER
#endif

#if !defined(SJPEG_DISABLE_CHUNKED_COMMIT)
#define SJPEG_USE_CHUNKED_COMMIT
#endif

#if !defined(SJPEG_DISABLE_FUSED_STATS)
#define SJPEG_USE_FUSED_STATS
#endif

#if !defined(SJPEG_DISABLE_TRELLIS_PRUNE)
#define SJPEG_USE_TRELLIS_PRUNE
#endif

#endif    // SJPEG_PERF_TOGGLES_H_
