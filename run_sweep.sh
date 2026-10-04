#!/usr/bin/env bash
# run_sweep.sh - FloatSimd (opLat + issueLat = 7) x thread-count sweep
# Usage:  ./run_sweep.sh                    (defaults below)
#         N=131072 JOBS=4 ./run_sweep.sh    (override any variable)
set -u

GEM5=${GEM5:-$HOME/gem5}                       # your gem5 source tree
GEM5_BIN=${GEM5_BIN:-$GEM5/build/X86/gem5.opt} # or build/ALL/gem5.opt
ISA=${ISA:-x86}                                # x86 | riscv | arm
WORK=${WORK:-$HOME/gem5-tlp}                   # this assignment's folder
BIN=${BIN:-$WORK/daxpy_mt}                     # static daxpy binary
N=${N:-65536}                                  # vector length
JOBS=${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}  # parallel sims
THREADS=${THREADS:-"1 2 4 8"}
OUT=$WORK/results/N$N

run_one() {   # $1=opLat $2=issueLat $3=threads
  local dir=$OUT/op$1_iss$2_t$3
  if "$GEM5_BIN" --outdir="$dir" "$WORK/daxpy_tlp.py" --isa "$ISA" \
       --num-cpus "$3" --op-lat "$1" --issue-lat "$2" \
       --binary "$BIN" --args "$N $3" > "$dir.log" 2>&1; then
    echo "done   op=$1 issue=$2 threads=$3"
  else
    echo "FAILED op=$1 issue=$2 threads=$3  (see $dir.log)"
  fi
}

mkdir -p "$OUT"
echo "gem5: $GEM5_BIN | N=$N | threads: $THREADS | $JOBS parallel jobs"
start=$(date +%s)
for op in 1 2 3 4 5 6; do
  issue=$((7 - op))
  for t in $THREADS; do
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 1; done
    run_one "$op" "$issue" "$t" &
  done
done
wait
echo "Sweep finished in $(( $(date +%s) - start )) s -> $OUT"
