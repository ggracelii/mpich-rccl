#!/bin/bash
# submit_scaling.sh — fan the scaling study out across node counts, REPS each.
# Run from a Frontier LOGIN node after builds pass on 2 nodes.
#
#   ./submit_scaling.sh              # default ladder 1..1024, 3 reps
#   ./submit_scaling.sh "1 2 4 8"    # custom ladder
#   REPS=5 ./submit_scaling.sh
#
# Each rep is a fresh allocation -> captures dragonfly placement variance
# (report median-of-medians across reps). All jobs go to the normal batch queue:
# the debug QOS caps queued jobs at ~1-2, which a multi-job sweep blows past.
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

# Default sweep is 1->1024 (~1.8k node-hrs, ~18% of budget). Stage the big runs
# separately after reviewing results: ./submit_scaling.sh "2048 4096"
# Reps taper by node count (3 <=1024, 2 at 2048, 1 at 4096) via reps_for below.
LADDER=${1:-"1 2 4 8 16 32 64 128 256 512 1024"}

reps_for() {                      # taper reps by node count
  local n=$1
  if   [ "$n" -le 1024 ]; then echo 3
  elif [ "$n" -le 2048 ]; then echo 2
  else                         echo 1
  fi
}

for N in $LADDER; do
  R=$(reps_for "$N")
  for r in $(seq 1 "$R"); do
    echo "submitting N=$N rep=$r/$R"
    sbatch -N "$N" "$HERE/run_allreduce.sbatch"
    sleep 1
  done
done
echo "submitted. track with: squeue --me"
