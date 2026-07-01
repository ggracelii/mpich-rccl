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

# 9984 node-hrs available; ladder below costs well under that even at 3 reps.
LADDER=${1:-"1 2 4 8 16 32 64 128 256 512 1024"}
REPS=${REPS:-3}

for N in $LADDER; do
  for r in $(seq 1 "$REPS"); do
    echo "submitting N=$N rep=$r"
    sbatch -N "$N" "$HERE/run_allreduce.sbatch"
    sleep 1
  done
done
echo "submitted. track with: squeue --me"
