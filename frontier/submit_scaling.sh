#!/bin/bash
# submit_scaling.sh — fan the scaling study out across node counts, REPS each.
# Run from a Frontier LOGIN node after builds pass on 2 nodes.
#
#   ./submit_scaling.sh              # default ladder 1..1024, 3 reps
#   ./submit_scaling.sh "1 2 4 8"    # custom ladder
#   REPS=5 ./submit_scaling.sh
#
# Each rep is a fresh allocation -> captures dragonfly placement variance
# (report median-of-medians across reps). Small counts go to the debug queue
# to conserve the CSC678 allocation (ends 2026-07-31).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

# 9984 node-hrs available; ladder below costs well under that even at 3 reps.
LADDER=${1:-"1 2 4 8 16 32 64 128 256 512 1024"}
REPS=${REPS:-3}

for N in $LADDER; do
  QOS=""; [ "$N" -le 8 ] && QOS="-q debug"      # [verify OLCF debug limits]
  for r in $(seq 1 "$REPS"); do
    echo "submitting N=$N rep=$r $QOS"
    sbatch -N "$N" $QOS "$HERE/run_allreduce.sbatch"
  done
done
echo "submitted. track with: squeue --me"
