#!/bin/bash
# submit_ml.sh — ML gradient-sync proxy (docs/ML_EXPERIMENT.md, angle B), weak-scaled.
# Fixed per-rank gradient size, grow node count = data-parallel weak scaling.
#
#   ./submit_ml.sh                  # default ladder 1 8 64 512 1024, 3 reps
#   ./submit_ml.sh "1 8 64 512"     # custom ladder
#   REPS=2 ./submit_ml.sh
#
# Much of this is also derivable ~free from the main sweep at the nearest power-of-2 sizes
# (32 MiB≈bucket, 128 MiB≈ResNet, 512 MiB/1 GiB≈BERT) — run this only for exact model sizes.
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export FRONTIER_HOME=$(cd "$HERE/.." && pwd)   # so the job finds env.sh regardless of submit CWD

LADDER=${1:-"1 8 64 512 1024 2048 4096"}
REPS=${REPS:-3}

for N in $LADDER; do
  # Tight caps: RCCL (run first) finishes fast even at 4096; the slow CPU baseline may get
  # clipped at scale, which is fine (RCCL is the result). 4096 runs in <4 min.
  WT="-t 00:03:00"                          # 128-1024
  [ "$N" -le 64 ]   && WT="-t 00:02:00"     # small
  [ "$N" -ge 2048 ] && WT="-t 00:04:00"     # 2048 / 4096
  for r in $(seq 1 "$REPS"); do
    echo "submit ML N=$N rep=$r/$REPS $WT"
    sbatch $WT -N "$N" "$HERE/run_ml_sync.sbatch"
  done
done
