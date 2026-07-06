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
  # Order is C -> D -> B, so C (RCCL) and D (Cray) are captured before the slow CPU path.
  # These caps give B room to finish too at <=512; at scale B (680 MB CPU) may still clip -- fine.
  WT="-t 00:05:00"                          # 128-1024
  [ "$N" -le 64 ]   && WT="-t 00:04:00"     # small
  [ "$N" -ge 2048 ] && WT="-t 00:08:00"     # giants: B at 680 MB is very slow (no Cray here anyway)
  for r in $(seq 1 "$REPS"); do
    echo "submit ML N=$N rep=$r/$REPS $WT"
    sbatch $WT -N "$N" "$HERE/run_ml_sync.sbatch"
  done
done
