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
export FRONTIER_HOME=$(cd "$HERE/.." && pwd)   # dir holding env.sh/bind_frontier.sh; propagated to the job so submit CWD never matters

# Staged for safety (3 reps everywhere; override with REPS=n). Node-hr estimates:
#   default 1-64 ................................. ~95
#   ./submit_scaling.sh "128 256 512 1024" ....... ~1.7k
#   ./submit_scaling.sh "2048" ................... ~2.0k
#   ./submit_scaling.sh "4096" ................... ~4.0k   (REPS=1 -> ~1.4k)
LADDER=${1:-"1 2 4 8 16 32 64"}

reps_for() {                      # 3 reps at every scale: placement/network variance is
  [ -n "$REPS" ] && { echo "$REPS"; return; }   # present at ALL multi-node scales (not less at 4096).
  echo 3                          # REPS=n overrides (e.g. REPS=1 at 4096 purely to save budget)
}

for N in $LADDER; do
  R=$(reps_for "$N")
  WT=""; [ "$N" -ge 2048 ] && WT="-t 00:20:00"   # giants: launch + RCCL comm-init at 2048/4096 nodes adds a few min over the ~2-min run; 20m = ~2-3x margin, far better backfill than the old 60m
  for r in $(seq 1 "$R"); do
    echo "submitting N=$N rep=$r/$R $WT"
    sbatch $WT -N "$N" "$HERE/run_allreduce.sbatch"
    sleep 1
  done
done
echo "submitted. track with: squeue --me"
