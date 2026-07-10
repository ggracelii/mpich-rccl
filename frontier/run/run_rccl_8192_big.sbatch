#!/bin/bash
#SBATCH -A csc678
#SBATCH -J rccl-8192big
#SBATCH -t 00:15:00
#SBATCH --network=single_node_vni
#SBATCH -o %x-%j.out
#
# 8192-node (65,536 GCD) RCCL flagship — LARGE-MESSAGE regime only.
# Correction to the earlier "comm-init wall": job 4948914 proved RCCL DOES init
# at 8192 (~3 min) and produces data; it only timed out because the full 4 B->4 GiB
# sweep at 50 iters is too long (tiny sizes are sync-bound, ~50 s each across 65,536
# ranks). This job skips the slow small sizes and captures 1 MiB -> 4 GiB with light
# iters — the bandwidth regime where RCCL's advantage is the whole story.

set -o pipefail
HERE=${FRONTIER_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}
[ -f "$HERE/env.sh" ] || { echo "FATAL: env.sh not found in '$HERE'" >&2; exit 1; }
source "$HERE/env.sh"
: "${WORK:?FATAL: env.sh did not set WORK}"
BIND="$HERE/bind_frontier.sh"

NODES=${SLURM_JOB_NUM_NODES:?}
PPN=8; NRANKS=$((NODES * PPN))
ITERS=10; WARMUP=3            # light: init is the fixed cost; keep the sweep short

RESULTS=$WORK/results/N${NODES}_job${SLURM_JOB_ID}
mkdir -p "$RESULTS"
{ echo "date=$(date -u +%FT%TZ)"; echo "nodes=$NODES ranks=$NRANKS ppn=$PPN";
  echo "large-only 1 MiB->4 GiB, i=$ITERS x=$WARMUP (single launch)"; module list; } \
  > "$RESULTS/env.txt" 2>&1

load_mine
module load rccl-net-plugin
MPIEXEC="mpiexec -bootstrap slurm -n $NRANKS -ppn $PPN -bind-to numa"
OSU_MINE_BIN=$OSU_MINE/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce

echo "=== C_mpich_rccl 1MiB-4GiB : N=$NODES ranks=$NRANKS ===" | tee "$RESULTS/C_mpich_rccl.txt"
MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=ccl MPIR_CVAR_ALLREDUCE_CCL=rccl MPIR_CVAR_DEVICE_COLLECTIVES=none \
  $MPIEXEC "$BIND" "$OSU_MINE_BIN" -m 1048576:4294967296 -i "$ITERS" -x "$WARMUP" -d rocm -f 2>&1 \
  | tee -a "$RESULTS/C_mpich_rccl.txt"

echo "[8192big] done -> $RESULTS"
