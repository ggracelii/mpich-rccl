#!/bin/bash
#SBATCH -A csc678
#SBATCH -J rccl-8192
#SBATCH -t 00:25:00
#SBATCH --network=single_node_vni
#SBATCH -o %x-%j.out
#
# Fire-and-forget: full RCCL allreduce curve at 8192 nodes (65,536 GCDs), 8 B -> 4 GiB.
# C-ONLY, SINGLE LAUNCH — the prior 8192 failures starved RCCL's communicator init
# (CPU baselines ran first / multiple launches); here one comm-init gets the whole
# walltime, and the measurement itself is only ~3 min. 25-min cap bounds a hang.
# Queue reality: needs ~87% of Frontier + we're deprioritized past allocation —
# it may never schedule; scancel it before 7/31 if still pending.

set -o pipefail
HERE=${FRONTIER_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}
[ -f "$HERE/env.sh" ] || { echo "FATAL: env.sh not found in '$HERE'" >&2; exit 1; }
source "$HERE/env.sh"
: "${WORK:?FATAL: env.sh did not set WORK}"
BIND="$HERE/bind_frontier.sh"

NODES=${SLURM_JOB_NUM_NODES:?}
PPN=8; NRANKS=$((NODES * PPN))

RESULTS=$WORK/results/N${NODES}_job${SLURM_JOB_ID}
mkdir -p "$RESULTS"
{ echo "date=$(date -u +%FT%TZ)"; echo "nodes=$NODES ranks=$NRANKS ppn=$PPN";
  echo "C-only single-launch, 8 B..4 GiB"; module list; } > "$RESULTS/env.txt" 2>&1

load_mine
module load rccl-net-plugin
MPIEXEC="mpiexec -bootstrap slurm -n $NRANKS -ppn $PPN -bind-to numa"
OSU_MINE_BIN=$OSU_MINE/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce

echo "=== C_mpich_rccl : N=$NODES ranks=$NRANKS (single launch, to 4 GiB) ===" \
  | tee "$RESULTS/C_mpich_rccl.txt"
MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=ccl MPIR_CVAR_ALLREDUCE_CCL=rccl MPIR_CVAR_DEVICE_COLLECTIVES=none \
  $MPIEXEC "$BIND" "$OSU_MINE_BIN" -m 0:4294967296 -i 50 -x 10 -d rocm -f 2>&1 \
  | tee -a "$RESULTS/C_mpich_rccl.txt"

echo "[8192] done -> $RESULTS"
