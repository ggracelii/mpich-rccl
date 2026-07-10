#!/bin/bash
#SBATCH -A csc678
#SBATCH -J rccl-ext2g4g
#SBATCH -t 00:08:00
#SBATCH --network=single_node_vni
#SBATCH -o %x-%j.out
#
# Trimmed 2-4 GiB RCCL one-shot for giant scale: ONLY the two extension sizes
# (2 GiB, 4 GiB), C-only, single launch => one RCCL comm-init + ~1 min of work.
# The full sweep (run_rccl_8192.sbatch) can't reach 4 GiB in the walltime at 4096
# because the small sizes + 50 iters eat the budget first; this skips straight to
# the sizes we still need reps of. Writes a normal results/N*_job* dir so the
# notebook loader merges the rows as extra reps.

set -o pipefail
HERE=${FRONTIER_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}
[ -f "$HERE/env.sh" ] || { echo "FATAL: env.sh not found in '$HERE'" >&2; exit 1; }
source "$HERE/env.sh"
: "${WORK:?FATAL: env.sh did not set WORK}"
BIND="$HERE/bind_frontier.sh"

NODES=${SLURM_JOB_NUM_NODES:?}
PPN=8; NRANKS=$((NODES * PPN))
ITERS=50; WARMUP=10           # match the main sweep

RESULTS=$WORK/results/N${NODES}_job${SLURM_JOB_ID}
mkdir -p "$RESULTS"
{ echo "date=$(date -u +%FT%TZ)"; echo "nodes=$NODES ranks=$NRANKS ppn=$PPN";
  echo "trimmed C-only: 2 GiB + 4 GiB (single launch)"; module list; } > "$RESULTS/env.txt" 2>&1

load_mine
module load rccl-net-plugin
MPIEXEC="mpiexec -bootstrap slurm -n $NRANKS -ppn $PPN -bind-to numa"
OSU_MINE_BIN=$OSU_MINE/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce

echo "=== C_mpich_rccl 2-4 GiB : N=$NODES ranks=$NRANKS ===" | tee "$RESULTS/C_mpich_rccl.txt"
MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=ccl MPIR_CVAR_ALLREDUCE_CCL=rccl MPIR_CVAR_DEVICE_COLLECTIVES=none \
  $MPIEXEC "$BIND" "$OSU_MINE_BIN" -m 2147483648:4294967296 -i "$ITERS" -x "$WARMUP" -d rocm -f 2>&1 \
  | tee -a "$RESULTS/C_mpich_rccl.txt"

echo "[ext2g4g] done -> $RESULTS"
