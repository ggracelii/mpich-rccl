#!/bin/bash
#SBATCH -A csc678
#SBATCH -J rccl8192sw
#SBATCH -t 00:20:00
#SBATCH --network=single_node_vni
#SBATCH -o %x-%j.out
#
# RCCL 8192-node (65,536 GCD) power-of-2 sweep, 1 KiB -> 4 GiB.
# Starts at 1 KiB: the 4 B-512 B tiny sizes are already captured (partial job 4948914)
# AND are the slowest (sync-bound, ~50 s each at 65,536 ranks), so skipping them keeps
# this single ascending sweep short enough to finish (~10 min sweep + ~3 min init) — the
# i=50 full-from-8B sweep is what timed out. One init; combined with the partial this
# gives complete 4 B -> 4 GiB coverage.

set -o pipefail
HERE=${FRONTIER_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}
[ -f "$HERE/env.sh" ] || { echo "FATAL: env.sh not found in '$HERE'" >&2; exit 1; }
source "$HERE/env.sh"
: "${WORK:?FATAL: env.sh did not set WORK}"
BIND="$HERE/bind_frontier.sh"

NODES=${SLURM_JOB_NUM_NODES:?}
PPN=8; NRANKS=$((NODES * PPN))
ITERS=10; WARMUP=3

RESULTS=$WORK/results/N${NODES}_job${SLURM_JOB_ID}
mkdir -p "$RESULTS"
{ echo "date=$(date -u +%FT%TZ)"; echo "nodes=$NODES ranks=$NRANKS ppn=$PPN";
  echo "ascending 1KiB-4GiB, i=$ITERS (4B-512B from partial 4948914)"; module list; } > "$RESULTS/env.txt" 2>&1

load_mine
module load rccl-net-plugin
MPIEXEC="mpiexec -bootstrap slurm -n $NRANKS -ppn $PPN -bind-to numa"
OSU=$OSU_MINE/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce
CVARS="MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=ccl MPIR_CVAR_ALLREDUCE_CCL=rccl MPIR_CVAR_DEVICE_COLLECTIVES=none"

echo "=== C_mpich_rccl : N=$NODES ranks=$NRANKS (1KiB-4GiB) ===" | tee "$RESULTS/C_mpich_rccl.txt"
env $CVARS $MPIEXEC "$BIND" "$OSU" -m 1024:4294967296 -i "$ITERS" -x "$WARMUP" -d rocm -f 2>&1 \
  | tee -a "$RESULTS/C_mpich_rccl.txt" || true

echo "[8192sw] done -> $RESULTS"
