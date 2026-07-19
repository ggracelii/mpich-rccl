#!/bin/bash
#SBATCH -A csc678
#SBATCH -J rccl8192full
#SBATCH -t 01:00:00
#SBATCH --network=single_node_vni
#SBATCH -o %x-%j.out
#
# RCCL 8192-node (65,536 GCD) FULL sweep — 1 KiB -> 4 GiB at the canonical ladder
# measurement protocol (i=50 x=10, from run_allreduce.sbatch). The 0-512 B tiny end
# is already captured by partial job 4948914, so combined this is a complete 4 B ->
# 4 GiB ladder point. It does NOT fit a short cap: RCCL comm-init (ncclCommInitRank,
# fired lazily on the first collective) at 65,536 ranks is a lottery — usually ~2 min
# (see ML job 4980915: two inits in <5 min) but occasionally catastrophic: job 4973420
# drew a ~30-min init and timed out a 30-min cap with only the 1024 B row. The sweep
# itself is only a few minutes once init lands, so wall time is pure init insurance.
# 60 min absorbs even a worst-case draw. Single launch => one init for the entire ladder.
# tee -a is incremental, so even a walltime clip preserves every completed size.

set -o pipefail
HERE=${FRONTIER_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}
[ -f "$HERE/env.sh" ] || { echo "FATAL: env.sh not found in '$HERE'" >&2; exit 1; }
source "$HERE/env.sh"
: "${WORK:?FATAL: env.sh did not set WORK}"
BIND="$HERE/bind_frontier.sh"

NODES=${SLURM_JOB_NUM_NODES:?}
PPN=8; NRANKS=$((NODES * PPN))
ITERS=50; WARMUP=10          # match the main ladder (run_allreduce.sbatch)

RESULTS=$WORK/results/N${NODES}_job${SLURM_JOB_ID}
mkdir -p "$RESULTS"
{ echo "date=$(date -u +%FT%TZ)"; echo "nodes=$NODES ranks=$NRANKS ppn=$PPN";
  echo "FULL sweep 1KiB-4GiB, i=$ITERS x=$WARMUP (single launch; 4B-512B from partial 4948914)"; module list; } > "$RESULTS/env.txt" 2>&1

load_mine
module load rccl-net-plugin
MPIEXEC="mpiexec -bootstrap slurm -n $NRANKS -ppn $PPN -bind-to numa"
OSU=$OSU_MINE/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce
CVARS="MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=ccl MPIR_CVAR_ALLREDUCE_CCL=rccl MPIR_CVAR_DEVICE_COLLECTIVES=none"

echo "=== C_mpich_rccl : N=$NODES ranks=$NRANKS (1KiB-4GiB, i=$ITERS) ===" | tee "$RESULTS/C_mpich_rccl.txt"
env $CVARS $MPIEXEC "$BIND" "$OSU" -m 1024:4294967296 -i "$ITERS" -x "$WARMUP" -d rocm -f 2>&1 \
  | tee -a "$RESULTS/C_mpich_rccl.txt" || true

echo "[8192full] done -> $RESULTS"
