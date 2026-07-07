#!/bin/bash
#SBATCH -A csc678
#SBATCH -J smallalg
#SBATCH -t 00:08:00
#SBATCH --network=single_node_vni
#SBATCH -o %x-%j.out
#
# Can ANY pure-MPIR allreduce algorithm beat RCCL at small device messages on
# Frontier (ch4:ofi)? On JLSE (ch4:ucx) recursive_doubling was fast; here the
# default pt2pt stages GPU->host per round. Force each candidate, w/ and w/o
# GPU-direct RDMA (FI_HMEM). Yardstick = forced RCCL. Small sizes only (8B..64KiB).
# WIN = any MPIR row dips below the RCCL row -> reopens pure-MPIR per-comm tuning.

set -o pipefail
HERE=${FRONTIER_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}
[ -f "$HERE/env.sh" ] || { echo "FATAL: env.sh not found in '$HERE'" >&2; exit 1; }
source "$HERE/env.sh"
: "${WORK:?FATAL: env.sh did not set WORK}"
BIND="$HERE/bind_frontier.sh"

NODES=${SLURM_JOB_NUM_NODES:?}
PPN=8; NRANKS=$((NODES * PPN))
ARGS="-m 8:65536 -i 100 -x 20 -d rocm -f"      # small regime only

RESULTS=$WORK/results_smallalg/N${NODES}_job${SLURM_JOB_ID}
mkdir -p "$RESULTS"
{ echo "date=$(date -u +%FT%TZ)"; echo "nodes=$NODES ranks=$NRANKS"; module list; } \
  > "$RESULTS/env.txt" 2>&1

load_mine
module load rccl-net-plugin
MPIEXEC="mpiexec -bootstrap slurm -n $NRANKS -ppn $PPN -bind-to numa"
OSU=$OSU_MINE/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce

# run <label> [extra env...]  — DEVICE_COLLECTIVES=none so the MPIR tree/algorithm is used directly
run() {
  local label=$1; shift
  echo "=== $label ===" | tee "$RESULTS/${label}.txt"
  env MPIR_CVAR_DEVICE_COLLECTIVES=none "$@" \
    $MPIEXEC "$BIND" "$OSU" $ARGS 2>&1 | tee -a "$RESULTS/${label}.txt" || true
}

# forced MPIR algorithms, default transport (per-round staging)
for alg in recursive_doubling smp recexch reduce_scatter_allgather tree; do
  run "mpir_${alg}"        MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=$alg
done
# same, with GPU-direct RDMA on (the JLSE-equivalent no-stage pt2pt path)
for alg in recursive_doubling smp recexch; do
  run "mpir_${alg}_hmem"   MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=$alg \
                           MPIR_CVAR_CH4_OFI_ENABLE_HMEM=1 MPIR_CVAR_CH4_OFI_ENABLE_MR_HMEM=1
done
# yardstick: forced RCCL
run "rccl"                 MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=ccl MPIR_CVAR_ALLREDUCE_CCL=rccl

echo "[smallalg] done -> $RESULTS"
