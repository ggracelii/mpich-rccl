#!/bin/bash
#SBATCH -A csc678
#SBATCH -J rccl-diag
#SBATCH -t 00:05:00
#SBATCH --network=single_node_vni
#SBATCH -o %x-%j.out
#
# Diagnose RCCL comm-init behavior at scale (the timeout wall). One small RCCL
# allreduce (just to trigger ncclCommInitRank) with full NCCL_DEBUG init tracing.
# Reveals: which NET plugin loads, which interface bootstrap uses, per-phase
# timing, and — if it hangs — the exact stall point. Also dumps the env the
# rccl-net-plugin module set (NCCL_*/FI_CXI_*), so we can see if SOCKET_IFNAME
# is unset/suboptimal. Not a benchmark; this is a probe.

set -o pipefail
HERE=${FRONTIER_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}
[ -f "$HERE/env.sh" ] || { echo "FATAL: env.sh not found in '$HERE'" >&2; exit 1; }
source "$HERE/env.sh"
: "${WORK:?FATAL: env.sh did not set WORK}"
BIND="$HERE/bind_frontier.sh"

NODES=${SLURM_JOB_NUM_NODES:?}
PPN=8; NRANKS=$((NODES * PPN))

RESULTS=$WORK/results_rccldiag/N${NODES}_job${SLURM_JOB_ID}
mkdir -p "$RESULTS"

load_mine
module load rccl-net-plugin

# 1. what did the plugin/module set? (the bootstrap-interface question)
{ echo "=== NCCL/FI/RCCL env after load_mine + rccl-net-plugin ==="
  env | grep -E "^(NCCL|FI_|RCCL|MPICH_OFI|HSN)" | sort; } > "$RESULTS/env_dump.txt" 2>&1

MPIEXEC="mpiexec -bootstrap slurm -n $NRANKS -ppn $PPN -bind-to numa"
OSU=$OSU_MINE/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce

# 2. one small RCCL allreduce with init tracing. Per-rank debug files (avoid interleave).
#    INIT+BOOTSTRAP+NET+ENV subsys = init path only (keeps volume sane at 512).
echo "[diag] launching traced RCCL init at N=$NODES ($NRANKS ranks) ..."
MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=ccl MPIR_CVAR_ALLREDUCE_CCL=rccl MPIR_CVAR_DEVICE_COLLECTIVES=none \
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,NET,ENV \
NCCL_DEBUG_FILE="$RESULTS/nccl_rank%g.log" \
  $MPIEXEC "$BIND" "$OSU" -m 1024:1024 -i 5 -x 2 -d rocm -f 2>&1 \
  | tee "$RESULTS/run.txt" || true

echo "[diag] done -> $RESULTS"
echo "--- rank 0 init trace (first 40 lines) ---"
head -40 "$RESULTS"/nccl_rank0.log 2>/dev/null || echo "(no rank0 log — init may not have reached NET)"
echo "--- env the plugin set ---"; cat "$RESULTS/env_dump.txt"
