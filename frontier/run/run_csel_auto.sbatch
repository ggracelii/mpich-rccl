#!/bin/bash
#SBATCH -A csc678
#SBATCH -J rccl-csel-auto
#SBATCH -t 00:15:00
#SBATCH --network=single_node_vni
#SBATCH -o %x-%j.out
#
# Validate JSON-driven automatic backend selection (csel):
# tuning/allreduce_rccl_auto.json routes allreduce with built-in ops and
# avg_msg_size >= 64 KiB to the CCL/RCCL algorithm, below that to MPICH's
# stock algorithms (64 KiB = measured RCCL-vs-MPICH crossover).
#
# Three runs on device buffers, same sizes (1 KiB..256 MiB):
#   1. AUTO_default : auto selection, stock tree (no JSON)   -> CPU-path everywhere
#   2. AUTO_json    : auto selection + our JSON              -> CPU-path <64 KiB, RCCL >=64 KiB
#   3. C_forced     : forced ccl/rccl reference              -> RCCL everywhere
# Success signature: AUTO_json tracks AUTO_default below 64 KiB and C_forced
# at/above it (large sizes orders of magnitude faster than AUTO_default).

set -o pipefail
HERE=${FRONTIER_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}
[ -f "$HERE/env.sh" ] || { echo "FATAL: env.sh not found in '$HERE'" >&2; exit 1; }
source "$HERE/env.sh"
: "${WORK:?FATAL: env.sh did not set WORK}"
BIND="$HERE/bind_frontier.sh"
JSON="$HERE/tuning/allreduce_rccl_auto.json"
[ -f "$JSON" ] || { echo "FATAL: $JSON missing" >&2; exit 1; }

NODES=${SLURM_JOB_NUM_NODES:?}
PPN=8; NRANKS=$((NODES * PPN))
ARGS="-m 8:268435456 -i 30 -x 10 -d rocm -f"   # 8 B..256 MiB — full range incl. the sub-1 KiB tail (crossover check)

RESULTS=$WORK/results_csel/N${NODES}_job${SLURM_JOB_ID}
mkdir -p "$RESULTS"
{ echo "date=$(date -u +%FT%TZ)"; echo "nodes=$NODES ranks=$NRANKS"; echo "json=$JSON";
  module list; } > "$RESULTS/env.txt" 2>&1
cp "$JSON" "$RESULTS/"    # snapshot the exact tree used

load_mine
MPIEXEC="mpiexec -bootstrap slurm -n $NRANKS -ppn $PPN -bind-to numa"
OSU_MINE_BIN=$OSU_MINE/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce

# 1. stock auto (no JSON, no rccl-net-plugin => untouched baseline networking)
echo "=== AUTO_default (stock tree) ===" | tee "$RESULTS/AUTO_default.txt"
MPIR_CVAR_DEVICE_COLLECTIVES=none \
  $MPIEXEC "$BIND" "$OSU_MINE_BIN" $ARGS 2>&1 | tee -a "$RESULTS/AUTO_default.txt" || true

# 1b. stock auto + GPU-direct RDMA (FI_HMEM): MPICH's NO-STAGING device path (off by
#     default). The fair "MPICH without staging vs RCCL" small-message comparison —
#     may error if the provider/MR support is incomplete in this build; || true captures it.
echo "=== AUTO_hmem (stock tree + FI_HMEM GPU-direct RDMA) ===" | tee "$RESULTS/AUTO_hmem.txt"
MPIR_CVAR_DEVICE_COLLECTIVES=none MPIR_CVAR_CH4_OFI_ENABLE_HMEM=1 MPIR_CVAR_CH4_OFI_ENABLE_MR_HMEM=1 \
  $MPIEXEC "$BIND" "$OSU_MINE_BIN" $ARGS 2>&1 | tee -a "$RESULTS/AUTO_hmem.txt" || true

# RCCL runs need the OFI plugin
module load rccl-net-plugin

# 2. auto + our tuning JSON => csel switches to RCCL at >=64 KiB automatically
echo "=== AUTO_json (RCCL >=64 KiB via csel) ===" | tee "$RESULTS/AUTO_json.txt"
MPIR_CVAR_DEVICE_COLLECTIVES=none \
MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE="$JSON" \
  $MPIEXEC "$BIND" "$OSU_MINE_BIN" $ARGS 2>&1 | tee -a "$RESULTS/AUTO_json.txt" || true

# 3. forced RCCL reference
echo "=== C_forced (rccl everywhere) ===" | tee "$RESULTS/C_forced.txt"
MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=ccl MPIR_CVAR_ALLREDUCE_CCL=rccl MPIR_CVAR_DEVICE_COLLECTIVES=none \
  $MPIEXEC "$BIND" "$OSU_MINE_BIN" $ARGS 2>&1 | tee -a "$RESULTS/C_forced.txt" || true

# 4. TWO-LEVEL HYBRID: CH4 tree = alpha (one-shot host staging) <=32 KiB, beta above;
#    beta delegates down to the MPIR tree = ccl/RCCL for built-in ops.
#    Expected: tracks the fast staged path at small sizes, RCCL at large — best of both.
JSON_CH4="$HERE/tuning/allreduce_ch4_hybrid.json"
if [ -f "$JSON_CH4" ]; then
  echo "=== AUTO_hybrid (ch4: alpha<=32K / beta->MPIR->rccl) ===" | tee "$RESULTS/AUTO_hybrid.txt"
  cp "$JSON_CH4" "$RESULTS/"
  MPIR_CVAR_DEVICE_COLLECTIVES=all \
  MPIR_CVAR_CH4_COLL_SELECTION_TUNING_JSON_FILE="$JSON_CH4" \
  MPIR_CVAR_CH4_COLL_SELECTION_TUNING_JSON_FILE_GPU="$JSON_CH4" \
  MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE="$JSON" \
    $MPIEXEC "$BIND" "$OSU_MINE_BIN" $ARGS 2>&1 | tee -a "$RESULTS/AUTO_hybrid.txt" || true
fi

echo "[csel] done -> $RESULTS"
