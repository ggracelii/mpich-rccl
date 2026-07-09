#!/bin/bash
#SBATCH -A csc678
#SBATCH -J hybrid-prod
#SBATCH -t 00:06:00
#SBATCH --network=single_node_vni
#SBATCH -o %x-%j.out
#
# Validate the PRODUCTION two-level hybrid: single 16 KiB threshold, SWAP=16 KiB
# (the only crash-free design — SWAP must equal the ch4/mpir split; per-comm-size
# thresholds fault at the alpha->beta handoff, verified job 4953409).
# Staged-CPU allreduce <=16 KiB, RCCL above; full auto (no forced algorithm).
# PASS = no crash + HYBRID tracks staged below 16 KiB and RCCL from 32 KiB up.

set -o pipefail
HERE=${FRONTIER_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}
[ -f "$HERE/env.sh" ] || { echo "FATAL: env.sh not found in '$HERE'" >&2; exit 1; }
source "$HERE/env.sh"
: "${WORK:?FATAL: env.sh did not set WORK}"
BIND="$HERE/bind_frontier.sh"
JCH4="$HERE/tuning/allreduce_ch4_prod.json"
JMPIR="$HERE/tuning/allreduce_mpir_prod.json"
[ -f "$JCH4" ] && [ -f "$JMPIR" ] || { echo "FATAL: production JSONs missing" >&2; exit 1; }

NODES=${SLURM_JOB_NUM_NODES:?}
PPN=8; NRANKS=$((NODES * PPN))
ARGS="-m 1024:262144 -i 30 -x 10 -d rocm -f"   # spans the 16 KiB threshold

RESULTS=$WORK/results_hybrid_prod/N${NODES}_job${SLURM_JOB_ID}
mkdir -p "$RESULTS"
{ echo "date=$(date -u +%FT%TZ)"; echo "nodes=$NODES ranks=$NRANKS thr=SWAP=16384"; module list; } \
  > "$RESULTS/env.txt" 2>&1
cp "$JCH4" "$JMPIR" "$RESULTS/"

load_mine
module load rccl-net-plugin
MPIEXEC="mpiexec -bootstrap slurm -n $NRANKS -ppn $PPN -bind-to numa"
OSU_MINE_BIN=$OSU_MINE/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce

echo "=== HYBRID_prod (16K threshold, SWAP=16K) ===" | tee "$RESULTS/HYBRID.txt"
MPIR_CVAR_DEVICE_COLLECTIVES=all \
MPIR_CVAR_CH4_GPU_COLL_SWAP_BUFFER_SZ=16384 \
MPIR_CVAR_CH4_COLL_SELECTION_TUNING_JSON_FILE="$JCH4" \
MPIR_CVAR_CH4_COLL_SELECTION_TUNING_JSON_FILE_GPU="$JCH4" \
MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE="$JMPIR" \
  $MPIEXEC "$BIND" "$OSU_MINE_BIN" $ARGS 2>&1 | tee -a "$RESULTS/HYBRID.txt" || true

echo "=== RCCL reference ===" | tee "$RESULTS/RCCL.txt"
MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=ccl MPIR_CVAR_ALLREDUCE_CCL=rccl MPIR_CVAR_DEVICE_COLLECTIVES=none \
  $MPIEXEC "$BIND" "$OSU_MINE_BIN" $ARGS 2>&1 | tee -a "$RESULTS/RCCL.txt" || true

echo "[hybrid-prod] done -> $RESULTS"
