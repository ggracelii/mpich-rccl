#!/bin/bash
#SBATCH -A csc678
#SBATCH -J hybrid-test
#SBATCH -t 00:08:00
#SBATCH --network=single_node_vni
#SBATCH -o %x-%j.out
#
# Per-comm-size hybrid validation: ONE ch4 + ONE mpir JSON with comm_size branches
# encode the measured per-scale thresholds (8K @<=32 ranks / 16K @<=8192 / 64K else).
# Single global SWAP=64K works because beta (>threshold path) never host-swaps (disabled
# in source), so no host buffer can reach the ccl leaf regardless of SWAP.
# Runs HYBRID (auto, two-level tree) + RCCL reference over sizes spanning all thresholds.
# PASS = no crash + staged wins below this comm's threshold, RCCL above.

set -o pipefail
HERE=${FRONTIER_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}
[ -f "$HERE/env.sh" ] || { echo "FATAL: env.sh not found in '$HERE'" >&2; exit 1; }
source "$HERE/env.sh"
: "${WORK:?FATAL: env.sh did not set WORK}"
BIND="$HERE/bind_frontier.sh"
JCH4="$HERE/tuning/allreduce_ch4_percomm.json"
JMPIR="$HERE/tuning/allreduce_mpir_percomm.json"
[ -f "$JCH4" ] && [ -f "$JMPIR" ] || { echo "FATAL: per-comm JSONs missing" >&2; exit 1; }

NODES=${SLURM_JOB_NUM_NODES:?}
PPN=8; NRANKS=$((NODES * PPN))
ARGS="-m 1024:262144 -i 30 -x 10 -d rocm -f"   # 1 KiB..256 KiB spans the 8/16/64 KiB thresholds

RESULTS=$WORK/results_hybrid_test/N${NODES}_job${SLURM_JOB_ID}
mkdir -p "$RESULTS"
{ echo "date=$(date -u +%FT%TZ)"; echo "nodes=$NODES ranks=$NRANKS SWAP=65536"; module list; } \
  > "$RESULTS/env.txt" 2>&1
cp "$JCH4" "$JMPIR" "$RESULTS/"

load_mine
module load rccl-net-plugin
MPIEXEC="mpiexec -bootstrap slurm -n $NRANKS -ppn $PPN -bind-to numa"
OSU_MINE_BIN=$OSU_MINE/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce

echo "=== HYBRID (per-comm tree, SWAP=64K) ===" | tee "$RESULTS/HYBRID.txt"
MPIR_CVAR_DEVICE_COLLECTIVES=all \
MPIR_CVAR_CH4_GPU_COLL_SWAP_BUFFER_SZ=65536 \
MPIR_CVAR_CH4_COLL_SELECTION_TUNING_JSON_FILE="$JCH4" \
MPIR_CVAR_CH4_COLL_SELECTION_TUNING_JSON_FILE_GPU="$JCH4" \
MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE="$JMPIR" \
  $MPIEXEC "$BIND" "$OSU_MINE_BIN" $ARGS 2>&1 | tee -a "$RESULTS/HYBRID.txt" || true

echo "=== RCCL reference ===" | tee "$RESULTS/RCCL.txt"
MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=ccl MPIR_CVAR_ALLREDUCE_CCL=rccl MPIR_CVAR_DEVICE_COLLECTIVES=none \
  $MPIEXEC "$BIND" "$OSU_MINE_BIN" $ARGS 2>&1 | tee -a "$RESULTS/RCCL.txt" || true

echo "[hybrid-test] done -> $RESULTS"
