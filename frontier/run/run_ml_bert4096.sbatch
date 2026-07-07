#!/bin/bash
#SBATCH -A csc678
#SBATCH -J rccl-ml-bert
#SBATCH -t 00:10:00
#SBATCH --network=single_node_vni
#SBATCH -o %x-%j.out
#
# One-shot: the single missing exact-size ML point — BERT-Large fp16 gradient
# (680 MB) on MPICH-RCCL at 4096 nodes. C-only (Cray faults >4 MiB at this scale,
# CPU baseline can't finish), single size => one RCCL comm-init + ~1 min of work.
# Walltime is pure comm-init headroom; a hang dies at 10 min.

set -o pipefail
HERE=${FRONTIER_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}
[ -f "$HERE/env.sh" ] || { echo "FATAL: env.sh not found in '$HERE'" >&2; exit 1; }
source "$HERE/env.sh"
: "${WORK:?FATAL: env.sh did not set WORK}"
BIND="$HERE/bind_frontier.sh"

NODES=${SLURM_JOB_NUM_NODES:?}
PPN=8; NRANKS=$((NODES * PPN))
SZ=${SZ:-680000000}   # BERT gradient bytes; override via --export=ALL,SZ=1360000000 for fp32
ITERS=30; WARMUP=10   # match run_ml_sync.sbatch

RESULTS=$WORK/results_ml/N${NODES}_job${SLURM_JOB_ID}
mkdir -p "$RESULTS"
{ echo "date=$(date -u +%FT%TZ)"; echo "nodes=$NODES ranks=$NRANKS ppn=$PPN";
  echo "single-size C-only: bert_large_fp16 $SZ B"; module list; } > "$RESULTS/env.txt" 2>&1

load_mine
module load rccl-net-plugin
MPIEXEC="mpiexec -bootstrap slurm -n $NRANKS -ppn $PPN -bind-to numa"
OSU_MINE_BIN=$OSU_MINE/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce

MODEL=bert_large_fp16; [ "$SZ" = 1360000000 ] && MODEL=bert_large_fp32
echo "### model=$MODEL bytes=$SZ N=$NODES" | tee -a "$RESULTS/C_mpich_rccl.txt"
MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=ccl MPIR_CVAR_ALLREDUCE_CCL=rccl MPIR_CVAR_DEVICE_COLLECTIVES=none \
  $MPIEXEC "$BIND" "$OSU_MINE_BIN" -m ${SZ}:${SZ} -i "$ITERS" -x "$WARMUP" -d rocm -f 2>&1 \
  | tee -a "$RESULTS/C_mpich_rccl.txt"

echo "[ml-bert] done -> $RESULTS"
