#!/bin/bash
#SBATCH -A csc678
#SBATCH -J cray-blk64
#SBATCH -t 00:15:00
#SBATCH --network=single_node_vni
#SBATCH -o %x-%j.out
#
# TUNED Cray: kernel path ON with MPICH_GPU_ALLREDUCE_BLK_SIZE=64MB (man-page-suggested,
# probe2 verdict: extends the fast path to 4 GiB at 1024 nodes where default faults >4 MiB).
# Full sweep + ML sizes -> results_crayblk64/ (non-default config, kept separate from D).
# Supersedes kernel-off (results_crayknob) as the practical workaround (~25x faster than it).

set -o pipefail
HERE=${FRONTIER_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}
[ -f "$HERE/env.sh" ] || { echo "FATAL: env.sh not found in '$HERE'" >&2; exit 1; }
source "$HERE/env.sh"
: "${WORK:?FATAL: env.sh did not set WORK}"

NODES=${SLURM_JOB_NUM_NODES:?}
PPN=8; NRANKS=$((NODES * PPN))
ITERS=20; WARMUP=5
BLK=67108864
ML=( "ddp_bucket_25MiB:26214400" "resnet50_fp32:102228128"
     "bert_large_fp16:680000000" "bert_large_fp32:1360000000" )

RESULTS=$WORK/results_crayblk64/N${NODES}_job${SLURM_JOB_ID}
mkdir -p "$RESULTS"
{ echo "date=$(date -u +%FT%TZ)"; echo "nodes=$NODES ranks=$NRANKS ppn=$PPN";
  echo "MPICH_GPU_ALLREDUCE_BLK_SIZE=$BLK (kernel path ON); i=$ITERS x=$WARMUP";
  module list; } > "$RESULTS/env.txt" 2>&1

load_cray
OSU_CRAY_BIN=$OSU_CRAY/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce
SRUN="srun -N $NODES -n $NRANKS --ntasks-per-node=$PPN --gpus-per-node=8 --gpu-bind=closest -c7"

echo "=== D_blk64 sweep : N=$NODES ranks=$NRANKS ===" | tee "$RESULTS/D_blk64_sweep.txt"
MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_ALLREDUCE_BLK_SIZE=$BLK \
  $SRUN "$OSU_CRAY_BIN" -m 0:4294967296 -i $ITERS -x $WARMUP -d rocm -f 2>&1 \
  | tee -a "$RESULTS/D_blk64_sweep.txt" || true

for e in "${ML[@]}"; do
  sz=${e##*:}
  echo "### model=${e%%:*} bytes=$sz N=$NODES" | tee -a "$RESULTS/D_blk64_ml.txt"
  MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_ALLREDUCE_BLK_SIZE=$BLK \
    $SRUN "$OSU_CRAY_BIN" -m ${sz}:${sz} -i $ITERS -x $WARMUP -d rocm -f 2>&1 \
    | tee -a "$RESULTS/D_blk64_ml.txt" || true
done

echo "[cray-blk64] done -> $RESULTS"
