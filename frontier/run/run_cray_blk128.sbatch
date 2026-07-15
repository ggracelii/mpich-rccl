#!/bin/bash
#SBATCH -A csc678
#SBATCH -J cray-blk128
#SBATCH -t 00:25:00
#SBATCH --network=single_node_vni
#SBATCH -o %x-%j.out
#
# CANONICAL tuned Cray: kernel path ON with MPICH_GPU_ALLREDUCE_BLK_SIZE=128MB.
# 128 MB is the smallest single staging buffer that avoids the GPU-allreduce fault at EVERY
# tested scale, 1->8192 nodes: BLK=64MB rescues only through 4096 and faults >32 MiB at 8192
# (job 4967908), while 128MB completes the full 4 GiB range at 8192 (probe job 4981401). We
# therefore standardize on 128MB across the whole dataset instead of escalating BLK mid-ladder.
# Full sweep + 4 ML sizes -> results_crayblk128/ (config T on all plots). srun => fast init,
# so this is cheap at every node count. Supersedes results_crayblk64/ as the plotted Cray line;
# the BLK=64 ladder + bigger-BLK probe are kept as evidence for the scale-dependent fault ceiling.

set -o pipefail
HERE=${FRONTIER_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}
[ -f "$HERE/env.sh" ] || { echo "FATAL: env.sh not found in '$HERE'" >&2; exit 1; }
source "$HERE/env.sh"
: "${WORK:?FATAL: env.sh did not set WORK}"

NODES=${SLURM_JOB_NUM_NODES:?}
PPN=8; NRANKS=$((NODES * PPN))
# Single rep per node count (submit once per -N); i=10 keeps it quick. Cray's staging path is
# steady-state stable, so 1 rep x 10 iters is enough for the plotted config-T line.
ITERS=10; WARMUP=3
BLK=134217728        # 128 MB
ML=( "ddp_bucket_25MiB:26214400" "resnet50_fp32:102228128"
     "bert_large_fp16:680000000" "bert_large_fp32:1360000000" )

RESULTS=$WORK/results_crayblk128/N${NODES}_job${SLURM_JOB_ID}
mkdir -p "$RESULTS"
{ echo "date=$(date -u +%FT%TZ)"; echo "nodes=$NODES ranks=$NRANKS ppn=$PPN";
  echo "MPICH_GPU_ALLREDUCE_BLK_SIZE=$BLK (kernel path ON); i=$ITERS x=$WARMUP";
  module list; } > "$RESULTS/env.txt" 2>&1

load_cray
OSU_CRAY_BIN=$OSU_CRAY/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce
SRUN="srun -N $NODES -n $NRANKS --ntasks-per-node=$PPN --gpus-per-node=8 --gpu-bind=closest -c7"

echo "=== D_blk128 sweep : N=$NODES ranks=$NRANKS ===" | tee "$RESULTS/D_blk128_sweep.txt"
MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_ALLREDUCE_BLK_SIZE=$BLK \
  $SRUN "$OSU_CRAY_BIN" -m 0:4294967296 -i $ITERS -x $WARMUP -d rocm -f 2>&1 \
  | tee -a "$RESULTS/D_blk128_sweep.txt" || true

for e in "${ML[@]}"; do
  sz=${e##*:}
  echo "### model=${e%%:*} bytes=$sz N=$NODES" | tee -a "$RESULTS/D_blk128_ml.txt"
  MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_ALLREDUCE_BLK_SIZE=$BLK \
    $SRUN "$OSU_CRAY_BIN" -m ${sz}:${sz} -i $ITERS -x $WARMUP -d rocm -f 2>&1 \
    | tee -a "$RESULTS/D_blk128_ml.txt" || true
done

echo "[cray-blk128] done -> $RESULTS"
