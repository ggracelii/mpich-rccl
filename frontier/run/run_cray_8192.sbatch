#!/bin/bash
#SBATCH -A csc678
#SBATCH -J cray8192
#SBATCH -t 00:20:00
#SBATCH --network=single_node_vni
#SBATCH -o %x-%j.out
#
# Tuned-Cray (BLK=64MB) at 8192 nodes (65,536 GCDs): 1 KiB->4 GiB sweep + 4 ML sizes,
# for the complete RCCL-vs-Cray comparison at full-machine scale (matching the RCCL 8192
# range, which starts at 1 KiB). Cray uses srun (fast MPI_Init), so init is cheap.
# Open question this answers: does the BLK=64MB fault rescue (verified 1024, ladder to
# 4096) still hold at 8192? A truncated sweep = the rescue's ceiling.

set -o pipefail
HERE=${FRONTIER_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}
[ -f "$HERE/env.sh" ] || { echo "FATAL: env.sh not found in '$HERE'" >&2; exit 1; }
source "$HERE/env.sh"
: "${WORK:?FATAL: env.sh did not set WORK}"

NODES=${SLURM_JOB_NUM_NODES:?}
PPN=8; NRANKS=$((NODES * PPN))
ITERS=10; WARMUP=3
BLK=67108864
ML=( "bert_large_fp32:1360000000" "bert_large_fp16:680000000"
     "resnet50_fp32:102228128"     "ddp_bucket_25MiB:26214400" )

RESULTS=$WORK/results_crayblk64/N${NODES}_job${SLURM_JOB_ID}
mkdir -p "$RESULTS"
{ echo "date=$(date -u +%FT%TZ)"; echo "nodes=$NODES ranks=$NRANKS ppn=$PPN";
  echo "BLK=$BLK; large-first sweep + ML; i=$ITERS"; module list; } > "$RESULTS/env.txt" 2>&1

load_cray
OSU=$OSU_CRAY/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce
SRUN="srun -N $NODES -n $NRANKS --ntasks-per-node=$PPN --gpus-per-node=8 --gpu-bind=closest -c7"
ENVV="MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_ALLREDUCE_BLK_SIZE=$BLK"

echo "=== D_blk64 sweep : N=$NODES (1KiB-4GiB) ===" | tee "$RESULTS/D_blk64_sweep.txt"
env $ENVV $SRUN "$OSU" -m 1024:4294967296 -i "$ITERS" -x "$WARMUP" -d rocm -f 2>&1 \
  | tee -a "$RESULTS/D_blk64_sweep.txt" || true

for e in "${ML[@]}"; do
  sz=${e##*:}
  echo "### model=${e%%:*} bytes=$sz N=$NODES" | tee -a "$RESULTS/D_blk64_ml.txt"
  env $ENVV $SRUN "$OSU" -m ${sz}:${sz} -i "$ITERS" -x "$WARMUP" -d rocm -f 2>&1 \
    | tee -a "$RESULTS/D_blk64_ml.txt" || true
done

echo "[cray8192] done -> $RESULTS"
