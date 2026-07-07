#!/bin/bash
#SBATCH -A csc678
#SBATCH -J cray-knob
#SBATCH -t 00:15:00
#SBATCH --network=single_node_vni
#SBATCH -o %x-%j.out
#
# Cray MPICH at giant scale with the fault workaround (probe verdict: only
# MPICH_GPU_ALLREDUCE_USE_KERNEL=0 survives >4 MiB at >=1024 nodes).
# Measures the full sweep (8 B -> 4 GiB, patched OSU) + the ML gradient sizes.
# NON-DEFAULT CRAY CONFIG: results go to results_crayknob/ (kept separate from
# the default-D dataset; plotted visually distinct). i=20 because the copy-based
# path is slow at giant sizes (~4 s/op at 4 GiB @1024).

set -o pipefail
HERE=${FRONTIER_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}
[ -f "$HERE/env.sh" ] || { echo "FATAL: env.sh not found in '$HERE'" >&2; exit 1; }
source "$HERE/env.sh"
: "${WORK:?FATAL: env.sh did not set WORK}"

NODES=${SLURM_JOB_NUM_NODES:?}
PPN=8; NRANKS=$((NODES * PPN))
ITERS=20; WARMUP=5
ML=( "ddp_bucket_25MiB:26214400" "resnet50_fp32:102228128"
     "bert_large_fp16:680000000" "bert_large_fp32:1360000000" )

RESULTS=$WORK/results_crayknob/N${NODES}_job${SLURM_JOB_ID}
mkdir -p "$RESULTS"
{ echo "date=$(date -u +%FT%TZ)"; echo "nodes=$NODES ranks=$NRANKS ppn=$PPN";
  echo "MPICH_GPU_ALLREDUCE_USE_KERNEL=0 (fault workaround); i=$ITERS x=$WARMUP";
  module list; } > "$RESULTS/env.txt" 2>&1

load_cray
OSU_CRAY_BIN=$OSU_CRAY/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce
SRUN="srun -N $NODES -n $NRANKS --ntasks-per-node=$PPN --gpus-per-node=8 --gpu-bind=closest -c7"

# full sweep with the workaround
echo "=== D_kernel0 sweep : N=$NODES ranks=$NRANKS ===" | tee "$RESULTS/D_kernel0_sweep.txt"
MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_ALLREDUCE_USE_KERNEL=0 \
  $SRUN "$OSU_CRAY_BIN" -m 0:4294967296 -i $ITERS -x $WARMUP -d rocm -f 2>&1 \
  | tee -a "$RESULTS/D_kernel0_sweep.txt" || true

# ML gradient sizes with the workaround
for e in "${ML[@]}"; do
  sz=${e##*:}
  echo "### model=${e%%:*} bytes=$sz N=$NODES" | tee -a "$RESULTS/D_kernel0_ml.txt"
  MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_ALLREDUCE_USE_KERNEL=0 \
    $SRUN "$OSU_CRAY_BIN" -m ${sz}:${sz} -i $ITERS -x $WARMUP -d rocm -f 2>&1 \
    | tee -a "$RESULTS/D_kernel0_ml.txt" || true
done

echo "[cray-knob] done -> $RESULTS"
