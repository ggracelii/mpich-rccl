#!/bin/bash
#SBATCH -A csc678
#SBATCH -J rccl-ext4g
#SBATCH -t 00:10:00
#SBATCH --network=single_node_vni
#SBATCH -o %x-%j.out
#
# Extension past OSU's old 1 GiB cap (patched osu_allreduce: size_t loop).
# Measures ONLY the new points and appends them as a fresh N*_job* dir, so the
# notebook loader merges them into the existing dataset:
#   - main sweep:  2 GiB + 4 GiB  -> $WORK/results/      (C all nodes; D <=512)
#   - ML exact:    BERT fp32 1.36 GB -> $WORK/results_ml/ (C all nodes; D <=512)
# 4 GiB = practical ceiling: 2^30 4-byte elements just fits MPI's 32-bit count.
# C runs first (comm-init gets priority); each config is a single launch.

set -o pipefail
HERE=${FRONTIER_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}
[ -f "$HERE/env.sh" ] || { echo "FATAL: env.sh not found in '$HERE'" >&2; exit 1; }
source "$HERE/env.sh"
: "${WORK:?FATAL: env.sh did not set WORK}"
BIND="$HERE/bind_frontier.sh"

NODES=${SLURM_JOB_NUM_NODES:?}
PPN=8; NRANKS=$((NODES * PPN))
ITERS=50; WARMUP=10                      # match the main sweep
EXT_RANGE="2147483648:4294967296"        # 2 GiB -> 4 GiB (doubling hits both)
BERT_FP32=1360000000                     # 340 M params x 4 B
ML_ITERS=30                              # match run_ml_sync.sbatch

RES=$WORK/results/N${NODES}_job${SLURM_JOB_ID}
RESML=$WORK/results_ml/N${NODES}_job${SLURM_JOB_ID}
mkdir -p "$RES" "$RESML"
{ echo "date=$(date -u +%FT%TZ)"; echo "nodes=$NODES ranks=$NRANKS ppn=$PPN";
  echo "ext4g: sweep $EXT_RANGE (i=$ITERS), bert_fp32 $BERT_FP32 (i=$ML_ITERS)";
  module list; } > "$RES/env.txt" 2>&1
cp "$RES/env.txt" "$RESML/env.txt"

MPIEXEC="mpiexec -bootstrap slurm -n $NRANKS -ppn $PPN -bind-to numa"
OSU_MINE_BIN=$OSU_MINE/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce
OSU_CRAY_BIN=$OSU_CRAY/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce
CVARS_C="MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=ccl MPIR_CVAR_ALLREDUCE_CCL=rccl MPIR_CVAR_DEVICE_COLLECTIVES=none"

# ---- C: MPICH + RCCL — 2-4 GiB sweep, then BERT fp32 point ----
load_mine
module load rccl-net-plugin
echo "=== C_mpich_rccl ext4g : N=$NODES ranks=$NRANKS ===" | tee "$RES/C_mpich_rccl.txt"
env $CVARS_C $MPIEXEC "$BIND" "$OSU_MINE_BIN" -m $EXT_RANGE -i $ITERS -x $WARMUP -d rocm -f 2>&1 \
  | tee -a "$RES/C_mpich_rccl.txt" || true
echo "### model=bert_large_fp32 bytes=$BERT_FP32 N=$NODES" | tee "$RESML/C_mpich_rccl.txt"
env $CVARS_C $MPIEXEC "$BIND" "$OSU_MINE_BIN" -m ${BERT_FP32}:${BERT_FP32} -i $ML_ITERS -x $WARMUP -d rocm -f 2>&1 \
  | tee -a "$RESML/C_mpich_rccl.txt" || true

# ---- D: Cray GPU-aware — only <=512 (faults >4 MiB at >=1024) ----
if [ "$NODES" -le 512 ]; then
  load_cray
  echo "=== D_cray_gpuaware ext4g : N=$NODES ranks=$NRANKS ===" | tee "$RES/D_cray_gpuaware.txt"
  MPICH_GPU_SUPPORT_ENABLED=1 \
    srun -N "$NODES" -n "$NRANKS" --ntasks-per-node=$PPN --gpus-per-node=8 --gpu-bind=closest -c7 \
      "$OSU_CRAY_BIN" -m $EXT_RANGE -i $ITERS -x $WARMUP -d rocm -f 2>&1 \
      | tee -a "$RES/D_cray_gpuaware.txt" || true
  echo "### model=bert_large_fp32 bytes=$BERT_FP32 N=$NODES" | tee "$RESML/D_cray_gpuaware.txt"
  MPICH_GPU_SUPPORT_ENABLED=1 \
    srun -N "$NODES" -n "$NRANKS" --ntasks-per-node=$PPN --gpus-per-node=8 --gpu-bind=closest -c7 \
      "$OSU_CRAY_BIN" -m ${BERT_FP32}:${BERT_FP32} -i $ML_ITERS -x $WARMUP -d rocm -f 2>&1 \
      | tee -a "$RESML/D_cray_gpuaware.txt" || true
fi

echo "[ext4g] done -> $RES , $RESML"
