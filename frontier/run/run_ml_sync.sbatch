#!/bin/bash
#SBATCH -A csc678
#SBATCH -J rccl-ml-sync
#SBATCH -t 00:15:00
#SBATCH --network=single_node_vni
#SBATCH -o %x-%j.out
#
# ML gradient-sync proxy (design: docs/ML_EXPERIMENT.md, angle B).
# Data-parallel training allreduces the gradient every step; we measure that allreduce
# at real model-gradient BYTE sizes, weak-scaled (per-rank size fixed, grow node count).
#
# Notes:
#  - Comm cost is byte-bound, so we measure at the gradient's byte size with the default
#    4-byte element (== fp32). A 680 MB allreduce thus proxies BERT-Large's fp16 gradient
#    just as well as its fp32 one — the type only relabels the same bytes. (OSU has no MPI
#    fp16 type; measuring by bytes sidesteps that and stays honest.)
#  - Cray (D) faults >4 MiB at >=1024 nodes and all ML sizes are >4 MiB, so D is run only
#    at <=512 nodes; at >=1024 the comparison is C (MPICH-RCCL) vs B (MPICH-CPU).
#  - Reuses the same configs/binding/CVARs as run_allreduce.sbatch.

set -o pipefail
HERE=${FRONTIER_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}
if [ ! -f "$HERE/env.sh" ]; then
  echo "FATAL: env.sh not found in '$HERE' — submit via submit_ml.sh or from the frontier/ dir" >&2
  exit 1
fi
source "$HERE/env.sh"
: "${WORK:?FATAL: env.sh did not set WORK}"
BIND="$HERE/bind_frontier.sh"

NODES=${SLURM_JOB_NUM_NODES:?submit with sbatch -N <nodes>}
PPN=8
NRANKS=$((NODES * PPN))
ITERS=${ITERS:-30}
WARMUP=${WARMUP:-10}

# model : gradient bytes  (weak scaling => fixed per-rank size, node count grows)
ML=(
  "ddp_bucket_25MiB:26214400"      # PyTorch/Horovod default gradient bucket
  "resnet50_fp32:102228128"        # 25,557,032 params x 4 B (~25.6 M)
  "bert_large_fp16:680000000"      # 340 M params x 2 B (fp16 gradient, byte-proxy)
)
# (GPT-2 XL etc. sync many 25 MiB buckets -> per-bucket time x bucket count; use ddp_bucket.)

RESULTS=$WORK/results_ml/N${NODES}_job${SLURM_JOB_ID}
mkdir -p "$RESULTS"
{ echo "date=$(date -u +%FT%TZ)"; echo "nodes=$NODES ranks=$NRANKS ppn=$PPN";
  echo "iters=$ITERS warmup=$WARMUP"; echo "sizes: ${ML[*]}"; module list; } \
  > "$RESULTS/env.txt" 2>&1

MPIEXEC="mpiexec -bootstrap slurm -n $NRANKS -ppn $PPN -bind-to numa"
OSU_MINE_BIN=$OSU_MINE/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce
OSU_CRAY_BIN=$OSU_CRAY/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce

osu_at() {  # binary size -> one exact size (min=max), rocm buffers, full min/med/max
  local bin=$1 sz=$2
  "$bin" -m ${sz}:${sz} -i "$ITERS" -x "$WARMUP" -d rocm -f
}

# ---- B: MPICH CPU-on-device (host-staged) ----
load_mine
for e in "${ML[@]}"; do
  echo "### model=${e%%:*} bytes=${e##*:} N=$NODES" | tee -a "$RESULTS/B_mpich_dev.txt"
  MPIR_CVAR_DEVICE_COLLECTIVES=all \
    $MPIEXEC "$BIND" $(osu_at "$OSU_MINE_BIN" "${e##*:}") 2>&1 | tee -a "$RESULTS/B_mpich_dev.txt" || true
done

# ---- C: MPICH + RCCL backend ----
load_mine
module load rccl-net-plugin
for e in "${ML[@]}"; do
  echo "### model=${e%%:*} bytes=${e##*:} N=$NODES" | tee -a "$RESULTS/C_mpich_rccl.txt"
  MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=ccl MPIR_CVAR_ALLREDUCE_CCL=rccl MPIR_CVAR_DEVICE_COLLECTIVES=none \
    $MPIEXEC "$BIND" $(osu_at "$OSU_MINE_BIN" "${e##*:}") 2>&1 | tee -a "$RESULTS/C_mpich_rccl.txt" || true
done

# ---- D: Cray GPU-aware (only <=512 nodes; faults >4 MiB at >=1024) ----
if [ "$NODES" -le 512 ]; then
  load_cray
  for e in "${ML[@]}"; do
    sz=${e##*:}
    echo "### model=${e%%:*} bytes=$sz N=$NODES" | tee -a "$RESULTS/D_cray_gpuaware.txt"
    MPICH_GPU_SUPPORT_ENABLED=1 \
      srun -N "$NODES" -n "$NRANKS" --ntasks-per-node=$PPN --gpus-per-node=8 --gpu-bind=closest -c7 \
        "$OSU_CRAY_BIN" -m ${sz}:${sz} -i "$ITERS" -x "$WARMUP" -d rocm -f 2>&1 \
        | tee -a "$RESULTS/D_cray_gpuaware.txt" || true
  done
else
  echo "N=$NODES > 512: skipping Cray (D) — faults >4 MiB, all ML sizes exceed that" \
    | tee "$RESULTS/D_cray_gpuaware.txt"
fi

echo "[ml] done -> $RESULTS"
