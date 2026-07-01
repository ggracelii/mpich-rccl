#!/bin/bash
#SBATCH -A csc678
#SBATCH -J rccl-allreduce
#SBATCH -t 00:20:00
#SBATCH -o %x-%j.out
# NOTE: submit with node count on the CLI so one script scales:
#   sbatch -N 2 run_allreduce.sbatch          (debug for <=8 nodes: add -q debug)
# Runs configs A/B/C/D back-to-back on the SAME allocation (same dragonfly
# placement) sweeping message sizes, so every comparison is apples-to-apples.
#
# 8 GCDs/node = 8 ranks/node, 1 rank per GCD. -c7 leaves core 0 of each L3
# reserved (Frontier low-noise). [verify OLCF] the exact -c/--gpu-bind recipe.

set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"

NODES=${SLURM_JOB_NUM_NODES:?submit with sbatch -N <nodes>}
PPN=8
NRANKS=$((NODES * PPN))
SIZES="0:1073741824"          # 0 B -> 1 GiB, OSU steps by powers of 2
ITERS=1000                    # OSU -i; drop for the 1 GiB tail if it's slow
WARMUP=200                    # OSU -x

RESULTS=$WORK/results/N${NODES}_job${SLURM_JOB_ID}
mkdir -p "$RESULTS"

# --- reproducibility: dump the full environment next to the numbers ---------
{ echo "date=$(date -u +%FT%TZ)"; echo "nodes=$NODES ranks=$NRANKS ppn=$PPN";
  echo "nodelist=$SLURM_JOB_NODELIST"; module list; rocm-smi --showdriverversion; } \
  > "$RESULTS/env.txt" 2>&1

SRUN="srun -N $NODES -n $NRANKS --ntasks-per-node=$PPN --gpus-per-node=8 \
      --gpu-bind=closest -c7 --mpi=pmix"     # [verify OLCF]; swap in bind_frontier.sh if closest misbinds
OSU_MINE_BIN=$OSU_MINE/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce
OSU_CRAY_BIN=$OSU_CRAY/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce
OSU_ARGS="-m $SIZES -i $ITERS -x $WARMUP -d rocm -f"

run_cfg() {  # $1=label  $2=osu_binary  (env/CVARs set by caller)
  echo "=== config $1 : $NODES nodes / $NRANKS ranks ===" | tee -a "$RESULTS/$1.txt"
  $SRUN "$2" $OSU_ARGS 2>&1 | tee -a "$RESULTS/$1.txt"
}

# ---- A: MPICH CPU, HOST buffers (traditional MPI, no GPU awareness) --------
load_mine
MPIR_CVAR_DEVICE_COLLECTIVES=none \
  $SRUN "$OSU_MINE_BIN" -m "$SIZES" -i $ITERS -x $WARMUP -f \
  2>&1 | tee "$RESULTS/A_mpich_host.txt"   # note: no -d rocm -> host buffers

# ---- B: MPICH CPU algorithm, DEVICE buffers (your run_mpich_multi.sh) -------
load_mine
MPIR_CVAR_DEVICE_COLLECTIVES=all \
  run_cfg B_mpich_dev "$OSU_MINE_BIN"

# ---- C: MPICH + RCCL backend  (your run_mpichccl_multi.sh) -----------------
load_mine
MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=ccl \
MPIR_CVAR_DEVICE_COLLECTIVES=none \
  run_cfg C_mpich_rccl "$OSU_MINE_BIN"

# ---- D: Cray MPICH GPU-aware (production baseline) -------------------------
load_cray
MPICH_GPU_SUPPORT_ENABLED=1 \
  run_cfg D_cray_gpuaware "$OSU_CRAY_BIN"

# ---- E: pure RCCL ceiling (only meaningful up to a few nodes) --------------
if [ "$NODES" -le 8 ]; then
  load_mine
  $SRUN "$RCCL_TESTS/build/all_reduce_perf" -b 8 -e 1G -f 2 -g 1 \
    2>&1 | tee "$RESULTS/E_rccl_tests.txt"
fi

echo "[run] done -> $RESULTS"
