#!/bin/bash
#SBATCH -A csc678
#SBATCH -J rccl-allreduce
#SBATCH -t 00:20:00
#SBATCH --network=single_node_vni
#SBATCH -o %x-%j.out
# Submit with node count on the CLI so one script scales:
#   sbatch -N 2 -q debug run_allreduce.sbatch      (debug for small counts)
#   sbatch -N 512        run_allreduce.sbatch
#
# Runs configs A/B/C/D(/E) back-to-back on the SAME allocation (same dragonfly
# placement) so every comparison is apples-to-apples.
#
# Launcher split: your MPICH is built --with-pm=hydra, so configs A/B/C/E use
# mpiexec (matching your JLSE run_*_multi.sh). Cray MPICH (D) uses srun. GPU
# pinning is done by bind_frontier.sh under either launcher (see [verify OLCF]).
#
# NOTE: no `set -e` — one config failing must not abort the rest.
set -o pipefail
HERE=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}
source "$HERE/env.sh"
BIND="$HERE/bind_frontier.sh"

NODES=${SLURM_JOB_NUM_NODES:?submit with sbatch -N <nodes>}
PPN=8
NRANKS=$((NODES * PPN))
SIZES="0:1073741824"          # 0 B -> 1 GiB (OSU steps by powers of 2)
ITERS=1000                    # OSU -i
WARMUP=200                    # OSU -x

RESULTS=$WORK/results/N${NODES}_job${SLURM_JOB_ID}
mkdir -p "$RESULTS"

# --- reproducibility: dump the full environment next to the numbers ---------
{ echo "date=$(date -u +%FT%TZ)"; echo "nodes=$NODES ranks=$NRANKS ppn=$PPN";
  echo "nodelist=$SLURM_JOB_NODELIST"; module list; rocm-smi --showdriverversion; } \
  > "$RESULTS/env.txt" 2>&1

# Hydra launcher for YOUR MPICH. -bootstrap slurm makes Hydra spawn its proxies
# via srun (no inter-node ssh needed). -bind-to numa gives coarse CPU locality;
# bind_frontier.sh handles the (critical) GPU pinning.
MPIEXEC="mpiexec -bootstrap slurm -n $NRANKS -ppn $PPN -bind-to numa"

OSU_MINE_BIN=$OSU_MINE/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce
OSU_CRAY_BIN=$OSU_CRAY/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce
DEV_ARGS="-m $SIZES -i $ITERS -x $WARMUP -d rocm -f"
HOST_ARGS="-m $SIZES -i $ITERS -x $WARMUP -f"

# ---- A: MPICH CPU, HOST buffers (traditional MPI, no GPU awareness) --------
load_mine
echo "=== A_mpich_host : N=$NODES ranks=$NRANKS ===" | tee "$RESULTS/A_mpich_host.txt"
MPIR_CVAR_DEVICE_COLLECTIVES=none \
  $MPIEXEC "$OSU_MINE_BIN" $HOST_ARGS 2>&1 | tee -a "$RESULTS/A_mpich_host.txt"

# ---- B: MPICH CPU algorithm, DEVICE buffers (your run_mpich_multi.sh) -------
load_mine
echo "=== B_mpich_dev : N=$NODES ranks=$NRANKS ===" | tee "$RESULTS/B_mpich_dev.txt"
MPIR_CVAR_DEVICE_COLLECTIVES=all \
  $MPIEXEC "$BIND" "$OSU_MINE_BIN" $DEV_ARGS 2>&1 | tee -a "$RESULTS/B_mpich_dev.txt"

# ---- C: MPICH + RCCL backend (your run_mpichccl_multi.sh) ------------------
load_mine
echo "=== C_mpich_rccl : N=$NODES ranks=$NRANKS ===" | tee "$RESULTS/C_mpich_rccl.txt"
MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=ccl MPIR_CVAR_ALLREDUCE_CCL=rccl MPIR_CVAR_DEVICE_COLLECTIVES=none \
  $MPIEXEC "$BIND" "$OSU_MINE_BIN" $DEV_ARGS 2>&1 | tee -a "$RESULTS/C_mpich_rccl.txt"

# ---- D: Cray MPICH GPU-aware (production baseline) — uses srun -------------
load_cray
echo "=== D_cray_gpuaware : N=$NODES ranks=$NRANKS ===" | tee "$RESULTS/D_cray_gpuaware.txt"
MPICH_GPU_SUPPORT_ENABLED=1 \
  srun -N "$NODES" -n "$NRANKS" --ntasks-per-node=$PPN --gpus-per-node=8 \
       --gpu-bind=closest -c7 \
       "$OSU_CRAY_BIN" $DEV_ARGS 2>&1 | tee -a "$RESULTS/D_cray_gpuaware.txt"   # [verify OLCF] -c/--gpu-bind

# ---- E: pure RCCL ceiling (only meaningful up to a few nodes) --------------
if [ "$NODES" -le 8 ]; then
  load_mine
  echo "=== E_rccl_tests : N=$NODES ===" | tee "$RESULTS/E_rccl_tests.txt"
  $MPIEXEC "$BIND" "$RCCL_TESTS/build/all_reduce_perf" -b 8 -e 1G -f 2 -g 1 \
    2>&1 | tee -a "$RESULTS/E_rccl_tests.txt"
fi

echo "[run] done -> $RESULTS"
