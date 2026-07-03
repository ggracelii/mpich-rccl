#!/bin/bash
#SBATCH -A csc678
#SBATCH -J rccl-allreduce
#SBATCH -t 00:10:00
#SBATCH --network=single_node_vni
#SBATCH -o %x-%j.out
# Submit with node count on the CLI so one script scales:
#   sbatch -N 2   run_allreduce.sbatch
#   sbatch -N 512 run_allreduce.sbatch
# (submit_scaling.sh fans this across node counts; all via the normal batch queue)
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
HERE=${FRONTIER_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}
if [ ! -f "$HERE/env.sh" ]; then
  echo "FATAL: env.sh not found in '$HERE' — submit via submit_scaling.sh, or run sbatch from the frontier/ dir" >&2
  exit 1
fi
source "$HERE/env.sh"
: "${WORK:?FATAL: env.sh did not set WORK — aborting so we never write to /results}"
BIND="$HERE/bind_frontier.sh"

NODES=${SLURM_JOB_NUM_NODES:?submit with sbatch -N <nodes>}
PPN=8
NRANKS=$((NODES * PPN))
WARMUP=10                     # OSU -x (covers one-time RCCL comm init + clock ramp; 5 is the floor)
ITERS=50                      # OSU -i, uniform across all configs (allreduce is low-variance; min/med/max reported)
# CPU configs (A,B) are catastrophically slow at large sizes (~6 s/op at 512 MiB
# inter-node), so cap them at 32 MiB. GPU configs (C,D,E) sweep to 4 GiB.
CPU_MAX=33554432              # 32 MiB
GPU_MAX=4294967296            # 4 GiB  (8 GiB send+recv per GCD; fits the 64 GB HBM easily)

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
HOST_ARGS="-m 0:$CPU_MAX -i $ITERS -x $WARMUP -f"
CPUDEV_ARGS="-m 0:$CPU_MAX -i $ITERS -x $WARMUP -d rocm -f"
GPU_ARGS="-m 0:$GPU_MAX -i $ITERS -x $WARMUP -d rocm -f"
# Cray's GPU allreduce faults (GPU memory access fault) above 4 MiB at >=1024 nodes, so cap
# config D there: it sweeps 8 B..4 MiB cleanly (Cray's survivable ceiling) and never attempts
# 8 MiB. At <=512 nodes Cray reaches 1 GiB fine, so D uses the full GPU range.
CRAY_MAX=$GPU_MAX
[ "$NODES" -ge 1024 ] && CRAY_MAX=4194304        # 4 MiB
CRAY_ARGS="-m 0:$CRAY_MAX -i $ITERS -x $WARMUP -d rocm -f"

# ---- A: MPICH CPU, HOST buffers (traditional MPI, no GPU awareness) --------
load_mine
echo "=== A_mpich_host : N=$NODES ranks=$NRANKS ===" | tee "$RESULTS/A_mpich_host.txt"
MPIR_CVAR_DEVICE_COLLECTIVES=none \
  $MPIEXEC "$OSU_MINE_BIN" $HOST_ARGS 2>&1 | tee -a "$RESULTS/A_mpich_host.txt"

# ---- B: MPICH CPU algorithm, DEVICE buffers (your run_mpich_multi.sh) -------
load_mine
echo "=== B_mpich_dev : N=$NODES ranks=$NRANKS ===" | tee "$RESULTS/B_mpich_dev.txt"
MPIR_CVAR_DEVICE_COLLECTIVES=all \
  $MPIEXEC "$BIND" "$OSU_MINE_BIN" $CPUDEV_ARGS 2>&1 | tee -a "$RESULTS/B_mpich_dev.txt"

# ---- C: MPICH + RCCL backend (your run_mpichccl_multi.sh) ------------------
load_mine
module load rccl-net-plugin   # aws-ofi-rccl: RCCL uses libfabric/CXI (not TCP) inter-node + OLCF-tuned FI_CXI/NCCL settings. Loaded ONLY for RCCL configs so A/B baselines stay on default MPICH networking.
echo "=== C_mpich_rccl : N=$NODES ranks=$NRANKS ===" | tee "$RESULTS/C_mpich_rccl.txt"
MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=ccl MPIR_CVAR_ALLREDUCE_CCL=rccl MPIR_CVAR_DEVICE_COLLECTIVES=none \
  $MPIEXEC "$BIND" "$OSU_MINE_BIN" $GPU_ARGS 2>&1 | tee -a "$RESULTS/C_mpich_rccl.txt"

# ---- D: Cray MPICH GPU-aware (production baseline) — uses srun -------------
# Cray's GPU allreduce faults (GPU memory access fault) above 4 MiB at >=1024 nodes,
# so RUN_CRAY=0 skips it at extreme scale (e.g. 8192) to avoid rank aborts / core dumps.
if [ "${RUN_CRAY:-1}" = 1 ]; then
  load_cray
  echo "=== D_cray_gpuaware : N=$NODES ranks=$NRANKS ===" | tee "$RESULTS/D_cray_gpuaware.txt"
  MPICH_GPU_SUPPORT_ENABLED=1 \
    srun -N "$NODES" -n "$NRANKS" --ntasks-per-node=$PPN --gpus-per-node=8 \
         --gpu-bind=closest -c7 \
         "$OSU_CRAY_BIN" $CRAY_ARGS 2>&1 | tee -a "$RESULTS/D_cray_gpuaware.txt"   # [verify OLCF] -c/--gpu-bind; CRAY_ARGS caps at 4 MiB for N>=1024
fi

# ---- E: pure RCCL ceiling (only meaningful up to a few nodes) --------------
if [ "$NODES" -le 8 ]; then
  load_mine
  module load rccl-net-plugin
  echo "=== E_rccl_tests : N=$NODES ===" | tee "$RESULTS/E_rccl_tests.txt"
  $MPIEXEC "$BIND" "$RCCL_TESTS/build/all_reduce_perf" -b 8 -e 4G -f 2 -g 1 -n $ITERS -w $WARMUP \
    2>&1 | tee -a "$RESULTS/E_rccl_tests.txt"
fi

echo "[run] done -> $RESULTS"
