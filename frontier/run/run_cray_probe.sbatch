#!/bin/bash
#SBATCH -A csc678
#SBATCH -J cray-probe
#SBATCH -t 00:12:00
#SBATCH --network=single_node_vni
#SBATCH -o %x-%j.out
#
# Probe: can ANY documented Cray GPU-path tuning avoid the >4 MiB GPU memory
# access fault at >=1024 nodes? One knob per run, 8 MiB only (first faulting
# size, from `man intro_mpi` candidates). Two outcomes, both publishable:
#   - a knob survives -> giant-scale Cray data becomes possible (with a
#     "non-default configuration" caveat); rerun full range with that knob.
#   - all fault      -> "persists under all documented GPU-path tunings",
#     the strongest form of the finding.

set -o pipefail
HERE=${FRONTIER_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}
[ -f "$HERE/env.sh" ] || { echo "FATAL: env.sh not found in '$HERE'" >&2; exit 1; }
source "$HERE/env.sh"
: "${WORK:?FATAL: env.sh did not set WORK}"

NODES=${SLURM_JOB_NUM_NODES:?}
PPN=8; NRANKS=$((NODES * PPN))
SZ=8388608                              # 8 MiB: smallest size that faults at >=1024
ARGS="-m ${SZ}:${SZ} -i 20 -x 5 -d rocm -f"

RESULTS=$WORK/results_crayprobe/N${NODES}_job${SLURM_JOB_ID}
mkdir -p "$RESULTS"
{ echo "date=$(date -u +%FT%TZ)"; echo "nodes=$NODES ranks=$NRANKS size=$SZ"; module list; } \
  > "$RESULTS/env.txt" 2>&1

load_cray
OSU_CRAY_BIN=$OSU_CRAY/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce
SRUN="srun -N $NODES -n $NRANKS --ntasks-per-node=$PPN --gpus-per-node=8 --gpu-bind=closest -c7"

probe() {  # probe <label> [VAR=VAL ...]
  local label=$1; shift
  echo "=== $label : $* ===" | tee "$RESULTS/${label}.txt"
  env MPICH_GPU_SUPPORT_ENABLED=1 "$@" \
    $SRUN "$OSU_CRAY_BIN" $ARGS 2>&1 | tee -a "$RESULTS/${label}.txt" || true
  # verdict: did the 8 MiB row appear?
  if grep -qE "^$SZ" "$RESULTS/${label}.txt"; then
    echo "[probe] $label : SURVIVED" | tee -a "$RESULTS/VERDICTS.txt"
  else
    echo "[probe] $label : FAULTED"  | tee -a "$RESULTS/VERDICTS.txt"
  fi
}

probe default                                                     # control: expect FAULTED
probe ipc0            MPICH_GPU_IPC_ENABLED=0
probe staging0        MPICH_GPU_COLL_STAGING_AREA_OPT=0
probe kernel0         MPICH_GPU_ALLREDUCE_USE_KERNEL=0
probe kernel1         MPICH_GPU_ALLREDUCE_USE_KERNEL=1
probe noasynccopy     MPICH_GPU_NO_ASYNC_COPY=1

echo "[probe] done -> $RESULTS"; cat "$RESULTS/VERDICTS.txt"
