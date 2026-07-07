#!/bin/bash
#SBATCH -A csc678
#SBATCH -J cray-probe2
#SBATCH -t 00:15:00
#SBATCH --network=single_node_vni
#SBATCH -o %x-%j.out
#
# Probe 2: can Cray's FAST (kernel) allreduce path be extended past 4 MiB at >=1024
# nodes? Fault coincides with payload >= BLK_SIZE (default 8MB staging buffer) =>
# suspect the multi-block path. Variants (all with kernel ON):
#   blk64M  - man-page-suggested 64MB staging buffer
#   blk4G   - staging buffer >= largest message: everything single-block
#   blk4M   - SMALLER buffer (diagnostic: if multi-block is the bug, faults should start EARLIER)
#   v8132   - cray-mpich/8.1.32, default env (bug may be fixed upstream)
#   v900    - cray-mpich/9.0.0,  default env (major version)
# Each runs a size RANGE; the last size reached before truncation = that variant's ceiling.

set -o pipefail
HERE=${FRONTIER_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}
[ -f "$HERE/env.sh" ] || { echo "FATAL: env.sh not found in '$HERE'" >&2; exit 1; }
source "$HERE/env.sh"
: "${WORK:?}"

NODES=${SLURM_JOB_NUM_NODES:?}
PPN=8; NRANKS=$((NODES * PPN))

RESULTS=$WORK/results_crayprobe2/N${NODES}_job${SLURM_JOB_ID}
mkdir -p "$RESULTS"
{ echo "date=$(date -u +%FT%TZ)"; echo "nodes=$NODES ranks=$NRANKS"; module list; } \
  > "$RESULTS/env.txt" 2>&1

load_cray
OSU_CRAY_BIN=$OSU_CRAY/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce
SRUN="srun -N $NODES -n $NRANKS --ntasks-per-node=$PPN --gpus-per-node=8 --gpu-bind=closest -c7"

probe() {  # probe <label> <mrange> [VAR=VAL ...]
  local label=$1 mrange=$2; shift 2
  echo "=== $label : -m $mrange $* ===" | tee "$RESULTS/${label}.txt"
  env MPICH_GPU_SUPPORT_ENABLED=1 "$@" \
    $SRUN "$OSU_CRAY_BIN" -m "$mrange" -i 20 -x 5 -d rocm -f 2>&1 \
    | tee -a "$RESULTS/${label}.txt" || true
  local last=$(grep -E '^[0-9]+ +[0-9]' "$RESULTS/${label}.txt" | tail -1 | awk '{print $1}')
  echo "[probe2] $label : reached ${last:-NOTHING}" | tee -a "$RESULTS/VERDICTS.txt"
}

probe blk64M "4194304:4294967296"  MPICH_GPU_ALLREDUCE_BLK_SIZE=67108864
probe blk4G  "4194304:4294967296"  MPICH_GPU_ALLREDUCE_BLK_SIZE=4294967296
probe blk4M  "1048576:16777216"    MPICH_GPU_ALLREDUCE_BLK_SIZE=4194304

module swap cray-mpich cray-mpich/8.1.32 2>&1 | head -2
probe v8132  "4194304:4294967296"

module swap cray-mpich cray-mpich/9.0.0 2>&1 | head -2
probe v900   "4194304:4294967296"

echo "[probe2] done -> $RESULTS"; cat "$RESULTS/VERDICTS.txt"
