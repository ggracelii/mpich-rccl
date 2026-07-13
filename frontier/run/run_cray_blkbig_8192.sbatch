#!/bin/bash
#SBATCH -A csc678
#SBATCH -J crayblkbig
#SBATCH -t 00:15:00
#SBATCH --network=single_node_vni
#SBATCH -o %x-%j.out
#
# Does a BIGGER staging buffer push the Cray fault ceiling back up at 8192 nodes?
# BLK=64MB rescued Cray to 4 GiB through 4096 nodes, but at 8192 it faulted trying
# 64 MiB (job 4967908). Here we retry the large end (32 MiB -> 4 GiB, ascending) with
# BLK=128 MB and then BLK=256 MB. If the ceiling tracks BLK, a bigger buffer restores
# the range; if it still dies ~the BLK boundary, the rescue is dead at 8192 and the
# underlying Cray kernel bug wins. Ascending => the last row printed per BLK is the
# largest size that survived. srun (fast init), so 15 min covers both BLKs easily.

set -o pipefail
HERE=${FRONTIER_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}
[ -f "$HERE/env.sh" ] || { echo "FATAL: env.sh not found in '$HERE'" >&2; exit 1; }
source "$HERE/env.sh"
: "${WORK:?FATAL: env.sh did not set WORK}"

NODES=${SLURM_JOB_NUM_NODES:?}
PPN=8; NRANKS=$((NODES * PPN))
ITERS=10; WARMUP=3
BLKS=( 134217728 268435456 )        # 128 MB, 256 MB

RESULTS=$WORK/results_crayblk64/N${NODES}_blkbig_job${SLURM_JOB_ID}
mkdir -p "$RESULTS"
{ echo "date=$(date -u +%FT%TZ)"; echo "nodes=$NODES ranks=$NRANKS ppn=$PPN";
  echo "BLK escalation {128MB,256MB}; large end 32MiB-4GiB; i=$ITERS"; module list; } > "$RESULTS/env.txt" 2>&1

load_cray
OSU=$OSU_CRAY/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce
SRUN="srun -N $NODES -n $NRANKS --ntasks-per-node=$PPN --gpus-per-node=8 --gpu-bind=closest -c7"

for BLK in "${BLKS[@]}"; do
  echo "=== BLK=$BLK : N=$NODES (32MiB-4GiB) ===" | tee -a "$RESULTS/D_blkbig.txt"
  env MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_ALLREDUCE_BLK_SIZE=$BLK \
    $SRUN "$OSU" -m 33554432:4294967296 -i "$ITERS" -x "$WARMUP" -d rocm -f 2>&1 \
    | tee -a "$RESULTS/D_blkbig.txt" || true
done

echo "[crayblkbig] done -> $RESULTS"
