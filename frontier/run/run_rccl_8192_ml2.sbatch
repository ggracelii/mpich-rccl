#!/bin/bash
#SBATCH -A csc678
#SBATCH -J rccl8192ml2
#SBATCH -t 00:20:00
#SBATCH --network=single_node_vni
#SBATCH -o %x-%j.out
#
# RCCL 8192-node (65,536 GCD) — the 2 smaller ML gradient sizes that the first ML
# job (4967902) did not reach: ResNet-50 fp32 (102 MB) and DDP bucket (25 MiB).
# The two BERT points (fp32 1.36 GB, fp16 680 MB) are already captured there.
# OSU takes one size per launch => one RCCL init each (~3-19 min lottery at this
# scale). Only 2 inits here, so 20 min has margin: the first ML job timed out only
# because it was already 2 BERT inits deep before reaching ResNet. Largest-first
# so any clip drops just the bucket point, keeping ResNet.

set -o pipefail
HERE=${FRONTIER_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}
[ -f "$HERE/env.sh" ] || { echo "FATAL: env.sh not found in '$HERE'" >&2; exit 1; }
source "$HERE/env.sh"
: "${WORK:?FATAL: env.sh did not set WORK}"
BIND="$HERE/bind_frontier.sh"

NODES=${SLURM_JOB_NUM_NODES:?}
PPN=8; NRANKS=$((NODES * PPN))
ITERS=10; WARMUP=3
ML=( "resnet50_fp32:102228128" "ddp_bucket_25MiB:26214400" )

RESULTS=$WORK/results_ml/N${NODES}_job${SLURM_JOB_ID}
mkdir -p "$RESULTS"
{ echo "date=$(date -u +%FT%TZ)"; echo "nodes=$NODES ranks=$NRANKS"; echo "resnet+bucket only"; module list; } > "$RESULTS/env.txt" 2>&1

load_mine
module load rccl-net-plugin
MPIEXEC="mpiexec -bootstrap slurm -n $NRANKS -ppn $PPN -bind-to numa"
OSU=$OSU_MINE/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce
CVARS="MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=ccl MPIR_CVAR_ALLREDUCE_CCL=rccl MPIR_CVAR_DEVICE_COLLECTIVES=none"

for e in "${ML[@]}"; do
  sz=${e##*:}
  echo "### model=${e%%:*} bytes=$sz N=$NODES" | tee -a "$RESULTS/C_mpich_rccl.txt"
  env $CVARS $MPIEXEC "$BIND" "$OSU" -m ${sz}:${sz} -i "$ITERS" -x "$WARMUP" -d rocm -f 2>&1 \
    | tee -a "$RESULTS/C_mpich_rccl.txt" || true
done

echo "[8192ml2] done -> $RESULTS"
