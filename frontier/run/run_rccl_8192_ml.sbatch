#!/bin/bash
#SBATCH -A csc678
#SBATCH -J rccl8192ml
#SBATCH -t 00:20:00
#SBATCH --network=single_node_vni
#SBATCH -o %x-%j.out
#
# RCCL 8192-node (65,536 GCD) exact ML gradient sizes. OSU takes one size per launch,
# so each of the 4 sizes is its own osu call => its own RCCL init (~3 min at this scale);
# ~4 inits + measurement fits a 20-min cap. Large sizes ordered first so a walltime clip
# drops the smallest (bucket) rather than the biggest (BERT-fp32).

set -o pipefail
HERE=${FRONTIER_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}
[ -f "$HERE/env.sh" ] || { echo "FATAL: env.sh not found in '$HERE'" >&2; exit 1; }
source "$HERE/env.sh"
: "${WORK:?FATAL: env.sh did not set WORK}"
BIND="$HERE/bind_frontier.sh"

NODES=${SLURM_JOB_NUM_NODES:?}
PPN=8; NRANKS=$((NODES * PPN))
ITERS=10; WARMUP=3
# largest first (BERT-fp32, BERT-fp16, ResNet, bucket) so a clip drops the cheapest point
ML=( "bert_large_fp32:1360000000" "bert_large_fp16:680000000"
     "resnet50_fp32:102228128"     "ddp_bucket_25MiB:26214400" )

RESULTS=$WORK/results_ml/N${NODES}_job${SLURM_JOB_ID}
mkdir -p "$RESULTS"
{ echo "date=$(date -u +%FT%TZ)"; echo "nodes=$NODES ranks=$NRANKS"; module list; } > "$RESULTS/env.txt" 2>&1

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

echo "[8192ml] done -> $RESULTS"
