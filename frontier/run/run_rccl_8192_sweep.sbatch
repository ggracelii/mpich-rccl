#!/bin/bash
#SBATCH -A csc678
#SBATCH -J rccl8192swps
#SBATCH -t 01:00:00
#SBATCH --network=single_node_vni
#SBATCH -o %x-%j.out
#
# RCCL 8192-node full sweep via PER-SIZE launches (one mpiexec / one RCCL init per size).
# WHY per-size, not one launch: the single-launch full sweep timed out TWICE — job 4967901
# (i=10, this script's old body) @20 min and job 4973420 (i=50) @30 min — BOTH producing only
# the 1024 B row before the wall. Two-for-two says the shared-comm single launch wedges right
# after the first size. The per-size pattern is the SAME one that already succeeded for the
# 8192 ML points (job 4980915: inits ~2 min, sizes measured up to 1.36 GB), so RCCL itself is
# fine at 65,536 ranks. Per-size => each init is independent, tee -a banks every completed
# size, and a slow/bad init on one size costs only that size (resubmit refills gaps; the loader
# merges all N8192_job* dirs by size). Covers 1 KiB -> 4 GiB; 4 B-512 B is partial job 4948914.
#
# LARGEST-FIRST: the headline giant sizes (where RCCL beats Cray) are measured first, so a
# walltime clip drops only small sizes (cheap, and partly covered by 4948914), never the GB end.
# Each size is timestamped and mpiexec flushes on exit, so `tail -f` shows live per-size progress
# and any stall is pinned to an exact size — no stdout block-buffering blind spot.
#
# CHUNKING: 8192 nodes is ~87% of Frontier, so a long job schedules terribly — short jobs backfill
# far better. Pass a subset of sizes as ARGS to run just those (each still one init), e.g.
#   sbatch -N 8192 -t 00:25:00 run/run_rccl_8192_sweep.sbatch 4294967296 2147483648 1073741824
# With no args it runs the full 4 GiB->1 KiB ladder. Each chunk writes its own N8192_job<id> dir;
# the notebook merges all such dirs by size, so chunks (and resubmits) compose into one 8192 row.

set -o pipefail
HERE=${FRONTIER_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}
[ -f "$HERE/env.sh" ] || { echo "FATAL: env.sh not found in '$HERE'" >&2; exit 1; }
source "$HERE/env.sh"
: "${WORK:?FATAL: env.sh did not set WORK}"
BIND="$HERE/bind_frontier.sh"

NODES=${SLURM_JOB_NUM_NODES:?}
PPN=8; NRANKS=$((NODES * PPN))
# Iterations default to i=50 (matches the main ladder), but are overridable via env: a 4 GiB
# allreduce at 65,536 ranks costs seconds PER CALL, so 60 calls (i=50 x=10) blew a 25-min cap
# on the giants. Export SWEEP_ITERS/SWEEP_WARMUP to cut iters for the giant sizes, e.g.
#   sbatch ... --export=ALL,SWEEP_ITERS=2,SWEEP_WARMUP=1 run/run_rccl_8192_sweep.sbatch 4294967296
ITERS=${SWEEP_ITERS:-50}; WARMUP=${SWEEP_WARMUP:-10}

# Sizes: a subset passed as args (a chunk), else the full 2^32 (4 GiB) -> 2^10 (1 KiB) ladder,
# largest-first (see header). Passing a chunk lets short jobs backfill on this near-full-machine run.
if [ "$#" -gt 0 ]; then
  SIZES=( "$@" )
else
  SIZES=( 4294967296 2147483648 1073741824 536870912 268435456 134217728 67108864 \
          33554432 16777216 8388608 4194304 2097152 1048576 524288 262144 131072 \
          65536 32768 16384 8192 4096 2048 1024 )
fi

RESULTS=$WORK/results/N${NODES}_job${SLURM_JOB_ID}
mkdir -p "$RESULTS"
{ echo "date=$(date -u +%FT%TZ)"; echo "nodes=$NODES ranks=$NRANKS ppn=$PPN";
  echo "per-size RCCL sweep 4GiB->1KiB, i=$ITERS x=$WARMUP (one RCCL init per size)"; module list; } > "$RESULTS/env.txt" 2>&1

load_mine
module load rccl-net-plugin
MPIEXEC="mpiexec -bootstrap slurm -n $NRANKS -ppn $PPN -bind-to numa"
OSU=$OSU_MINE/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce
CVARS="MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=ccl MPIR_CVAR_ALLREDUCE_CCL=rccl MPIR_CVAR_DEVICE_COLLECTIVES=none"

OUT="$RESULTS/C_mpich_rccl.txt"
echo "=== C_mpich_rccl per-size : N=$NODES ranks=$NRANKS (4GiB->1KiB, i=$ITERS) ===" | tee "$OUT"
for sz in "${SIZES[@]}"; do
  echo "# --- size=$sz start=$(date -u +%FT%TZ) ---" | tee -a "$OUT"
  env $CVARS $MPIEXEC "$BIND" "$OSU" -m ${sz}:${sz} -i "$ITERS" -x "$WARMUP" -d rocm -f 2>&1 \
    | tee -a "$OUT" || true
  echo "# --- size=$sz done=$(date -u +%FT%TZ) ---" | tee -a "$OUT"
done
echo "[8192swps] done -> $RESULTS"
