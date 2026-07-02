#!/bin/bash
#SBATCH -A csc678
#SBATCH -J rccl-confirm
#SBATCH -N 2
#SBATCH -t 00:20:00
#SBATCH --network=single_node_vni
#SBATCH -o %x-%j.out
# Confirms each config runs its INTENDED mechanism (not just produces numbers),
# with log-level + kernel-level evidence, at a single 1 MiB size:
#
#   NCCL_DEBUG : C invokes RCCL over OFI/CXI ; B never touches RCCL.
#   rocprof    : C = RCCL collective kernels ; B = host-staging hipMemcpy ;
#                A = no GPU kernels ; D = Cray GPU-aware path.
#
# Small evidence (NCCL logs, kernel-stats CSVs, SUMMARY) -> ./results/confirm_<id>/
# (committable). Full rocprof traces -> $WORK (scratch, large).
#
#   sbatch confirm.sbatch
set -o pipefail
HERE=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}
source "$HERE/env.sh"
BIND="$HERE/bind_frontier.sh"

NRANKS=$((SLURM_JOB_NUM_NODES * 8))
MPIEXEC="mpiexec -bootstrap slurm -n $NRANKS -ppn 8 -bind-to numa"
OSU=$OSU_MINE/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce
OSU_CRAY_BIN=$OSU_CRAY/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce
SZ=1048576                                   # 1 MiB — regime where RCCL wins
RUN="-d rocm -m $SZ:$SZ -i 5 -x 2"           # few iters; we want traces, not timing

REPO=$HERE/results/confirm_${SLURM_JOB_ID}   # small, commit this
SCR=$WORK/results/confirm_${SLURM_JOB_ID}    # big rocprof traces
mkdir -p "$REPO" "$SCR"

# rank-0-only rocprof wrapper (works under both Hydra MPI_LOCALRANKID and srun SLURM_LOCALID)
cat > "$SCR/prof0.sh" <<'EOF'
#!/bin/bash
LID=${MPI_LOCALRANKID:-${SLURM_LOCALID:-0}}
if [ "$LID" = 0 ] && [ -n "$PROF_OUT" ]; then
  exec rocprof --hip-trace --stats -o "$PROF_OUT" "$@"
fi
exec "$@"
EOF
chmod +x "$SCR/prof0.sh"

echo "=== confirm.sbatch : N=$SLURM_JOB_NUM_NODES ranks=$NRANKS size=$SZ ==="

# ============ Part 1: NCCL_DEBUG (library-level truth) ============
load_mine
echo "[1/2] NCCL_DEBUG for B (expect NO 'RCCL version')"
MPIR_CVAR_DEVICE_COLLECTIVES=all NCCL_DEBUG=INFO \
  $MPIEXEC "$BIND" "$OSU" $RUN > "$REPO/B_nccldebug.log" 2>&1

load_mine
module load rccl-net-plugin
echo "[1/2] NCCL_DEBUG for C (expect 'RCCL version' + OFI/libfabric net)"
MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=ccl MPIR_CVAR_ALLREDUCE_CCL=rccl MPIR_CVAR_DEVICE_COLLECTIVES=none NCCL_DEBUG=INFO \
  $MPIEXEC "$BIND" "$OSU" $RUN > "$REPO/C_nccldebug.log" 2>&1

# ============ Part 2: rocprof (kernel-level truth, rank 0) ============
load_mine
echo "[2/2] rocprof A (host CPU — expect no GPU kernels)"
PROF_OUT=$SCR/A.csv MPIR_CVAR_DEVICE_COLLECTIVES=none \
  $MPIEXEC "$SCR/prof0.sh" "$OSU" -m $SZ:$SZ -i 5 -x 2 > "$REPO/A_run.log" 2>&1

load_mine
echo "[2/2] rocprof B (MPICH device path — expect hipMemcpy staging, no nccl)"
PROF_OUT=$SCR/B.csv MPIR_CVAR_DEVICE_COLLECTIVES=all \
  $MPIEXEC "$BIND" "$SCR/prof0.sh" "$OSU" $RUN > "$REPO/B_run.log" 2>&1

load_mine
module load rccl-net-plugin
echo "[2/2] rocprof C (RCCL — expect nccl AllReduce kernels)"
PROF_OUT=$SCR/C.csv MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=ccl MPIR_CVAR_ALLREDUCE_CCL=rccl MPIR_CVAR_DEVICE_COLLECTIVES=none \
  $MPIEXEC "$BIND" "$SCR/prof0.sh" "$OSU" $RUN > "$REPO/C_run.log" 2>&1

load_cray
echo "[2/2] rocprof D (Cray GPU-aware)"
PROF_OUT=$SCR/D.csv MPICH_GPU_SUPPORT_ENABLED=1 \
  srun -N "$SLURM_JOB_NUM_NODES" -n "$NRANKS" --ntasks-per-node=8 --gpus-per-node=8 \
       --gpu-bind=closest -c7 "$SCR/prof0.sh" "$OSU_CRAY_BIN" $RUN > "$REPO/D_run.log" 2>&1

# copy the small kernel-stats CSVs into the repo dir
cp "$SCR"/*.stats.csv "$REPO/" 2>/dev/null

# ============ Summary ============
S="$REPO/SUMMARY.txt"
{
  echo "==== CONFIRMATION SUMMARY (job $SLURM_JOB_ID, size $SZ) ===="
  echo
  echo "-- B should NOT invoke RCCL --"
  grep -c "RCCL version" "$REPO/B_nccldebug.log" | sed 's/^/  RCCL banner count (want 0): /'
  echo
  echo "-- C should invoke RCCL over OFI/CXI --"
  grep -m1 "RCCL version" "$REPO/C_nccldebug.log" | sed 's/^/  /' || echo "  (no RCCL banner - PROBLEM)"
  grep -iE "Using network|NET/OFI|NET/Socket|Libfabric" "$REPO/C_nccldebug.log" | head -3 | sed 's/^/  /'
  echo
  echo "-- Kernel-level (rank 0) — grep for nccl / hipMemcpy in *.stats.csv --"
  for cfg in A B C D; do
    echo "  [$cfg]"
    grep -iE "nccl|allreduce|Memcpy" "$REPO/${cfg}.stats.csv" 2>/dev/null | head -5 | sed 's/^/    /' \
      || echo "    (no stats file)"
  done
} | tee "$S"

echo "[confirm] evidence -> $REPO"
