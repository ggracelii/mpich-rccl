#!/bin/bash
#SBATCH -A csc678
#SBATCH -J rccl-validate
#SBATCH -N 2
#SBATCH -t 00:15:00
#SBATCH -q debug
#SBATCH -o %x-%j.out
# validate.sbatch — Tier-0 CORRECTNESS GATE. Run this ONCE and confirm it passes
# before spending any node-hours on timed runs. Uses your correctness benchmark
# (allreduce_benchmark_mpi: fills each rank with its id, allreduce-SUM, checks
# every element == (P-1)*P/2). Validates BOTH the CPU-device path (config B) and
# the RCCL backend (config C) — same binary, backend chosen by CVARs — across
# int/float/double. Then a --corrupt self-test proves the checker actually
# detects bad data. Exits non-zero if anything is wrong.
#
#   sbatch validate.sbatch
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
load_mine
BIND="$HERE/bind_frontier.sh"
BENCH=$WORK/bench/allreduce_benchmark_mpi
[ -x "$BENCH" ] || { echo "missing $BENCH — run ./build_bench.sh first"; exit 1; }

NRANKS=$((SLURM_JOB_NUM_NODES * 8))
MPIEXEC="mpiexec -bootstrap slurm -n $NRANKS -ppn 8 -bind-to numa"
LOG=$WORK/results/validate_job${SLURM_JOB_ID}.log
mkdir -p "$(dirname "$LOG")"
fail=0

run_case() {  # $1=label  $2=bytes  $3=type-flag  $4=cvars(as string)
  echo "=== $1 (bytes=$2 $3) ===" | tee -a "$LOG"
  out=$(env $4 $MPIEXEC "$BIND" "$BENCH" "$2" $3 2>&1); echo "$out" >> "$LOG"
  if echo "$out" | grep -q "Validation passed"; then
    echo "  PASS $1" | tee -a "$LOG"
  else
    echo "  FAIL $1 (expected pass)" | tee -a "$LOG"; fail=1
  fi
}

B_CVARS="MPIR_CVAR_DEVICE_COLLECTIVES=all"
C_CVARS="MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=ccl MPIR_CVAR_ALLREDUCE_CCL=rccl MPIR_CVAR_DEVICE_COLLECTIVES=none"

for BYTES in 1024 1048576; do
  for T in --int --float --double; do
    run_case "B_mpich_dev"  "$BYTES" "$T" "$B_CVARS"
    run_case "C_mpich_rccl" "$BYTES" "$T" "$C_CVARS"
  done
done

# --- checker self-test: corruption MUST be detected (expect a FAILED result) --
echo "=== corruption self-test (expect 'Validation failed') ===" | tee -a "$LOG"
out=$(env $C_CVARS $MPIEXEC "$BIND" "$BENCH" 1048576 --float --corrupt 2>&1); echo "$out" >> "$LOG"
if echo "$out" | grep -q "Validation failed"; then
  echo "  PASS checker-detects-corruption" | tee -a "$LOG"
else
  echo "  FAIL checker did NOT detect corruption — validation is unreliable" | tee -a "$LOG"; fail=1
fi

echo "-----"; [ "$fail" -eq 0 ] && echo "ALL VALIDATION PASSED -> safe to run timed sweeps" \
  || echo "VALIDATION PROBLEMS — do NOT trust timed numbers until fixed"
echo "log: $LOG"
exit $fail
