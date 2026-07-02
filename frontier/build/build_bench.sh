#!/bin/bash
# build_bench.sh — build your correctness benchmark (grace_suli/benchmark/
# allreduce_benchmark_mpi.c). This is the Tier-0 validator: each rank fills its
# buffer with its rank id, allreduces SUM, and checks every element equals
# (P-1)*P/2. The SAME binary validates config B and config C — the RCCL backend
# is selected at RUNTIME via CVARs (see validate.sbatch), so one build covers both.
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../env.sh"
load_mine                       # your MPICH mpicc + ROCm hipcc + RCCL

# Source lives in the grace_suli checkout this script was cloned from.
REPO=$(cd "$HERE/../.." && pwd)    # .../grace_suli
BSRC=$REPO/benchmark
BENCH=$WORK/bench
mkdir -p "$BENCH"

# Compile the MPI validator directly (avoids the JLSE-hardcoded Makefile paths).
# Device buffers -> needs HIP; the allreduce itself goes through MPICH.
"$MPICC" -O2 -I"$ROCM_PATH/include" "$BSRC/allreduce_benchmark_mpi.c" \
  -L"$ROCM_PATH/lib" -lamdhip64 \
  -o "$BENCH/allreduce_benchmark_mpi"

echo "[build_bench] -> $BENCH/allreduce_benchmark_mpi"
