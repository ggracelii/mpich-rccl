#!/bin/bash
# build_rccl_tests.sh — config E: the pure-RCCL ceiling (no MPI overhead).
# rccl-tests is the AMD analog of nccl-tests; all_reduce_perf is the reference.
# (Your grace_suli/benchmark/allreduce_benchmark_rccl.c does the same job; this
#  is the standard tool reviewers expect for the ceiling number.)
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
load_mine   # ROCm + RCCL (from module) + an MPI launcher for the multi-node driver

if [ ! -d "$RCCL_TESTS" ]; then
  git clone https://github.com/ROCm/rccl-tests.git "$RCCL_TESTS"
fi
cd "$RCCL_TESTS"
# MPI=1 lets one all_reduce_perf span multiple nodes via MPI bootstrap.
# RCCL ("NCCL_HOME") comes from the rocm module, same as the MPICH build.
make MPI=1 MPI_HOME="$MPICH_MINE" HIP_HOME="$ROCM_PATH" NCCL_HOME="$ROCM_PATH" -j16
echo "[build_rccl_tests] -> $RCCL_TESTS/build/all_reduce_perf"
