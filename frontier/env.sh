#!/bin/bash
# env.sh — central Frontier environment for the RCCL-allreduce experiments.
# Source this from every build/run script:  source "$(dirname "$0")/env.sh"
#
# Ported from grace_suli/omb/build.sh (JLSE). JLSE hard-coded ROCm at
# /soft/compilers/rocm/rocm-6.3.2; on Frontier we get ROCm + libfabric(CXI)
# from modules instead. Everything marked [verify OLCF] must be confirmed on a
# Frontier login node against the current user guide before the first run.

set -euo pipefail

# ---- Project / filesystem -------------------------------------------------
export PROJ=csc678                         # ends 2026-07-31 — watch the clock
export SCRATCH=/lustre/orion/$PROJ/scratch/$USER   # member scratch (confirmed present)
export WORK=$SCRATCH/rccl-frontier          # all builds + results live here
mkdir -p "$WORK"

# ---- Versions ------------------------------------------------------------
# JLSE used 6.3.2; Frontier doesn't have it — 6.3.1 is the closest available
# (6.4.0/6.4.1/6.4.2 and 7.x also present if 6.3.1 misbehaves).
export ROCM_VERSION=6.3.1

# ---- Install prefixes -----------------------------------------------------
export MPICH_MINE=$WORK/mpich-rccl/install   # config C: your MPICH+RCCL build
export RCCL_BASE=$WORK/rccl/build/release     # if building RCCL from source; else use module
export OSU_MINE=$WORK/osu-mine/install        # OSU built against YOUR MPICH
export OSU_CRAY=$WORK/osu-cray/install        # OSU built against Cray MPICH
export RCCL_TESTS=$WORK/rccl-tests            # config E ceiling

# ---- Frontier module base (shared by all configs) ------------------------
# GCD arch is gfx90a (MI250X). craype-accel-amd-gfx90a wires GPU-aware flags
# into the cc/CC compiler wrappers and is required for GPU-aware Cray MPICH.
load_base_modules() {
  module reset                                        # [verify OLCF]
  module load PrgEnv-gnu                               # or PrgEnv-amd
  module load rocm/${ROCM_VERSION}
  module load craype-accel-amd-gfx90a
  module load libfabric                                # provides the CXI provider
  export ROCM_PATH=${ROCM_PATH:-$OLCF_ROCM_ROOT}       # set by the rocm module [verify OLCF]
}

# ---- Config C/B/A: YOUR MPICH ---------------------------------------------
load_mine() {
  load_base_modules
  export PATH=$MPICH_MINE/bin:$PATH
  export LD_LIBRARY_PATH=$MPICH_MINE/lib:$RCCL_BASE:$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}
  export MPICC=$MPICH_MINE/bin/mpicc
  export MPICXX=$MPICH_MINE/bin/mpicxx
  export FI_PROVIDER=cxi                                # force Slingshot provider
}

# ---- Config D: Cray MPICH (production baseline) ---------------------------
# Uses the vendor MPI already on the system — just modules + GPU-support flag.
load_cray() {
  load_base_modules
  module load cray-mpich                                # [verify OLCF: usually default]
  export MPICC=cc
  export MPICXX=CC
  export MPICH_GPU_SUPPORT_ENABLED=1                    # enable GPU-aware Cray MPICH
}

echo "[env] PROJ=$PROJ WORK=$WORK ROCM=$ROCM_VERSION" >&2
