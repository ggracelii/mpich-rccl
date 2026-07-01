#!/bin/bash
# build_mpich.sh — build YOUR MPICH (PR pmodels/mpich#7493, fork grace_mpich)
# with the RCCL allreduce backend, on Frontier, over Slingshot-11.
#
# THIS IS THE CRITICAL-PATH / HIGHEST-RISK STEP. A non-vendor MPICH must talk to
# Slingshot through the libfabric CXI provider (ch4:ofi) and must speak a PMI
# that Slurm's srun provides (pmix). If this doesn't build+run on 2 nodes, the
# whole comparison is blocked — do it first, in the debug queue.
#
# Usage:  ./build_mpich.sh            # clone (if needed) + configure + make + install
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
load_mine

SRC=$WORK/mpich-rccl/src
mkdir -p "$(dirname "$SRC")"

# 1) Source: your fork with the RCCL backend (submodule of grace_suli).
if [ ! -d "$SRC" ]; then
  git clone --recurse-submodules https://github.com/ggracelii/grace_mpich.git "$SRC"
fi
cd "$SRC"
git submodule update --init --recursive
./autogen.sh                                  # generates configure from git checkout

# 2) libfabric with the CXI provider — the Slingshot-11 transport on Frontier.
#    Use the SYSTEM cray libfabric, not MPICH's embedded copy.
LIBFABRIC_DIR=${OLCF_LIBFABRIC_ROOT:-/opt/cray/libfabric/$CRAY_LIBFABRIC_VERSION}  # [verify OLCF]

# 3) Configure.
#    --with-device=ch4:ofi + --with-libfabric  -> Slingshot via CXI
#    --with-hip / gfx90a                        -> MI250X device buffers
#    --with-rccl                                -> YOUR allreduce backend (PR #7493)
#    --with-pmi=pmix                            -> launchable by `srun --mpi=pmix`
#    CC=cc CXX=CC                               -> Cray wrappers pull in craype flags
mkdir -p "$SRC/build" && cd "$SRC/build"
../configure \
  --prefix="$MPICH_MINE" \
  --with-device=ch4:ofi \
  --with-libfabric="$LIBFABRIC_DIR" \
  --with-hip="$ROCM_PATH" \
  --with-rccl="$RCCL_BASE" \
  --with-pmi=pmix --with-pmix=/usr \
  --enable-fast=O2 --disable-fortran \
  CC=cc CXX=CC \
  2>&1 | tee "$WORK/mpich-configure.log"

# 4) Build + install.
make -j$(nproc)   2>&1 | tee "$WORK/mpich-make.log"
make install      2>&1 | tee "$WORK/mpich-install.log"

echo "[build_mpich] installed to $MPICH_MINE"
"$MPICH_MINE/bin/mpichversion" || true
