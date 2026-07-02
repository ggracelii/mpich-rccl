#!/bin/bash
# build_mpich.sh — build YOUR MPICH (grace_mpich, PR pmodels/mpich#7493) with the
# RCCL allreduce backend on Frontier.
#
# Ported from grace_mpich/build.sh (JLSE). The JLSE build used ch4:ucx + UCX
# (InfiniBand). Frontier's Slingshot-11 has no native UCX transport, so the ONE
# structural change is the transport: ch4:ofi + libfabric CXI. The Cray compiler
# wrappers (cc/CC) inject libfabric+CXI and the gfx90a GPU flags automatically —
# so, unlike the JLSE build, we set NO explicit UCX/libfabric paths. RCCL comes
# from the rocm module. Everything else (RCCL flags, offload-arch, -DENABLE_RCCL,
# -lrccl -lamdhip64, hydra, yaksa=embedded) matches your working build.
#
# CRITICAL PATH: prove this launches on 2 nodes (debug queue) before scaling.
# Run it plainly so you SEE output live (it also tees to $WORK/mpich-*.log):
#     ./build_mpich.sh
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../env.sh"
load_mine                       # modules + ROCM_PATH + RCCL_INC/RCCL_LIB

GFX_ARCH=gfx90a                 # MI250X

# Build with amdclang DIRECTLY, not the Cray `cc` wrapper. cc links Cray MPICH
# (libmpi_amd / GTL) into our libmpi so Cray's version symbols win and every run
# would secretly be Cray MPICH. amdclang + explicit SYSTEM libfabric (which
# carries the Slingshot CXI provider; embedded libfabric may not) gives a clean,
# self-contained MPICH. Mirrors your JLSE clang build.
CC_BIN=$ROCM_PATH/llvm/bin/amdclang
CXX_BIN=$ROCM_PATH/llvm/bin/amdclang++
LIBFABRIC_DIR=$(pkg-config --variable=prefix libfabric 2>/dev/null || true)
[ -n "$LIBFABRIC_DIR" ] || LIBFABRIC_DIR=/opt/cray/libfabric/2.3.1   # [verify OLCF]
echo "[build_mpich] CC=$CC_BIN  libfabric=$LIBFABRIC_DIR"

# 1) Source: UPSTREAM MPICH. Your RCCL allreduce backend (PR #7493) was merged
#    into pmodels/mpich main on 2025-09-16 (commit d8176a3), so upstream has an
#    API-consistent version. Building the fork instead fails: grace_mpich/main is
#    stale/half-migrated (errflag drift). Pin MPICH_REF for reproducibility.
MPICH_REF=${MPICH_REF:-main}
SRC=$WORK/mpich-upstream/src
mkdir -p "$(dirname "$SRC")"
if [ ! -d "$SRC/.git" ]; then
  git clone --recurse-submodules https://github.com/pmodels/mpich.git "$SRC"
fi
cd "$SRC"
git fetch --tags origin
git checkout "$MPICH_REF"
git submodule update --init --recursive
git rev-parse HEAD | tee "$WORK/mpich-commit.txt"      # exact source, for the paper
[ -x ./configure ] || ./autogen.sh 2>&1 | tee "$WORK/mpich-autogen.log"

# 2) Confirm RCCL is actually in the rocm module before configuring.
if [ ! -f "$RCCL_INC/rccl.h" ]; then
  echo "ERROR: rccl.h not found at $RCCL_INC — check that the rocm module ships RCCL"
  echo "  try: find \$ROCM_PATH -name rccl.h" ; exit 1
fi

# 3) Configure (out-of-tree). cc/CC == Cray wrappers -> libfabric+CXI+gfx90a.
mkdir -p "$SRC/build" && cd "$SRC/build"
../configure \
  --prefix="$MPICH_MINE" \
  --with-device=ch4:ofi \
  --with-libfabric="$LIBFABRIC_DIR" \
  --with-hip="$ROCM_PATH" \
  --with-rccl-include="$RCCL_INC" \
  --with-rccl-lib="$RCCL_LIB" \
  --with-pm=hydra \
  --with-yaksa=embedded \
  --with-ch4-shmmods=posix \
  --enable-fast=all,O2 \
  --disable-fortran \
  --disable-weak-symbols \
  CC="$CC_BIN" CXX="$CXX_BIN" HIPCC="$ROCM_PATH/bin/hipcc" \
  CXXFLAGS="--offload-arch=$GFX_ARCH" \
  HIPCCFLAGS="--offload-arch=$GFX_ARCH" \
  CPPFLAGS="-DENABLE_CCLCOMM -DENABLE_RCCL -I$ROCM_PATH/include -I$RCCL_INC" \
  CFLAGS="-I$ROCM_PATH/include" \
  LDFLAGS="-L$RCCL_LIB -L$ROCM_PATH/lib -Wl,-rpath,$RCCL_LIB -Wl,-rpath,$ROCM_PATH/lib" \
  LIBS="-lrccl -lamdhip64" \
  2>&1 | tee "$WORK/mpich-configure.log"

# 4) Build + install. -j16 (not nproc) to be a good citizen on the shared login
#    node; move to an salloc compute node if OLCF throttles it.
make -j16    2>&1 | tee "$WORK/mpich-make.log"
make install 2>&1 | tee "$WORK/mpich-install.log"

echo "[build_mpich] installed -> $MPICH_MINE"
"$MPICH_MINE/bin/mpichversion" || true
