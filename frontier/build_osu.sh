#!/bin/bash
# build_osu.sh — build the OSU Micro-Benchmarks (your grace_suli/omb fork, which
# has the ROCm/RCCL additions) against a chosen MPI stack.
#
# We build TWICE, into separate prefixes, so config C (your MPICH) and config D
# (Cray MPICH) each get an osu_allreduce linked against the right MPI:
#     ./build_osu.sh mine     # -> $OSU_MINE  (configs A/B/C)
#     ./build_osu.sh cray     # -> $OSU_CRAY  (config D)
#
# Compile flags ported verbatim from grace_suli/omb/build.sh.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"

STACK=${1:?usage: build_osu.sh <mine|cray>}
case "$STACK" in
  mine) load_mine; PREFIX=$OSU_MINE ;;
  cray) load_cray; PREFIX=$OSU_CRAY ;;
  *) echo "unknown stack: $STACK" >&2; exit 1 ;;
esac

SRC=$WORK/osu-$STACK/src
if [ ! -d "$SRC" ]; then
  git clone https://github.com/ggracelii/grace_suli.git "$WORK/grace_suli" 2>/dev/null || true
  cp -r "$WORK/grace_suli/omb" "$SRC"
fi
cd "$SRC"
[ -x ./configure ] || ./autogen.sh 2>/dev/null || autoreconf -fi

# ROCm/RCCL compile flags (from grace_suli/omb/build.sh). For -d rocm allreduce
# only --enable-rocm is strictly required; the RCCL defines are kept for the
# xccl variants you added.
export CPPFLAGS="-DENABLE_CCLCOMM -DENABLE_RCCL -D_ENABLE_ROCM -DROCM_ENABLED=1 -I${ROCM_PATH}/include"
export LDFLAGS="-L${ROCM_PATH}/lib"
export LIBS="-lamdhip64"

./configure \
  CC="$MPICC" CXX="$MPICXX" \
  --enable-rocm --with-rocm="$ROCM_PATH" \
  --prefix="$PREFIX" \
  2>&1 | tee "$WORK/osu-$STACK-configure.log"

make -j$(nproc)
make install
echo "[build_osu:$STACK] -> $PREFIX/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce"
