#!/bin/bash
# build_osu.sh — build OSU Micro-Benchmarks against a chosen MPI stack.
#
# Uses UPSTREAM OSU (complete, with full ROCm support). The grace_suli/omb fork
# is incomplete in this checkout (only Makefile.am files, no .c sources), and we
# don't need its custom bits: configs A-D just run `osu_allreduce -d rocm`, and
# the RCCL backend is picked by MPICH CVARs at RUNTIME, not by OSU code.
#
#     ./build_osu.sh mine     # -> $OSU_MINE  (configs A/B/C)
#     ./build_osu.sh cray     # -> $OSU_CRAY  (config D)
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"

STACK=${1:?usage: build_osu.sh <mine|cray>}
case "$STACK" in
  mine) load_mine; PREFIX=$OSU_MINE ;;
  cray) load_cray; PREFIX=$OSU_CRAY ;;
  *) echo "unknown stack: $STACK" >&2; exit 1 ;;
esac

OSU_VER=${OSU_VER:-7.5}
BASE=$WORK/osu-$STACK
SRC=$BASE/osu-micro-benchmarks-$OSU_VER
mkdir -p "$BASE"
if [ ! -d "$SRC" ]; then
  cd "$BASE"
  curl -L -O "https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-${OSU_VER}.tar.gz"
  tar xzf "osu-micro-benchmarks-${OSU_VER}.tar.gz"
fi
cd "$SRC"

./configure \
  CC="$MPICC" CXX="$MPICXX" \
  --enable-rocm --with-rocm="$ROCM_PATH" \
  --prefix="$PREFIX" \
  2>&1 | tee "$WORK/osu-$STACK-configure.log"

make -j16
make install
echo "[build_osu:$STACK] -> $PREFIX/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce"
