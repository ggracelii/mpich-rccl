# RCCL backend for MPICH `MPI_Allreduce`

Reference copies of the source files I contributed to MPICH to add an RCCL
(AMD GPU) backend for `MPI_Allreduce`, merged upstream in
[pmodels/mpich#7493](https://github.com/pmodels/mpich/pull/7493).

These are **read-only reference copies** — kept here so the backend is browsable
in this repo (and counts toward its language stats). The canonical, buildable
versions live in the `mpich/` submodule and in upstream MPICH; build from there,
not from this folder.

## Files

- `mpir_cclcomm.h` — CCL communicator type and the CCL collective interface.
- `nccl.c` — NCCL/RCCL wrapper: comm bootstrap, datatype/reduction-op mapping,
  the device-side allreduce call. NCCL symbol names are retained because RCCL is
  API-compatible with NCCL.
- `rccl.c` — RCCL-specific glue (HIP stream/device management, ported from CUDA).
- `allreduce_intra_ccl.c` — the `MPI_Allreduce` hook that dispatches to the CCL
  backend (`MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=ccl`, `MPIR_CVAR_ALLREDUCE_CCL=rccl`).

See `../frontier/docs/NOTES.md` for the build recipe and the Frontier evaluation.
