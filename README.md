# MPICH-RCCL Integration

An RCCL backend for MPICH's `MPI_Allreduce` (contributed to upstream MPICH,
PR [pmodels/mpich#7493](https://github.com/pmodels/mpich/pull/7493)) and its
performance evaluation at scale on OLCF Frontier (AMD MI250X, Slingshot-11).

## What this is

MPICH normally reduces GPU data on the CPU, staging buffers device↔host. This work
offloads `MPI_Allreduce` to RCCL so the reduction runs on the GPUs directly (XGMI
intra-node, libfabric/CXI inter-node). The current effort measures that backend on
Frontier against:

- **Cray MPICH** — the production GPU-aware MPI on Frontier,
- **MPICH's own CPU path** — host buffers, and device buffers staged through host,
- **pure RCCL** (`rccl-tests`) — a no-MPI ceiling,

sweeping message size (8 B → 1 GiB) and node count (1 → 4096) to map where the RCCL
backend wins, by how much, and how the crossover moves with scale.

## Where things are

- **`frontier/`** — the current Frontier evaluation (active work). See `frontier/README.md` for the full how-to.
  - `build/` — build MPICH+RCCL, OSU, rccl-tests, the validator on Frontier
  - `run/` — the A–E sweep job + `submit_scaling.sh`
  - `check/` — correctness + mechanism gates, queue/result monitors
  - `docs/` — `NOTES.md` (build recipe, fix log, findings), `ML_EXPERIMENT.md`
  - `plots/` — `plot.ipynb` (parses `results/` → latency / speedup / scaling / crossover)
  - `results/` — committed sweep data + mechanism-confirmation evidence
  - `out/` — captured job logs
- **`mpich/`** — submodule: MPICH fork with the RCCL backend (since merged upstream).
- **`benchmark/`** — small allreduce correctness/latency programs (C + HIP).
- **`omb/`** — OSU Micro-Benchmarks automation from earlier work.

## Key implementation changes - MPICH

**Switched from CUDA to HIP** (port to AMD GPUs):
- `cudaStreamCreate` → `hipStreamCreate`
- `cudaSetDevice` → `hipSetDevice`
- `cudaStreamSynchronize` → `hipStreamSynchronize`
- CUDA error types and pointer-attribute APIs replaced with HIP equivalents

**Switched from NCCL to RCCL** (GPU collective backend on AMD hardware):
- Retained NCCL constant/function names (`ncclRedOp_t`, `ncclDataType_t`, `ncclCommInitRank`, …) since RCCL is API-compatible with NCCL — avoids conditional compilation or wrapper macros for most symbols

**Preserved and adjusted the NCCL implementation:**
- Kept `nccl.c` with a minor fix: added a `break` to the `float16` switch case to avoid fall-through

(A later upstream-API change — `MPIR_Errflag_t` removal — is handled by building from upstream MPICH; see `frontier/docs/NOTES.md`.)

## Key notes - Frontier (in progress)

*Evaluation is ongoing — numbers below are preliminary (1–64 nodes measured; 128–4096 running).*

- **Build:** upstream MPICH built with **amdclang + system libfabric/CXI** (not the Cray `cc` wrapper, which links Cray MPI into it); `--disable-weak-symbols`; `module unload cray-mpich` at runtime so our `libmpi` isn't shadowed.
- **RCCL needs the OFI plugin:** without OLCF's `rccl-net-plugin` (aws-ofi-rccl), RCCL falls back to TCP and runs ~4× *slower* than Cray inter-node; with it (CXI), RCCL is competitive/faster.
- **GPU binding:** under Hydra (`mpiexec -bootstrap slurm`) pin GPUs via `MPI_LOCALRANKID`, not `SLURM_LOCALID` (which is 0 for every rank → RCCL "duplicate GPU" crash); single-node jobs also need `--network=single_node_vni` for CXI.
- **Intra-node (1 node, 8 GCDs):** the RCCL backend beats MPICH's CPU path by up to **~18× at 1 MiB**.
- **Inter-node is a crossover surface (message size × node count):** RCCL wins large messages (**~4× at 1 GiB**, holding across scale) but loses tiny messages to Cray; the crossover slides to smaller sizes as node count grows (~16 KiB at 2 nodes).

See `frontier/docs/NOTES.md` for the full build recipe, fix log, and evidence.

## Background

Originally developed during a 2025 SULI internship at Argonne National Laboratory; the
allreduce backend was merged into upstream MPICH and is now being evaluated at exascale
on Frontier. Full build recipe, fix log, and findings are in `frontier/docs/NOTES.md`.
