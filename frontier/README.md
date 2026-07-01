# Frontier RCCL-Allreduce Experiments

Port of the JLSE `grace_suli/omb` pipeline to OLCF Frontier (MI250X / Slingshot-11),
testing the MPICH RCCL allreduce backend (PR pmodels/mpich#7493) at scale.

**Deadline: 2026-07-31, no renewal.** 9,984 node-hours available (not the constraint —
calendar time is). Priority: comprehensive MAIN allreduce numbers first; ML proxy is a
stretch goal; all analysis/plots happen off-cluster later.

## The five configs compared (sweep 0 B → 1 GiB)

| ID | What | How it's selected | Launcher |
|----|------|-------------------|----------|
| A | MPICH CPU, host buffers | your MPICH, no `-d rocm` | mpiexec |
| B | MPICH CPU algo, device buffers | `MPIR_CVAR_DEVICE_COLLECTIVES=all` | mpiexec |
| **C** | **MPICH + RCCL backend** | `MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=ccl` + `DEVICE_COLLECTIVES=none` | mpiexec |
| D | Cray MPICH GPU-aware | `cray-mpich` + `MPICH_GPU_SUPPORT_ENABLED=1` | **srun** |
| E | pure RCCL ceiling | `rccl-tests all_reduce_perf` | mpiexec |

Your MPICH is built `--with-pm=hydra`, so A/B/C/E launch with `mpiexec -bootstrap slurm`
(matching your JLSE `run_*_multi.sh`). Only Cray MPICH (D) uses `srun`. GPU pinning for
all configs is done by `bind_frontier.sh`.

## Order of operations

```bash
# 1. Verify [verify OLCF] markers on a login node (module names, RCCL path, binding).
vim env.sh bind_frontier.sh

# 2. Build. CRITICAL PATH = build_mpich.sh. Run plainly so you SEE output (also tees to $WORK).
./build_mpich.sh          # your MPICH + RCCL over ch4:ofi/CXI   <-- highest risk
./build_osu.sh mine       # OSU vs your MPICH  (A/B/C)
./build_osu.sh cray       # OSU vs Cray MPICH  (D)
./build_rccl_tests.sh     # ceiling (E)
./build_bench.sh          # correctness validator (Tier 0)

# 3. CORRECTNESS GATE — must pass before any timed run:
sbatch validate.sbatch
#   -> tail the *.out: "ALL VALIDATION PASSED" across int/float/double for B & C,
#      and the corruption self-test correctly reports "Validation failed".

# 4. Smoke + binding check on 2 nodes:
sbatch -N 2 -q debug run_allreduce.sbatch
#   -> results/N2_job*/C_mpich_rccl.txt has sane latencies; 1-node intra-node BW
#      should approach XGMI peak. If not, the bind_frontier.sh GCD map is wrong.

# 5. Full comprehensive scaling study (the main result):
./submit_scaling.sh
```

## Key differences from the JLSE setup

| JLSE (grace_suli/omb) | Frontier (here) |
|---|---|
| `--with-device=ch4:ucx` + UCX (InfiniBand) | `--with-device=ch4:ofi` + libfabric **CXI** (Slingshot) |
| `--with-rccl-include/lib` from source RCCL build | same flags, RCCL from `rocm/6.3.1` module |
| clang + explicit UCX/HIP paths | Cray `cc`/`CC` wrappers (auto libfabric + gfx90a) |
| `mpiexec --hostfile hosts.txt -ppn 4` | `mpiexec -bootstrap slurm -ppn 8` (A/B/C/E); `srun` (D) |
| ROCm at `/soft/compilers/rocm/rocm-6.3.2` | `module load rocm/6.3.1` |
| `map_rank_to_gpu` (MV2 + CUDA_VISIBLE_DEVICES) | `bind_frontier.sh` (SLURM_LOCALID/MPI_LOCALRANKID + ROCR_VISIBLE_DEVICES, non-linear GCD map) |
| 4 GPUs/node | 8 GCDs/node |
| no vendor-MPI baseline | + config D (Cray MPICH GPU-aware) |

## ⚠️ Things that will bite you (in priority order)
1. **`build_mpich.sh` over CXI** — non-vendor MPICH on Slingshot via `ch4:ofi` + Cray
   `cc` wrappers + hydra. If it won't build/launch, this is why. Debug on 2 nodes first.
2. **GCD binding map** — `bind_frontier.sh` map `[4 5 2 3 6 7 0 1]` is `[verify OLCF]`.
   Wrong = silent 2-3x error. This is why step 4 checks intra-node bandwidth.
3. **Calendar** — hard stop 2026-07-31, no renewal. Do dev in `-q debug`; launch the big
   sweeps early so reruns fit before the cutoff.

All `[verify OLCF]` markers = confirm against docs.olcf.ornl.gov/systems/frontier_user_guide.html.
