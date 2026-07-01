# Frontier RCCL-Allreduce Experiments

Port of the JLSE `grace_suli/omb` pipeline to OLCF Frontier (MI250X / Slingshot-11),
testing the MPICH RCCL allreduce backend (PR pmodels/mpich#7493) at scale.

**Drop this folder into `grace_suli/frontier/`** (same submodule, same benchmarks).

## The five configs compared (all sweep 0 B → 1 GiB, `-d rocm`, fp32)

| ID | What | How it's selected |
|----|------|-------------------|
| A | MPICH CPU, host buffers | your MPICH, no `-d rocm` |
| B | MPICH CPU algo, device buffers | `MPIR_CVAR_DEVICE_COLLECTIVES=all` |
| **C** | **MPICH + RCCL backend** | `MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=ccl` + `DEVICE_COLLECTIVES=none` |
| D | Cray MPICH GPU-aware | `cray-mpich` + `MPICH_GPU_SUPPORT_ENABLED=1` |
| E | pure RCCL ceiling | `rccl-tests all_reduce_perf` |

## Order of operations

```bash
# 0. Confirm the allocation actually has hours (ends 2026-07-31!)
showusage

# 1. Edit env.sh — verify every [verify OLCF] path/module on a login node.
vim env.sh

# 2. Build (debug queue / login node). CRITICAL PATH = build_mpich.sh.
./build_mpich.sh          # your MPICH + RCCL over CXI  <-- highest risk
./build_osu.sh mine       # OSU vs your MPICH  (A/B/C)
./build_osu.sh cray       # OSU vs Cray MPICH  (D)
./build_rccl_tests.sh     # ceiling (E)

# 3. PROVE IT on 2 nodes before spending hours (correctness + nonzero inter-node BW):
sbatch -N 2 -q debug run_allreduce.sbatch
#   -> check results/N2_job*/C_mpich_rccl.txt has sane latencies

# 4. Validate GCD binding on 1 node (intra-node BW should hit XGMI peak).
#   If --gpu-bind=closest looks wrong, switch the SRUN line in run_allreduce.sbatch
#   to wrap ranks with ./bind_frontier.sh and re-check.

# 5. Full scaling study.
./submit_scaling.sh
```

## Key differences from the JLSE setup

| JLSE (grace_suli/omb) | Frontier (here) |
|---|---|
| `mpiexec --hostfile hosts.txt -ppn 4` (Hydra) | `srun --mpi=pmix -N.. -n.. --gpus-per-node=8` (Slurm) |
| ROCm at `/soft/compilers/rocm/rocm-6.3.2` | `module load rocm/6.3.2` |
| `map_rank_to_gpu` (MV2 localrank + CUDA_VISIBLE_DEVICES) | `bind_frontier.sh` (SLURM_LOCALID + ROCR_VISIBLE_DEVICES, non-linear GCD map) |
| 4 GPUs/node | 8 GCDs/node |
| no vendor-MPI baseline | + config D (Cray MPICH GPU-aware) |

## ⚠️ Things that will bite you (in priority order)
1. **`build_mpich.sh` over CXI** — non-vendor MPICH on Slingshot via `ch4:ofi` +
   libfabric CXI + `--mpi=pmix`. If it won't launch, this is why. Debug on 2 nodes first.
2. **GCD binding map** — `bind_frontier.sh` map is `[verify OLCF]`. Wrong = silent 2-3x error.
3. **Allocation clock** — CSC678 ends 2026-07-31. Do dev in `-q debug`; ask the PI about renewal.

All `[verify OLCF]` markers = confirm against docs.olcf.ornl.gov/systems/frontier_user_guide.html.
