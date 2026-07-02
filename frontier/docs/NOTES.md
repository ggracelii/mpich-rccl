# Frontier RCCL-Allreduce — Working Notes

Running log of setup, fixes, findings, and how-to for evaluating the MPICH RCCL
allreduce backend (PR pmodels/mpich#7493, merged upstream) on OLCF Frontier.
Doubles as the reproducibility / methods record for the paper.

## Goal
Measure the MPICH RCCL allreduce backend vs alternatives on Frontier
(MI250X / Slingshot-11), across message sizes and node counts.

## System & constraints
- Frontier: 9,408 nodes; per node 1× EPYC "Trento" + 4× MI250X = **8 GCDs**, 4× Slingshot NICs.
- Project **csc678**: ~9,984 node-hours, **ends 2026-07-31 (no renewal)**. graceli is the only active user.
- ROCm **6.2.4** (must match PrgEnv-amd's amd module), MPICH **upstream main** (5.1.0a1, commit 1524d1c), libfabric **2.3.1** (system, CXI), RCCL **2.20.5**, rccl-net-plugin (aws-ofi-nccl 1.19.2).

## Configs under test
| | Config | Buffers | Mechanism | Selected by |
|---|---|---|---|---|
| A | MPICH CPU, host | host | CPU allreduce, no GPU | your MPICH, no `-d rocm` |
| B | MPICH CPU, device | device | CPU algo + host staging (GPU-aware, NOT accelerated) | `MPIR_CVAR_DEVICE_COLLECTIVES=all` |
| **C** | **MPICH + RCCL** | device | RCCL on GPUs (XGMI intra, CXI inter) — **the contribution** | `ALLREDUCE_INTRA_ALGORITHM=ccl` + `ALLREDUCE_CCL=rccl` + `DEVICE_COLLECTIVES=none` |
| D | Cray MPICH GPU-aware | device | vendor production MPI | `cray-mpich` + `MPICH_GPU_SUPPORT_ENABLED=1` |
| E | pure rccl-tests | device | RCCL, no MPI — ceiling | `all_reduce_perf` |

## Build recipe (config C) — the working configuration
Built with `build_mpich.sh`. Key choices (each forced by a failure below):
- Source: **upstream pmodels/mpich** (not the fork — fork's main is stale/half-migrated).
- Compilers: **amdclang / amdclang++** directly (NOT the Cray `cc` wrapper).
- `--with-device=ch4:ofi --with-libfabric=/opt/cray/libfabric/2.3.1` (system CXI, not embedded).
- `--with-rccl-include/lib` from the rocm module; `--offload-arch=gfx90a`; `--with-pm=hydra`.
- `--disable-weak-symbols` (amdclang/lld weak-alias issue).
- Runtime: `module unload cray-mpich` so our libmpi isn't shadowed.

## Fix chronology (what broke → the fix)
1. **rocm/6.3.2 not on Frontier** → use 6.3.1, then **6.2.4** (rocm must match PrgEnv-amd's `amd/6.2.4`).
2. **"C++ compiler does not work"** — PrgEnv-gnu's g++ rejects `--offload-arch` → **PrgEnv-amd** (amdclang++).
3. **`unknown type MPIR_Errflag_t` / `undeclared errflag`** — the fork lagged an upstream API change → **build upstream mpich** (PR is merged there, API-consistent).
4. **mpichversion says "CRAY MPICH" + libmpi_amd baked in** — the `cc` wrapper linked Cray MPI into our libmpi → build with **amdclang + explicit system libfabric**, no cc wrapper.
5. **`undefined symbol MPI_Get_library_version`** — amdclang/lld weak-alias failure → **`--disable-weak-symbols`**.
6. **OSU won't configure (missing `osu_latency.c`)** — the grace_suli/omb fork is incomplete → **upstream OSU 7.5** (RCCL selected by CVARs at runtime, so fork not needed).
7. **`env.sh: No such file` in sbatch** — `dirname $0` points to spool dir → use **`$SLURM_SUBMIT_DIR`**.
8. **`OFI opendomain: Function not implemented` on 1 node** — single-node jobs get no Slingshot VNI → **`#SBATCH --network=single_node_vni`**.
9. **RCCL "Duplicate GPU detected" segfault** — `bind_frontier.sh` read `SLURM_LOCALID`, which is 0 for all ranks under `mpiexec -bootstrap slurm` (Hydra forks from one per-node srun proxy) → bind uses **`MPI_LOCALRANKID`**.
10. **RCCL 4× SLOWER than Cray inter-node** — RCCL fell back to TCP (`librccl-net.so` missing) → load **`rccl-net-plugin`** (aws-ofi-rccl) so RCCL uses CXI. Loaded for C/E only (keeps A/B clean baselines).
11. **Walltime blowout** — CPU allreduce is ~6 s/op at 512 MiB, and `-i 1000` to 1 GiB never finished → **cap CPU (A,B) at 32 MiB**, GPU (C,D,E) to 1 GiB; **10 warmup / 50 iters** uniform.

## Key findings so far
- **Intra-node (1 node, 8 GCDs):** RCCL vs MPICH-CPU (B) — up to **~18.5× faster** at 1 MiB (126 µs vs 2340 µs), scaling from ~2.4× at 8 B.
- **Inter-node (2 nodes) with the OFI plugin:** RCCL (C) vs Cray (D):
  - Cray wins tiny messages (RCCL fixed kernel/launch overhead); **crossover ≈16 KiB**.
  - RCCL wins above: 1 MiB 1.25×, 16 MiB 2.2×, 256 MiB 3.8×, **1 GiB ~4×** (19.2 ms vs 77.9 ms).
- **The OFI plugin is essential**: without `rccl-net-plugin`, RCCL uses TCP sockets and is ~4× *slower* than Cray inter-node (1 GiB: 308 ms → 19 ms with the plugin, a 16× jump).
- Correctness validated (int/float/double + corruption self-test) before timed runs.

## Mechanism confirmation (confirm.sbatch, job 4929423, 1 MiB, 2 nodes)
Verified each config does what it claims, not just that it produces numbers:
- **B ≠ RCCL**: NCCL_DEBUG "RCCL banner count = 0". Kernel trace: `B.stats.csv` empty (no GPU
  compute kernel); `B.copy_stats.csv` = CopyHostToDevice + CopyDeviceToHost → host-staged CPU
  reduction on device buffers, exactly as labeled.
- **C = RCCL over CXI**: NCCL_DEBUG shows `NET/OFI Initializing aws-ofi-nccl 1.19.2` +
  `Using Libfabric version 2.3`; kernel trace `ncclDevKernel_Generic` = 99% of GPU time.
- **A = host CPU**: only HIP-API calls, no GPU kernels.
- **D = Cray**: confirmed via `ldd` (links /opt/cray `libmpi_amd`); rocprof-under-srun hit a
  known ROCm-6.2 rocprof sqlite bug (tooling, not the config).
Evidence: `results/confirm_4929423/` (NCCL logs + rocprof stats + SUMMARY.txt).

## Preliminary results — 1–64 node sweep (jobs 4929427–4929448; 3 reps; 10 warmup/50 iters)
Clean run (no failures in any file). Key finding: the RCCL(C)-vs-Cray(D) winner depends on
BOTH message size and node count — a crossover *surface*, not a flat result. Avg latency (µs):

| size | N=1 C / D | N=8 C / D | N=64 C / D | crossover vs Cray |
|---|---|---|---|---|
| 1 GiB  | 13.7k / 66.5k | 25.9k / 105k | 36.5k / 126k | **none — RCCL wins 3.5–4.8× at all scales** |
| 16 MiB | 303 / 1029 | 941 / 1584 | 1953 / 1863 | Cray overtakes ~64 nodes |
| 1 MiB  | 126 / 158 | 385 / 309 | 613 / 410 | Cray overtakes ~4 nodes |

Monotonic: the smaller the message, the fewer nodes before Cray overtakes. Config B (MPICH
CPU-on-device) is flat-slow (~2.2–2.8 ms at 1 MiB, all scales). Interpretation: RCCL is the
**large-message / bandwidth champion** (XGMI + tuned rings; ~4× at 1 GiB, holding across
scale), while Cray-MPICH has **lower fixed/latency overhead and scales better for small–mid
messages**. The 128–1024 sweep will show where the crossover contour lands at larger scale.

## `[verify OLCF]` items still to confirm
- GCD→rank map in `bind_frontier.sh` `[4 5 2 3 6 7 0 1]` (validate via intra-node bandwidth).
- Config D `--gpu-bind=closest -c7` recipe.

## How to run
```
./build_mpich.sh ; ./build_osu.sh mine ; ./build_osu.sh cray ; ./build_rccl_tests.sh ; ./build_bench.sh
sbatch validate.sbatch            # correctness gate
sbatch confirm.sbatch             # mechanism check (NCCL_DEBUG + rocprof)
sbatch -N 2 run_allreduce.sbatch  # smoke
./submit_scaling.sh               # main sweep 1->1024 (x3/2/1 reps)
./submit_scaling.sh "2048 4096"   # big scaling points (after reviewing)
```

## Results provenance
- Authoritative data = runs made with the FINAL config (upstream mpich + amdclang build,
  MPI_LOCALRANKID binding, rccl-net-plugin, single_node_vni, 10/50 iters). The scaling sweep
  produces the clean matched dataset.
- Early exploratory jobs (FAILED / CANCELLED / TIMEOUT / pre-plugin RCCL-over-TCP) are the
  debugging journey above — NOT valid comparison data; do not analyze them as results.
