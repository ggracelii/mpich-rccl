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

## Full sweep — 1–4096 nodes (jobs 4934xxx + reruns; 3 reps, 2 at 4096)
Ran the full ladder 1→4096. Two hard limits surfaced, both worth recording as methods/results.

### OSU caps at 1 GiB (32-bit message size)
Tried to extend the GPU range to 4 GiB (`GPU_MAX=4 GiB`, `-m 0:4294967296`), but OSU silently
stops at **1 GiB**: its message-size loop variable is a 32-bit `int`, so 2 GiB (2^31) overflows
and the loop exits at 1 GiB. Confirmed — every `C_mpich_rccl.txt` ends at exactly `1073741824`,
no error and no 2 GiB row. Measuring >1 GiB would need OSU patched to 64-bit sizes and rebuilt.
Decision: **cap the study at 1 GiB** (real allreduce workloads rarely exceed ~1 GiB; the size
trend is already clear across ~8 orders of magnitude). Note: Slurm also snapshots the batch
script at *submit* time, so the 4 GiB `GPU_MAX` edit didn't take until a resubmit — irrelevant
once the OSU cap was understood.

### Cray MPICH GPU allreduce FAULTS above 4 MiB at ≥1024 nodes — the headline scale result
At **≥1024 nodes, Cray MPICH's default GPU-aware `MPI_Allreduce` (config D) crashes with a GPU
memory access fault for messages larger than 4 MiB**, while the MPICH-RCCL backend (C) completes
the full 8 B→1 GiB range at every scale up to 4096 nodes.
- Last size Cray completes: **4 MiB** (`4194304`); the 8 MiB op faults. **Uniform cutoff — the
  same 4 MiB at 1024, 2048, and 4096** (not scale-dependent). Cray reaches **1 GiB** cleanly at
  ≤512 nodes, so the fault threshold sits **between 512 and 1024 nodes**.
- Error (config D, N=2048, job 4934659): `Memory access fault by GPU ... on address
  0x7ffb89e00000. Reason: Unknown.` → `task 8191: Aborted (core dumped)` →
  `STEP 4934659.3 CANCELLED ... DUE TO TASK FAILURE`. (These crashes produced the `gpucore.*`
  dumps — now gitignored.)
- **Not a harness/binding artifact:** config C runs the *same* OSU program, buffers, and GPU
  binding to 1 GiB without fault; the fault is inside Cray's large-message GPU allreduce path
  (a host-range address touched by the GPU → a pointer not device-accessible at scale).
- **Not a walltime artifact:** the job `COMPLETED` in ~2:48 (D died fast, the script continued) —
  a crash, not a timeout. Reproducible across all ≥1024-node reps.
- Scope: this is the **production default** GPU-aware Cray (`MPICH_GPU_SUPPORT_ENABLED=1`); the
  RCCL comparison uses the identical harness — a fair comparison. (Untested: whether a Cray
  algorithm CVAR dodges the fault; the default-config result stands on its own.)

**Dataset consequence:** RCCL (C) is complete 8 B→1 GiB at all scales; Cray (D) is complete to
1 GiB at ≤512 nodes but only to 4 MiB at ≥1024 (Cray faults beyond). The C-vs-Cray crossover at
≥1024 nodes is bounded by where Cray survives.

**Rep status (final):** **3 reps at every node count, 1–4096.** **Top scale = 4096 nodes = 32,768 GCDs.**
All kept reps have complete RCCL (C) data to 1 GiB; jobs that hung during config C or at startup were discarded.
The 3rd 4096 rep (job 4939751) is RCCL-complete to 1 GiB; its Cray (D) reached only 2 MiB (Cray faulted *at*
4 MiB that allocation vs 8 MiB in the other two — the fault threshold varies around ~4 MiB). **8192 not pursued**
(comm-init wall + budget — see below). The loader averages over whatever reps exist per size and records `nreps`.

### 8192 attempted, abandoned (near-full-machine limit)
Both 8192-node jobs (65,536 GCDs) timed out at 12 min. Config C printed its OSU header but completed
**zero sizes** — the *first* allreduce (8 B) stalled in **RCCL communicator init across 65,536 GCDs** and
never returned. Two compounding causes:
- **CPU baselines starve the fast config.** Run order is A→B→C→D, and at 65,536 ranks the CPU allreduce is
  catastrophic (**config B = 445 ms/op at 32 MiB**), so A+B consumed much of the walltime before C ran.
- **RCCL comm-init does not complete in a practical walltime at 65,536 GCDs.** A run long enough for C to
  finish would have exceeded the remaining csc678 allocation, so 8192 was dropped.
- **Lesson if ever retried:** run **C-only** (or C first) at extreme scale so the RCCL bootstrap gets the full
  walltime — don't let the slow CPU baselines run ahead of it. Each 8192 timeout costs ~1,640 node-hrs.
This is itself a result: the RCCL backend completes 8 B→1 GiB at 4096 nodes, but its communicator bootstrap
hits a practical wall at 8192 (near-full Frontier) within a 12-min budget.

### Giant-scale run hygiene (lessons)
- Real run times are tiny (~2 min ≤1024, ~3–4 min at 2048/4096); a hung/faulting job otherwise
  burns to the walltime. Set the walltime to **8 min for ≥512 nodes** so a stall dies fast
  (a 4096 hang at 8 min ≈ 550 node-hrs vs ~2000 at 30 min). ~3,400 node-hrs were lost to hangs
  before this cap — watch the giants and `scancel` a job sitting past ~5 min.
- **Submit from `frontier/`.** `run_allreduce.sbatch` sources `$SLURM_SUBMIT_DIR/env.sh`;
  submitting from elsewhere ran with `$WORK` unset (results → `/results`, every config failed
  in ~5 s but exited 0). Hardened: the sbatch now aborts loudly if `env.sh`/`WORK` don't load,
  and `submit_scaling.sh` exports `FRONTIER_HOME` so submit location no longer matters.

## `[verify OLCF]` items still to confirm
- GCD→rank map in `bind_frontier.sh` `[4 5 2 3 6 7 0 1]` (validate via intra-node bandwidth).
- Config D `--gpu-bind=closest -c7` recipe.

## How to run (all commands from the `frontier/` directory)
```
./build/build_mpich.sh
./build/build_osu.sh mine ; ./build/build_osu.sh cray
./build/build_rccl_tests.sh ; ./build/build_bench.sh
sbatch check/validate.sbatch             # correctness gate
sbatch check/confirm.sbatch              # mechanism check (NCCL_DEBUG + rocprof)
sbatch -N 2 run/run_allreduce.sbatch     # smoke
./run/submit_scaling.sh                  # sweep 1-64 (default ladder)
./run/submit_scaling.sh "128 256 512 1024"
./run/submit_scaling.sh "2048 4096"      # REPS=n overrides the default 3 reps
watch -n 30 ./check/monitor.sh           # live health ; ./check/check_results.sh to scan
```

## Results provenance
- Authoritative data = runs made with the FINAL config (upstream mpich + amdclang build,
  MPI_LOCALRANKID binding, rccl-net-plugin, single_node_vni, 10/50 iters). The scaling sweep
  produces the clean matched dataset.
- Early exploratory jobs (FAILED / CANCELLED / TIMEOUT / pre-plugin RCCL-over-TCP) are the
  debugging journey above — NOT valid comparison data; do not analyze them as results.
