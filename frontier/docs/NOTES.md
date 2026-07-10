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
Evidence: `archive/confirm_4929423/` (NCCL logs + rocprof stats + SUMMARY.txt).

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
  RCCL comparison uses the identical harness — a fair comparison.
- **Knob probe (job 4948927, 8 MiB @1024, all documented GPU-path tunings):** default, `IPC_ENABLED=0`,
  `COLL_STAGING_AREA_OPT=0`, `ALLREDUCE_USE_KERNEL=1`, `NO_ASYNC_COPY=1` all **FAULT**; only
  **`MPICH_GPU_ALLREDUCE_USE_KERNEL=0` survives** — the faulting component is Cray's GPU-kernel
  reduction path; disabling it falls back to a copy-based reduce. **The workaround costs 2.4×:**
  7,824 µs vs RCCL's 3,250 µs at 8 MiB/1024 nodes — and the gap explodes with size (kernel-off
  1 GiB @1024 = **1,000 ms vs RCCL 43.7 ms, 23×**). Kernel-off data (full ladder, 2 reps, sweep +
  ML sizes) lives in `results_crayknob/`, kept separate from default-D everywhere.
- **Probe 2 (job 4952482, kernel path ON, 1024 nodes): `MPICH_GPU_ALLREDUCE_BLK_SIZE=64MB`
  RESCUES the fast path.** The kernel path stages through a GPU-attached block buffer
  (`BLK_SIZE`, default 8 MB); the fault tracks the payload≥block boundary at small blocks
  (`BLK=4MB` → faults at 4 MiB; default 8 MB → faults at 8 MiB), yet **64 MB blocks (the man
  page's own suggested value for large payloads) complete the entire 8 B→4 GiB range**. Newer
  releases do NOT fix the default: **cray-mpich 8.1.32 and 9.0.0 both still fault at 8 MiB.**
- **Tuned Cray (BLK=64MB) vs RCCL at 1024 nodes:** the rescued kernel path is fast — it *beats*
  RCCL in the mid range (**8 MiB: 1,114 vs 3,253 µs = 2.9×; 64 MiB: 1.7×**) while **RCCL wins the
  large regime (1 GiB: 43.7 vs 94.5 ms = 2.2×; 4 GiB: 3.1×)** — crossover ≈128–512 MB. It is also
  ~25× faster than kernel-off, which is hereby obsolete as a workaround. Honest framing: *default
  production Cray crashes >4 MiB at ≥1024 nodes (unfixed through v9.0.0); a documented non-default
  tuning rescues it and is competitive to ~100 MB; RCCL needs no tuning, never crashes, and wins
  2–3× at the sizes where large-model gradients live (0.1–1.4 GB).*
- **BLK=64MB is Pareto-better than the default even where the default works** (measured 1–64
  nodes): identical below the 128 KiB kernel threshold, equal mid-range, and **up to 1.5×
  faster at ≥256 MB** (2.5× at 64 MiB/N=64) — plus it doesn't crash at scale. There is no
  size/scale where the default configuration is meaningfully better.
- **Plot convention (repo-wide):** "Cray MPICH" on all plots = the **BLK=64MB tuned config**
  (config T); the default config is retired from plots (explicitly callable, labeled
  "default 8MB†"). READMEs and the paper define this once; graph labels stay clean.
- **In flight:** tuned-Cray (BLK=64MB) full ladder **1→4096 nodes × 2 reps** (sweep 8 B→4 GiB +
  the 4 ML gradient sizes each) → `results_crayblk64/`, plotted as config **T** ("Cray
  (BLK=64 MB†)") with a `Dt` best-of-default/tuned composite heatmap. Open question the ladder
  answers: whether the BLK=64MB rescue holds at 2048/4096 (probe verified 1024 only — a
  truncated sweep at another scale would itself be a finding).

**Dataset consequence:** RCCL (C) is complete 8 B→1 GiB at all scales; Cray (D) is complete to
1 GiB at ≤512 nodes but only to 4 MiB at ≥1024 (Cray faults beyond). The C-vs-Cray crossover at
≥1024 nodes is bounded by where Cray survives.

**Rep status (final):** **3 reps at every node count, 1–4096.** **Top scale = 4096 nodes = 32,768 GCDs.**
All kept reps have complete RCCL (C) data to 1 GiB; jobs that hung during config C or at startup were discarded.
The 3rd 4096 rep (job 4939751) is RCCL-complete to 1 GiB; its Cray (D) reached only 2 MiB (Cray faulted *at*
4 MiB that allocation vs 8 MiB in the other two — the fault threshold varies around ~4 MiB). **8192: RCCL
init succeeds (~3 min); large-message flagship run in progress — see below.** The loader averages over whatever reps exist per size and records `nreps`.

### 8192 (65,536 GCDs) — NOT a comm-init wall; it's sweep length (corrected)
Earlier 8192 jobs (A→B→C→D order, or full C sweep) timed out, and we initially blamed **RCCL
communicator init**. That was WRONG. The C-only single-launch retry (job 4948914, 10-min cap)
**proved RCCL initializes fine at 65,536 GCDs**: init took ~3 min, then it produced real data —
8 sizes (4 B–512 B) before the cap. It timed out on **sweep length, not bootstrap**:
- **Init works** (~3 min at 8192) — the "wall" was a misdiagnosis from earlier runs where the
  slow CPU baselines (A→B) ate the walltime before C even started (config B = 445 ms/op at 32 MiB
  at this scale), so C's *init* never got a turn. C-only removes that.
- **Tiny sizes are sync-bound at 65,536 ranks** (~50 s each — the per-iteration barrier across
  65,536 ranks dominates), so a 50-iter full 4 B→4 GiB sweep can't finish in ~10 min.
- **Fix / flagship run:** `run_rccl_8192_big.sbatch` — C-only, **1 MiB→4 GiB only**, i=10, 15-min
  cap. Skips the sync-bound small sizes; captures the bandwidth regime where RCCL's advantage is
  the whole story. The 8 small-size rows from 4948914 are kept as a partial.
So the honest 8192 result is: **RCCL scales to 65,536 GCDs — init succeeds and it runs**; the only
constraint is that a full fine-grained sweep doesn't fit a short walltime at near-full-machine scale.

## 2–4 GiB extension (patched OSU) — the win grows with size
OSU's silent 1 GiB cap was a **32-bit `int` size loop** in `osu_allreduce.c` (`size *= 2`
overflows at 2^31 and exits); `-m` parsing was already 64-bit. Patched `size` (and
`print_stats`/`print_stats_validate`) to `size_t`, rebuilt both OSU stacks, and measured
**only the new points** (2 GiB, 4 GiB; plus BERT-fp32 1.36 GB for ML): they merge into the
existing dataset as extra rows. 4 GiB is the practical ceiling (2^30 4-byte elements just
fits MPI's 32-bit count). Findings (2 nodes; ladder in progress):
- **RCCL is bandwidth-flat past 1 GiB**: 2 GiB = 37.6 ms, 4 GiB = 74.5 ms (clean 2×/4× of 1 GiB).
- **Cray survives 2–4 GiB at small scale** (160/321 ms at 2 nodes) → its >4 MiB crash at ≥1024
  nodes is **scale-triggered, not size-triggered**.
- **RCCL vs Cray at 4 GiB: 4.3×** — the headline speedup grows with message size.

## csel: JSON-driven automatic backend selection (the summer "auto" TODO, closed)
MPICH's collective selection (csel) natively supports the CCL leaf:
`"algorithm=MPIR_Allreduce_intra_ccl": {"ccl=rccl": {}}`. We generated a tuning tree
(`tuning/allreduce_rccl_auto.json`, from `maint/tuning/coll/mpir/generic.json`) and pointed
`MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE` at it with `MPIR_CVAR_DEVICE_COLLECTIVES=none`.
- **v1 validation (2 nodes, job 4948886):** with a 64 KiB threshold, `auto` tracks the stock
  algorithms below the threshold and sits exactly on forced-RCCL above it — **47× at 256 MiB
  from a JSON file alone**, zero switching overhead.
- **Threshold study (1–512 nodes, jobs 4949326–4949335):** the selector's true baseline is the
  **stock csel tree under `DEVICE_COLLECTIVES=none`** — NOT config B (`=all`, the CH4 composition
  path); at 2 nodes the stock path is ~3× faster than B, so thresholds derived from C-vs-B would
  be wrong. Measured against the right baseline, **RCCL wins the entire measured range
  (1 KiB–256 MiB) at every node count** — 2.3–4.7× even at 1 KiB, 18–50× at 256 MiB. Sub-1 KiB
  tail + 1024–4096 confirmation runs pending; v2 JSON (route all built-in-op device allreduce to
  ccl, runtime requirements-check falls back for host buffers/unsupported ops) to follow.
- **Staged-hybrid crash (the real mechanism, corrected):** under `DEVICE_COLLECTIVES=all` the
  all-ccl MPIR tree faults (bisected to `=all` + all-ccl MPIR JSON, no CH4 JSON needed). NOTE:
  the ccl fallback is NOT the cause — `MPIR_Allreduce_intra_ccl` calls `recursive_doubling`
  directly (not auto-selection), and `MPIR_RCCL_check_requirements_red_op` DOES reject host
  buffers. The fault is the composition's host-swap path interacting with the ccl leaf when the
  global swap buffer and the routing threshold disagree (see below). Workaround: keep the swap
  buffer equal to the threshold (threshold-gated `tuning/allreduce_mpir_hybrid.json`).
- **GPU-direct RDMA (`MPIR_CVAR_CH4_OFI_ENABLE_HMEM=1`) works but is insufficient:** ~13–25%
  faster than staging at both 2 and 64 nodes, yet still 2.5–3× slower than RCCL at 8 B and ~40×
  at 256 MiB — even MPICH's no-staging path never beats RCCL at any size.
- **Why there is no crossover here (though there is one vs Cray):** MPICH's stock algorithms pay
  a per-operation host-staging floor on device buffers (~330 µs even at 1 KiB), while RCCL's
  floor is ~60 µs — two flat floors, one above the other. **MPICH's small-message device path is
  so slow that RCCL always wins; a faster small-message path (like Cray's, ~60 µs floor, which
  beats RCCL below ~16 KiB) would re-introduce a real crossover and move the threshold.**
  Improving MPICH's small-message device path is future work that would make the selection
  genuinely size-dependent.

### Per-communicator thresholds: why they worked at JLSE but not (as a staged hybrid) on Frontier
The SULI paper (JLSE, **ch4:ucx**) had two thresholds — 8 KB single-node, 32 KB multi-node —
routing `recursive_doubling` (small) vs RCCL (large) in the MPIR tuning tree. That mechanism is
intact: `comm_size` + `avg_msg_size` branches in the MPIR tree are crash-free and per-communicator.
The reason it doesn't reproduce here is the **interconnect/device layer**, not the tree:
- **Small-alg probe (jobs 4953516/17, forced algorithm, 8 B–64 KiB, 1+2 nodes):** on Frontier
  **ch4:ofi, NO pure-MPIR algorithm beats RCCL at any small size.** Best (recexch /
  recursive_doubling) ≈ **117–160 µs at 8 B vs RCCL ≈ 53–56 µs** (2–3× slower); smp/tree/
  reduce_scatter worse (to ~515 µs). Enabling GPU-direct RDMA (`FI_HMEM=1`) is a wash (118 vs 118).
  JLSE's UCX rocm transport did GPU-direct pt2pt (fast small); Frontier's ofi path stages
  GPU→host per communication round → log₂(P)× the tax → no MPIR algorithm is competitive.
- **So on Frontier the ONLY sub-RCCL small path is the CPU-staged composition** (one bulk copy,
  ~39 µs). That path is gated by a single global CVAR (`CH4_GPU_COLL_SWAP_BUFFER_SZ`), which must
  equal the routing threshold (verified: decoupling them — per-comm ch4 branches + large global
  swap — faults at the alpha→beta handoff, job 4953409, EXIT 9). **Hence per-communicator
  thresholds for the *staged* hybrid are not realizable without a source change** (expose the
  staged reduce as a per-comm tree algorithm, or make the swap buffer per-communicator).
- **Measured staged-vs-RCCL crossover, all 13 node counts** (confirm runs; staged = alpha/CPU
  composition, compared against committed RCCL per scale):

  | N | thr | N | thr | N | thr |
  |---|---|---|---|---|---|
  | 1 | 16K | 32 | 16K | 512 | 16K |
  | 2 | 8K | 64 | 16K | 1024 | 16K |
  | 4 | 8K | 128 | 16K | **2048** | **64K** |
  | 8 | 16K | 256 | 16K | **4096** | **32K** |
  | 16 | 16K | | | | |

  Shape: **flat ~16 KiB across the middle, 8 KiB at tiny scale, rising to 32–64 KiB at the
  giants** (8× spread). The staged path's winning band widens at extreme scale — the opposite of
  the plot-read guess, and a stronger "optima vary with scale" result than a flat threshold.
- **Production decision:** single **16 KiB** threshold (SWAP = ch4 split = mpir gate = 16 KiB) —
  the only crash-free design (per-comm faults, job 4953409). The per-scale table above is the
  **characterization result** motivating a per-communicator swap buffer as future work. Contrast
  with JLSE, where per-node-class thresholds were both realizable and beneficial because the
  small path (`recursive_doubling`) was itself fast.

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
