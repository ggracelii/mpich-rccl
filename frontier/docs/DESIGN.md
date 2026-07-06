# RCCL Allreduce Backend for MPICH — Frontier Experiment Design

**Target:** conference paper (SC / IPDPS / ISC / EuroMPI class)
**Contribution under test:** MPICH `MPI_Allreduce` offloaded to RCCL (PR pmodels/mpich#7493)
**System:** OLCF Frontier (HPE Cray EX, AMD EPYC + MI250X, Slingshot-11)

> NOTE: exact queue limits, module versions, and the GCD↔NUMA binding table drift over
> time. Every place marked **[verify OLCF]** must be checked against the current Frontier
> User Guide (docs.olcf.ornl.gov/systems/frontier_user_guide.html) at run time.

---

## 1. Research questions & hypotheses

| # | Question | Hypothesis |
|---|----------|-----------|
| RQ1 | Where is the message-size crossover between RCCL and CPU recursive-doubling? | RCCL loses below some small size (HIP stream + kernel-launch overhead), wins above it; crossover ~4–64 KiB. |
| RQ2 | Does the intra-node (~7×) advantage hold as node count grows? | Advantage decays with scale as Slingshot latency dominates, but stays >1× for medium/large messages. |
| RQ3 | Is MPICH+RCCL competitive with **Cray MPICH GPU-aware** (production)? | Comparable or better at large messages; this is the headline result. |
| RQ4 | How much MPICH overhead sits above pure RCCL? | MPICH+RCCL within a small constant of `rccl-tests` ceiling; gap grows at small sizes. |
| RQ5 | Does the win translate to a realistic ML gradient-allreduce workload? | Yes — data-parallel training is allreduce-bound, so effective step-time improves at model-scale buffer sizes. |

Each RQ maps to specific plots (§8). Design experiments to *answer these*, not to produce
numbers for their own sake.

---

## 2. System under test (Frontier)

- 9,408 nodes. Per node: 1× 64-core AMD EPYC 7A53 "Trento" + 4× MI250X.
- Each MI250X = 2 GCDs ⇒ **8 GCDs per node**, presented as 8 GPUs, 64 GB HBM2e each.
- **Intra-node** GPU↔GPU: Infinity Fabric / XGMI. **Inter-node**: 4× Slingshot-11 NICs
  (200 Gb/s each), dragonfly topology.
- **Critical:** GCD↔NUMA mapping is non-linear. Wrong binding costs 2–3× and invalidates
  comparisons. Use the OLCF mapping **[verify OLCF]**; validate with a 1-node bandwidth
  sanity check before trusting any number.

---

## 3. Software configurations (the four baselines + contribution)

All comparisons sweep the **same message sizes** on the **same node allocation** back-to-back
to control for dragonfly placement. Five configs:

| ID | Config | Buffer | Algorithm | Build / env |
|----|--------|--------|-----------|-------------|
| **A** | MPICH-CPU, host buffers | host | CPU allreduce | your MPICH, copy device→host→device manually |
| **B** | MPICH-CPU, device buffers | device | CPU allreduce, GPU-aware pt2pt | your MPICH, `MPIR_CVAR_ALLREDUCE_CCL=` unset |
| **C** | **MPICH + RCCL (contribution)** | device | `MPIR_Allreduce_intra_ccl` → RCCL | your MPICH `--with-rccl`, `MPIR_CVAR_ALLREDUCE_CCL=rccl` |
| **D** | Cray MPICH GPU-aware | device | vendor | `module load PrgEnv-*` + `craype-accel-amd-gfx90a`, `MPICH_GPU_SUPPORT_ENABLED=1` |
| **E** | rccl-tests (ceiling) | device | pure RCCL, no MPI | `all_reduce_perf` |

**Pin and record** for reproducibility: MPICH git SHA (your PR branch), ROCm version,
RCCL version, libfabric/CXI version, Cray PE version, and the exact `configure` line.

### Build risk — your MPICH over Slingshot
Running **non-vendor** MPICH on Frontier requires `ch4:ofi` against the **libfabric CXI
provider**. This is the hardest setup step. Rough configure shape (**[verify OLCF]** paths):
```bash
./configure \
  --prefix=$MPICH_INSTALL \
  --with-device=ch4:ofi \
  --with-libfabric=$OLCF_LIBFABRIC_PATH \    # CXI provider for Slingshot-11
  --with-hip=$ROCM_PATH \
  --with-rccl=$RCCL_PATH \
  CC=cc CXX=CC
```
Prove it works on **2 nodes in the debug queue** (correctness + a nonzero inter-node
bandwidth) before spending any node-hours on scaling.

---

## 4. Benchmark ladder (cheap → expensive)

| Tier | What | Tool | Nodes | Purpose |
|------|------|------|-------|---------|
| 0 | Correctness | your test suite + fp16/fp32/fp64 allreduce vs CPU reference | 1–2 | it runs correctly on Frontier |
| 1 | Ceiling | `rccl-tests all_reduce_perf` (config E) | 1, 2, 8 | upper bound, zero MPI overhead |
| 2 | **Head-to-head** | **OSU `osu_allreduce -d rocm`** (A–D) | 1…1024+ | the main result |
| 3 | ML proxy | gradient-allreduce proxy (§7) | 8…512 | realistic-workload impact (RQ5) |

Use OSU Micro-Benchmarks built against **each** MPI (A/B/C use your MPICH; D uses Cray
MPICH). rccl-tests is separate. Keep binaries per-config to avoid link contamination.

---

## 5. Sweep matrix (Tier 2)

- **Message sizes:** 8 B → 1 GiB, powers of 2 (≈28 points). Covers latency-bound → bandwidth-bound.
- **Datatypes:** fp32 (primary), fp16 (ML relevance), fp64 (HPC relevance). Op = SUM.
- **Node counts (strong+weak):** 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 — extend to
  2048/4096 if hours allow. 8 ranks/node (1 per GCD).
- **Placement isolation:** the 1-node point isolates XGMI; ≥2 nodes exercise Slingshot.
- **Scaling modes:**
  - *Strong:* fixed total buffer (e.g. 256 MiB), more GPUs → per-GPU shrinks.
  - *Weak:* fixed **per-GPU** bytes (e.g. 32 MiB/GPU), total grows with GPU count.

For each (size × config × nodecount): warmup ≥20 iters, measure ≥50 iters (fewer at 1 GiB),
record **min / median / p95 / max**. Repeat the whole node-count series **≥3× on fresh
allocations** to capture dragonfly placement variance — report median-of-medians + spread.

---

## 6. Methodology / rigor (reviewers will check these)

- **Binding validated, not assumed.** Show the srun binding line; sanity-check intra-node
  bandwidth against XGMI peak before trusting data.
- **Steady state.** Discard warmup; RCCL lazily initializes communicators/streams on first call.
- **Same allocation, back-to-back configs.** A/B/C/D for a given size run in one job so they
  see the same nodes; don't compare across jobs allocated hours apart.
- **Variance reported.** Error bars from the ≥3 repetitions; note that Slingshot is shared.
- **Fair CPU baseline.** Tune CPU allreduce (config B) — don't cripple it. Use the same rank
  count and best available CPU algorithm so the win is honest.
- **Environment captured.** Dump `module list`, all `MPIR_CVAR_*`/`MPICH_*`, `rocm-smi`,
  git SHAs into every result directory.

---

## 7. ML gradient-allreduce proxy (Tier 3, RQ5)

**Why it's applicable:** data-parallel DL training synchronizes gradients every step via one
allreduce over the flattened gradient buffer (frameworks bucket it into a few large
allreduces). That is *exactly* the collective your backend accelerates — no other collective
is on the critical path. So this proxy is a faithful stand-in for real training, without the
cost/complexity of a full framework.

**Design:** an MPI program that, per "step," allreduces buffers sized like real models:
- ResNet-50: ~25.5 M params → ~102 MiB fp32 / ~51 MiB fp16
- BERT-Large: ~340 M params → ~1.3 GiB fp32
- GPT-2 1.5 B: ~6 GiB fp32 (bucketed)

Mimic framework behavior: **bucket** into ~25 MiB chunks and allreduce each (like DDP/Horovod
gradient bucketing). Report **allreduce time per step** and **effective allreduce bandwidth**
across configs A–D and node counts. Headline: "communication phase of a training step is N×
faster with the RCCL backend at 128 nodes."

Keep it a proxy (buffers of the right size + dtype + bucketing), not a real model — that's the
defensible, reproducible choice and reviewers accept it for communication studies.

---

## 8. Metrics & figures

1. **Latency vs message size**, 1 node, all configs — shows the RQ1 crossover. (log-log)
2. **Speedup C/B and C/D vs message size**, per node count — the core win. (=1 line marked)
3. **Latency vs node count** at fixed sizes (small/medium/large) — RQ2 scaling. (log-x)
4. **MPICH overhead:** C vs E (rccl-tests) — RQ4, gap vs ceiling.
5. **Weak-scaling efficiency** vs node count.
6. **ML proxy:** per-step allreduce time vs node count, per model size — RQ5.
7. Table: crossover size per node count; peak speedup; where C beats D.

---

## 9. Node-hour budget (rough — for the allocation ask)

Node-hours = nodes × wallclock. Microbench jobs are short but large-node jobs dominate cost.

| Node count | Wall/job (A–D full size sweep) | Reps | Node-hours |
|-----------:|-------------------------------:|-----:|-----------:|
| 1–64 (per pt) | ~15 min | 3 | small (<200 total) |
| 128 | ~15 min | 3 | ~96 |
| 256 | ~15 min | 3 | ~192 |
| 512 | ~15 min | 3 | ~384 |
| 1024 | ~20 min | 3 | ~1,024 |
| 2048 | ~20 min | 3 | ~2,048 |
| 4096 | ~20 min | 3 | ~4,096 |
| ML proxy | — | — | ~1,000 |

**Ballpark: ~2–4k node-hours** for the core paper (through 1024 nodes + ML proxy);
**~8–10k** if you push to 4096 and add datatype sweeps. Cheap in Frontier terms — a DD
allocation covers it. Do all small-node development in the **debug queue** to preserve hours.

---

## 10. Reproducibility artifact (paper requirement)

Ship a repo with: build scripts (all 5 configs), Slurm job scripts (parametrized by node
count), the ML proxy source, a results schema (one JSON/CSV per run with full env dump), and
the plotting harness. This doubles as your own experiment automation.

---

## 11. Phased execution plan

- **P0 — Access & allocation.** Confirm Frontier hours; get on a project or request DD.
- **P1 — Build.** Your MPICH (`ch4:ofi`+CXI+RCCL) on Frontier; Cray MPICH env; OSU ×MPIs;
  rccl-tests. Debug queue only.
- **P2 — Correctness + binding.** Tier 0/1 on 1–2 nodes; validate GCD binding.
- **P3 — Microbench scaling.** Tier 2, 1→1024 nodes, ≥3 reps. Main data.
- **P4 — ML proxy.** Tier 3 at model sizes.
- **P5 — Analysis & plots.** Figures §8, crossover/scaling tables, variance.
- **P6 — Writeup + artifact.**
