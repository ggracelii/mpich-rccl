# ML Gradient-Sync Experiment — Design & Rationale

Design for the ML portion of the RCCL-allreduce study, grounded in the 2024–2026
distributed-training literature. Read the framing section first — it determines
what claims are honest.

## 1. The honest framing (critical)
The contribution accelerates **`MPI_Allreduce`** via RCCL. Native PyTorch
DDP/DeepSpeed/FSDP call **RCCL/NCCL directly and bypass MPI entirely**, so this
work does **not** speed up native-PyTorch LLM training — claiming that would be
wrong and reviewers will catch it. What it *does* accelerate:
- **Horovod with the MPI backend** (Horovod uses `MPI_Allreduce` for gradient sync).
- **MPI-collective frameworks** (e.g. MXNet KVStore, some DeepSpeed paths, Intel oneCCL-over-MPI).
- **Scientific ML / HPC data-parallel codes** that already use MPI and operate on GPU buffers.
- Any code that wants **RCCL-class collective performance through the portable MPI API** without rewriting to call RCCL directly.

So the ML story is: *"for the large class of HPC/scientific-ML and Horovod-MPI
data-parallel workloads that sync gradients via MPI collectives, the RCCL backend
closes the gap to native RCCL."* That is true, useful, and defensible.

## 2. What the literature says (2024–2026)
- **DDP uses allreduce; FSDP/ZeRO-3 use reduce-scatter + all-gather** (RS+AG = a
  decomposed allreduce, ~50% more volume). Modern LLM pretraining is mostly FSDP/ZeRO,
  i.e. RS+AG — *not* monolithic allreduce. Your backend is allreduce-only → it serves
  the **DDP-style** path directly; RS+AG is the natural extension (§6).
- **Low precision makes communication the bottleneck.** FP8/BF16 training (DeepSeek-V3
  etc.) cuts compute ~2×, so the *communication fraction* of each step rises; FP8 roughly
  halves comm volume but comm becomes the critical path. → Accelerating the (now fp16/bf16)
  allreduce matters more than ever. **This is the timely angle.**
- **On Frontier specifically**, the two collective options are Cray-MPICH and RCCL; prior
  work ("The Big Send-off", ORNL LLM-on-Frontier papers) benchmarks allgather/reduce-scatter
  and finds *both* libraries have shortcomings at scale. They explicitly leave a gap:
  **deeper MPI↔RCCL integration** — which is exactly your backend.
- NCCL/RCCL beats MPI allreduce by up to ~3–4× generally, but Horovod-MPI has been shown
  competitive (within a few %, 90% scaling on 64 GPUs). Your C-vs-D data already shows the
  MPI+RCCL path *beating* Cray at ≥16 KiB — a strong result in this context.

## 3. Three angles worth including (pick 1–3)
- **A. Low-precision gradient allreduce (recommended, novel+timely).** Sweep fp32 vs **fp16**
  (and bf16 if the backend maps it) at model-gradient sizes. Story: as training goes low-precision
  and becomes comm-bound, the RCCL backend accelerates the now-critical fp16/bf16 allreduce.
- **B. Realistic DDP gradient-sync proxy (recommended, concrete).** Allreduce at real
  model-gradient sizes, bucketed like DDP/Horovod, reporting per-step comm time + effective
  bandwidth, weak-scaled across nodes.
- **C. Honest positioning vs FSDP/ZeRO (cheap, strengthens the paper).** Note RS+AG is the
  modern pattern; optionally measure that MPICH's CPU RS/AG are slow → motivates extending the
  CCL backend beyond allreduce. Mostly a discussion/future-work section + 1 supporting plot.

## 4. Experiment design
**Configs:** reuse B (MPICH-CPU), C (MPICH+RCCL), D (Cray GPU-aware) — same as the main sweep.

**Workloads = real model gradient sizes:**
| Model | Params | fp32 grad | fp16 grad |
|---|---|---|---|
| ResNet-50 | 25.5 M | 102 MiB | 51 MiB |
| BERT-Large | 340 M | 1.36 GiB | 680 MiB |
| GPT-2 XL | 1.5 B | (bucketed) | (bucketed) |
| DDP bucket | — | 25 MiB (PyTorch default) | 25 MiB |

**Datatypes:** fp32 and **fp16** (the low-precision angle). Add bf16 only if the backend maps it.

**Bucketing:** mimic DDP/Horovod — instead of one allreduce of the whole gradient, do repeated
allreduce of ~25 MiB buckets summing to model size, and report total per-step comm time. (This
matches how frameworks overlap backprop with bucketed gradient sync.)

**Scaling:** **weak scaling** — data-parallel training holds a full model replica per rank and
allreduces gradients over all ranks, so per-rank size is fixed and you grow node count.
Node ladder 1→1024 (extend to 4096 if budget allows), 3 reps ≤1024.

**Metrics:** per-step gradient-allreduce time (ms); effective (bus) bandwidth; speedup C/B and
C/D; and "communication fraction of a step" using a representative compute time for context.

**Program:** extend `benchmark/allreduce_benchmark_mpi.c` (already validates correctness) to add
a `--bucket <bytes>` loop and `--fp16`, or just script `osu_allreduce -d rocm -T <type>` at the
model/bucket sizes (OSU supports datatype + full min/med/max). The osu route needs no new code.

## 5. Positioning vs FSDP/ZeRO (scope statement for the paper)
Modern LLM pretraining uses FSDP/ZeRO (RS+AG), so frame the allreduce backend as: (a) directly
serving DDP-style and MPI-collective training, and (b) the foundational first collective — with
reduce-scatter/all-gather offload as clearly-scoped future work. Cite that even production
Frontier LLM runs wrestle with RCCL/Cray-MPICH RS+AG shortcomings, so an MPI-integrated CCL path
is a real contribution, not a toy.

## 6. Node-hour budget
Reuses the sweep infra; only ~3 model sizes × {fp32,fp16} × configs B/C/D, weak-scaled:
- to 1024 nodes, 3 reps: **~600–800 node-hrs**
- to 4096 nodes (2 reps at giants): **~2–2.5k**
Much of it is also derivable from the main sweep at the corresponding sizes (≈ free). Fits easily
in the remaining budget after the main sweep.

## 7. Sources
- The Big Send-off: Collectives for DL on GPU supercomputers — https://arxiv.org/pdf/2504.18658
- Optimizing Distributed Training on Frontier for LLMs — https://arxiv.org/html/2312.12705v2
- Scaling LLM Training on Frontier w/ Low-Bandwidth Partitioning — https://arxiv.org/html/2501.04266v1
- Collective Comm Perf Eval for Distributed DL Training (MDPI) — https://www.mdpi.com/2076-3417/14/12/5100
- FP8-LM: Training FP8 LLMs — https://arxiv.org/html/2310.18313v2
- ZeRO-3 vs FSDP (RS+AG decomposition) — https://denny.hashnode.dev/understanding-reduce-scatter-all-gather-and-all-reduce-in-distributed-computing-for-llm-training
- GPU-aware MPI Allreduce via direct sendrecv (ICPP'25) — https://dl.acm.org/doi/10.1145/3754598.3754666
