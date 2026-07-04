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
| ResNet-50 | 25.6 M | 102 MB | 51 MB |
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

## 7. Sources (titles/links verified 2026-07-04)
Collectives / distributed-training literature:
- Singh et al., *The Big Send-off: Scalable and Performant Collectives for Deep Learning*, arXiv:2504.18658 — https://arxiv.org/abs/2504.18658  (PCCL; up to 10× all-reduce over RCCL on 2048 GCDs of Frontier; documents RCCL/Cray-MPICH limits — direct motivation)
- Dash et al., *Optimizing Distributed Training on Frontier for Large Language Models*, arXiv:2312.12705 — https://arxiv.org/abs/2312.12705
- Xu et al., *Scaling Large Language Model Training on Frontier with Low-Bandwidth Partitioning*, arXiv:2501.04266 — https://arxiv.org/abs/2501.04266
- *Collective Communication Performance Evaluation for Distributed Deep Learning Training*, MDPI Applied Sciences 14(12):5100 (2024), doi:10.3390/app14125100 — https://www.mdpi.com/2076-3417/14/12/5100
- Peng et al., *FP8-LM: Training FP8 Large Language Models*, arXiv:2310.18313 — https://arxiv.org/abs/2310.18313
- Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*, arXiv:1910.02054 — https://arxiv.org/abs/1910.02054  (RS+AG / FSDP-ZeRO reference — cite this, not a blog)
- Chen et al., *Design and Optimization of GPU-Aware MPI Allreduce Using Direct Sendrecv Communication*, ICPP '25, doi:10.1145/3754598.3754666 — https://dl.acm.org/doi/10.1145/3754598.3754666

Model specs (gradient bytes = params × bytes/element):
- ResNet-50 — **25,557,032 params (≈25.6 M)** — He et al., *Deep Residual Learning for Image Recognition*, arXiv:1512.03385
- BERT-Large — **340 M params** — Devlin et al., *BERT*, arXiv:1810.04805
- GPT-2 XL — **1.5 B params** — Radford et al., *Language Models are Unsupervised Multitask Learners* (2019)
- DDP 25 MiB gradient bucket — PyTorch `DistributedDataParallel` `bucket_cap_mb=25` (MiB) default — https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html

## 8. Implementation (as built)
Harness added under `run/`:
- **`run/run_ml_sync.sbatch`** — runs configs **B / C / D** at real model-gradient byte sizes
  (`ddp_bucket` 25 MiB, `resnet50` 102 MB, `bert_large_fp16` 680 MB) at a given `-N`, writing
  `results_ml/N<nodes>_job<id>/{B,C,D}_*.txt` (same OSU format the notebook loader already parses).
- **`run/submit_ml.sh`** — weak-scales it across a node ladder (default `1 8 64 512 1024`, 3 reps).
  Run from anywhere (`chmod +x` or `bash run/submit_ml.sh`); it exports `FRONTIER_HOME`.

Decisions baked in (all honesty-preserving):
- **Byte-proxy for precision.** Comm cost is byte-bound, so we measure at the gradient's byte
  count with the default 4-byte element. A 680 MB allreduce proxies BERT's fp16 gradient exactly
  (same bytes on the wire); this sidesteps OSU/MPI having no fp16 type. State it as such in the paper.
- **Cray (D) only at ≤512 nodes** — it faults >4 MiB at ≥1024, and every ML size exceeds 4 MiB, so
  at scale the ML comparison is **C (MPICH-RCCL) vs B (MPICH-CPU)**; C-vs-Cray holds only ≤512.
- **Weak scaling** — fixed per-rank gradient size, growing node count (data-parallel replica model).

**Cheapest path first (recommended):** most of this is already in the main sweep. The gradient-sync
story can be told **for free** by reading the main-sweep latencies at the nearest power-of-2 sizes
(**32 MiB ≈ bucket, 128 MiB ≈ ResNet, 512 MiB–1 GiB ≈ BERT**) and reframing them as per-step comm
time + speedup. Only run `submit_ml.sh` if you want the *exact* model sizes; keep the ladder short
(`1 8 64 512`) to stay well inside the remaining budget. Metrics to report from either path:
per-step gradient-allreduce time (ms), effective bus bandwidth, C/B and C/D speedup, and comm-fraction
of a step using a representative compute time.
