# Frontier RCCL-Allreduce Experiments

Evaluating the MPICH RCCL allreduce backend (PR pmodels/mpich#7493, merged upstream)
vs alternatives on OLCF Frontier (MI250X / Slingshot-11).

**Run every command from this `frontier/` directory.** Scripts locate `env.sh` and
`bind_frontier.sh` at this root, and the sbatch jobs resolve them via `$SLURM_SUBMIT_DIR`
(the directory you run `sbatch` from) — so staying in `frontier/` is what makes the paths work.

## Layout
```
frontier/
├── env.sh                  # shared config (paths, modules, load_mine/load_cray) — sourced by all
├── bind_frontier.sh        # per-rank GCD pinning (used by the sbatch jobs)
├── build/
│   ├── build_mpich.sh      # MPICH + RCCL over ch4:ofi/CXI (config C) — critical path
│   ├── build_osu.sh        # OSU micro-benchmarks vs your MPICH (mine) and Cray (cray)
│   ├── build_rccl_tests.sh # rccl-tests (config E ceiling)
│   └── build_bench.sh      # correctness validator (allreduce_benchmark_mpi)
├── run/
│   ├── run_allreduce.sbatch  # one job = configs A–E on the same allocation
│   └── submit_scaling.sh     # fan the sweep across node counts (3 reps; REPS=n overrides)
├── check/
│   ├── validate.sbatch     # Tier-0 correctness gate (int/float/double + corruption self-test)
│   ├── confirm.sbatch      # mechanism check (NCCL_DEBUG + rocprof): C=RCCL/CXI, B≠RCCL
│   ├── monitor.sh          # live sweep health (queue / states / file completeness)
│   └── check_results.sh    # scan result files for hidden errors + latency summary
├── docs/
│   ├── NOTES.md            # build recipe, full fix chronology, findings (methods notes)
│   └── ML_EXPERIMENT.md    # ML gradient-sync experiment design
├── plots/
│   └── plot.ipynb          # parses results/ → latency / speedup / scaling / crossover plots
├── out/                    # captured job stdout logs (*.out)
└── results/
    ├── sweep/              # committed sweep data: N<nodes>_job<id>/{A..E}_*.txt
    └── confirm_*/          # mechanism-confirmation evidence
```

## The five configs (sweep 0 B → 1 GiB)
| ID | What | Selected by |
|----|------|-------------|
| A | MPICH CPU, host buffers | your MPICH, no `-d rocm` |
| B | MPICH CPU algo, device buffers | `MPIR_CVAR_DEVICE_COLLECTIVES=all` |
| **C** | **MPICH + RCCL backend** | `ALLREDUCE_INTRA_ALGORITHM=ccl` + `ALLREDUCE_CCL=rccl` + `DEVICE_COLLECTIVES=none` + `rccl-net-plugin` |
| D | Cray MPICH GPU-aware | `cray-mpich` + `MPICH_GPU_SUPPORT_ENABLED=1` |
| E | pure RCCL ceiling | `rccl-tests all_reduce_perf` |

## Order of operations (all from `frontier/`)
```bash
vim env.sh bind_frontier.sh              # verify the [verify OLCF] markers on a login node

./build/build_mpich.sh                   # critical path
./build/build_osu.sh mine
./build/build_osu.sh cray
./build/build_rccl_tests.sh
./build/build_bench.sh

sbatch check/validate.sbatch             # correctness gate
sbatch check/confirm.sbatch              # mechanism check
sbatch -N 2 run/run_allreduce.sbatch     # smoke

./run/submit_scaling.sh                  # sweep 1-64 (default ladder)
./run/submit_scaling.sh "128 256 512 1024"
./run/submit_scaling.sh "2048 4096"

watch -n 30 ./check/monitor.sh           # live health while the sweep drains
./check/check_results.sh                 # scan committed results
```

## Plotting
Open `plots/plot.ipynb` in Jupyter or VSCode (Python + Jupyter extensions; needs
`pandas numpy matplotlib`). Run cell 1 (loads `results_sweep/`), then any plot cell —
`plot_latency_vs_size(nodes)`, `plot_speedup_vs_size("D")`, `plot_scaling(size)`,
`plot_crossover("D")`. Re-run cell 1 after adding new results and every plot updates.

## Gotchas (full chronology in docs/NOTES.md)
1. Build MPICH with **amdclang + system libfabric (CXI)**, not the `cc` wrapper (avoids Cray-MPI contamination); `--disable-weak-symbols`; `module unload cray-mpich` at runtime.
2. **Single-node** jobs need `--network=single_node_vni` for CXI.
3. GPU binding uses **`MPI_LOCALRANKID`** (Hydra), not `SLURM_LOCALID` (which is 0 for all ranks under mpiexec) — else RCCL "Duplicate GPU" crash.
4. RCCL needs the **`rccl-net-plugin`** (aws-ofi-rccl) module for CXI, else it silently falls back to slow TCP. Loaded for configs C/E only.
5. `[verify OLCF]` markers = confirm against the current Frontier User Guide.
