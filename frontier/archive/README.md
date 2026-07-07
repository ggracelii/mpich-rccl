# Archive — superseded runs & job logs (provenance only)

**Do not analyze anything here as results.** The canonical dataset is
`../results_sweep/` (main sweep, `4934xxx` + rerun jobs) and `../results_ml/`
(ML gradient-sync). This folder preserves the run history before the csc678
allocation ended (2026-07-31, Lustre scratch purged).

- **`results_1gib_older/`** — two superseded sweep generations, kept as backup:
  `4929xxx` (original 1–64 node run) and `4932xxx` (early 128–1024 run, includes
  the walltime-clipped reps). Same 8 B→1 GiB protocol as the canonical set.
- **`job_logs/`** — Slurm stdout (`*.out`) from jobs since 2026-07-03. Notable
  evidence: `rccl-allreduce-4938124/4938125.out` (the 8192-node attempts —
  RCCL comm-init wall), `rccl-allreduce-4937457.out` (2048 run whose Cray step
  was force-terminated — GPU memory-access-fault evidence), `4938077` (hung
  4096 rep, discarded), and all `rccl-ml-*` gradient-sync jobs. Logs from before
  7/3 (incl. the canonical sweep's own stdout) were deleted in a cleanup; the
  fault evidence for those lives in the committed result files and
  `../docs/NOTES.md`.
