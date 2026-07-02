#!/bin/bash
# monitor.sh — at-a-glance sweep health. Run it, or `watch -n 30 ./monitor.sh`.
# Shows: queued/running jobs, a tally of finished states, any non-OK jobs to
# investigate, and how many config files each result dir has (should reach 5,
# or 4 for N>8 where config E is skipped).
source "$(dirname "$0")/env.sh" >/dev/null 2>&1

echo "== queue (mine) =="
squeue --me -o "%.10i %.14j %.5D %.4t %.11M %R"

echo; echo "== finished-job state tally (today) =="
sacct -X -S today -n -o State 2>/dev/null | sort | uniq -c

echo; echo "== NON-OK finished jobs (investigate) =="
sacct -X -S today -o JobID,JobName%16,NNodes,State,Elapsed 2>/dev/null \
  | grep -viE "COMPLETED|RUNNING|PENDING|^JobID|^---" || echo "  (none)"

echo; echo "== result completeness (config files per N dir) =="
for d in "$WORK"/results/N*/; do
  [ -d "$d" ] || continue
  n=$(ls "$d"[ABCDE]_*.txt 2>/dev/null | wc -l)
  echo "  $(basename "$d"): $n files"
done
