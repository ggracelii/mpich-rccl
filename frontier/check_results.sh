#!/bin/bash
# check_results.sh — sanity-scan sweep result files so you don't eyeball each one.
# Flags empty/truncated files and hidden error dumps, and prints key latencies.
#   ./check_results.sh [dir]        (default: results/sweep)
DIR=${1:-results/sweep}
lat() { awk -v s="$2" '$1==s{print $2; exit}' "$1" 2>/dev/null; }   # OSU avg latency (col2) at size $1B

echo "== health: data-rows / error-hits / max-size-reached (want err=0; A,B max=33554432; C,D max=1073741824) =="
for f in "$DIR"/N*/[ABCDE]_*.txt; do
  [ -f "$f" ] || continue
  rows=$(grep -cE '^[0-9]' "$f")
  err=$(grep -ciE 'abort|error|fatal|segmentation|opendomain|invalid|duplicate|failed' "$f")
  maxsz=$(grep -E '^[0-9]' "$f" | tail -1 | awk '{print $1}')
  flag=""; [ "$rows" -eq 0 ] && flag="  <== EMPTY"; [ "${err:-0}" -gt 0 ] && flag="$flag  <== ERROR TEXT"
  printf "%-42s rows=%-4s err=%-3s max=%-11s%s\n" "${f#"$DIR"/}" "$rows" "$err" "${maxsz:-none}" "$flag"
done

echo; echo "== avg latency (us): B/C/D @1MiB, C/D @1GiB (expect C<D at 1MiB+ ; B slowest) =="
printf "%-22s %9s %9s %9s | %11s %11s\n" "run" "B@1M" "C@1M" "D@1M" "C@1G" "D@1G"
for d in "$DIR"/N*/; do
  printf "%-22s %9s %9s %9s | %11s %11s\n" "$(basename "$d")" \
    "$(lat "$d/B_mpich_dev.txt" 1048576)" \
    "$(lat "$d/C_mpich_rccl.txt" 1048576)" \
    "$(lat "$d/D_cray_gpuaware.txt" 1048576)" \
    "$(lat "$d/C_mpich_rccl.txt" 1073741824)" \
    "$(lat "$d/D_cray_gpuaware.txt" 1073741824)"
done
