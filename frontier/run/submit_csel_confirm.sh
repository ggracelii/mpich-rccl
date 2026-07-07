#!/bin/bash
# submit_csel_confirm.sh — targeted staged-vs-RCCL crossover confirmation.
# Per node count: probe split set ABOVE Grace's plot-read bracket, OSU window
# spanning bracket/4 .. bracket*4. Generates per-N probe JSONs, then submits.
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export FRONTIER_HOME=$(cd "$HERE/.." && pwd)
TUN=$FRONTIER_HOME/tuning/confirm
mkdir -p "$TUN"

# nodes : plot-read bracket upper edge (bytes). Probe split = 4x upper edge.
#   N=1,2,4: 32-64K | N=8: 64-128K | N=16,32,64: 256-512K
#   N=128,256: 512K-1M | N=512: 256-512K | N=1024,2048: 32-64K | N=4096: 8-16K
# round 1 (plot-read B-curve brackets) ran high: direct staged-vs-RCCL measurement puts
# thresholds at 8-16 KiB for N=1..8 and BELOW the 64K+ windows for N>=16. Round-2 map:
HI=( [1]=65536 [2]=65536 [4]=65536 [8]=131072
     [16]=16384 [32]=16384 [64]=16384
     [128]=16384 [256]=16384 [512]=16384
     [1024]=16384 [2048]=262144 [4096]=16384 )   # 2048 raised: staged won its whole 2-64K window (crossover >64K)

LADDER=${1:-"1 2 4 8 16 32 64 128 256 512 1024 2048 4096"}

for N in $LADDER; do
  hi=${HI[$N]}; [ -z "$hi" ] && { echo "no bracket for N=$N, skipping"; continue; }
  split=$((hi * 4))                 # measure alpha well past the bracket
  lo=$((hi / 8))                    # window: bracket/8 .. bracket*4
  mhi=$((hi * 4))
  jc="$TUN/ch4_N${N}.json"; jm="$TUN/mpir_N${N}.json"
  python3 - "$split" "$jc" "$jm" "$FRONTIER_HOME/tuning" <<'PY'
import json, sys, collections
split, jc, jm, tun = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
# templates: the full trees already in the repo (no mpich submodule needed on Frontier)
ch4 = json.load(open(f"{tun}/allreduce_ch4_hybrid.json"), object_pairs_hook=collections.OrderedDict)
ch4["collective=allreduce"] = collections.OrderedDict([
    ("comm_type=intra", collections.OrderedDict([
        (f"avg_msg_size<={split}", {"composition=MPIDI_Allreduce_intra_composition_alpha": {}}),
        ("avg_msg_size=any",       {"composition=MPIDI_Allreduce_intra_composition_beta": {}}),
    ])),
])
json.dump(ch4, open(jc, "w"), indent=2)
mp = json.load(open(f"{tun}/allreduce_mpir_hybrid.json"), object_pairs_hook=collections.OrderedDict)
mp["collective=allreduce"]["comm_type=intra"] = collections.OrderedDict([
    (f"avg_msg_size<={split}", {"algorithm=MPIR_Allreduce_intra_recursive_doubling": {}}),
    ("avg_msg_size=any", collections.OrderedDict([
        ("is_op_built_in=no",  {"algorithm=MPIR_Allreduce_intra_recursive_doubling": {}}),
        ("is_op_built_in=yes", {"algorithm=MPIR_Allreduce_intra_ccl": {"ccl=rccl": {}}}),
    ])),
])
json.dump(mp, open(jm, "w"), indent=2)
PY
  [ -s "$jc" ] && [ -s "$jm" ] || { echo "SKIP N=$N: JSON generation failed" >&2; continue; }
  echo "submit N=$N split=$split window=$lo:$mhi"
  sbatch -N "$N" \
    --export=ALL,FRONTIER_HOME="$FRONTIER_HOME",PROBE_SPLIT="$split",MRANGE="$lo:$mhi",JSON_CH4_P="$jc",JSON_MPIR_P="$jm" \
    "$HERE/run_csel_confirm.sbatch"
done
