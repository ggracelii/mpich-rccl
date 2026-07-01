#!/bin/bash
# bind_frontier.sh — pin each rank to the CORRECT GCD on a Frontier node.
#
# Replaces grace_suli/omb/map_rank_to_gpu (which used MV2_COMM_WORLD_LOCAL_RANK
# + CUDA_VISIBLE_DEVICES — both wrong on Frontier). On Frontier:
#   - local rank comes from SLURM_LOCALID
#   - GPU selection is ROCR_VISIBLE_DEVICES (not CUDA_VISIBLE_DEVICES)
#   - the GCD<->NUMA numbering is NON-LINEAR: rank i does NOT map to GCD i.
#
# The map below is the OLCF-documented optimal CPU-die -> GCD affinity for
# 8 ranks/node.  ***[verify OLCF]*** against the current Frontier User Guide
# ("AMD GPUs and MI250X" / "Slurm and GPU mapping") BEFORE trusting numbers —
# a wrong map silently costs 2-3x and invalidates the whole comparison.
#
# Usage (as an srun wrapper):
#   srun ... ./bind_frontier.sh ./osu_allreduce -d rocm ...
set -euo pipefail

# OLCF optimal local-rank -> GCD map for Frontier [verify OLCF]:
GPU_MAP=(4 5 2 3 6 7 0 1)
LID=${SLURM_LOCALID:-0}
export ROCR_VISIBLE_DEVICES=${GPU_MAP[$LID]}

exec "$@"
