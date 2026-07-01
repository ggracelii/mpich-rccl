#!/bin/bash
# bind_frontier.sh — pin each rank to the CORRECT GCD on a Frontier node.
#
# Replaces grace_suli/omb/map_rank_to_gpu (MV2_COMM_WORLD_LOCAL_RANK +
# CUDA_VISIBLE_DEVICES — both wrong on Frontier). Works under BOTH launchers:
#   - Hydra/mpiexec sets MPI_LOCALRANKID
#   - Slurm/srun     sets SLURM_LOCALID
# GPU selection is ROCR_VISIBLE_DEVICES (not CUDA_VISIBLE_DEVICES), and the
# GCD<->NUMA numbering is NON-LINEAR: rank i does NOT map to GCD i.
#
# The map below is the OLCF-documented optimal local-rank -> GCD affinity for
# 8 ranks/node.  ***[verify OLCF]*** against the current Frontier User Guide
# ("AMD GPUs and MI250X" / GPU mapping) BEFORE trusting numbers — a wrong map
# silently costs 2-3x and invalidates the whole comparison.
#
# Usage (wrap the ranked binary under either launcher):
#   mpiexec ... ./bind_frontier.sh ./osu_allreduce -d rocm ...
#   srun    ... ./bind_frontier.sh ./osu_allreduce -d rocm ...
set -o pipefail

GPU_MAP=(4 5 2 3 6 7 0 1)                                    # [verify OLCF]
LID=${SLURM_LOCALID:-${MPI_LOCALRANKID:-${PMI_LOCALRANKID:-0}}}
export ROCR_VISIBLE_DEVICES=${GPU_MAP[$((LID % 8))]}

exec "$@"
