#!/usr/bin/env bash
set -euo pipefail

DATA=/data/yf23/datasets/ICRA27-ROBOT
TMP="$DATA/hf_missing_seen_parts"
CHECKPOINTS=/data/yf23/checkpoints/ICRA27-ROBOT
LOG=/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/download_seen_missing_hf.log

export HF_HOME="$CHECKPOINTS/hf_home"
export HF_HUB_ENABLE_HF_TRANSFER=0

if [ -f /data/yf23/miniforge3/etc/profile.d/conda.sh ]; then
  source /data/yf23/miniforge3/etc/profile.d/conda.sh
elif [ -f /data/yf23/miniconda3/etc/profile.d/conda.sh ]; then
  source /data/yf23/miniconda3/etc/profile.d/conda.sh
else
  echo "Could not find CAIR conda activation script" >&2
  exit 1
fi

conda activate /data/yf23/conda/envs/icra27-robot
mkdir -p "$TMP"

attempt=1
while true; do
  {
    echo "[$(date)] HF fallback attempt $attempt for seen_tasks.part_ab/ad"
    huggingface-cli download Jiaming2472/AGNOSTOS \
      --repo-type dataset \
      --include "seen_tasks.part_ab" "seen_tasks.part_ad" \
      --local-dir "$TMP" \
      --resume-download
    echo "[$(date)] HF fallback complete"
  } >> "$LOG" 2>&1 && break

  echo "[$(date)] HF fallback failed; retrying in 60s" >> "$LOG"
  attempt=$((attempt + 1))
  sleep 60
done
