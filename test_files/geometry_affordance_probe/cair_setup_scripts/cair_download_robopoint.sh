#!/usr/bin/env bash
set -euo pipefail

CHECKPOINTS=/data/yf23/checkpoints/ICRA27-ROBOT
OUT="$CHECKPOINTS/robopoint-v1-vicuna-v1.5-13b"
LOG=/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/download_robopoint.log

export HF_HOME="$CHECKPOINTS/hf_home"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1

if [ -f /data/yf23/miniforge3/etc/profile.d/conda.sh ]; then
  source /data/yf23/miniforge3/etc/profile.d/conda.sh
elif [ -f /data/yf23/miniconda3/etc/profile.d/conda.sh ]; then
  source /data/yf23/miniconda3/etc/profile.d/conda.sh
else
  echo "Could not find CAIR conda activation script" >&2
  exit 1
fi

conda activate /data/yf23/conda/envs/icra27-robot
mkdir -p "$OUT"

attempt=1
while true; do
  if {
    echo "[$(date)] RoboPoint download attempt $attempt"
    huggingface-cli download wentao-yuan/robopoint-v1-vicuna-v1.5-13b \
      --local-dir "$OUT" \
      --resume-download
    test -f "$OUT/config.json"
    shard_count="$(find "$OUT" -maxdepth 1 -name "model-*.safetensors" | wc -l | tr -d ' ')"
    if [ "$shard_count" -lt 6 ]; then
      echo "Only found $shard_count/6 RoboPoint model shards; retrying"
      false
    else
      echo "[$(date)] RoboPoint download complete"
    fi
  } >> "$LOG" 2>&1; then
    break
  fi

  echo "[$(date)] RoboPoint download failed; retrying in 60s" >> "$LOG"
  attempt=$((attempt + 1))
  sleep 60
done
