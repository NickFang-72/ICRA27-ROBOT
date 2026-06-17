#!/usr/bin/env bash
set -euo pipefail

PROBE=/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe
DATA=/data/yf23/datasets/ICRA27-ROBOT
CHECKPOINTS=/data/yf23/checkpoints/ICRA27-ROBOT
QWEN="$CHECKPOINTS/Qwen2.5-VL-7B-Instruct"
ROBOPOINT="$CHECKPOINTS/robopoint-v1-vicuna-v1.5-13b"
LOG="$PROBE/run_probe_when_ready.log"

export HF_HOME="$CHECKPOINTS/hf_home"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
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

{
  echo "[$(date)] Waiting for AGNOSTOS seen_tasks and Qwen checkpoint"

  until [ -d "$DATA/seen_tasks" ] && find "$DATA/seen_tasks" -path "*/front_rgb/*.png" -print -quit | grep -q .; do
    echo "[$(date)] seen_tasks not ready yet"
    sleep 300
  done

  until [ -f "$QWEN/config.json" ] && ls "$QWEN"/model-*.safetensors >/dev/null 2>&1; do
    echo "[$(date)] Qwen checkpoint not ready yet"
    sleep 300
  done

  echo "[$(date)] Inputs ready. Sampling demos."
  python "$PROBE/scripts/sample_seen_demos.py" \
    --train-json "$PROBE/train.json" \
    --data-root "$DATA" \
    --out "$PROBE/results/manifest.json" \
    --per-task 1 \
    --max-demos 12 \
    --copy-images

  echo "[$(date)] Running Qwen geometry."
  CUDA_VISIBLE_DEVICES=0 python "$PROBE/scripts/run_qwen_geometry.py" \
    --manifest "$PROBE/results/manifest.json" \
    --model "$QWEN"

  until [ -f "$ROBOPOINT/config.json" ] && [ "$(find "$ROBOPOINT" -maxdepth 1 -name "model-*.safetensors" | wc -l | tr -d ' ')" -ge 6 ]; do
    echo "[$(date)] RoboPoint checkpoint not ready yet"
    sleep 300
  done

  echo "[$(date)] Running RoboPoint affordance."
  CUDA_VISIBLE_DEVICES=1 python "$PROBE/scripts/run_robopoint_affordance.py" \
    --manifest "$PROBE/results/manifest.json" \
    --model-path "$ROBOPOINT"

  echo "[$(date)] Building review index."
  python "$PROBE/scripts/build_review_index.py"
  echo "[$(date)] Probe complete: $PROBE/results"
} 2>&1 | tee -a "$LOG"
