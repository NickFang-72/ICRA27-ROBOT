#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/yf23/projects/ICRA27-ROBOT}"
XICM_ROOT="${XICM_ROOT:-$PROJECT_ROOT/X-ICM}"
DATA_ROOT="${DATA_ROOT:-/data/yf23/datasets/ICRA27-ROBOT}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data/yf23/checkpoints/ICRA27-ROBOT}"

SEEN_DIR="$DATA_ROOT/seen_tasks"
UNSEEN_DIR="$DATA_ROOT/unseen_tasks"
UNSEEN_TAR="$DATA_ROOT/unseen_tasks.tar"
MODEL_DIR="$CHECKPOINT_ROOT/dynamics_diffusion"
MODEL_TAR="$CHECKPOINT_ROOT/dynamics_diffusion.tar"

mkdir -p "$DATA_ROOT" "$CHECKPOINT_ROOT" "$XICM_ROOT/data"

download_hf_file() {
  local repo_id="$1"
  local repo_type="$2"
  local filename="$3"
  local out_dir="$4"

  HF_REPO_ID="$repo_id" HF_REPO_TYPE="$repo_type" HF_FILENAME="$filename" HF_OUT_DIR="$out_dir" \
    HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-0}" \
    HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}" \
    python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

repo_id = os.environ["HF_REPO_ID"]
repo_type = os.environ["HF_REPO_TYPE"]
filename = os.environ["HF_FILENAME"]
out_dir = Path(os.environ["HF_OUT_DIR"])
out_dir.mkdir(parents=True, exist_ok=True)

path = hf_hub_download(
    repo_id=repo_id,
    repo_type=repo_type,
    filename=filename,
    local_dir=str(out_dir),
)
print(path)
PY
}

echo "[1/4] Checking seen tasks"
if [ ! -d "$SEEN_DIR" ] || [ "$(find "$SEEN_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)" -lt 18 ]; then
  echo "Seen tasks are missing or incomplete at $SEEN_DIR"
  echo "Expected the 18 AGNOSTOS seen task folders before continuing."
  exit 1
fi
echo "Seen tasks present: $(find "$SEEN_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l) task folders"

echo "[2/4] Downloading/extracting unseen tasks"
if [ -d "$UNSEEN_DIR" ] && [ "$(find "$UNSEEN_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)" -ge 23 ]; then
  echo "Unseen tasks already present at $UNSEEN_DIR"
else
  download_hf_file "Jiaming2472/AGNOSTOS" "dataset" "unseen_tasks.tar" "$DATA_ROOT"

  rm -rf "$DATA_ROOT/unseen_tasks_temp"
  mkdir -p "$DATA_ROOT/unseen_tasks_temp"
  tar -xf "$UNSEEN_TAR" -C "$DATA_ROOT/unseen_tasks_temp"
  mkdir -p "$UNSEEN_DIR"

  if [ -d "$DATA_ROOT/unseen_tasks_temp/unseen_tasks/test" ]; then
    find "$UNSEEN_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
    mv "$DATA_ROOT/unseen_tasks_temp/unseen_tasks/test/"* "$UNSEEN_DIR/"
  elif [ -d "$DATA_ROOT/unseen_tasks_temp/unseen_tasks" ]; then
    find "$UNSEEN_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
    mv "$DATA_ROOT/unseen_tasks_temp/unseen_tasks/"* "$UNSEEN_DIR/"
  else
    find "$UNSEEN_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
    mv "$DATA_ROOT/unseen_tasks_temp/"* "$UNSEEN_DIR/"
  fi
  rm -rf "$DATA_ROOT/unseen_tasks_temp"
fi
echo "Unseen tasks present: $(find "$UNSEEN_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l) task folders"

echo "[3/4] Downloading/extracting X-ICM dynamics diffusion model"
if [ -d "$MODEL_DIR" ] && [ "$(find "$MODEL_DIR" -mindepth 1 | head -1)" ]; then
  echo "Dynamics diffusion model already present at $MODEL_DIR"
else
  download_hf_file "Jiaming2472/X-ICM" "model" "dynamics_diffusion.tar" "$CHECKPOINT_ROOT"
  tar -xf "$MODEL_TAR" -C "$CHECKPOINT_ROOT"
fi

if [ ! -d "$MODEL_DIR" ]; then
  echo "Expected model directory $MODEL_DIR after extraction."
  exit 1
fi

echo "[4/4] Linking data into X-ICM"
ln -sfn "$SEEN_DIR" "$XICM_ROOT/data/seen_tasks"
ln -sfn "$UNSEEN_DIR" "$XICM_ROOT/data/unseen_tasks"
ln -sfn "$MODEL_DIR" "$XICM_ROOT/data/dynamics_diffusion"

echo "Done."
echo "X-ICM data links:"
ls -la "$XICM_ROOT/data"
