#!/usr/bin/env bash
set -u

ROOT_DIR="${ROOT_DIR:-/Users/nicholas/Documents/ICRA27 ROBOT}"
LOG_FILE="$ROOT_DIR/test_files/geometry_affordance_probe/agnostos_xicm_download_loop.log"
PID_FILE="$ROOT_DIR/test_files/geometry_affordance_probe/agnostos_xicm_download_loop.pid"
SCRIPT="$ROOT_DIR/test_files/geometry_affordance_probe/cair_setup_scripts/stream_archives_to_cair_from_local.sh"
QWEN_RELAY="$ROOT_DIR/test_files/geometry_affordance_probe/cair_setup_scripts/stream_qwen25_7b_to_cair_from_local.sh"
REMOTE="${REMOTE:-cair}"
REMOTE_XICM_ROOT="${REMOTE_XICM_ROOT:-/data/yf23/projects/ICRA27-ROBOT/X-ICM}"
REMOTE_QWEN_DIR="${REMOTE_QWEN_DIR:-/data/yf23/models/Qwen2.5-7B-Instruct}"
REMOTE_CHECKPOINT_ROOT="${REMOTE_CHECKPOINT_ROOT:-/data/yf23/checkpoints/ICRA27-ROBOT}"

mkdir -p "$(dirname "$LOG_FILE")"
echo "$$" > "$PID_FILE"
cd "$ROOT_DIR" || exit 1

while true; do
  printf '[%s] starting/resuming DOWNLOAD_TARGET=unseen\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
  DOWNLOAD_TARGET=unseen CHUNK_SIZE=67108864 PARALLEL_CHUNKS=1 "$SCRIPT" >> "$LOG_FILE" 2>&1
  rc=$?
  if [ "$rc" = 0 ]; then
    printf '[%s] AGNOSTOS unseen relay finished; checking dynamics checkpoint\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
    if ! ssh -n -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=12 "$REMOTE" "[ -f '$REMOTE_XICM_ROOT/data/dynamics_diffusion/all_diffusion_features.pkl' ]"; then
      printf '[%s] dynamics checkpoint missing; relaying to CAIR\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
      REMOTE="$REMOTE" REMOTE_CHECKPOINT_ROOT="$REMOTE_CHECKPOINT_ROOT" REMOTE_XICM_ROOT="$REMOTE_XICM_ROOT" DOWNLOAD_TARGET=dynamics CHUNK_SIZE=67108864 PARALLEL_CHUNKS=1 "$SCRIPT" >> "$LOG_FILE" 2>&1 || {
        printf '[%s] dynamics relay failed; restarting in 30s\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
        sleep 30
        continue
      }
    fi

    printf '[%s] AGNOSTOS unseen and dynamics ready; checking Qwen and launching baseline\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
    if ! ssh -n -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=12 "$REMOTE" "[ -f '$REMOTE_QWEN_DIR/model-00004-of-00004.safetensors' ]"; then
      printf '[%s] Qwen2.5-7B-Instruct missing; relaying to CAIR\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
      REMOTE="$REMOTE" REMOTE_MODEL_DIR="$REMOTE_QWEN_DIR" PARALLEL_CHUNKS=1 "$QWEN_RELAY" >> "$LOG_FILE" 2>&1 || {
        printf '[%s] Qwen relay failed; restarting in 30s\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
        sleep 30
        continue
      }
    fi

    ssh -n -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=12 "$REMOTE" "cd '$REMOTE_XICM_ROOT' && mkdir -p logs/baseline_xicm_original_prompt && nohup bash -lc 'source /data/yf23/miniconda3/etc/profile.d/conda.sh && conda activate /data/yf23/conda/envs/zero-shot && SEEDS=\"\${SEEDS:-0}\" EPISODES=\"\${EPISODES:-25}\" GPU_IDS=\"\${GPU_IDS:-0}\" NUM_ICLS=\"\${NUM_ICLS:-18}\" RANKING_METHOD=\"\${RANKING_METHOD:-lang_vis.out}\" MODELNAME=\"\${MODELNAME:-Qwen2.5.7B.instruct}\" ./run_baseline_xicm_after_data_ready.sh' > logs/baseline_xicm_original_prompt/nohup_baseline_\$(date +%Y%m%d_%H%M%S).log 2>&1 & echo baseline_pid=\$!" >> "$LOG_FILE" 2>&1
    printf '[%s] Baseline launch attempted; exiting loop\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
    exit 0
  fi
  printf '[%s] relay exited rc=%s; restarting in 30s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$rc" >> "$LOG_FILE"
  sleep 30
done
