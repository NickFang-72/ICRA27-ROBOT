#!/usr/bin/env bash
set -u

ROOT_DIR="${ROOT_DIR:-/Users/nicholas/Documents/ICRA27 ROBOT}"
REMOTE="${REMOTE:-cair}"
REMOTE_XICM_ROOT="${REMOTE_XICM_ROOT:-/data/yf23/projects/ICRA27-ROBOT/X-ICM}"
REMOTE_QWEN_DIR="${REMOTE_QWEN_DIR:-/data/yf23/models/Qwen2.5-7B-Instruct}"
LOG_FILE="$ROOT_DIR/test_files/geometry_affordance_probe/xicm_goal_runner.log"
PID_FILE="$ROOT_DIR/test_files/geometry_affordance_probe/xicm_goal_runner.pid"
QWEN_RELAY="$ROOT_DIR/test_files/geometry_affordance_probe/cair_setup_scripts/stream_qwen25_7b_to_cair_from_local.sh"

SSH_OPTS=(-n -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=12)

mkdir -p "$(dirname "$LOG_FILE")"
echo "$$" > "$PID_FILE"
cd "$ROOT_DIR" || exit 1

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_FILE"
}

remote_cmd() {
  ssh "${SSH_OPTS[@]}" "$REMOTE" "$@"
}

remote_ready_report() {
  remote_cmd "cd '$REMOTE_XICM_ROOT' && \
    seen=\$(find -L data/seen_tasks/train -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l); \
    unseen=\$(find -L data/unseen_tasks/test -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l); \
    dyn=0; [ -f data/dynamics_diffusion/all_diffusion_features.pkl ] && dyn=1; \
    qwen=0; [ -f '$REMOTE_QWEN_DIR/model-00004-of-00004.safetensors' ] && qwen=1; \
    printf 'seen=%s unseen=%s dynamics=%s qwen=%s\n' \"\$seen\" \"\$unseen\" \"\$dyn\" \"\$qwen\""
}

stop_archive_relay_loop() {
  log "Stopping archive relay loop now that AGNOSTOS/dynamics are ready."
  for pid_file in \
    "$ROOT_DIR/test_files/geometry_affordance_probe/agnostos_xicm_download_loop.pid" \
    "$ROOT_DIR/test_files/geometry_affordance_probe/agnostos_xicm_download_loop.wrapper.pid"; do
    if [ -f "$pid_file" ]; then
      pid=$(cat "$pid_file" 2>/dev/null || true)
      if [[ "${pid:-}" =~ ^[0-9]+$ ]]; then
        kill "$pid" 2>/dev/null || true
      fi
    fi
  done
  pgrep -f "$ROOT_DIR.*/run_agnostos_xicm_download_loop.sh" | while read -r pid; do
    [ "$pid" = "$$" ] || kill "$pid" 2>/dev/null || true
  done
  pgrep -f "$ROOT_DIR.*/stream_archives_to_cair_from_local.sh" | while read -r pid; do
    [ "$pid" = "$$" ] || kill "$pid" 2>/dev/null || true
  done
}

log "Waiting for AGNOSTOS unseen data and X-ICM dynamics checkpoint."
while true; do
  report=$(remote_ready_report 2>/dev/null || true)
  log "${report:-remote check failed}"
  if echo "$report" | grep -q 'seen=18' &&
     echo "$report" | grep -q 'unseen=23' &&
     echo "$report" | grep -q 'dynamics=1'; then
    break
  fi
  sleep 300
done

stop_archive_relay_loop

report=$(remote_ready_report 2>/dev/null || true)
if ! echo "$report" | grep -q 'qwen=1'; then
  log "Qwen2.5-7B-Instruct is missing on CAIR; starting local-to-CAIR relay."
  REMOTE="$REMOTE" REMOTE_MODEL_DIR="$REMOTE_QWEN_DIR" PARALLEL_CHUNKS=1 "$QWEN_RELAY" 2>&1 | tee -a "$LOG_FILE"
else
  log "Qwen2.5-7B-Instruct already present on CAIR."
fi

log "Launching original-prompt X-ICM baseline on CAIR."
remote_cmd "cd '$REMOTE_XICM_ROOT' && mkdir -p logs/baseline_xicm_original_prompt && \
  nohup bash -lc 'source /data/yf23/miniconda3/etc/profile.d/conda.sh && conda activate /data/yf23/conda/envs/zero-shot && SEEDS=\"\${SEEDS:-0}\" EPISODES=\"\${EPISODES:-25}\" GPU_IDS=\"\${GPU_IDS:-0}\" NUM_ICLS=\"\${NUM_ICLS:-18}\" RANKING_METHOD=\"\${RANKING_METHOD:-lang_vis.out}\" MODELNAME=\"\${MODELNAME:-Qwen2.5.7B.instruct}\" ./run_baseline_xicm_after_data_ready.sh' \
  > logs/baseline_xicm_original_prompt/nohup_baseline_\$(date +%Y%m%d_%H%M%S).log 2>&1 & echo baseline_pid=\$!"

log "Baseline launch command completed."
