#!/usr/bin/env bash
set -u

ROOT_DIR="${ROOT_DIR:-/Users/nicholas/Documents/ICRA27 ROBOT}"
REMOTE="${REMOTE:-cair}"
CHUNK_SIZE="${CHUNK_SIZE:-67108864}"
PARALLEL_CHUNKS="${PARALLEL_CHUNKS:-1}"
INTERVAL="${INTERVAL:-60}"
LOG_DIR="$ROOT_DIR/test_files/geometry_affordance_probe"
PIPELINE_LOG="$LOG_DIR/stream_archives_to_cair.nohup.log"
SUPERVISOR_LOG="$LOG_DIR/full_cair_pipeline_supervisor.log"
PIPELINE_PID_FILE="$LOG_DIR/stream_archives_to_cair.pid"

REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/data/yf23/datasets/ICRA27-ROBOT}"
REMOTE_CHECKPOINT_ROOT="${REMOTE_CHECKPOINT_ROOT:-/data/yf23/checkpoints/ICRA27-ROBOT}"
REMOTE_XICM_ROOT="${REMOTE_XICM_ROOT:-/data/yf23/projects/ICRA27-ROBOT/X-ICM}"
UNSEEN_SIZE=20184780800
MODEL_SIZE=10436526080

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$SUPERVISOR_LOG"
}

remote_cmd() {
  local attempt
  for attempt in $(seq 1 12); do
    if ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=12 "$REMOTE" "$@"; then
      return 0
    fi
    log "remote command failed (attempt $attempt/12); retrying"
    sleep 10
  done
  return 1
}

active_relay_pids() {
  ps -axo pid,command 2>/dev/null |
    awk '/stream_archives_to_cair_from_local[.]sh/ && $0 !~ /supervise_full_cair_pipeline/ {print $1}' || true
}

remote_complete() {
  ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=12 "$REMOTE" "set -e
unseen_tar='$REMOTE_DATA_ROOT/unseen_tasks.tar'
model_tar='$REMOTE_CHECKPOINT_ROOT/dynamics_diffusion.tar'
test -f \"\$unseen_tar\" && test \"\$(stat -c %s \"\$unseen_tar\")\" = '$UNSEEN_SIZE'
test -f \"\$model_tar\" && test \"\$(stat -c %s \"\$model_tar\")\" = '$MODEL_SIZE'
test -d '$REMOTE_DATA_ROOT/unseen_tasks'
test -d '$REMOTE_CHECKPOINT_ROOT/dynamics_diffusion'
test -L '$REMOTE_XICM_ROOT/data/seen_tasks'
test -L '$REMOTE_XICM_ROOT/data/unseen_tasks'
test -L '$REMOTE_XICM_ROOT/data/dynamics_diffusion'
" >/dev/null 2>&1
  return $?
}

cd "$ROOT_DIR"
mkdir -p "$LOG_DIR"
log "supervisor started"

while true; do
  if remote_complete; then
    log "complete: AGNOSTOS unseen, dynamics checkpoint, and X-ICM links are present"
    exit 0
  fi

  pids="$(active_relay_pids | tr '\n' ' ' | sed 's/[[:space:]]*$//')"
  if [[ -n "$pids" ]]; then
    log "active relay already running: $pids"
    sleep "$INTERVAL"
    continue
  fi

  log "starting/resuming full all-pipeline"
  DOWNLOAD_TARGET=all \
    CHUNK_SIZE="$CHUNK_SIZE" \
    PARALLEL_CHUNKS="$PARALLEL_CHUNKS" \
    REMOTE="$REMOTE" \
    REMOTE_DATA_ROOT="$REMOTE_DATA_ROOT" \
    REMOTE_CHECKPOINT_ROOT="$REMOTE_CHECKPOINT_ROOT" \
    REMOTE_XICM_ROOT="$REMOTE_XICM_ROOT" \
    "$ROOT_DIR/test_files/geometry_affordance_probe/cair_setup_scripts/stream_archives_to_cair_from_local.sh" \
    >> "$PIPELINE_LOG" 2>&1 &
  child_pid=$!
  echo "$child_pid" > "$PIPELINE_PID_FILE"
  log "all-pipeline pid: $child_pid"

  if wait "$child_pid"; then
    log "all-pipeline exited successfully; verifying"
  else
    rc=$?
    log "all-pipeline exited with code $rc; will retry after ${INTERVAL}s"
    sleep "$INTERVAL"
  fi
done
