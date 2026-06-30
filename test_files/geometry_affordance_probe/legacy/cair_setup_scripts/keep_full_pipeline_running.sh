#!/usr/bin/env bash

ROOT_DIR="${ROOT_DIR:-/Users/nicholas/Documents/ICRA27 ROBOT}"
REMOTE="${REMOTE:-cair}"
CHUNK_SIZE="${CHUNK_SIZE:-67108864}"
PARALLEL_CHUNKS="${PARALLEL_CHUNKS:-1}"
INTERVAL="${INTERVAL:-120}"
LOG_DIR="$ROOT_DIR/test_files/geometry_affordance_probe"
WATCHDOG_LOG="$LOG_DIR/full_pipeline_watchdog.log"
PIPELINE_LOG="$LOG_DIR/stream_archives_to_cair.nohup.log"
PIPELINE_PID_FILE="$LOG_DIR/stream_archives_to_cair.pid"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/data/yf23/datasets/ICRA27-ROBOT}"
REMOTE_CHECKPOINT_ROOT="${REMOTE_CHECKPOINT_ROOT:-/data/yf23/checkpoints/ICRA27-ROBOT}"
REMOTE_XICM_ROOT="${REMOTE_XICM_ROOT:-/data/yf23/projects/ICRA27-ROBOT/X-ICM}"

exec </dev/null

log() {
  mkdir -p "$LOG_DIR"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$WATCHDOG_LOG"
}

relay_pids() {
  ps -axo pid,command 2>/dev/null |
    awk '/stream_archives_to_cair_from_local[.]sh/ && $0 !~ /keep_full_pipeline_running/ {print $1}' |
    tr '\n' ' ' |
    sed 's/[[:space:]]*$//'
}

remote_complete() {
  ssh -n -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 "$REMOTE" "
    test -f '$REMOTE_DATA_ROOT/unseen_tasks.tar' &&
    test -f '$REMOTE_CHECKPOINT_ROOT/dynamics_diffusion.tar' &&
    test -d '$REMOTE_DATA_ROOT/unseen_tasks' &&
    test -d '$REMOTE_CHECKPOINT_ROOT/dynamics_diffusion' &&
    test -L '$REMOTE_XICM_ROOT/data/seen_tasks' &&
    test -L '$REMOTE_XICM_ROOT/data/unseen_tasks' &&
    test -L '$REMOTE_XICM_ROOT/data/dynamics_diffusion'
  " >/dev/null 2>&1
}

cd "$ROOT_DIR" || exit 1
log "watchdog started"

while true; do
  pids="$(relay_pids)"
  if [ -n "$pids" ]; then
    log "relay active: $pids"
    sleep "$INTERVAL"
    continue
  fi

  if remote_complete; then
    log "complete: all required archives, extracted folders, and X-ICM links exist"
    exit 0
  fi

  log "no relay active; starting DOWNLOAD_TARGET=all"
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
  log "started all-pipeline pid=$child_pid"
  wait "$child_pid"
  rc=$?
  log "all-pipeline exited rc=$rc"
  sleep "$INTERVAL"
done
