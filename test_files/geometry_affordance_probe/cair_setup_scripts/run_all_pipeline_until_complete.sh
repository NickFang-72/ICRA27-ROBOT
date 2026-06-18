#!/usr/bin/env bash
set -u

ROOT_DIR="${ROOT_DIR:-/Users/nicholas/Documents/ICRA27 ROBOT}"
REMOTE="${REMOTE:-cair}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/data/yf23/datasets/ICRA27-ROBOT}"
REMOTE_CHECKPOINT_ROOT="${REMOTE_CHECKPOINT_ROOT:-/data/yf23/checkpoints/ICRA27-ROBOT}"
REMOTE_XICM_ROOT="${REMOTE_XICM_ROOT:-/data/yf23/projects/ICRA27-ROBOT/X-ICM}"
LOG_DIR="$ROOT_DIR/test_files/geometry_affordance_probe"
RESTART_LOG="$LOG_DIR/all_pipeline_restart_loop.log"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"

log() {
  mkdir -p "$LOG_DIR"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$RESTART_LOG"
}

complete() {
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
log "restart loop started"

while true; do
  if complete; then
    log "complete"
    exit 0
  fi

  log "starting DOWNLOAD_TARGET=all relay"
  DOWNLOAD_TARGET=all \
    CHUNK_SIZE="${CHUNK_SIZE:-67108864}" \
    PARALLEL_CHUNKS="${PARALLEL_CHUNKS:-1}" \
    REMOTE="$REMOTE" \
    REMOTE_DATA_ROOT="$REMOTE_DATA_ROOT" \
    REMOTE_CHECKPOINT_ROOT="$REMOTE_CHECKPOINT_ROOT" \
    REMOTE_XICM_ROOT="$REMOTE_XICM_ROOT" \
    "$ROOT_DIR/test_files/geometry_affordance_probe/cair_setup_scripts/stream_archives_to_cair_from_local.sh"
  rc=$?
  log "relay exited rc=$rc"
  sleep "$SLEEP_SECONDS"
done
