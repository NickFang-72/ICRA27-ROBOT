#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-cair}"
REMOTE_CACHE_ROOT="${REMOTE_CACHE_ROOT:-/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_target_pose_v2_full_cache_20260626}"
INTERVAL="${INTERVAL:-30}"
ONCE="${ONCE:-0}"
LOG_LINES="${LOG_LINES:-30}"
REMOTE_RUNNER="$REMOTE_CACHE_ROOT/scripts/cache_all_seen_geometry_affordance.py"
REMOTE_LOG="${REMOTE_LOG:-$REMOTE_CACHE_ROOT/full_geometry_target_pose_v2_cache.log}"
REMOTE_PID="$REMOTE_CACHE_ROOT/full_geometry_target_pose_v2_cache.pid"

render_once() {
  clear 2>/dev/null || true
  date '+%Y-%m-%d %H:%M:%S %Z'
  echo
  echo "remote root: $REMOTE_CACHE_ROOT"
  ssh -o BatchMode=yes "$REMOTE" "if test -f '$REMOTE_PID'; then printf 'pid: '; cat '$REMOTE_PID'; else echo 'pid: missing'; fi; \
    if test -f '$REMOTE_PID' && kill -0 \$(cat '$REMOTE_PID') 2>/dev/null; then echo 'process: running'; else echo 'process: not running or unavailable'; fi"
  echo
  ssh -o BatchMode=yes "$REMOTE" "python3 '$REMOTE_RUNNER' --stage status --root '$REMOTE_CACHE_ROOT' 2>/dev/null || \
    (echo 'progress status is not ready yet'; test -f '$REMOTE_CACHE_ROOT/progress.json' && cat '$REMOTE_CACHE_ROOT/progress.json' || true)"
  echo
  ssh -o BatchMode=yes "$REMOTE" "if test -f '$REMOTE_CACHE_ROOT/cache_summary.md'; then sed -n '1,80p' '$REMOTE_CACHE_ROOT/cache_summary.md'; fi"
  echo
  echo "log tail: $REMOTE_LOG"
  ssh -o BatchMode=yes "$REMOTE" "tail -n '$LOG_LINES' '$REMOTE_LOG' 2>/dev/null || true"
}

if [ "$ONCE" = "1" ]; then
  render_once
  exit 0
fi

while true; do
  render_once
  sleep "$INTERVAL"
done
