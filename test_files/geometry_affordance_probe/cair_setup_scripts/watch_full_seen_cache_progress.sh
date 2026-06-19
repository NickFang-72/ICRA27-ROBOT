#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-cair}"
REMOTE_CACHE_ROOT="${REMOTE_CACHE_ROOT:-/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_full_cache}"
INTERVAL="${INTERVAL:-15}"
ONCE="${ONCE:-0}"
LOG_LINES="${LOG_LINES:-20}"
REMOTE_RUNNER="$REMOTE_CACHE_ROOT/scripts/cache_all_seen_geometry_affordance.py"
REMOTE_LOG="${REMOTE_LOG:-$REMOTE_CACHE_ROOT/full_cache_all.log}"

render_once() {
  clear 2>/dev/null || true
  date '+%Y-%m-%d %H:%M:%S %Z'
  echo
  ssh -o BatchMode=yes "$REMOTE" "python3 '$REMOTE_RUNNER' --stage status --root '$REMOTE_CACHE_ROOT' 2>/dev/null || \
    (echo 'progress status is not ready yet'; test -f '$REMOTE_CACHE_ROOT/progress.json' && cat '$REMOTE_CACHE_ROOT/progress.json' || true)"
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
