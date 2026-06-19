#!/usr/bin/env bash
set -Eeuo pipefail

# Watch X-ICM geometry/affordance ablation progress from this Mac.
#
# Usage:
#   bash test_files/geometry_affordance_probe/cair_setup_scripts/watch_xicm_ablation_progress_from_local.sh
#   ONCE=1 bash test_files/geometry_affordance_probe/cair_setup_scripts/watch_xicm_ablation_progress_from_local.sh
#   INTERVAL_SECONDS=120 bash test_files/geometry_affordance_probe/cair_setup_scripts/watch_xicm_ablation_progress_from_local.sh

CAIR_HOST="${CAIR_HOST:-cair}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-60}"
ONCE="${ONCE:-0}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_ablations}"
REMOTE_XICM_ROOT="${REMOTE_XICM_ROOT:-/data/yf23/projects/ICRA27-ROBOT/X-ICM}"
REMOTE_WATCHER="${REMOTE_WATCHER:-/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_full_cache/scripts/watch_xicm_ablation_progress.sh}"

run_once() {
    printf "Local time: %s\n" "$(date '+%Y-%m-%d %H:%M:%S %Z')"
    printf "Remote host: %s\n\n" "$CAIR_HOST"
    ssh "$CAIR_HOST" \
        "RUN_ROOT='$REMOTE_RUN_ROOT' XICM_ROOT='$REMOTE_XICM_ROOT' bash '$REMOTE_WATCHER'"
}

if [[ "$ONCE" == "1" || "$ONCE" == "true" ]]; then
    run_once
    exit 0
fi

while true; do
    clear || true
    if ! run_once; then
        echo
        echo "Watcher failed. If this mentions Tailscale, approve/login and rerun:"
        echo "  ssh cair 'echo ok'"
    fi
    echo
    echo "Refreshing in ${INTERVAL_SECONDS}s. Press Ctrl-C to stop watching."
    sleep "$INTERVAL_SECONDS"
done
