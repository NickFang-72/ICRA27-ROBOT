#!/usr/bin/env bash
set -Eeuo pipefail

# Interval watcher for the improved v4 geometry+affordance ablation.
# It performs read-only CAIR status checks, and when a new strict final score
# appears, it pulls completed logs and regenerates the local wide tables.
#
# Usage:
#   INTERVAL_SECONDS=120 bash test_files/geometry_affordance_probe/cair_setup_scripts/watch_and_update_xicm_v4_progress_from_local.sh
#   ONCE=1 bash test_files/geometry_affordance_probe/cair_setup_scripts/watch_and_update_xicm_v4_progress_from_local.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CAIR_HOST="${CAIR_HOST:-cair}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-120}"
ONCE="${ONCE:-0}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-/data/yf23/projects/ICRA27-ROBOT/X-ICM/logs}"
METHOD="${METHOD:-XICM_Cross.ZS_Ranking.lang_vis.out.geo_aff_v4_Qwen2.5.7B.instruct_icl.6_test}"
STATE_FILE="${STATE_FILE:-$REPO_ROOT/test_files/geometry_affordance_probe/ablation_results/.v4_k6_last_strict_count}"

strict_count() {
    ssh "$CAIR_HOST" "REMOTE_LOG_ROOT='$REMOTE_LOG_ROOT' METHOD='$METHOD' python3 - <<'PY'
from pathlib import Path
import os
import re

method_dir = Path(os.environ['REMOTE_LOG_ROOT']) / os.environ['METHOD']
finish_re = re.compile(r'Finished\s+[^|]+?\s+\|\s+Final Score:\s*[-+]?\d+(?:\.\d+)?')
count = 0
if method_dir.exists():
    for path in method_dir.glob('*/seed0/test_data.csv'):
        if finish_re.search(path.read_text(errors='replace')):
            count += 1
print(count)
PY"
}

run_once() {
    ONCE=1 bash "$SCRIPT_DIR/watch_xicm_v4_progress_from_local.sh"

    local count
    count="$(strict_count | tail -n 1 | tr -d '[:space:]')"
    if [[ -z "$count" || ! "$count" =~ ^[0-9]+$ ]]; then
        echo "Could not read strict final-score count."
        return 0
    fi

    mkdir -p "$(dirname "$STATE_FILE")"
    local previous="-1"
    if [[ -f "$STATE_FILE" ]]; then
        previous="$(cat "$STATE_FILE" 2>/dev/null || echo -1)"
    fi
    if [[ ! "$previous" =~ ^-?[0-9]+$ ]]; then
        previous="-1"
    fi

    if (( count > previous && count > 0 )); then
        echo
        echo "Strict v4 final-score count increased: ${previous} -> ${count}. Pulling logs and regenerating tables..."
        bash "$SCRIPT_DIR/pull_xicm_ablation_results_from_cair.sh"
        python3 "$REPO_ROOT/test_files/geometry_affordance_probe/scripts/collect_xicm_ablation_results.py"
    fi

    echo "$count" > "$STATE_FILE"

    if (( count >= 23 )); then
        echo
        echo "v4 reached 23/23 strict final scores. Running final table collection..."
        bash "$SCRIPT_DIR/pull_xicm_ablation_results_from_cair.sh"
        python3 "$REPO_ROOT/test_files/geometry_affordance_probe/scripts/collect_xicm_ablation_results.py" --require-complete
        echo "Complete."
        return 2
    fi
}

if [[ "$ONCE" == "1" || "$ONCE" == "true" ]]; then
    run_once || true
    exit 0
fi

while true; do
    clear || true
    set +e
    run_once
    status=$?
    set -e
    if [[ "$status" == "2" ]]; then
        exit 0
    fi
    echo
    echo "Refreshing in ${INTERVAL_SECONDS}s. Press Ctrl-C to stop watching."
    sleep "$INTERVAL_SECONDS"
done
