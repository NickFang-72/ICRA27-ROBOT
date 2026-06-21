#!/usr/bin/env bash
set -Eeuo pipefail

# Read-only local watcher for the v3 geometry+affordance X-ICM ablation.
#
# Usage:
#   ONCE=1 bash test_files/geometry_affordance_probe/cair_setup_scripts/watch_xicm_v3_progress_from_local.sh
#   INTERVAL_SECONDS=120 bash test_files/geometry_affordance_probe/cair_setup_scripts/watch_xicm_v3_progress_from_local.sh

CAIR_HOST="${CAIR_HOST:-cair}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-120}"
ONCE="${ONCE:-0}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_ablations}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-/data/yf23/projects/ICRA27-ROBOT/X-ICM/logs}"
METHOD="${METHOD:-XICM_Cross.ZS_Ranking.lang_vis.out.geo_aff_v3_Qwen2.5.7B.instruct_icl.6_test}"
LOCAL_TABLE="${LOCAL_TABLE:-/Users/nicholas/Documents/ICRA27 ROBOT/test_files/geometry_affordance_probe/ablation_results/xicm_geometry_affordance_ablation_paper_style_scores.csv}"

run_once() {
    printf "Local time: %s\n" "$(date '+%Y-%m-%d %H:%M:%S %Z')"
    printf "Remote host: %s\n" "$CAIR_HOST"
    printf "Method: %s\n\n" "$METHOD"

    ssh "$CAIR_HOST" "REMOTE_LOG_ROOT='$REMOTE_LOG_ROOT' REMOTE_RUN_ROOT='$REMOTE_RUN_ROOT' METHOD='$METHOD' python3 - <<'PY'
from pathlib import Path
import json
import os
import re
import subprocess

log_root = Path(os.environ['REMOTE_LOG_ROOT'])
run_root = Path(os.environ['REMOTE_RUN_ROOT'])
method = os.environ['METHOD']
finish_re = re.compile(r'Finished\\s+([^|]+?)\\s+\\|\\s+Final Score:\\s+([-+]?\\d+(?:\\.\\d+)?)')

print('progress_v3.json:')
progress_path = run_root / 'progress_v3.json'
if progress_path.exists():
    try:
        print(json.dumps(json.loads(progress_path.read_text()), indent=2))
    except Exception as exc:
        print(f'  could not parse {progress_path}: {exc}')
else:
    print(f'  missing at {progress_path}')

print('\\nprocesses:')
try:
    out = subprocess.check_output(
        ['pgrep', '-af', 'geo_aff_v3|run_geometry_affordance_v3|eval_XICM.sh|python main.py'],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    lines = [
        line for line in out.splitlines()
        if 'python3 - <<' not in line and 'pgrep -af' not in line
    ]
    print('\\n'.join(lines) if lines else '  none')
except subprocess.CalledProcessError:
    print('  none')

print('\\nstrict finals:')
method_dir = log_root / method
csv_paths = sorted(method_dir.glob('*/seed0/test_data.csv'))
finals = []
for path in csv_paths:
    text = path.read_text(errors='replace')
    matches = list(finish_re.finditer(text))
    if matches:
        task = path.parts[-3]
        score = float(matches[-1].group(2))
        finals.append((task, score))
print(f'  {len(finals)}/23 final task scores, {len(csv_paths)}/23 task CSV files')
if finals:
    for task, score in finals[-8:]:
        print(f'  {task}: {score:g}')

logs = sorted((run_root / 'logs').glob('*v3*.log'), key=lambda path: path.stat().st_mtime)
if logs:
    latest = logs[-1]
    print(f'\\nlatest v3 log: {latest}')
    print('tail:')
    for line in latest.read_text(errors='replace').splitlines()[-14:]:
        print('  ' + line)
PY"

    echo
    if [[ -f "$LOCAL_TABLE" ]]; then
        python3 - "$LOCAL_TABLE" <<'PY'
from pathlib import Path
import csv
import sys

path = Path(sys.argv[1])
rows = list(csv.DictReader(path.open()))
row = next((item for item in rows if item.get('run') == 'geometry_affordance_v3_k6'), None)
if not row:
    print('local table: geometry_affordance_v3_k6 row not present yet')
else:
    tasks = [
        key for key in row
        if key not in {'method', 'run', 'Level 1 Avg', 'Level 2 Avg', 'Average'}
    ]
    filled = [(task, row[task]) for task in tasks if row.get(task)]
    print(f"local table: geometry_affordance_v3_k6 {len(filled)}/23 scores, Average={row.get('Average') or '(blank)'}")
    for task, score in filled[-8:]:
        print(f'  {task}: {score}')
PY
    else
        echo "local table missing: $LOCAL_TABLE"
    fi
}

if [[ "$ONCE" == "1" || "$ONCE" == "true" ]]; then
    run_once
    exit 0
fi

while true; do
    clear || true
    run_once || true
    echo
    echo "Refreshing in ${INTERVAL_SECONDS}s. Press Ctrl-C to stop watching."
    sleep "$INTERVAL_SECONDS"
done
