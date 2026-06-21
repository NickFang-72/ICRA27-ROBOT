#!/usr/bin/env bash
set -Eeuo pipefail

# Read-only watcher for the compact v2 geometry+affordance K sweep.
#
# Usage:
#   ONCE=1 bash test_files/geometry_affordance_probe/cair_setup_scripts/watch_xicm_v2_k_sweep_progress_from_local.sh
#   bash test_files/geometry_affordance_probe/cair_setup_scripts/watch_xicm_v2_k_sweep_progress_from_local.sh

CAIR_HOST="${CAIR_HOST:-cair}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-120}"
ONCE="${ONCE:-0}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_ablations}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-/data/yf23/projects/ICRA27-ROBOT/X-ICM/logs}"
LOCAL_TABLE="${LOCAL_TABLE:-/Users/nicholas/Documents/ICRA27 ROBOT/test_files/geometry_affordance_probe/ablation_results/xicm_geometry_affordance_ablation_paper_style_scores.csv}"

run_once() {
    printf "Local time: %s\n" "$(date '+%Y-%m-%d %H:%M:%S %Z')"
    printf "Remote host: %s\n\n" "$CAIR_HOST"

    ssh "$CAIR_HOST" "REMOTE_LOG_ROOT='$REMOTE_LOG_ROOT' REMOTE_RUN_ROOT='$REMOTE_RUN_ROOT' python3 - <<'PY'
from pathlib import Path
import json
import os
import re
import subprocess

log_root = Path(os.environ['REMOTE_LOG_ROOT'])
run_root = Path(os.environ['REMOTE_RUN_ROOT'])
methods = [
    ('k=6', 'XICM_Cross.ZS_Ranking.lang_vis.out.geo_aff_v2_Qwen2.5.7B.instruct_icl.6_test'),
    ('k=8', 'XICM_Cross.ZS_Ranking.lang_vis.out.geo_aff_v2_Qwen2.5.7B.instruct_icl.8_test'),
    ('k=10', 'XICM_Cross.ZS_Ranking.lang_vis.out.geo_aff_v2_Qwen2.5.7B.instruct_icl.10_test'),
]
finish_re = re.compile(r'Finished\\s+([^|]+?)\\s+\\|\\s+Final Score:\\s+([-+]?\\d+(?:\\.\\d+)?)')

progress_path = run_root / 'progress.json'
print('progress.json:')
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
        ['pgrep', '-af', 'geo_aff_v2|run_geometry_affordance_v2|eval_XICM.sh|python main.py'],
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

print('\\nstrict finals by run:')
for label, method in methods:
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
    print(f'  {label}: {len(finals)}/23 final task scores, {len(csv_paths)}/23 task CSV files')
    if finals:
        recent = ', '.join(f'{task}={score:g}' for task, score in finals[-5:])
        print(f'    recent: {recent}')

logs = sorted((run_root / 'logs').glob('*v2*.log'), key=lambda path: path.stat().st_mtime)
if logs:
    latest = logs[-1]
    print(f'\\nlatest v2 log: {latest}')
    print('tail:')
    for line in latest.read_text(errors='replace').splitlines()[-12:]:
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
wanted = [
    'geometry_affordance_v2_k6',
    'geometry_affordance_v2_k8',
    'geometry_affordance_v2_k10',
]
tasks = None
for run in wanted:
    row = next((item for item in rows if item.get('run') == run), None)
    if not row:
        print(f'local table: {run} row not present yet')
        continue
    if tasks is None:
        tasks = [
            key for key in row
            if key not in {'method', 'run', 'Level 1 Avg', 'Level 2 Avg', 'Average'}
        ]
    filled = [(task, row[task]) for task in tasks if row.get(task)]
    print(f"local table: {run} {len(filled)}/23 scores, Average={row.get('Average') or '(blank)'}")
    for task, score in filled[-5:]:
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
