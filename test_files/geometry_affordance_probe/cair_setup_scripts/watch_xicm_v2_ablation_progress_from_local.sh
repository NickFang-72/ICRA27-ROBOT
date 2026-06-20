#!/usr/bin/env bash
set -Eeuo pipefail

# Watch only the X-ICM geometry+affordance v2 ablation from this Mac.
# This script is read-only: it does not pull logs, restart jobs, or edit files.
#
# Usage:
#   ONCE=1 bash test_files/geometry_affordance_probe/cair_setup_scripts/watch_xicm_v2_ablation_progress_from_local.sh
#   bash test_files/geometry_affordance_probe/cair_setup_scripts/watch_xicm_v2_ablation_progress_from_local.sh
#   INTERVAL_SECONDS=120 bash test_files/geometry_affordance_probe/cair_setup_scripts/watch_xicm_v2_ablation_progress_from_local.sh

CAIR_HOST="${CAIR_HOST:-cair}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-60}"
ONCE="${ONCE:-0}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_ablations}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-/data/yf23/projects/ICRA27-ROBOT/X-ICM/logs}"
METHOD="${METHOD:-XICM_Cross.ZS_Ranking.lang_vis.out.geo_aff_v2_Qwen2.5.7B.instruct_icl.18_test}"
LOCAL_TABLE="${LOCAL_TABLE:-/Users/nicholas/Documents/ICRA27 ROBOT/test_files/geometry_affordance_probe/ablation_results/xicm_geometry_affordance_ablation_paper_style_scores.csv}"

run_once() {
    printf "Local time: %s\n" "$(date '+%Y-%m-%d %H:%M:%S %Z')"
    printf "Remote host: %s\n" "$CAIR_HOST"
    printf "V2 method: %s\n\n" "$METHOD"

    ssh "$CAIR_HOST" "METHOD='$METHOD' REMOTE_LOG_ROOT='$REMOTE_LOG_ROOT' REMOTE_RUN_ROOT='$REMOTE_RUN_ROOT' python3 - <<'PY'
from pathlib import Path
import csv
import glob
import json
import os
import re
import subprocess

method = os.environ['METHOD']
log_root = Path(os.environ['REMOTE_LOG_ROOT'])
run_root = Path(os.environ['REMOTE_RUN_ROOT'])
method_dir = log_root / method
progress_path = run_root / 'progress.json'

finish_re = re.compile(r'Finished\\s+([^|]+?)\\s+\\|\\s+Final Score:\\s+([-+]?\\d+(?:\\.\\d+)?)')
eval_re = re.compile(
    r'Evaluating\\s+([^|]+?)\\s+\\|\\s+Episode\\s+(\\d+)\\s+\\|\\s+Step:\\s+(\\d+)\\s+'
    r'\\|\\s+Score:\\s+([-+]?\\d+(?:\\.\\d+)?)\\s+\\|\\s+Lang Goal:\\s+(.*)$'
)

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
        ['pgrep', '-af', 'geo_aff_v2|run_geometry_affordance_v2_on_cair|scripts/eval_XICM.sh|python main.py'],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    lines = [
        line for line in out.splitlines()
        if 'python3 - <<' not in line and 'pgrep -af' not in line
    ]
    print('\n'.join(lines) if lines else '  none')
except subprocess.CalledProcessError:
    print('  none')

csv_paths = sorted(method_dir.glob('*/seed0/test_data.csv'))
finals = []
current = None
for path in csv_paths:
    text = path.read_text(errors='replace')
    matches = list(finish_re.finditer(text))
    if matches:
        match = matches[-1]
        finals.append((path.parts[-3], float(match.group(2))))
    for line in text.splitlines():
        match = eval_re.search(line.strip())
        if match:
            current = {
                'task': match.group(1).strip(),
                'episode': int(match.group(2)),
                'step': int(match.group(3)),
                'score': float(match.group(4)),
                'lang_goal': match.group(5),
            }

print('\\nstrict finals:')
print(f'  {len(finals)}/23 final task scores, {len(csv_paths)}/23 task CSV files')
if finals:
    print('  recent finished:')
    for task, score in finals[-5:]:
        print(f'    {task}: {score:g}')

if current:
    print('\\ncurrent task:')
    print('  {} episode {}/25, last_step={}, last_score={:g}, lang_goal={}'.format(
        current['task'],
        min(current['episode'] + 1, 25),
        current['step'],
        current['score'],
        current['lang_goal'],
    ))
else:
    latest_log = None
    if progress_path.exists():
        try:
            log_value = json.loads(progress_path.read_text()).get('log_path')
            latest_log = Path(log_value) if log_value else None
        except Exception:
            latest_log = None
    if latest_log is None or not latest_log.exists():
        logs = sorted(glob.glob(str(run_root / 'logs' / '*v2*.log')), key=os.path.getmtime)
        latest_log = Path(logs[-1]) if logs else None
    print('\\ncurrent task:')
    print('  no evaluation line found yet')
    if latest_log and latest_log.exists():
        print(f'  latest log: {latest_log}')
        print('  tail:')
        for line in latest_log.read_text(errors='replace').splitlines()[-12:]:
            print('    ' + line)
PY"

    echo
    if [[ -f "$LOCAL_TABLE" ]]; then
        python3 - "$LOCAL_TABLE" <<'PY'
from pathlib import Path
import csv
import sys

path = Path(sys.argv[1])
rows = list(csv.DictReader(path.open()))
v2 = next((row for row in rows if row.get('run') == 'geometry_affordance_v2'), None)
if not v2:
    print('local table: v2 row not present yet')
    raise SystemExit(0)
tasks = [
    key for key in v2
    if key not in {'method', 'run', 'Level 1 Avg', 'Level 2 Avg', 'Average'}
]
filled = sum(1 for task in tasks if v2.get(task))
print(f'local table: {filled}/23 v2 scores filled')
print('  Average={} Level1={} Level2={}'.format(
    v2.get('Average') or '(blank)',
    v2.get('Level 1 Avg') or '(blank)',
    v2.get('Level 2 Avg') or '(blank)',
))
if filled:
    last = [(task, v2[task]) for task in tasks if v2.get(task)][-5:]
    print('  recent filled cells:')
    for task, score in last:
        print(f'    {task}: {score}')
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
