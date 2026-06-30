#!/usr/bin/env bash
set -Eeuo pipefail

# Watch the clean vanilla X-ICM baseline rerun, pull completed logs, and
# regenerate the baseline-vs-paper comparison files.
#
# Usage:
#   ONCE=1 bash test_files/geometry_affordance_probe/cair_setup_scripts/watch_and_update_xicm_baseline_rerun_from_local.sh
#   INTERVAL_SECONDS=120 bash test_files/geometry_affordance_probe/cair_setup_scripts/watch_and_update_xicm_baseline_rerun_from_local.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CAIR_HOST="${CAIR_HOST:-cair}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-120}"
ONCE="${ONCE:-0}"
REMOTE_BASELINE_ROOT="${REMOTE_BASELINE_ROOT:-/data/yf23/projects/ICRA27-ROBOT/experiments/vanilla_baseline_rerun_3seed_20260622}"
REMOTE_RUN_XICM="${REMOTE_RUN_XICM:-$REMOTE_BASELINE_ROOT/X-ICM_vanilla}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-$REMOTE_RUN_XICM/logs}"
METHOD="${METHOD:-XICM_Cross.ZS_Ranking.lang_vis.out_Qwen2.5.7B.instruct_icl.18_test}"
SEEDS="${SEEDS:-0,50,99}"
LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-$REPO_ROOT/test_files/xicm_baseline_results/vanilla_rerun_3seed_2026-06-22}"
LOCAL_LOG_ROOT="${LOCAL_LOG_ROOT:-$LOCAL_OUTPUT_DIR/cair_logs}"
STATE_FILE="${STATE_FILE:-$LOCAL_OUTPUT_DIR/.last_strict_count}"

strict_count() {
  ssh "$CAIR_HOST" "REMOTE_LOG_ROOT='$REMOTE_LOG_ROOT' METHOD='$METHOD' SEEDS='$SEEDS' python3 - <<'PY'
from pathlib import Path
import os
import re

method_dir = Path(os.environ['REMOTE_LOG_ROOT']) / os.environ['METHOD']
seeds = [item.strip() for item in os.environ['SEEDS'].split(',') if item.strip()]
finish_re = re.compile(r'Finished\s+[^|]+?\s+\|\s+Final Score:\s*[-+]?\d+(?:\.\d+)?')
count = 0
if method_dir.exists():
    for seed in seeds:
        for path in method_dir.glob(f'*/seed{seed}/test_data.csv'):
            if finish_re.search(path.read_text(errors='replace')):
                count += 1
print(count)
PY"
}

pull_logs() {
  mkdir -p "$LOCAL_LOG_ROOT/$METHOD"
  if ssh "$CAIR_HOST" "test -d '$REMOTE_LOG_ROOT/$METHOD'"; then
    rsync -a "$CAIR_HOST:$REMOTE_LOG_ROOT/$METHOD/" "$LOCAL_LOG_ROOT/$METHOD/"
  fi
}

collect_tables() {
  python3 "$REPO_ROOT/test_files/geometry_affordance_probe/scripts/collect_xicm_baseline_rerun_results.py" \
    --logs-root "$LOCAL_LOG_ROOT" \
    --output-dir "$LOCAL_OUTPUT_DIR" \
    --seeds "$SEEDS"
}

print_status() {
  printf "Local time: %s\n" "$(date '+%Y-%m-%d %H:%M:%S %Z')"
  printf "Remote host: %s\n" "$CAIR_HOST"
  printf "Remote baseline root: %s\n" "$REMOTE_BASELINE_ROOT"
  printf "Method: %s\n\n" "$METHOD"

  ssh "$CAIR_HOST" "REMOTE_BASELINE_ROOT='$REMOTE_BASELINE_ROOT' REMOTE_LOG_ROOT='$REMOTE_LOG_ROOT' METHOD='$METHOD' SEEDS='$SEEDS' python3 - <<'PY'
from pathlib import Path
import json
import os
import re
import subprocess

baseline_root = Path(os.environ['REMOTE_BASELINE_ROOT'])
log_root = Path(os.environ['REMOTE_LOG_ROOT'])
method = os.environ['METHOD']
seeds = [item.strip() for item in os.environ['SEEDS'].split(',') if item.strip()]
finish_re = re.compile(r'Finished\s+([^|]+?)\s+\|\s+Final Score:\s*([-+]?\d+(?:\.\d+)?)')

progress_path = baseline_root / 'progress_baseline.json'
print('progress_baseline.json:')
if progress_path.exists():
    try:
        print(json.dumps(json.loads(progress_path.read_text()), indent=2))
    except Exception as exc:
        print(f'  could not parse {progress_path}: {exc}')
else:
    print(f'  missing at {progress_path}')

print('\nprocesses:')
try:
    out = subprocess.check_output(
        ['pgrep', '-af', 'vanilla_baseline_rerun_3seed_20260622|X-ICM_vanilla|eval_XICM.sh|python main.py'],
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

method_dir = log_root / method
finals = []
csv_paths = []
per_seed_counts = {}
for seed in seeds:
    seed_paths = sorted(method_dir.glob(f'*/seed{seed}/test_data.csv'))
    csv_paths.extend(seed_paths)
    per_seed_counts[seed] = 0
    for path in seed_paths:
        matches = list(finish_re.finditer(path.read_text(errors='replace')))
        if matches:
            task = path.parts[-3]
            score = float(matches[-1].group(2))
            finals.append((seed, task, score))
            per_seed_counts[seed] += 1
print('\nstrict finals:')
print(f'  {len(finals)}/{len(seeds) * 23} final seed-task scores, {len(csv_paths)}/{len(seeds) * 23} seed-task CSV files')
for seed in seeds:
    print(f'  seed{seed}: {per_seed_counts.get(seed, 0)}/23 finals')
for seed, task, score in finals[-10:]:
    print(f'  seed{seed} {task}: {score:g}')

logs = sorted((baseline_root / 'logs').glob('vanilla_baseline_seed*.log'), key=lambda path: path.stat().st_mtime)
if logs:
    latest = logs[-1]
    print(f'\nlatest log: {latest}')
    print('tail:')
    for line in latest.read_text(errors='replace').splitlines()[-18:]:
        print('  ' + line)
PY"

  if [[ -f "$LOCAL_OUTPUT_DIR/vanilla_baseline_rerun_3seed_paper_style_scores.csv" ]]; then
    echo
    echo "Local comparison table:"
    python3 - "$LOCAL_OUTPUT_DIR/vanilla_baseline_rerun_3seed_paper_style_scores.csv" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = list(csv.DictReader(path.open()))
for row in rows:
    print(f"  {row['method']}: Average={row.get('Average') or '(pending)'}, Level 1={row.get('Level 1 Avg') or '(pending)'}, Level 2={row.get('Level 2 Avg') or '(pending)'}")
PY
  fi
}

run_once() {
  print_status

  local count
  count="$(strict_count | tail -n 1 | tr -d '[:space:]')"
  if [[ -z "$count" || ! "$count" =~ ^[0-9]+$ ]]; then
    echo "Could not read strict final-score count."
    return 0
  fi

  mkdir -p "$LOCAL_OUTPUT_DIR"
  local previous="-1"
  if [[ -f "$STATE_FILE" ]]; then
    previous="$(cat "$STATE_FILE" 2>/dev/null || echo -1)"
  fi
  if [[ ! "$previous" =~ ^-?[0-9]+$ ]]; then
    previous="-1"
  fi

  if (( count > previous && count > 0 )); then
    echo
    echo "Strict baseline final-score count increased: ${previous} -> ${count}. Pulling logs and regenerating comparison..."
    pull_logs
    collect_tables
  elif (( count == 0 && previous < 0 )); then
    collect_tables
  fi

  echo "$count" > "$STATE_FILE"

  local total_expected
  total_expected="$(python3 - <<PY
seeds=[item for item in "$SEEDS".split(",") if item]
print(len(seeds) * 23)
PY
)"

  if (( count >= total_expected )); then
    echo
    echo "Baseline rerun reached ${total_expected}/${total_expected} strict seed-task final scores. Pulling final logs and requiring complete collection..."
    pull_logs
    python3 "$REPO_ROOT/test_files/geometry_affordance_probe/scripts/collect_xicm_baseline_rerun_results.py" \
      --logs-root "$LOCAL_LOG_ROOT" \
      --output-dir "$LOCAL_OUTPUT_DIR" \
      --seeds "$SEEDS" \
      --require-complete
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
