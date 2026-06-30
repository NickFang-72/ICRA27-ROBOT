#!/usr/bin/env bash
set -Eeuo pipefail

# Watch the clean v1 3-seed ablation benchmark, pull completed CAIR logs, and
# regenerate the paper-style baseline-plus-v1 comparison files.
#
# Usage:
#   ONCE=1 bash test_files/geometry_affordance_probe/cair_setup_scripts/watch_and_update_xicm_v1_ablation_from_local.sh
#   INTERVAL_SECONDS=120 bash test_files/geometry_affordance_probe/cair_setup_scripts/watch_and_update_xicm_v1_ablation_from_local.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CAIR_HOST="${CAIR_HOST:-cair}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-120}"
ONCE="${ONCE:-0}"
REMOTE_V1_ROOT="${REMOTE_V1_ROOT:-/data/yf23/projects/ICRA27-ROBOT/experiments/v1_ablation_3seed_20260624}"
REMOTE_RUN_XICM="${REMOTE_RUN_XICM:-$REMOTE_V1_ROOT/X-ICM_v1}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-$REMOTE_RUN_XICM/logs}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5.7B.instruct}"
DEMO_NUM_PER_ICL="${DEMO_NUM_PER_ICL:-18}"
SEEDS="${SEEDS:-0,50,99}"
RANKING_METHODS="${RANKING_METHODS:-lang_vis.out.geo,lang_vis.out.geo.aff}"
LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-$REPO_ROOT/test_files/geometry_affordance_probe/ablation_results/v1_3seed_2026-06-24}"
LOCAL_LOG_ROOT="${LOCAL_LOG_ROOT:-$LOCAL_OUTPUT_DIR/cair_logs}"
STATE_FILE="${STATE_FILE:-$LOCAL_OUTPUT_DIR/.last_strict_count}"

method_for_ranking() {
  local ranking="$1"
  printf "XICM_Cross.ZS_Ranking.%s_%s_icl.%s_test" "$ranking" "$MODEL_NAME" "$DEMO_NUM_PER_ICL"
}

strict_count() {
  ssh "$CAIR_HOST" "REMOTE_LOG_ROOT='$REMOTE_LOG_ROOT' RANKING_METHODS='$RANKING_METHODS' MODEL_NAME='$MODEL_NAME' DEMO_NUM_PER_ICL='$DEMO_NUM_PER_ICL' SEEDS='$SEEDS' python3 - <<'PY'
from pathlib import Path
import os
import re

log_root = Path(os.environ['REMOTE_LOG_ROOT'])
rankings = [item.strip() for item in os.environ['RANKING_METHODS'].split(',') if item.strip()]
seeds = [item.strip() for item in os.environ['SEEDS'].split(',') if item.strip()]
finish_re = re.compile(r'Finished\s+[^|]+?\s+\|\s+Final Score:\s*[-+]?\d+(?:\.\d+)?')
count = 0
for ranking in rankings:
    method = f\"XICM_Cross.ZS_Ranking.{ranking}_{os.environ['MODEL_NAME']}_icl.{os.environ['DEMO_NUM_PER_ICL']}_test\"
    method_dir = log_root / method
    if not method_dir.exists():
        continue
    for seed in seeds:
        for path in method_dir.glob(f'*/seed{seed}/test_data.csv'):
            if finish_re.search(path.read_text(errors='replace')):
                count += 1
print(count)
PY"
}

pull_logs() {
  mkdir -p "$LOCAL_LOG_ROOT"
  IFS=',' read -r -a rankings <<< "$RANKING_METHODS"
  for ranking in "${rankings[@]}"; do
    method="$(method_for_ranking "$ranking")"
    mkdir -p "$LOCAL_LOG_ROOT/$method"
    if ssh "$CAIR_HOST" "test -d '$REMOTE_LOG_ROOT/$method'"; then
      rsync -a \
        --include '*/' \
        --include 'test_data.csv' \
        --exclude '*' \
        "$CAIR_HOST:$REMOTE_LOG_ROOT/$method/" "$LOCAL_LOG_ROOT/$method/"
    fi
  done
  mkdir -p "$LOCAL_OUTPUT_DIR/remote_runner_logs"
  if ssh "$CAIR_HOST" "test -d '$REMOTE_V1_ROOT/logs'"; then
    rsync -a --include '*.log' --exclude '*' "$CAIR_HOST:$REMOTE_V1_ROOT/logs/" "$LOCAL_OUTPUT_DIR/remote_runner_logs/"
  fi
}

collect_tables() {
  python3 "$REPO_ROOT/test_files/geometry_affordance_probe/scripts/collect_xicm_v1_ablation_results.py" \
    --logs-root "$LOCAL_LOG_ROOT" \
    --output-dir "$LOCAL_OUTPUT_DIR" \
    --seeds "$SEEDS"
}

print_status() {
  printf "Local time: %s\n" "$(date '+%Y-%m-%d %H:%M:%S %Z')"
  printf "Remote host: %s\n" "$CAIR_HOST"
  printf "Remote v1 root: %s\n" "$REMOTE_V1_ROOT"
  printf "Rankings: %s\n\n" "$RANKING_METHODS"

  ssh "$CAIR_HOST" "REMOTE_V1_ROOT='$REMOTE_V1_ROOT' REMOTE_LOG_ROOT='$REMOTE_LOG_ROOT' RANKING_METHODS='$RANKING_METHODS' MODEL_NAME='$MODEL_NAME' DEMO_NUM_PER_ICL='$DEMO_NUM_PER_ICL' SEEDS='$SEEDS' python3 - <<'PY'
from pathlib import Path
import json
import os
import re
import subprocess

root = Path(os.environ['REMOTE_V1_ROOT'])
log_root = Path(os.environ['REMOTE_LOG_ROOT'])
rankings = [item.strip() for item in os.environ['RANKING_METHODS'].split(',') if item.strip()]
seeds = [item.strip() for item in os.environ['SEEDS'].split(',') if item.strip()]
finish_re = re.compile(r'Finished\s+([^|]+?)\s+\|\s+Final Score:\s*([-+]?\d+(?:\.\d+)?)')

progress_path = root / 'progress_v1.json'
print('progress_v1.json:')
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
        ['pgrep', '-af', 'v1_ablation_3seed_20260624|X-ICM_v1|eval_XICM.sh|python main.py'],
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

print('\nstrict finals:')
total = 0
for ranking in rankings:
    method = f\"XICM_Cross.ZS_Ranking.{ranking}_{os.environ['MODEL_NAME']}_icl.{os.environ['DEMO_NUM_PER_ICL']}_test\"
    method_dir = log_root / method
    count = 0
    latest = []
    per_seed = {seed: 0 for seed in seeds}
    if method_dir.exists():
        for seed in seeds:
            for path in method_dir.glob(f'*/seed{seed}/test_data.csv'):
                matches = list(finish_re.finditer(path.read_text(errors='replace')))
                if matches:
                    count += 1
                    per_seed[seed] += 1
                    latest.append((path.stat().st_mtime, seed, path.parts[-3], float(matches[-1].group(2))))
    total += count
    print(f'  {ranking}: {count}/{len(seeds) * 23} final seed-task scores')
    for seed in seeds:
        print(f'    seed{seed}: {per_seed[seed]}/23')
    for _mtime, seed, task, score in sorted(latest)[-5:]:
        print(f'    latest seed{seed} {task}: {score:g}')
print(f'  total: {total}/{len(rankings) * len(seeds) * 23}')

logs = sorted((root / 'logs').glob('*.log'), key=lambda path: path.stat().st_mtime)
if logs:
    latest_log = logs[-1]
    print(f'\nlatest runner log: {latest_log}')
    for line in latest_log.read_text(errors='replace').splitlines()[-18:]:
        print('  ' + line)
PY"

  if [[ -f "$LOCAL_OUTPUT_DIR/xicm_v1_ablation_3seed_paper_style_scores.csv" ]]; then
    echo
    echo "Local v1 comparison table:"
    python3 - "$LOCAL_OUTPUT_DIR/xicm_v1_ablation_3seed_paper_style_scores.csv" <<'PY'
import csv
import sys
from pathlib import Path

rows = list(csv.DictReader(Path(sys.argv[1]).open()))
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
    echo "Strict v1 final-score count increased: ${previous} -> ${count}. Pulling logs and regenerating comparison..."
    pull_logs
    collect_tables
  elif (( count == 0 && previous < 0 )); then
    collect_tables
  fi

  echo "$count" > "$STATE_FILE"

  local total_expected
  total_expected="$(python3 - <<PY
seeds=[item for item in "$SEEDS".split(",") if item]
rankings=[item for item in "$RANKING_METHODS".split(",") if item]
print(len(seeds) * len(rankings) * 23)
PY
)"

  if (( count >= total_expected )); then
    echo
    echo "v1 ablations reached ${total_expected}/${total_expected} strict seed-task final scores. Pulling final logs and requiring complete collection..."
    pull_logs
    python3 "$REPO_ROOT/test_files/geometry_affordance_probe/scripts/collect_xicm_v1_ablation_results.py" \
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
