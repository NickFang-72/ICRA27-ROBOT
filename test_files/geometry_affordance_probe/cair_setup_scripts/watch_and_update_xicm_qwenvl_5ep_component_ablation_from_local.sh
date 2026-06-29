#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CAIR_HOST="${CAIR_HOST:-cair}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-120}"
ONCE="${ONCE:-0}"
REMOTE_COMPONENT_ROOT="${REMOTE_COMPONENT_ROOT:-/data/yf23/projects/ICRA27-ROBOT/experiments/qwenvl_component_5eps_seed0_20260627}"
REMOTE_RUN_XICM="${REMOTE_RUN_XICM:-$REMOTE_COMPONENT_ROOT/X-ICM_qwenvl_component_5eps}"
REMOTE_METHOD_LOG_ROOT="${REMOTE_METHOD_LOG_ROOT:-$REMOTE_RUN_XICM/logs}"
REMOTE_RUNNER_LOG_ROOT="${REMOTE_RUNNER_LOG_ROOT:-$REMOTE_COMPONENT_ROOT/runner_logs}"
REMOTE_PROGRESS_JSON="${REMOTE_PROGRESS_JSON:-$REMOTE_COMPONENT_ROOT/progress_v1_10ep_component.json}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5.VL.7B.instruct}"
DEMO_NUM_PER_ICL="${DEMO_NUM_PER_ICL:-18}"
SEEDS="${SEEDS:-0}"
EPISODES="${EPISODES:-5}"
TOTAL_TASKS="${TOTAL_TASKS:-23}"
RANKING_METHODS="${RANKING_METHODS:-lang_vis.out.geo,lang_vis.out.aff,lang_vis.out.geo.aff}"
COLLECT_METHOD_SET="${COLLECT_METHOD_SET:-default}"
LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-$REPO_ROOT/test_files/geometry_affordance_probe/ablation_results/qwenvl_component_5eps_seed0_2026-06-27}"
LOCAL_LOG_ROOT="${LOCAL_LOG_ROOT:-$LOCAL_OUTPUT_DIR/cair_logs}"
STATE_FILE="${STATE_FILE:-$LOCAL_OUTPUT_DIR/.last_strict_count}"

method_for_ranking() {
  local ranking="$1"
  printf "XICM_Cross.ZS_Ranking.%s_%s_icl.%s_test" "$ranking" "$MODEL_NAME" "$DEMO_NUM_PER_ICL"
}

strict_count() {
  ssh "$CAIR_HOST" "REMOTE_METHOD_LOG_ROOT='$REMOTE_METHOD_LOG_ROOT' RANKING_METHODS='$RANKING_METHODS' MODEL_NAME='$MODEL_NAME' DEMO_NUM_PER_ICL='$DEMO_NUM_PER_ICL' SEEDS='$SEEDS' python3 - <<'PY'
from pathlib import Path
import os
import re

log_root = Path(os.environ['REMOTE_METHOD_LOG_ROOT'])
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
    if ssh "$CAIR_HOST" "test -d '$REMOTE_METHOD_LOG_ROOT/$method'"; then
      rsync -a --include '*/' --include 'test_data.csv' --exclude '*' \
        "$CAIR_HOST:$REMOTE_METHOD_LOG_ROOT/$method/" "$LOCAL_LOG_ROOT/$method/"
    fi
  done
  mkdir -p "$LOCAL_OUTPUT_DIR/remote_runner_logs"
  if ssh "$CAIR_HOST" "test -d '$REMOTE_RUNNER_LOG_ROOT'"; then
    rsync -a --include '*.log' --exclude '*' \
      "$CAIR_HOST:$REMOTE_RUNNER_LOG_ROOT/" "$LOCAL_OUTPUT_DIR/remote_runner_logs/"
  fi
  if ssh "$CAIR_HOST" "test -f '$REMOTE_PROGRESS_JSON'"; then
    rsync -a "$CAIR_HOST:$REMOTE_PROGRESS_JSON" \
      "$LOCAL_OUTPUT_DIR/progress_qwenvl_5ep_component.json"
  fi
  if ssh "$CAIR_HOST" "find '$REMOTE_COMPONENT_ROOT' -maxdepth 1 -name 'retrieval_audit_*.jsonl' -print -quit | grep -q ."; then
    mkdir -p "$LOCAL_OUTPUT_DIR/retrieval_audits"
    rsync -a --include 'retrieval_audit_*.jsonl' --exclude '*' \
      "$CAIR_HOST:$REMOTE_COMPONENT_ROOT/" "$LOCAL_OUTPUT_DIR/retrieval_audits/"
  fi
}

collect_tables() {
  python3 "$REPO_ROOT/test_files/geometry_affordance_probe/scripts/collect_xicm_qwenvl_5ep_component_ablation_results.py" \
    --logs-root "$LOCAL_LOG_ROOT" \
    --output-dir "$LOCAL_OUTPUT_DIR" \
    --seeds "$SEEDS" \
    --episodes "$EPISODES" \
    --demo-num-per-icl "$DEMO_NUM_PER_ICL" \
    --method-set "$COLLECT_METHOD_SET"
}

print_status() {
  printf "Local time: %s\n" "$(date '+%Y-%m-%d %H:%M:%S %Z')"
  printf "Remote component root: %s\n" "$REMOTE_COMPONENT_ROOT"
  printf "Model: %s\n" "$MODEL_NAME"
  printf "Rankings: %s\n\n" "$RANKING_METHODS"
  printf "Collector method set: %s\n\n" "$COLLECT_METHOD_SET"
  ssh "$CAIR_HOST" "REMOTE_COMPONENT_ROOT='$REMOTE_COMPONENT_ROOT' REMOTE_METHOD_LOG_ROOT='$REMOTE_METHOD_LOG_ROOT' REMOTE_RUNNER_LOG_ROOT='$REMOTE_RUNNER_LOG_ROOT' REMOTE_PROGRESS_JSON='$REMOTE_PROGRESS_JSON' RANKING_METHODS='$RANKING_METHODS' MODEL_NAME='$MODEL_NAME' DEMO_NUM_PER_ICL='$DEMO_NUM_PER_ICL' SEEDS='$SEEDS' TOTAL_TASKS='$TOTAL_TASKS' python3 - <<'PY'
from pathlib import Path
import json
import os
import re
import subprocess

root = Path(os.environ['REMOTE_COMPONENT_ROOT'])
log_root = Path(os.environ['REMOTE_METHOD_LOG_ROOT'])
runner_log_root = Path(os.environ['REMOTE_RUNNER_LOG_ROOT'])
progress_path = Path(os.environ['REMOTE_PROGRESS_JSON'])
rankings = [item.strip() for item in os.environ['RANKING_METHODS'].split(',') if item.strip()]
seeds = [item.strip() for item in os.environ['SEEDS'].split(',') if item.strip()]
total_tasks = int(os.environ.get('TOTAL_TASKS', '23'))
finish_re = re.compile(r'Finished\s+([^|]+?)\s+\|\s+Final Score:\s*([-+]?\d+(?:\.\d+)?)')

print('progress:')
if progress_path.exists():
    try:
        print(json.dumps(json.loads(progress_path.read_text()), indent=2))
    except Exception as exc:
        print(f'  could not parse {progress_path}: {exc}')
else:
    print(f'  missing at {progress_path}')

print('\nprocesses:')
try:
    out = subprocess.check_output(['ps', '-eo', 'pid=,args='], text=True, stderr=subprocess.DEVNULL)
    lines = [
        line for line in out.splitlines()
        if str(root) in line
        and ('run_v1_10ep_component_ablation.sh' in line or 'eval_XICM.sh' in line or 'main.py' in line)
        and 'ps -eo' not in line
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
    print(f'  {ranking}: {count}/{len(seeds) * total_tasks}')
    for seed in seeds:
        print(f'    seed{seed}: {per_seed[seed]}/{total_tasks}')
    for _mtime, seed, task, score in sorted(latest)[-4:]:
        print(f'    latest seed{seed} {task}: {score:g}')
print(f'  total: {total}/{len(rankings) * len(seeds) * total_tasks}')

logs = sorted(runner_log_root.glob('*.log'), key=lambda path: path.stat().st_mtime) if runner_log_root.exists() else []
if logs:
    latest_log = logs[-1]
    print(f'\nlatest runner log: {latest_log}')
    for line in latest_log.read_text(errors='replace').splitlines()[-18:]:
        print('  ' + line)
PY"

  if [[ -f "$LOCAL_OUTPUT_DIR/qwenvl_5ep_component_ablation_paper_style_scores.csv" ]]; then
    echo
    echo "Local QwenVL component table:"
    python3 - "$LOCAL_OUTPUT_DIR/qwenvl_5ep_component_ablation_paper_style_scores.csv" <<'PY'
import csv
import sys
from pathlib import Path

for row in csv.DictReader(Path(sys.argv[1]).open()):
    print(f"  {row['run']}: Average={row.get('Average') or '(pending)'}")
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
    echo "Strict QwenVL component final-score count increased: ${previous} -> ${count}. Pulling logs and regenerating CSVs..."
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
print(len(seeds) * len(rankings) * int("$TOTAL_TASKS"))
PY
)"
  if (( count >= total_expected )); then
    echo
    echo "QwenVL component ablations reached ${total_expected}/${total_expected} strict seed-task final scores."
    pull_logs
    python3 "$REPO_ROOT/test_files/geometry_affordance_probe/scripts/collect_xicm_qwenvl_5ep_component_ablation_results.py" \
      --logs-root "$LOCAL_LOG_ROOT" \
      --output-dir "$LOCAL_OUTPUT_DIR" \
      --seeds "$SEEDS" \
      --episodes "$EPISODES" \
      --demo-num-per-icl "$DEMO_NUM_PER_ICL" \
      --method-set "$COLLECT_METHOD_SET" \
      --require-complete
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
