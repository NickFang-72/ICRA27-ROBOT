#!/usr/bin/env bash
set -Eeuo pipefail

# Watch the quick k=4/k=8 top-50 rerank run on CAIR.
#
# Usage:
#   bash test_files/geometry_affordance_probe/cair_setup_scripts/watch_rerank_top50_k4_k8_5ep_from_local.sh
#   INTERVAL_SECONDS=30 bash test_files/geometry_affordance_probe/cair_setup_scripts/watch_rerank_top50_k4_k8_5ep_from_local.sh
#   ONCE=1 bash test_files/geometry_affordance_probe/cair_setup_scripts/watch_rerank_top50_k4_k8_5ep_from_local.sh

CAIR_HOST="${CAIR_HOST:-cair}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-60}"
ONCE="${ONCE:-0}"
TAIL_LINES="${TAIL_LINES:-80}"
REMOTE_COMPONENT_ROOT="${REMOTE_COMPONENT_ROOT:-/data/yf23/projects/ICRA27-ROBOT/experiments/rerank_top50_k4_k8_5eps_20260701}"
REMOTE_RUN_XICM="${REMOTE_RUN_XICM:-$REMOTE_COMPONENT_ROOT/X-ICM_rerank_top50_k4_k8_5eps}"
REMOTE_METHOD_LOG_ROOT="${REMOTE_METHOD_LOG_ROOT:-$REMOTE_RUN_XICM/logs}"
REMOTE_RUNNER_LOG_ROOT="${REMOTE_RUNNER_LOG_ROOT:-$REMOTE_COMPONENT_ROOT/runner_logs}"
REMOTE_PROGRESS_JSON="${REMOTE_PROGRESS_JSON:-$REMOTE_COMPONENT_ROOT/progress_rerank_top50_k4_k8_5eps.json}"

print_remote_status() {
  ssh "$CAIR_HOST" \
    "REMOTE_COMPONENT_ROOT='$REMOTE_COMPONENT_ROOT' REMOTE_RUN_XICM='$REMOTE_RUN_XICM' REMOTE_METHOD_LOG_ROOT='$REMOTE_METHOD_LOG_ROOT' REMOTE_RUNNER_LOG_ROOT='$REMOTE_RUNNER_LOG_ROOT' REMOTE_PROGRESS_JSON='$REMOTE_PROGRESS_JSON' TAIL_LINES='$TAIL_LINES' python3 - <<'PY'
from pathlib import Path
import json
import os
import re
import subprocess

component_root = Path(os.environ['REMOTE_COMPONENT_ROOT'])
run_xicm = Path(os.environ['REMOTE_RUN_XICM'])
method_log_root = Path(os.environ['REMOTE_METHOD_LOG_ROOT'])
runner_log_root = Path(os.environ['REMOTE_RUNNER_LOG_ROOT'])
progress_path = Path(os.environ['REMOTE_PROGRESS_JSON'])
tail_lines = int(os.environ.get('TAIL_LINES', '80'))
finish_re = re.compile(r'Finished\s+([^|]+?)\s+\|\s+Final Score:\s*([-+]?\d+(?:\.\d+)?)')

def run_cmd(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ''

progress = {}
if progress_path.exists():
    try:
        progress = json.loads(progress_path.read_text())
    except Exception as exc:
        progress = {'status': 'unparseable', 'message': str(exc)}

print(f'Remote time: {run_cmd([\"date\", \"+%Y-%m-%d %H:%M:%S %Z\"])}')
print(f'Component root: {component_root}')
print(f'Progress JSON:  {progress_path}')
print()

if progress:
    print('Progress summary:')
    print(f\"  status: {progress.get('status', 'unknown')}\")
    print(f\"  message: {progress.get('message', '')}\")
    print(f\"  active_k: {progress.get('active_k', '')}\")
    print(f\"  ranking: {progress.get('ranking_method', '')}\")
    print(f\"  model: {progress.get('model_name', '')}\")
    print(f\"  episodes: {progress.get('episodes', '')}\")
    print(f\"  rerank_candidates: {progress.get('rerank_candidates', '')}\")
    print(f\"  updated_utc: {progress.get('updated_utc', '')}\")
else:
    print('Progress summary: missing progress JSON')
print()

print('GPU:')
gpu = run_cmd([
    'nvidia-smi',
    '--query-gpu=index,memory.used,memory.total,utilization.gpu',
    '--format=csv,noheader,nounits',
])
print(gpu or '  nvidia-smi unavailable')
print()

print('Processes:')
ps = run_cmd(['ps', '-eo', 'pid=,etime=,args='])
process_lines = []
for line in ps.splitlines():
    if str(component_root) in line or 'lang_vis.out.geo.aff_v3' in line:
        if 'ps -eo' not in line and 'python3 - <<' not in line:
            process_lines.append(line.strip())
print('\\n'.join(process_lines[:12]) if process_lines else '  no matching processes')
print()

ranking = progress.get('ranking_method', 'lang_vis.out.geo.aff_v3.rerank_top50')
model_name = progress.get('model_name', 'Qwen2.5.VL.7B.instruct')
k_values = progress.get('k_values') or ['4', '8']
seeds = [item.strip() for item in str(progress.get('seeds', '0')).split(',') if item.strip()]
per_k = progress.get('per_k') or {}

print('Completion by k:')
for k in k_values:
    key = f'k{k}'
    method = (per_k.get(key) or {}).get('method') or f'XICM_Cross.ZS_Ranking.{ranking}_{model_name}_icl.{k}_test'
    method_dir = method_log_root / method
    finished = []
    for seed in seeds:
        if method_dir.exists():
            for path in method_dir.glob(f'*/seed{seed}/test_data.csv'):
                text = path.read_text(errors='replace')
                match = finish_re.search(text)
                if match:
                    finished.append({
                        'task': path.parts[-3],
                        'seed': seed,
                        'score': match.group(2),
                        'mtime': path.stat().st_mtime,
                    })
    total = (per_k.get(key) or {}).get('total_seed_task_csvs') or (23 * max(1, len(seeds)))
    print(f'  k={k}: {len(finished)}/{total} finished')
    for item in sorted(finished, key=lambda row: row['mtime'], reverse=True)[:8]:
        print(f\"    {item['task']} seed{item['seed']} score={item['score']}\")
print()

active_log = progress.get('active_log_path') or ''
if not active_log and runner_log_root.exists():
    logs = sorted(runner_log_root.glob('*.log'), key=lambda path: path.stat().st_mtime, reverse=True)
    active_log = str(logs[0]) if logs else ''

print('Active log:')
print(f'  {active_log or \"none\"}')
if active_log and Path(active_log).exists():
    lines = Path(active_log).read_text(errors='replace').splitlines()
    print(f'--- tail -{tail_lines} {active_log} ---')
    for line in lines[-tail_lines:]:
        print(line)
PY"
}

while true; do
  clear 2>/dev/null || true
  printf "Local time: %s\n" "$(date '+%Y-%m-%d %H:%M:%S %Z')"
  print_remote_status
  if [[ "$ONCE" == "1" ]]; then
    break
  fi
  printf "\nRefreshing in %ss. Press Ctrl-C to stop watching.\n" "$INTERVAL_SECONDS"
  sleep "$INTERVAL_SECONDS"
done
