#!/usr/bin/env bash
set -Eeuo pipefail

CAIR_HOST="${CAIR_HOST:-cair}"
REMOTE_EXPERIMENT_ROOT="${REMOTE_EXPERIMENT_ROOT:-/data/yf23/projects/ICRA27-ROBOT/experiments/qwen_vs_qwenvl_front_top_baseline_20260626}"
PROGRESS_JSON="${PROGRESS_JSON:-$REMOTE_EXPERIMENT_ROOT/progress_qwen_vs_qwenvl.json}"
REMOTE_SUMMARY_CSV="${REMOTE_SUMMARY_CSV:-$REMOTE_EXPERIMENT_ROOT/qwen_vs_qwenvl_front_top_results.csv}"
REMOTE_TASK_MEAN_CSV="${REMOTE_TASK_MEAN_CSV:-$REMOTE_EXPERIMENT_ROOT/qwen_vs_qwenvl_front_top_task_means.csv}"
LOCAL_OUT_DIR="${LOCAL_OUT_DIR:-test_files/geometry_affordance_probe/ablation_results/qwen_vs_qwenvl_front_top_2026-06-26}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-120}"
ONCE="${ONCE:-0}"

mkdir -p "$LOCAL_OUT_DIR"

pull_if_exists() {
  local remote_path="$1"
  local local_name="$2"
  if ssh -o BatchMode=yes -o ConnectTimeout=10 "$CAIR_HOST" "test -f '$remote_path'" >/dev/null 2>&1; then
    rsync -a "$CAIR_HOST:$remote_path" "$LOCAL_OUT_DIR/$local_name" >/dev/null
    echo "pulled: $LOCAL_OUT_DIR/$local_name"
  fi
}

show_once() {
  echo "$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "remote experiment: $REMOTE_EXPERIMENT_ROOT"
  if ssh -o BatchMode=yes -o ConnectTimeout=10 "$CAIR_HOST" "test -f '$PROGRESS_JSON'" >/dev/null 2>&1; then
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$CAIR_HOST" "cat '$PROGRESS_JSON'" | python3 -m json.tool
  else
    echo "No progress JSON yet: $PROGRESS_JSON"
  fi
  pull_if_exists "$REMOTE_SUMMARY_CSV" "qwen_vs_qwenvl_front_top_results.csv"
  pull_if_exists "$REMOTE_TASK_MEAN_CSV" "qwen_vs_qwenvl_front_top_task_means.csv"
}

while true; do
  show_once
  if [[ "$ONCE" == "1" ]]; then
    break
  fi
  echo
  echo "Refreshing in ${INTERVAL_SECONDS}s. Press Ctrl-C to stop watching."
  sleep "$INTERVAL_SECONDS"
done
