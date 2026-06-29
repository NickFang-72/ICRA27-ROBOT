#!/usr/bin/env bash
set -Eeuo pipefail

# Prepare/launch a baseline-only comparison:
#   1. Qwen2.5-7B-Instruct text-only final LLM
#   2. Qwen2.5-VL-7B-Instruct final LLM with current front + overhead images
#
# Retrieval remains the original X-ICM lang_vis.out baseline for both rows.
# No geometry descriptors, target-pose descriptors, or contact points are used
# for retrieval or prompt augmentation in this comparison.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOCAL_XICM_ROOT="${LOCAL_XICM_ROOT:-$ROOT_DIR/X-ICM}"

CAIR_HOST="${CAIR_HOST:-cair}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/data/yf23/projects/ICRA27-ROBOT}"
REMOTE_SOURCE_XICM="${REMOTE_SOURCE_XICM:-$REMOTE_PROJECT_ROOT/X-ICM}"
REMOTE_EXPERIMENT_ROOT="${REMOTE_EXPERIMENT_ROOT:-$REMOTE_PROJECT_ROOT/experiments/qwen_vs_qwenvl_front_top_baseline_20260626}"
REMOTE_RUN_XICM="${REMOTE_RUN_XICM:-$REMOTE_EXPERIMENT_ROOT/X-ICM_qwen_vs_qwenvl_front_top}"
REMOTE_RUNNER_LOG_ROOT="${REMOTE_RUNNER_LOG_ROOT:-$REMOTE_EXPERIMENT_ROOT/runner_logs}"
CONDA_ENV="${CONDA_ENV:-/data/yf23/conda/envs/zero-shot}"

SEEDS="${SEEDS:-0}"
EPISODES="${EPISODES:-5}"
DEMO_NUM_PER_ICL="${DEMO_NUM_PER_ICL:-18}"
GPU_ID="${GPU_ID:-0}"
RANKING_METHOD="${RANKING_METHOD:-lang_vis.out}"
TOTAL_TASKS="${TOTAL_TASKS:-23}"
WAIT_FOR_GPU="${WAIT_FOR_GPU:-1}"
MIN_FREE_GPU_MEMORY_MB="${MIN_FREE_GPU_MEMORY_MB:-38000}"
MAX_GPU_UTIL_PERCENT="${MAX_GPU_UTIL_PERCENT:-10}"
GPU_WAIT_INTERVAL_SECONDS="${GPU_WAIT_INTERVAL_SECONDS:-300}"
XICM_VLLM_GPU_MEMORY_UTILIZATION="${XICM_VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
XICM_VLLM_MAX_MODEL_LEN="${XICM_VLLM_MAX_MODEL_LEN:-24576}"
XICM_QWEN_7B_PATH="${XICM_QWEN_7B_PATH:-/data/yf23/models/Qwen2.5-7B-Instruct}"
XICM_QWEN25_VL_7B_PATH="${XICM_QWEN25_VL_7B_PATH:-/data/yf23/checkpoints/ICRA27-ROBOT/Qwen2.5-VL-7B-Instruct}"

PROGRESS_JSON="$REMOTE_EXPERIMENT_ROOT/progress_qwen_vs_qwenvl.json"
RUN_SCRIPT="$REMOTE_EXPERIMENT_ROOT/run_qwen_vs_qwenvl_front_top_baseline.sh"
PID_FILE="$REMOTE_EXPERIMENT_ROOT/qwen_vs_qwenvl_front_top.pid"
SUMMARY_CSV="$REMOTE_EXPERIMENT_ROOT/qwen_vs_qwenvl_front_top_results.csv"
TASK_MEAN_CSV="$REMOTE_EXPERIMENT_ROOT/qwen_vs_qwenvl_front_top_task_means.csv"

IFS=',' read -r -a SEED_LIST <<< "$SEEDS"
TOTAL_SEED_TASKS=$(( ${#SEED_LIST[@]} * TOTAL_TASKS ))
TOTAL_ALL_SEED_TASKS=$(( 2 * TOTAL_SEED_TASKS ))

echo "Preparing isolated remote run tree at $REMOTE_RUN_XICM"
ssh "$CAIR_HOST" \
  "REMOTE_SOURCE_XICM='$REMOTE_SOURCE_XICM' REMOTE_EXPERIMENT_ROOT='$REMOTE_EXPERIMENT_ROOT' REMOTE_RUN_XICM='$REMOTE_RUN_XICM' REMOTE_RUNNER_LOG_ROOT='$REMOTE_RUNNER_LOG_ROOT' bash -s" <<'REMOTE_SETUP'
set -Eeuo pipefail
mkdir -p "$REMOTE_EXPERIMENT_ROOT" "$REMOTE_RUNNER_LOG_ROOT"
if [[ ! -d "$REMOTE_RUN_XICM" ]]; then
  mkdir -p "$REMOTE_RUN_XICM"
  rsync -a --delete \
    --exclude 'logs' \
    --exclude 'outputs' \
    "$REMOTE_SOURCE_XICM/" "$REMOTE_RUN_XICM/"
fi
mkdir -p "$REMOTE_RUN_XICM/logs"
REMOTE_SETUP

echo "Syncing patched local Qwen/QwenVL hooks to CAIR"
rsync -a \
  "$LOCAL_XICM_ROOT/main.py" \
  "$LOCAL_XICM_ROOT/crosstask_icl_agent.py" \
  "$CAIR_HOST:$REMOTE_RUN_XICM/"

ssh "$CAIR_HOST" \
  "REMOTE_EXPERIMENT_ROOT='$REMOTE_EXPERIMENT_ROOT' REMOTE_RUN_XICM='$REMOTE_RUN_XICM' REMOTE_RUNNER_LOG_ROOT='$REMOTE_RUNNER_LOG_ROOT' CONDA_ENV='$CONDA_ENV' SEEDS='$SEEDS' EPISODES='$EPISODES' DEMO_NUM_PER_ICL='$DEMO_NUM_PER_ICL' GPU_ID='$GPU_ID' RANKING_METHOD='$RANKING_METHOD' TOTAL_TASKS='$TOTAL_TASKS' TOTAL_SEED_TASKS='$TOTAL_SEED_TASKS' TOTAL_ALL_SEED_TASKS='$TOTAL_ALL_SEED_TASKS' WAIT_FOR_GPU='$WAIT_FOR_GPU' MIN_FREE_GPU_MEMORY_MB='$MIN_FREE_GPU_MEMORY_MB' MAX_GPU_UTIL_PERCENT='$MAX_GPU_UTIL_PERCENT' GPU_WAIT_INTERVAL_SECONDS='$GPU_WAIT_INTERVAL_SECONDS' XICM_VLLM_GPU_MEMORY_UTILIZATION='$XICM_VLLM_GPU_MEMORY_UTILIZATION' XICM_VLLM_MAX_MODEL_LEN='$XICM_VLLM_MAX_MODEL_LEN' XICM_QWEN_7B_PATH='$XICM_QWEN_7B_PATH' XICM_QWEN25_VL_7B_PATH='$XICM_QWEN25_VL_7B_PATH' PROGRESS_JSON='$PROGRESS_JSON' RUN_SCRIPT='$RUN_SCRIPT' PID_FILE='$PID_FILE' SUMMARY_CSV='$SUMMARY_CSV' TASK_MEAN_CSV='$TASK_MEAN_CSV' bash -s" <<'REMOTE_SCRIPT'
set -Eeuo pipefail

method_name() {
  local model_name="$1"
  printf "XICM_Cross.ZS_Ranking.%s_%s_icl.%s_test" "$RANKING_METHOD" "$model_name" "$DEMO_NUM_PER_ICL"
}

count_method_completed() {
  local method="$1"
  local method_dir="$REMOTE_RUN_XICM/logs/$method"
  local count=0
  if [[ -d "$method_dir" ]]; then
    IFS=',' read -r -a seeds <<< "$SEEDS"
    for seed in "${seeds[@]}"; do
      while IFS= read -r score_file; do
        if grep -Eq "Finished .*Final Score: [-+]?[0-9]+(\\.[0-9]+)?" "$score_file"; then
          count=$((count + 1))
        fi
      done < <(find "$method_dir" -path "*/seed${seed}/test_data.csv" -type f 2>/dev/null)
    done
  fi
  printf "%s" "$count"
}

write_progress() {
  local status="$1"
  local active_run="$2"
  local active_method="$3"
  local log_path="$4"
  local message="$5"
  local qwen_method qwenvl_method qwen_completed qwenvl_completed
  qwen_method="$(method_name Qwen2.5.7B.instruct)"
  qwenvl_method="$(method_name Qwen2.5.VL.7B.instruct)"
  qwen_completed="$(count_method_completed "$qwen_method")"
  qwenvl_completed="$(count_method_completed "$qwenvl_method")"
  cat > "$PROGRESS_JSON" <<JSON
{
  "status": "$status",
  "condition": "qwen_vs_qwenvl_front_top_baseline",
  "active_run": "$active_run",
  "active_method": "$active_method",
  "ranking_method": "$RANKING_METHOD",
  "retrieval": "baseline_lang_vis_out_only",
  "geometry_in_retrieval": false,
  "contact_points_in_retrieval": false,
  "target_pose_in_retrieval": false,
  "qwenvl_query_images": ["front_rgb_initial", "overhead_rgb_initial"],
  "seeds": "$SEEDS",
  "episodes": $EPISODES,
  "demo_num_per_icl": $DEMO_NUM_PER_ICL,
  "gpu_id": "$GPU_ID",
  "qwen_text_completed_seed_task_csvs": $qwen_completed,
  "qwenvl_front_top_completed_seed_task_csvs": $qwenvl_completed,
  "total_seed_task_csvs_per_row": $TOTAL_SEED_TASKS,
  "total_seed_task_csvs_all_rows": $TOTAL_ALL_SEED_TASKS,
  "run_xicm_root": "$REMOTE_RUN_XICM",
  "summary_csv": "$SUMMARY_CSV",
  "task_mean_csv": "$TASK_MEAN_CSV",
  "log_path": "$log_path",
  "message": "$message",
  "updated_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON
}

wait_for_gpu_capacity() {
  local run_id="$1"
  local model_name="$2"
  local log_path="$3"
  if [[ "$WAIT_FOR_GPU" != "1" ]]; then
    return 0
  fi
  while true; do
    local line free used total util
    line="$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.free,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    IFS=',' read -r free used total util <<< "$line"
    if [[ "${free:-0}" -ge "$MIN_FREE_GPU_MEMORY_MB" && "${util:-100}" -le "$MAX_GPU_UTIL_PERCENT" ]]; then
      echo "gpu_ready_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) gpu_id=$GPU_ID free_memory_mb=$free used_memory_mb=$used total_memory_mb=$total util_percent=$util" >> "$log_path"
      return 0
    fi
    echo "gpu_wait_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) run_id=$run_id model=$model_name gpu_id=$GPU_ID free_memory_mb=$free used_memory_mb=$used total_memory_mb=$total util_percent=$util threshold_free_mb=$MIN_FREE_GPU_MEMORY_MB threshold_util_percent=$MAX_GPU_UTIL_PERCENT" >> "$log_path"
    write_progress "waiting_for_gpu" "$run_id" "$(method_name "$model_name")" "$log_path" "Waiting for GPU $GPU_ID capacity before starting $run_id."
    sleep "$GPU_WAIT_INTERVAL_SECONDS"
  done
}

write_summary_csv() {
  python3 - <<'PY'
import csv
import os
import re
from collections import defaultdict
from pathlib import Path

run_root = Path(os.environ["REMOTE_RUN_XICM"])
summary_csv = Path(os.environ["SUMMARY_CSV"])
task_mean_csv = Path(os.environ["TASK_MEAN_CSV"])
ranking = os.environ["RANKING_METHOD"]
icl = os.environ["DEMO_NUM_PER_ICL"]
rows = []
for run_id, model_name, image_mode in [
    ("qwen_text_baseline", "Qwen2.5.7B.instruct", "text_only"),
    ("qwenvl_front_top_baseline", "Qwen2.5.VL.7B.instruct", "front_plus_overhead_query_images"),
]:
    method = f"XICM_Cross.ZS_Ranking.{ranking}_{model_name}_icl.{icl}_test"
    method_dir = run_root / "logs" / method
    for score_file in method_dir.glob("*/seed*/test_data.csv"):
        seed_match = re.search(r"seed(\d+)", str(score_file))
        seed = seed_match.group(1) if seed_match else ""
        text = score_file.read_text(errors="replace")
        matches = re.findall(r"Finished\\s+(.+?)\\s+\\|\\s+Final Score:\\s+([-+]?[0-9]+(?:\\.[0-9]+)?)", text)
        if not matches:
            continue
        task, score = matches[-1]
        rows.append({
            "run_id": run_id,
            "model_name": model_name,
            "image_mode": image_mode,
            "method": method,
            "seed": seed,
            "task": task.strip(),
            "episodes": os.environ["EPISODES"],
            "final_score": score,
            "score_file": str(score_file),
        })

summary_csv.parent.mkdir(parents=True, exist_ok=True)
with summary_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=[
        "run_id", "model_name", "image_mode", "method", "seed", "task",
        "episodes", "final_score", "score_file",
    ])
    writer.writeheader()
    writer.writerows(rows)

groups = defaultdict(list)
for row in rows:
    groups[(row["run_id"], row["model_name"], row["image_mode"], row["task"])].append(float(row["final_score"]))
with task_mean_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=[
        "run_id", "model_name", "image_mode", "task", "num_seeds", "mean_final_score",
    ])
    writer.writeheader()
    for (run_id, model_name, image_mode, task), values in sorted(groups.items()):
        writer.writerow({
            "run_id": run_id,
            "model_name": model_name,
            "image_mode": image_mode,
            "task": task,
            "num_seeds": len(values),
            "mean_final_score": sum(values) / len(values),
        })
PY
}

if pgrep -af "qwen_vs_qwenvl_front_top_baseline_20260626|X-ICM_qwen_vs_qwenvl_front_top" | grep -v pgrep >/dev/null 2>&1; then
  write_progress "already_running" "" "" "" "The Qwen vs QwenVL baseline comparison already appears to be running."
  echo "Qwen vs QwenVL comparison already appears to be running."
  exit 0
fi

cat > "$RUN_SCRIPT" <<'RUNNER'
#!/usr/bin/env bash
set -Eeuo pipefail

method_name() {
  local model_name="$1"
  printf "XICM_Cross.ZS_Ranking.%s_%s_icl.%s_test" "$RANKING_METHOD" "$model_name" "$DEMO_NUM_PER_ICL"
}

count_method_completed() {
  local method="$1"
  local method_dir="$REMOTE_RUN_XICM/logs/$method"
  local count=0
  if [[ -d "$method_dir" ]]; then
    IFS=',' read -r -a seeds <<< "$SEEDS"
    for seed in "${seeds[@]}"; do
      while IFS= read -r score_file; do
        if grep -Eq "Finished .*Final Score: [-+]?[0-9]+(\\.[0-9]+)?" "$score_file"; then
          count=$((count + 1))
        fi
      done < <(find "$method_dir" -path "*/seed${seed}/test_data.csv" -type f 2>/dev/null)
    done
  fi
  printf "%s" "$count"
}

write_progress() {
  local status="$1"
  local active_run="$2"
  local active_method="$3"
  local log_path="$4"
  local message="$5"
  local qwen_method qwenvl_method qwen_completed qwenvl_completed
  qwen_method="$(method_name Qwen2.5.7B.instruct)"
  qwenvl_method="$(method_name Qwen2.5.VL.7B.instruct)"
  qwen_completed="$(count_method_completed "$qwen_method")"
  qwenvl_completed="$(count_method_completed "$qwenvl_method")"
  cat > "$PROGRESS_JSON" <<JSON
{
  "status": "$status",
  "condition": "qwen_vs_qwenvl_front_top_baseline",
  "active_run": "$active_run",
  "active_method": "$active_method",
  "ranking_method": "$RANKING_METHOD",
  "retrieval": "baseline_lang_vis_out_only",
  "geometry_in_retrieval": false,
  "contact_points_in_retrieval": false,
  "target_pose_in_retrieval": false,
  "qwenvl_query_images": ["front_rgb_initial", "overhead_rgb_initial"],
  "seeds": "$SEEDS",
  "episodes": $EPISODES,
  "demo_num_per_icl": $DEMO_NUM_PER_ICL,
  "gpu_id": "$GPU_ID",
  "qwen_text_completed_seed_task_csvs": $qwen_completed,
  "qwenvl_front_top_completed_seed_task_csvs": $qwenvl_completed,
  "total_seed_task_csvs_per_row": $TOTAL_SEED_TASKS,
  "total_seed_task_csvs_all_rows": $TOTAL_ALL_SEED_TASKS,
  "run_xicm_root": "$REMOTE_RUN_XICM",
  "summary_csv": "$SUMMARY_CSV",
  "task_mean_csv": "$TASK_MEAN_CSV",
  "log_path": "$log_path",
  "message": "$message",
  "updated_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON
}

wait_for_gpu_capacity() {
  local run_id="$1"
  local model_name="$2"
  local log_path="$3"
  if [[ "$WAIT_FOR_GPU" != "1" ]]; then
    return 0
  fi
  while true; do
    local line free used total util
    line="$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.free,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    IFS=',' read -r free used total util <<< "$line"
    if [[ "${free:-0}" -ge "$MIN_FREE_GPU_MEMORY_MB" && "${util:-100}" -le "$MAX_GPU_UTIL_PERCENT" ]]; then
      echo "gpu_ready_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) gpu_id=$GPU_ID free_memory_mb=$free used_memory_mb=$used total_memory_mb=$total util_percent=$util" >> "$log_path"
      return 0
    fi
    echo "gpu_wait_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) run_id=$run_id model=$model_name gpu_id=$GPU_ID free_memory_mb=$free used_memory_mb=$used total_memory_mb=$total util_percent=$util threshold_free_mb=$MIN_FREE_GPU_MEMORY_MB threshold_util_percent=$MAX_GPU_UTIL_PERCENT" >> "$log_path"
    write_progress "waiting_for_gpu" "$run_id" "$(method_name "$model_name")" "$log_path" "Waiting for GPU $GPU_ID capacity before starting $run_id."
    sleep "$GPU_WAIT_INTERVAL_SECONDS"
  done
}

write_summary_csv() {
  python3 - <<'PY'
import csv
import os
import re
from collections import defaultdict
from pathlib import Path

run_root = Path(os.environ["REMOTE_RUN_XICM"])
summary_csv = Path(os.environ["SUMMARY_CSV"])
task_mean_csv = Path(os.environ["TASK_MEAN_CSV"])
ranking = os.environ["RANKING_METHOD"]
icl = os.environ["DEMO_NUM_PER_ICL"]
rows = []
for run_id, model_name, image_mode in [
    ("qwen_text_baseline", "Qwen2.5.7B.instruct", "text_only"),
    ("qwenvl_front_top_baseline", "Qwen2.5.VL.7B.instruct", "front_plus_overhead_query_images"),
]:
    method = f"XICM_Cross.ZS_Ranking.{ranking}_{model_name}_icl.{icl}_test"
    method_dir = run_root / "logs" / method
    for score_file in method_dir.glob("*/seed*/test_data.csv"):
        seed_match = re.search(r"seed(\d+)", str(score_file))
        seed = seed_match.group(1) if seed_match else ""
        text = score_file.read_text(errors="replace")
        matches = re.findall(r"Finished\s+(.+?)\s+\|\s+Final Score:\s+([-+]?[0-9]+(?:\.[0-9]+)?)", text)
        if not matches:
            continue
        task, score = matches[-1]
        rows.append({
            "run_id": run_id,
            "model_name": model_name,
            "image_mode": image_mode,
            "method": method,
            "seed": seed,
            "task": task.strip(),
            "episodes": os.environ["EPISODES"],
            "final_score": score,
            "score_file": str(score_file),
        })

summary_csv.parent.mkdir(parents=True, exist_ok=True)
with summary_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=[
        "run_id", "model_name", "image_mode", "method", "seed", "task",
        "episodes", "final_score", "score_file",
    ])
    writer.writeheader()
    writer.writerows(rows)

groups = defaultdict(list)
for row in rows:
    groups[(row["run_id"], row["model_name"], row["image_mode"], row["task"])].append(float(row["final_score"]))
with task_mean_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=[
        "run_id", "model_name", "image_mode", "task", "num_seeds", "mean_final_score",
    ])
    writer.writeheader()
    for (run_id, model_name, image_mode, task), values in sorted(groups.items()):
        writer.writerow({
            "run_id": run_id,
            "model_name": model_name,
            "image_mode": image_mode,
            "task": task,
            "num_seeds": len(values),
            "mean_final_score": sum(values) / len(values),
        })
PY
}

source /data/yf23/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$REMOTE_RUN_XICM"

export COPPELIASIM_ROOT="$REMOTE_RUN_XICM/CoppeliaSim"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export HF_HOME="${HF_HOME:-/data/yf23/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/data/yf23/huggingface/transformers}"
export HF_HUB_DISABLE_XET=1
export XICM_QWEN_7B_PATH
export XICM_QWEN25_VL_7B_PATH
export XICM_VLLM_GPU_MEMORY_UTILIZATION
export XICM_VLLM_MAX_MODEL_LEN
export XICM_VL_MAX_IMAGES=2
export XICM_SD2_BASE_PATH="${XICM_SD2_BASE_PATH:-/data/yf23/models/Manojb-stable-diffusion-2-base}"
export MULTIPROCESSING_START_METHOD=spawn

run_row() {
  local run_id="$1"
  local model_name="$2"
  local image_mode="$3"
  local method
  method="XICM_Cross.ZS_Ranking.${RANKING_METHOD}_${model_name}_icl.${DEMO_NUM_PER_ICL}_test"
  local log_path="$REMOTE_RUNNER_LOG_ROOT/${run_id}_seed${SEEDS}_$(date -u +%Y%m%d_%H%M%S).log"

  local completed
  completed="$(count_method_completed "$method")"
  if [[ "$completed" -ge "$TOTAL_SEED_TASKS" ]]; then
    write_progress "skipped_completed" "$run_id" "$method" "$log_path" "$run_id already has all strict seed-task final scores."
    return 0
  fi

  write_progress "running" "$run_id" "$method" "$log_path" "Starting $run_id."
  wait_for_gpu_capacity "$run_id" "$model_name" "$log_path"
  (
    echo "run_id=$run_id"
    echo "model_name=$model_name"
    echo "image_mode=$image_mode"
    echo "method=$method"
    echo "ranking=$RANKING_METHOD"
    echo "retrieval=baseline_lang_vis_out_only"
    echo "geometry_in_retrieval=false"
    echo "contact_points_in_retrieval=false"
    echo "target_pose_in_retrieval=false"
    echo "seed=$SEEDS"
    echo "episodes=$EPISODES"
    echo "demo_num_per_icl=$DEMO_NUM_PER_ICL"
    echo "gpu_id=$GPU_ID"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    set +e
    bash scripts/eval_XICM.sh "$SEEDS" "$EPISODES" "$model_name" "$DEMO_NUM_PER_ICL" "$GPU_ID" "$RANKING_METHOD" "true"
    status=$?
    set -e
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "exit_status=$status"
    exit "$status"
  ) > "$log_path" 2>&1

  completed="$(count_method_completed "$method")"
  write_summary_csv
  if [[ "$completed" -ge "$TOTAL_SEED_TASKS" ]]; then
    write_progress "row_completed" "$run_id" "$method" "$log_path" "$run_id finished all strict seed-task final scores."
  else
    write_progress "failed_or_partial" "$run_id" "$method" "$log_path" "$run_id exited before all strict seed-task final scores were present."
    return 1
  fi
}

run_row "qwen_text_baseline" "Qwen2.5.7B.instruct" "text_only"
run_row "qwenvl_front_top_baseline" "Qwen2.5.VL.7B.instruct" "front_plus_overhead_query_images"
write_summary_csv
write_progress "completed" "" "" "" "Qwen vs QwenVL front/top baseline comparison finished."
RUNNER
chmod +x "$RUN_SCRIPT"

write_progress "prepared" "" "" "" "Launcher prepared. It will run only after this script is started and GPU wait conditions are met."

nohup env \
  REMOTE_EXPERIMENT_ROOT="$REMOTE_EXPERIMENT_ROOT" \
  REMOTE_RUN_XICM="$REMOTE_RUN_XICM" \
  REMOTE_RUNNER_LOG_ROOT="$REMOTE_RUNNER_LOG_ROOT" \
  CONDA_ENV="$CONDA_ENV" \
  SEEDS="$SEEDS" \
  EPISODES="$EPISODES" \
  DEMO_NUM_PER_ICL="$DEMO_NUM_PER_ICL" \
  GPU_ID="$GPU_ID" \
  RANKING_METHOD="$RANKING_METHOD" \
  TOTAL_TASKS="$TOTAL_TASKS" \
  TOTAL_SEED_TASKS="$TOTAL_SEED_TASKS" \
  TOTAL_ALL_SEED_TASKS="$TOTAL_ALL_SEED_TASKS" \
  WAIT_FOR_GPU="$WAIT_FOR_GPU" \
  MIN_FREE_GPU_MEMORY_MB="$MIN_FREE_GPU_MEMORY_MB" \
  MAX_GPU_UTIL_PERCENT="$MAX_GPU_UTIL_PERCENT" \
  GPU_WAIT_INTERVAL_SECONDS="$GPU_WAIT_INTERVAL_SECONDS" \
  XICM_QWEN_7B_PATH="$XICM_QWEN_7B_PATH" \
  XICM_QWEN25_VL_7B_PATH="$XICM_QWEN25_VL_7B_PATH" \
  XICM_VLLM_GPU_MEMORY_UTILIZATION="$XICM_VLLM_GPU_MEMORY_UTILIZATION" \
  XICM_VLLM_MAX_MODEL_LEN="$XICM_VLLM_MAX_MODEL_LEN" \
  PROGRESS_JSON="$PROGRESS_JSON" \
  SUMMARY_CSV="$SUMMARY_CSV" \
  TASK_MEAN_CSV="$TASK_MEAN_CSV" \
  bash -lc "source '$RUN_SCRIPT'" \
  > "$REMOTE_RUNNER_LOG_ROOT/qwen_vs_qwenvl_launcher_$(date -u +%Y%m%d_%H%M%S).log" 2>&1 &

pid="$!"
echo "$pid" > "$PID_FILE"
echo "Launched Qwen vs QwenVL baseline comparison runner PID $pid"
echo "Progress: $PROGRESS_JSON"
echo "Run root: $REMOTE_RUN_XICM"
echo "Runner logs: $REMOTE_RUNNER_LOG_ROOT"
echo "Summary CSV: $SUMMARY_CSV"
echo "Task means CSV: $TASK_MEAN_CSV"
REMOTE_SCRIPT
