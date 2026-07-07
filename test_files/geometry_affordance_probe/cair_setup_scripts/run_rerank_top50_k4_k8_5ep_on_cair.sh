#!/usr/bin/env bash
set -Eeuo pipefail

# Launch a quick sequential CAIR run for the two-stage retrieval reranker:
# dynamic-diffusion top-N shortlist first, then geometry/profile rerank.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CAIR_HOST="${CAIR_HOST:-cair}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/data/yf23/projects/ICRA27-ROBOT}"
REMOTE_SOURCE_XICM="${REMOTE_SOURCE_XICM:-$REMOTE_PROJECT_ROOT/X-ICM}"
REMOTE_COMPONENT_ROOT="${REMOTE_COMPONENT_ROOT:-$REMOTE_PROJECT_ROOT/experiments/rerank_top50_k4_k8_5eps_20260701}"
REMOTE_RUN_XICM="${REMOTE_RUN_XICM:-$REMOTE_COMPONENT_ROOT/X-ICM_rerank_top50_k4_k8_5eps}"
REMOTE_RUNNER_LOG_ROOT="${REMOTE_RUNNER_LOG_ROOT:-$REMOTE_COMPONENT_ROOT/runner_logs}"
CONDA_ENV="${CONDA_ENV:-/data/yf23/conda/envs/zero-shot}"
SEEDS="${SEEDS:-0}"
EPISODES="${EPISODES:-5}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5.VL.7B.instruct}"
GPU_ID="${GPU_ID:-0}"
K_VALUES="${K_VALUES:-4,8}"
RANKING_METHOD="${RANKING_METHOD:-lang_vis.out.geo.aff_v3.rerank_top50}"
RERANK_CANDIDATES="${RERANK_CANDIDATES:-50}"
TOTAL_TASKS="${TOTAL_TASKS:-23}"
WAIT_FOR_GPU="${WAIT_FOR_GPU:-1}"
MIN_FREE_GPU_MEMORY_MB="${MIN_FREE_GPU_MEMORY_MB:-20000}"
MAX_GPU_UTIL_PERCENT="${MAX_GPU_UTIL_PERCENT:-15}"
GPU_WAIT_INTERVAL_SECONDS="${GPU_WAIT_INTERVAL_SECONDS:-120}"
XICM_QWEN25_VL_7B_PATH="${XICM_QWEN25_VL_7B_PATH:-/data/yf23/checkpoints/ICRA27-ROBOT/Qwen2.5-VL-7B-Instruct}"
XICM_VLLM_GPU_MEMORY_UTILIZATION="${XICM_VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
XICM_VLLM_MAX_MODEL_LEN="${XICM_VLLM_MAX_MODEL_LEN:-24576}"
XICM_VL_MAX_IMAGES="${XICM_VL_MAX_IMAGES:-2}"
XICM_GA_REVIEW_BUNDLE="${XICM_GA_REVIEW_BUNDLE:-$REMOTE_PROJECT_ROOT/experiments/geometry_affordance_full_cache/review_bundle.jsonl}"
XICM_TASKS_OVERRIDE="${XICM_TASKS_OVERRIDE:-}"

LOCAL_FORM="$REPO_ROOT/X-ICM/form_icl_demonstrations_crosstask_ranking.py"
LOCAL_AGENT="$REPO_ROOT/X-ICM/crosstask_icl_agent.py"
LOCAL_MAIN="$REPO_ROOT/X-ICM/main.py"
LOCAL_EVAL_SCRIPT="$REPO_ROOT/X-ICM/scripts/eval_XICM.sh"

ssh "$CAIR_HOST" \
  "REMOTE_SOURCE_XICM='$REMOTE_SOURCE_XICM' REMOTE_COMPONENT_ROOT='$REMOTE_COMPONENT_ROOT' REMOTE_RUN_XICM='$REMOTE_RUN_XICM' REMOTE_RUNNER_LOG_ROOT='$REMOTE_RUNNER_LOG_ROOT' bash -s" <<'REMOTE_SETUP'
set -Eeuo pipefail
mkdir -p "$REMOTE_COMPONENT_ROOT" "$REMOTE_RUNNER_LOG_ROOT"
if [[ ! -d "$REMOTE_RUN_XICM" ]]; then
  echo "Creating isolated X-ICM tree at $REMOTE_RUN_XICM"
  mkdir -p "$REMOTE_RUN_XICM"
  rsync -a --delete \
    --exclude 'logs' \
    --exclude 'outputs' \
    "$REMOTE_SOURCE_XICM/" "$REMOTE_RUN_XICM/"
fi
mkdir -p "$REMOTE_RUN_XICM/logs"
REMOTE_SETUP

rsync -a "$LOCAL_MAIN" "$LOCAL_FORM" "$LOCAL_AGENT" "$CAIR_HOST:$REMOTE_RUN_XICM/"
rsync -a "$LOCAL_EVAL_SCRIPT" "$CAIR_HOST:$REMOTE_RUN_XICM/scripts/eval_XICM.sh"

ssh "$CAIR_HOST" \
  "REMOTE_COMPONENT_ROOT='$REMOTE_COMPONENT_ROOT' REMOTE_RUN_XICM='$REMOTE_RUN_XICM' REMOTE_RUNNER_LOG_ROOT='$REMOTE_RUNNER_LOG_ROOT' CONDA_ENV='$CONDA_ENV' SEEDS='$SEEDS' EPISODES='$EPISODES' MODEL_NAME='$MODEL_NAME' GPU_ID='$GPU_ID' K_VALUES='$K_VALUES' RANKING_METHOD='$RANKING_METHOD' RERANK_CANDIDATES='$RERANK_CANDIDATES' TOTAL_TASKS='$TOTAL_TASKS' WAIT_FOR_GPU='$WAIT_FOR_GPU' MIN_FREE_GPU_MEMORY_MB='$MIN_FREE_GPU_MEMORY_MB' MAX_GPU_UTIL_PERCENT='$MAX_GPU_UTIL_PERCENT' GPU_WAIT_INTERVAL_SECONDS='$GPU_WAIT_INTERVAL_SECONDS' XICM_QWEN25_VL_7B_PATH='$XICM_QWEN25_VL_7B_PATH' XICM_VLLM_GPU_MEMORY_UTILIZATION='$XICM_VLLM_GPU_MEMORY_UTILIZATION' XICM_VLLM_MAX_MODEL_LEN='$XICM_VLLM_MAX_MODEL_LEN' XICM_VL_MAX_IMAGES='$XICM_VL_MAX_IMAGES' XICM_GA_REVIEW_BUNDLE='$XICM_GA_REVIEW_BUNDLE' XICM_TASKS_OVERRIDE='$XICM_TASKS_OVERRIDE' bash -s" <<'REMOTE_SCRIPT'
set -Eeuo pipefail

RUN_SCRIPT="$REMOTE_COMPONENT_ROOT/run_rerank_top50_k4_k8_5eps.sh"
PROGRESS_JSON="$REMOTE_COMPONENT_ROOT/progress_rerank_top50_k4_k8_5eps.json"
PID_FILE="$REMOTE_COMPONENT_ROOT/rerank_top50_k4_k8_5eps.pid"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && ps -p "$old_pid" -o args= | grep -q "$RUN_SCRIPT"; then
    echo "rerank k4/k8 runner already active as PID $old_pid"
    echo "Progress: $PROGRESS_JSON"
    exit 0
  fi
fi

cat > "$RUN_SCRIPT" <<'RUNNER'
#!/usr/bin/env bash
set -Eeuo pipefail

IFS=',' read -r -a K_LIST <<< "$K_VALUES"
IFS=',' read -r -a SEED_LIST <<< "$SEEDS"
TOTAL_SEED_TASKS=$(( ${#SEED_LIST[@]} * TOTAL_TASKS ))
TOTAL_ALL=$(( ${#K_LIST[@]} * TOTAL_SEED_TASKS ))

method_name_for_k() {
  local k="$1"
  printf "XICM_Cross.ZS_Ranking.%s_%s_icl.%s_test" "$RANKING_METHOD" "$MODEL_NAME" "$k"
}

count_completed() {
  local k="$1"
  local method
  method="$(method_name_for_k "$k")"
  METHOD="$method" REMOTE_RUN_XICM="$REMOTE_RUN_XICM" SEEDS="$SEEDS" python3 - <<'PY'
from pathlib import Path
import os
import re

method_dir = Path(os.environ["REMOTE_RUN_XICM"]) / "logs" / os.environ["METHOD"]
seeds = [item.strip() for item in os.environ["SEEDS"].split(",") if item.strip()]
finish_re = re.compile(r"Finished\s+[^|]+?\s+\|\s+Final Score:\s*[-+]?\d+(?:\.\d+)?")
count = 0
if method_dir.exists():
    for seed in seeds:
        for path in method_dir.glob(f"*/seed{seed}/test_data.csv"):
            if finish_re.search(path.read_text(errors="replace")):
                count += 1
print(count)
PY
}

write_progress() {
  local status="$1"
  local active_k="$2"
  local active_log="$3"
  local message="$4"
  STATUS="$status" ACTIVE_K="$active_k" ACTIVE_LOG="$active_log" MESSAGE="$message" \
  REMOTE_COMPONENT_ROOT="$REMOTE_COMPONENT_ROOT" REMOTE_RUN_XICM="$REMOTE_RUN_XICM" PROGRESS_JSON="$PROGRESS_JSON" \
  RANKING_METHOD="$RANKING_METHOD" MODEL_NAME="$MODEL_NAME" K_VALUES="$K_VALUES" SEEDS="$SEEDS" EPISODES="$EPISODES" \
  GPU_ID="$GPU_ID" TOTAL_SEED_TASKS="$TOTAL_SEED_TASKS" TOTAL_ALL="$TOTAL_ALL" RERANK_CANDIDATES="$RERANK_CANDIDATES" \
  XICM_GA_REVIEW_BUNDLE="$XICM_GA_REVIEW_BUNDLE" XICM_TASKS_OVERRIDE="$XICM_TASKS_OVERRIDE" \
  python3 - <<'PY'
from pathlib import Path
import json
import os
import re
from datetime import datetime, timezone

run_root = Path(os.environ["REMOTE_RUN_XICM"])
seeds = [item.strip() for item in os.environ["SEEDS"].split(",") if item.strip()]
ks = [item.strip() for item in os.environ["K_VALUES"].split(",") if item.strip()]
finish_re = re.compile(r"Finished\s+[^|]+?\s+\|\s+Final Score:\s*[-+]?\d+(?:\.\d+)?")
per_k = {}
total = 0
for k in ks:
    method = f"XICM_Cross.ZS_Ranking.{os.environ['RANKING_METHOD']}_{os.environ['MODEL_NAME']}_icl.{k}_test"
    method_dir = run_root / "logs" / method
    count = 0
    per_seed = {}
    for seed in seeds:
        seed_count = 0
        if method_dir.exists():
            for path in method_dir.glob(f"*/seed{seed}/test_data.csv"):
                if finish_re.search(path.read_text(errors="replace")):
                    count += 1
                    seed_count += 1
        per_seed[f"seed{seed}"] = seed_count
    total += count
    per_k[f"k{k}"] = {
        "method": method,
        "completed_seed_task_csvs": count,
        "total_seed_task_csvs": int(os.environ["TOTAL_SEED_TASKS"]),
        "per_seed": per_seed,
    }
payload = {
    "status": os.environ["STATUS"],
    "active_k": os.environ["ACTIVE_K"],
    "active_log_path": os.environ["ACTIVE_LOG"],
    "ranking_method": os.environ["RANKING_METHOD"],
    "model_name": os.environ["MODEL_NAME"],
    "k_values": ks,
    "seeds": os.environ["SEEDS"],
    "episodes": int(os.environ["EPISODES"]),
    "rerank_candidates": os.environ["RERANK_CANDIDATES"],
    "review_bundle": os.environ["XICM_GA_REVIEW_BUNDLE"],
    "tasks_override": os.environ.get("XICM_TASKS_OVERRIDE", ""),
    "gpu_id": os.environ["GPU_ID"],
    "per_k": per_k,
    "completed_seed_task_csvs": total,
    "total_seed_task_csvs": int(os.environ["TOTAL_ALL"]),
    "component_root": os.environ["REMOTE_COMPONENT_ROOT"],
    "run_xicm_root": os.environ["REMOTE_RUN_XICM"],
    "message": os.environ["MESSAGE"],
    "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
Path(os.environ["PROGRESS_JSON"]).write_text(json.dumps(payload, indent=2) + "\n")
PY
}

wait_for_gpu_capacity() {
  local k="$1"
  local log_path="$2"
  if [[ "$WAIT_FOR_GPU" != "1" && "$WAIT_FOR_GPU" != "true" && "$WAIT_FOR_GPU" != "yes" ]]; then
    return 0
  fi
  while true; do
    local stats used total util free
    stats="$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    IFS=',' read -r used total util <<< "$stats"
    free=$(( total - used ))
    if (( free >= MIN_FREE_GPU_MEMORY_MB && util <= MAX_GPU_UTIL_PERCENT )); then
      echo "gpu_wait_ready_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) gpu_id=$GPU_ID free_memory_mb=$free used_memory_mb=$used total_memory_mb=$total util_percent=$util" >> "$log_path"
      return 0
    fi
    echo "gpu_wait_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) gpu_id=$GPU_ID free_memory_mb=$free used_memory_mb=$used total_memory_mb=$total util_percent=$util" >> "$log_path"
    write_progress "waiting_for_gpu" "$k" "$log_path" "Waiting for GPU $GPU_ID before k=$k."
    sleep "$GPU_WAIT_INTERVAL_SECONDS"
  done
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
export XICM_QWEN25_VL_7B_PATH="$XICM_QWEN25_VL_7B_PATH"
export XICM_VLLM_GPU_MEMORY_UTILIZATION="$XICM_VLLM_GPU_MEMORY_UTILIZATION"
export XICM_VLLM_MAX_MODEL_LEN="$XICM_VLLM_MAX_MODEL_LEN"
export XICM_VL_MAX_IMAGES="$XICM_VL_MAX_IMAGES"
export XICM_GA_REVIEW_BUNDLE="$XICM_GA_REVIEW_BUNDLE"
export XICM_GA_RERANK_CANDIDATES="$RERANK_CANDIDATES"
export XICM_TASKS_OVERRIDE="$XICM_TASKS_OVERRIDE"
export MULTIPROCESSING_START_METHOD=spawn

for k in "${K_LIST[@]}"; do
  method="$(method_name_for_k "$k")"
  completed="$(count_completed "$k")"
  if [[ "$completed" -ge "$TOTAL_SEED_TASKS" ]]; then
    write_progress "skipped_completed" "$k" "" "k=$k already has all strict seed-task final scores."
    continue
  fi
  export XICM_GA_AUDIT_JSONL="$REMOTE_COMPONENT_ROOT/retrieval_audit_k${k}.jsonl"
  if [[ "$completed" -eq 0 ]]; then
    : > "$XICM_GA_AUDIT_JSONL"
  fi
  log_path="$REMOTE_RUNNER_LOG_ROOT/rerank_top50_k${k}_seed${SEEDS}_$(date -u +%Y%m%d_%H%M%S).log"
  write_progress "running" "$k" "$log_path" "Started k=$k."
  wait_for_gpu_capacity "$k" "$log_path"
  (
    echo "ranking=$RANKING_METHOD"
    echo "method=$method"
    echo "k=$k"
    echo "seeds=$SEEDS"
    echo "episodes=$EPISODES"
    echo "gpu_id=$GPU_ID"
    echo "rerank_candidates=$XICM_GA_RERANK_CANDIDATES"
    echo "review_bundle=$XICM_GA_REVIEW_BUNDLE"
    echo "retrieval_audit_jsonl=$XICM_GA_AUDIT_JSONL"
    echo "tasks_override=${XICM_TASKS_OVERRIDE:-}"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    set +e
    bash scripts/eval_XICM.sh "$SEEDS" "$EPISODES" "$MODEL_NAME" "$k" "$GPU_ID" "$RANKING_METHOD" "true"
    status=$?
    set -e
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "exit_status=$status"
    exit "$status"
  ) > "$log_path" 2>&1

  completed="$(count_completed "$k")"
  if [[ "$completed" -ge "$TOTAL_SEED_TASKS" ]]; then
    write_progress "k_completed" "$k" "$log_path" "k=$k finished all strict seed-task final scores."
  else
    write_progress "failed_or_partial" "$k" "$log_path" "k=$k exited before all strict seed-task final scores were present."
    exit 1
  fi
done

write_progress "completed" "" "" "k4/k8 rerank run completed."
RUNNER
chmod +x "$RUN_SCRIPT"

write_initial_progress() {
  REMOTE_COMPONENT_ROOT="$REMOTE_COMPONENT_ROOT" REMOTE_RUN_XICM="$REMOTE_RUN_XICM" PROGRESS_JSON="$PROGRESS_JSON" \
  RANKING_METHOD="$RANKING_METHOD" MODEL_NAME="$MODEL_NAME" K_VALUES="$K_VALUES" SEEDS="$SEEDS" EPISODES="$EPISODES" \
  GPU_ID="$GPU_ID" RERANK_CANDIDATES="$RERANK_CANDIDATES" XICM_GA_REVIEW_BUNDLE="$XICM_GA_REVIEW_BUNDLE" \
  XICM_TASKS_OVERRIDE="$XICM_TASKS_OVERRIDE" TOTAL_TASKS="$TOTAL_TASKS" python3 - <<'PY'
from pathlib import Path
import json
import os
from datetime import datetime, timezone

ks = [item.strip() for item in os.environ["K_VALUES"].split(",") if item.strip()]
seeds = [item.strip() for item in os.environ["SEEDS"].split(",") if item.strip()]
per_k = {}
for k in ks:
    per_k[f"k{k}"] = {
        "method": f"XICM_Cross.ZS_Ranking.{os.environ['RANKING_METHOD']}_{os.environ['MODEL_NAME']}_icl.{k}_test",
        "completed_seed_task_csvs": 0,
        "total_seed_task_csvs": len(seeds) * int(os.environ["TOTAL_TASKS"]),
        "per_seed": {f"seed{seed}": 0 for seed in seeds},
    }
payload = {
    "status": "launching",
    "active_k": "",
    "active_log_path": "",
    "ranking_method": os.environ["RANKING_METHOD"],
    "model_name": os.environ["MODEL_NAME"],
    "k_values": ks,
    "seeds": os.environ["SEEDS"],
    "episodes": int(os.environ["EPISODES"]),
    "rerank_candidates": os.environ["RERANK_CANDIDATES"],
    "review_bundle": os.environ["XICM_GA_REVIEW_BUNDLE"],
    "tasks_override": os.environ.get("XICM_TASKS_OVERRIDE", ""),
    "gpu_id": os.environ["GPU_ID"],
    "per_k": per_k,
    "completed_seed_task_csvs": 0,
    "total_seed_task_csvs": len(ks) * len(seeds) * int(os.environ["TOTAL_TASKS"]),
    "component_root": os.environ["REMOTE_COMPONENT_ROOT"],
    "run_xicm_root": os.environ["REMOTE_RUN_XICM"],
    "message": "Launching k4/k8 rerank run.",
    "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
Path(os.environ["PROGRESS_JSON"]).write_text(json.dumps(payload, indent=2) + "\n")
PY
}

write_initial_progress
nohup env \
  REMOTE_COMPONENT_ROOT="$REMOTE_COMPONENT_ROOT" \
  REMOTE_RUN_XICM="$REMOTE_RUN_XICM" \
  REMOTE_RUNNER_LOG_ROOT="$REMOTE_RUNNER_LOG_ROOT" \
  CONDA_ENV="$CONDA_ENV" \
  SEEDS="$SEEDS" \
  EPISODES="$EPISODES" \
  MODEL_NAME="$MODEL_NAME" \
  GPU_ID="$GPU_ID" \
  K_VALUES="$K_VALUES" \
  RANKING_METHOD="$RANKING_METHOD" \
  RERANK_CANDIDATES="$RERANK_CANDIDATES" \
  TOTAL_TASKS="$TOTAL_TASKS" \
  WAIT_FOR_GPU="$WAIT_FOR_GPU" \
  MIN_FREE_GPU_MEMORY_MB="$MIN_FREE_GPU_MEMORY_MB" \
  MAX_GPU_UTIL_PERCENT="$MAX_GPU_UTIL_PERCENT" \
  GPU_WAIT_INTERVAL_SECONDS="$GPU_WAIT_INTERVAL_SECONDS" \
  XICM_QWEN25_VL_7B_PATH="$XICM_QWEN25_VL_7B_PATH" \
  XICM_VLLM_GPU_MEMORY_UTILIZATION="$XICM_VLLM_GPU_MEMORY_UTILIZATION" \
  XICM_VLLM_MAX_MODEL_LEN="$XICM_VLLM_MAX_MODEL_LEN" \
  XICM_VL_MAX_IMAGES="$XICM_VL_MAX_IMAGES" \
  XICM_GA_REVIEW_BUNDLE="$XICM_GA_REVIEW_BUNDLE" \
  XICM_TASKS_OVERRIDE="$XICM_TASKS_OVERRIDE" \
  PROGRESS_JSON="$PROGRESS_JSON" \
  "$RUN_SCRIPT" \
  > "$REMOTE_RUNNER_LOG_ROOT/rerank_top50_launcher_$(date -u +%Y%m%d_%H%M%S).log" 2>&1 &

pid="$!"
echo "$pid" > "$PID_FILE"
echo "Launched rerank k4/k8 runner PID $pid"
echo "Progress: $PROGRESS_JSON"
echo "Run root: $REMOTE_RUN_XICM"
echo "Runner logs: $REMOTE_RUNNER_LOG_ROOT"
echo "Method logs: $REMOTE_RUN_XICM/logs"
REMOTE_SCRIPT
