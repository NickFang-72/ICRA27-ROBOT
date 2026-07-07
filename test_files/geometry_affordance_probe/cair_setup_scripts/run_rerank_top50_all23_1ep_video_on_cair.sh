#!/usr/bin/env bash
set -Eeuo pipefail

# Launch the frozen double-retrieval rerank checkpoint on all 23 unseen
# RLBench tasks, one episode per task, with runtime MP4 recording enabled.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

UNSEEN_TASKS=(
  "put_toilet_roll_on_stand"
  "put_knife_on_chopping_board"
  "close_fridge"
  "close_microwave"
  "close_laptop_lid"
  "phone_on_base"
  "toilet_seat_down"
  "lamp_off"
  "lamp_on"
  "put_books_on_bookshelf"
  "put_umbrella_in_umbrella_stand"
  "open_grill"
  "put_rubbish_in_bin"
  "take_usb_out_of_computer"
  "take_lid_off_saucepan"
  "take_plate_off_colored_dish_rack"
  "basketball_in_hoop"
  "scoop_with_spatula"
  "straighten_rope"
  "turn_oven_on"
  "beat_the_buzz"
  "water_plants"
  "unplug_charger"
)

TASKS_DEFAULT="$(IFS=,; echo "${UNSEEN_TASKS[*]}")"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"

CAIR_HOST="${CAIR_HOST:-cair}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-10}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/data/yf23/projects/ICRA27-ROBOT}"
REMOTE_SOURCE_XICM="${REMOTE_SOURCE_XICM:-$REMOTE_PROJECT_ROOT/X-ICM}"
REMOTE_COMPONENT_ROOT="${REMOTE_COMPONENT_ROOT:-$REMOTE_PROJECT_ROOT/experiments/rerank_top50_all23_1ep_video_$RUN_STAMP}"
REMOTE_RUN_XICM="${REMOTE_RUN_XICM:-$REMOTE_COMPONENT_ROOT/X-ICM_rerank_top50_all23_1ep_video}"
REMOTE_RUNNER_LOG_ROOT="${REMOTE_RUNNER_LOG_ROOT:-$REMOTE_COMPONENT_ROOT/runner_logs}"
REMOTE_REVIEW_ROOT="${REMOTE_REVIEW_ROOT:-$REMOTE_COMPONENT_ROOT/review/rerank_top50_all23_1ep_video_$RUN_STAMP}"

CONDA_ENV="${CONDA_ENV:-/data/yf23/conda/envs/zero-shot}"
SEED="${SEED:-0}"
EPISODES="${EPISODES:-1}"
EPISODE_LENGTH="${EPISODE_LENGTH:-25}"
K="${K:-8}"
TASKS="${TASKS:-$TASKS_DEFAULT}"
RLBENCH_DEMO_PATH="${RLBENCH_DEMO_PATH:-data/unseen_tasks/test}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5.VL.7B.instruct}"
GPU_ID="${GPU_ID:-0}"
RANKING_METHOD="${RANKING_METHOD:-lang_vis.out.geo.aff_v3.rerank_top50}"
METHOD_SUFFIX="${METHOD_SUFFIX:-rerank_top50_all23_1ep_video_$RUN_STAMP}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-21600}"
ENABLE_VIDEO="${ENABLE_VIDEO:-1}"
RECORD_EVERY_N="${RECORD_EVERY_N:-1}"
RERANK_CANDIDATES="${RERANK_CANDIDATES:-50}"
WAIT_FOR_GPU="${WAIT_FOR_GPU:-1}"
MIN_FREE_GPU_MEMORY_MB="${MIN_FREE_GPU_MEMORY_MB:-20000}"
MAX_GPU_UTIL_PERCENT="${MAX_GPU_UTIL_PERCENT:-15}"
GPU_WAIT_INTERVAL_SECONDS="${GPU_WAIT_INTERVAL_SECONDS:-120}"

XICM_QWEN25_VL_7B_PATH="${XICM_QWEN25_VL_7B_PATH:-/data/yf23/checkpoints/ICRA27-ROBOT/Qwen2.5-VL-7B-Instruct}"
XICM_VLLM_GPU_MEMORY_UTILIZATION="${XICM_VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
XICM_VLLM_MAX_MODEL_LEN="${XICM_VLLM_MAX_MODEL_LEN:-24576}"
XICM_VL_MAX_IMAGES="${XICM_VL_MAX_IMAGES:-2}"
XICM_GA_REVIEW_BUNDLE="${XICM_GA_REVIEW_BUNDLE:-$REMOTE_PROJECT_ROOT/experiments/geometry_affordance_full_cache/review_bundle.jsonl}"

LOCAL_FORM="$REPO_ROOT/X-ICM/form_icl_demonstrations_crosstask_ranking.py"
LOCAL_AGENT="$REPO_ROOT/X-ICM/crosstask_icl_agent.py"
LOCAL_MAIN="$REPO_ROOT/X-ICM/main.py"
LOCAL_RERANK_REVIEW="$REPO_ROOT/X-ICM/scripts/generate_rerank_review_trace.py"

SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout="$SSH_CONNECT_TIMEOUT"
  -o ServerAliveInterval=5
  -o ServerAliveCountMax=1
)
RSYNC_SSH="ssh -o BatchMode=yes -o ConnectTimeout=$SSH_CONNECT_TIMEOUT -o ServerAliveInterval=5 -o ServerAliveCountMax=1"

ssh "${SSH_OPTS[@]}" "$CAIR_HOST" \
  "REMOTE_SOURCE_XICM='$REMOTE_SOURCE_XICM' REMOTE_COMPONENT_ROOT='$REMOTE_COMPONENT_ROOT' REMOTE_RUN_XICM='$REMOTE_RUN_XICM' REMOTE_RUNNER_LOG_ROOT='$REMOTE_RUNNER_LOG_ROOT' REMOTE_REVIEW_ROOT='$REMOTE_REVIEW_ROOT' bash -s" <<'REMOTE_SETUP'
set -Eeuo pipefail
mkdir -p "$REMOTE_COMPONENT_ROOT" "$REMOTE_RUNNER_LOG_ROOT" "$REMOTE_REVIEW_ROOT"
if [[ ! -d "$REMOTE_RUN_XICM" ]]; then
  echo "Creating isolated X-ICM tree at $REMOTE_RUN_XICM"
  mkdir -p "$REMOTE_RUN_XICM"
  rsync -a --delete \
    --exclude 'logs' \
    --exclude 'outputs' \
    "$REMOTE_SOURCE_XICM/" "$REMOTE_RUN_XICM/"
fi
mkdir -p "$REMOTE_RUN_XICM/logs" "$REMOTE_RUN_XICM/scripts"
REMOTE_SETUP

rsync -e "$RSYNC_SSH" -a "$LOCAL_MAIN" "$LOCAL_FORM" "$LOCAL_AGENT" "$CAIR_HOST:$REMOTE_RUN_XICM/"
rsync -e "$RSYNC_SSH" -a "$LOCAL_RERANK_REVIEW" "$CAIR_HOST:$REMOTE_RUN_XICM/scripts/generate_rerank_review_trace.py"

ssh "${SSH_OPTS[@]}" "$CAIR_HOST" \
  "REMOTE_COMPONENT_ROOT='$REMOTE_COMPONENT_ROOT' REMOTE_RUN_XICM='$REMOTE_RUN_XICM' REMOTE_RUNNER_LOG_ROOT='$REMOTE_RUNNER_LOG_ROOT' REMOTE_REVIEW_ROOT='$REMOTE_REVIEW_ROOT' CONDA_ENV='$CONDA_ENV' SEED='$SEED' EPISODES='$EPISODES' EPISODE_LENGTH='$EPISODE_LENGTH' K='$K' TASKS='$TASKS' RLBENCH_DEMO_PATH='$RLBENCH_DEMO_PATH' MODEL_NAME='$MODEL_NAME' GPU_ID='$GPU_ID' RANKING_METHOD='$RANKING_METHOD' METHOD_SUFFIX='$METHOD_SUFFIX' RUN_TIMEOUT_SECONDS='$RUN_TIMEOUT_SECONDS' ENABLE_VIDEO='$ENABLE_VIDEO' RECORD_EVERY_N='$RECORD_EVERY_N' RERANK_CANDIDATES='$RERANK_CANDIDATES' WAIT_FOR_GPU='$WAIT_FOR_GPU' MIN_FREE_GPU_MEMORY_MB='$MIN_FREE_GPU_MEMORY_MB' MAX_GPU_UTIL_PERCENT='$MAX_GPU_UTIL_PERCENT' GPU_WAIT_INTERVAL_SECONDS='$GPU_WAIT_INTERVAL_SECONDS' XICM_QWEN25_VL_7B_PATH='$XICM_QWEN25_VL_7B_PATH' XICM_VLLM_GPU_MEMORY_UTILIZATION='$XICM_VLLM_GPU_MEMORY_UTILIZATION' XICM_VLLM_MAX_MODEL_LEN='$XICM_VLLM_MAX_MODEL_LEN' XICM_VL_MAX_IMAGES='$XICM_VL_MAX_IMAGES' XICM_GA_REVIEW_BUNDLE='$XICM_GA_REVIEW_BUNDLE' bash -s" <<'REMOTE_SCRIPT'
set -Eeuo pipefail

RUN_SCRIPT="$REMOTE_COMPONENT_ROOT/run_rerank_top50_all23_1ep_video.sh"
PROGRESS_JSON="$REMOTE_COMPONENT_ROOT/progress_rerank_top50_all23_1ep_video.json"
PID_FILE="$REMOTE_COMPONENT_ROOT/rerank_top50_all23_1ep_video.pid"
METHOD_NAME="XICM_Cross.ZS_Ranking.${RANKING_METHOD}_${MODEL_NAME}_icl.${K}_${METHOD_SUFFIX}"
LOG_PATH="$REMOTE_RUNNER_LOG_ROOT/rerank_top50_all23_k${K}_seed${SEED}_$(date -u +%Y%m%d_%H%M%S).log"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && ps -p "$old_pid" -o args= | grep -q "$RUN_SCRIPT"; then
    echo "rerank video runner already active as PID $old_pid"
    echo "Progress: $PROGRESS_JSON"
    exit 0
  fi
fi

cat > "$RUN_SCRIPT" <<'RUNNER'
#!/usr/bin/env bash
set -Eeuo pipefail

write_progress() {
  local status="$1"
  local message="$2"
  local exit_status="${3:-}"
  STATUS="$status" MESSAGE="$message" EXIT_STATUS="$exit_status" python3 - <<'PY'
from pathlib import Path
import json
import os
import re
from datetime import datetime, timezone

tasks = [item.strip() for item in os.environ["TASKS"].split(",") if item.strip()]
method_dir = Path(os.environ["REMOTE_RUN_XICM"]) / "logs" / os.environ["METHOD_NAME"]
video_dir = Path(os.environ["REMOTE_REVIEW_ROOT"]) / "runtime_videos"
finish_re = re.compile(r"Finished\s+[^|]+?\s+\|\s+Final Score:\s*([^\r\n]+)")
step_score_re = re.compile(r"Score:\s*([-+]?\d+(?:\.\d+)?)")
finished = []
if method_dir.exists():
    for task in tasks:
        csv_path = method_dir / task / f"seed{os.environ['SEED']}" / "test_data.csv"
        score = None
        if csv_path.exists():
            text = csv_path.read_text(errors="replace")
            match = finish_re.search(text)
            if match:
                final = match.group(1).strip()
                if final.lower() == "unknown":
                    step_scores = step_score_re.findall(text)
                    score = float(step_scores[-1]) if step_scores else "unknown"
                else:
                    try:
                        score = float(final)
                    except ValueError:
                        score = final
        if score is not None:
            finished.append({"task": task, "score": score})
videos = sorted(video_dir.glob("*.mp4")) if video_dir.exists() else []
payload = {
    "status": os.environ["STATUS"],
    "message": os.environ["MESSAGE"],
    "exit_status": os.environ["EXIT_STATUS"],
    "method_name": os.environ["METHOD_NAME"],
    "ranking_method": os.environ["RANKING_METHOD"],
    "model_name": os.environ["MODEL_NAME"],
    "tasks": tasks,
    "rlbench_demo_path": os.environ["RLBENCH_DEMO_PATH"],
    "seed": int(os.environ["SEED"]),
    "episodes": int(os.environ["EPISODES"]),
    "episode_length": int(os.environ["EPISODE_LENGTH"]),
    "k": int(os.environ["K"]),
    "rerank_candidates": os.environ["RERANK_CANDIDATES"],
    "review_bundle": os.environ["XICM_GA_REVIEW_BUNDLE"],
    "component_root": os.environ["REMOTE_COMPONENT_ROOT"],
    "run_xicm_root": os.environ["REMOTE_RUN_XICM"],
    "review_root": os.environ["REMOTE_REVIEW_ROOT"],
    "runtime_video_dir": str(video_dir),
    "log_path": os.environ["LOG_PATH"],
    "enable_video": os.environ["ENABLE_VIDEO"],
    "record_every_n": os.environ["RECORD_EVERY_N"],
    "finished_task_count": len(finished),
    "total_task_count": len(tasks),
    "video_count": len(videos),
    "finished_tasks": finished,
    "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
Path(os.environ["PROGRESS_JSON"]).write_text(json.dumps(payload, indent=2) + "\n")
PY
}

wait_for_gpu_capacity() {
  if [[ "$WAIT_FOR_GPU" != "1" && "$WAIT_FOR_GPU" != "true" && "$WAIT_FOR_GPU" != "yes" ]]; then
    return 0
  fi
  while true; do
    local stats used total util free
    stats="$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    IFS=',' read -r used total util <<< "$stats"
    free=$(( total - used ))
    if (( free >= MIN_FREE_GPU_MEMORY_MB && util <= MAX_GPU_UTIL_PERCENT )); then
      echo "gpu_wait_ready_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) gpu_id=$GPU_ID free_memory_mb=$free used_memory_mb=$used total_memory_mb=$total util_percent=$util" >> "$LOG_PATH"
      return 0
    fi
    echo "gpu_wait_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) gpu_id=$GPU_ID free_memory_mb=$free used_memory_mb=$used total_memory_mb=$total util_percent=$util" >> "$LOG_PATH"
    write_progress "waiting_for_gpu" "Waiting for GPU $GPU_ID before rerank video run."
    sleep "$GPU_WAIT_INTERVAL_SECONDS"
  done
}

collect_videos_and_scores() {
  local video_review_dir="$REMOTE_REVIEW_ROOT/runtime_videos"
  mkdir -p "$video_review_dir"
  IFS=',' read -ra task_list <<< "$TASKS"
  for task_name in "${task_list[@]}"; do
    task_name="$(echo "$task_name" | xargs)"
    [[ -z "$task_name" ]] && continue
    match="$(
      find "$REMOTE_RUN_XICM/logs/videos" -type f -path "*/${task_name}_w/*_s${SEED}_*.mp4" -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr \
        | head -1 \
        | cut -d' ' -f2-
    )"
    if [[ -n "$match" && -f "$match" ]]; then
      result="$(basename "$match")"
      result="${result##*_s${SEED}_}"
      result="${result%.mp4}"
      cp "$match" "$video_review_dir/${task_name}_seed${SEED}_${result}.mp4"
    else
      echo "No video found for task=$task_name seed=$SEED" >> "$LOG_PATH"
    fi
  done

  python3 - <<'PY'
from pathlib import Path
import csv
import os
import re

tasks = [item.strip() for item in os.environ["TASKS"].split(",") if item.strip()]
method_dir = Path(os.environ["REMOTE_RUN_XICM"]) / "logs" / os.environ["METHOD_NAME"]
review_root = Path(os.environ["REMOTE_REVIEW_ROOT"])
video_dir = review_root / "runtime_videos"
finish_re = re.compile(r"Finished\s+[^|]+?\s+\|\s+Final Score:\s*([^\r\n]+)")
step_score_re = re.compile(r"Score:\s*([-+]?\d+(?:\.\d+)?)")
rows = []
for task in tasks:
    csv_path = method_dir / task / f"seed{os.environ['SEED']}" / "test_data.csv"
    score = ""
    if csv_path.exists():
        text = csv_path.read_text(errors="replace")
        match = finish_re.search(text)
        if match:
            final = match.group(1).strip()
            if final.lower() == "unknown":
                step_scores = step_score_re.findall(text)
                score = step_scores[-1] if step_scores else "unknown"
            else:
                score = final
    videos = sorted(video_dir.glob(f"{task}_seed{os.environ['SEED']}_*.mp4"))
    rows.append({
        "task": task,
        "seed": os.environ["SEED"],
        "score": score,
        "video_file": videos[-1].name if videos else "",
        "test_data_csv": str(csv_path) if csv_path.exists() else "",
    })
out = review_root / "all23_scores_and_videos.csv"
with out.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["task", "seed", "score", "video_file", "test_data_csv"])
    writer.writeheader()
    writer.writerows(rows)
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
export XICM_QWEN25_VL_7B_PATH="$XICM_QWEN25_VL_7B_PATH"
export XICM_VLLM_GPU_MEMORY_UTILIZATION="$XICM_VLLM_GPU_MEMORY_UTILIZATION"
export XICM_VLLM_MAX_MODEL_LEN="$XICM_VLLM_MAX_MODEL_LEN"
export XICM_VL_MAX_IMAGES="$XICM_VL_MAX_IMAGES"
export XICM_GA_REVIEW_BUNDLE="$XICM_GA_REVIEW_BUNDLE"
export XICM_GA_RERANK_CANDIDATES="$RERANK_CANDIDATES"
export MULTIPROCESSING_START_METHOD=spawn

mkdir -p "$REMOTE_REVIEW_ROOT/runtime_videos"
write_progress "running" "Started rerank top-50 all-task one-episode video run."

{
  echo "method=$METHOD_NAME"
  echo "ranking=$RANKING_METHOD"
  echo "tasks=$TASKS"
  echo "rlbench_demo_path=$RLBENCH_DEMO_PATH"
  echo "seed=$SEED"
  echo "episodes=$EPISODES"
  echo "episode_length=$EPISODE_LENGTH"
  echo "k=$K"
  echo "rerank_candidates=$RERANK_CANDIDATES"
  echo "enable_video=$ENABLE_VIDEO"
  echo "record_every_n=$RECORD_EVERY_N"
  echo "review_root=$REMOTE_REVIEW_ROOT"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "$LOG_PATH"

if ! command -v xvfb-run >/dev/null 2>&1; then
  echo "xvfb-run is not available on this CAIR host." >> "$LOG_PATH"
  write_progress "failed" "xvfb-run is missing on CAIR." "127"
  exit 127
fi

wait_for_gpu_capacity

set +e
CUDA_VISIBLE_DEVICES="$GPU_ID" timeout "$RUN_TIMEOUT_SECONDS" xvfb-run -a python3 main.py \
  "method.name=$METHOD_NAME" \
  "rlbench.tasks=[$TASKS]" \
  "rlbench.demo_path=$RLBENCH_DEMO_PATH" \
  "framework.start_seed=$SEED" \
  "framework.eval_episodes=$EPISODES" \
  "rlbench.episode_length=$EPISODE_LENGTH" \
  "framework.demo_num_per_icl=$K" \
  "framework.ranking_method=$RANKING_METHOD" \
  "framework.record_every_n=$RECORD_EVERY_N" \
  "cinematic_recorder.enabled=$ENABLE_VIDEO" \
  >> "$LOG_PATH" 2>&1
status=$?
set -e

{
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "exit_status=$status"
} >> "$LOG_PATH"

collect_videos_and_scores

if [[ "$status" -eq 0 ]]; then
  write_progress "finished" "Rerank top-50 all-task video run finished." "$status"
else
  write_progress "failed" "Rerank top-50 all-task video run exited with status $status." "$status"
fi
exit "$status"
RUNNER

chmod +x "$RUN_SCRIPT"
cat > "$PROGRESS_JSON" <<JSON
{
  "status": "launching",
  "method_name": "$METHOD_NAME",
  "ranking_method": "$RANKING_METHOD",
  "model_name": "$MODEL_NAME",
  "tasks": "$TASKS",
  "rlbench_demo_path": "$RLBENCH_DEMO_PATH",
  "seed": $SEED,
  "episodes": $EPISODES,
  "episode_length": $EPISODE_LENGTH,
  "k": $K,
  "rerank_candidates": "$RERANK_CANDIDATES",
  "review_bundle": "$XICM_GA_REVIEW_BUNDLE",
  "component_root": "$REMOTE_COMPONENT_ROOT",
  "run_xicm_root": "$REMOTE_RUN_XICM",
  "review_root": "$REMOTE_REVIEW_ROOT",
  "runtime_video_dir": "$REMOTE_REVIEW_ROOT/runtime_videos",
  "log_path": "$LOG_PATH",
  "enable_video": "$ENABLE_VIDEO",
  "record_every_n": "$RECORD_EVERY_N"
}
JSON

(
  export REMOTE_COMPONENT_ROOT REMOTE_RUN_XICM REMOTE_REVIEW_ROOT PROGRESS_JSON LOG_PATH
  export CONDA_ENV SEED EPISODES EPISODE_LENGTH K TASKS RLBENCH_DEMO_PATH MODEL_NAME GPU_ID RANKING_METHOD METHOD_NAME RUN_TIMEOUT_SECONDS ENABLE_VIDEO RECORD_EVERY_N
  export RERANK_CANDIDATES WAIT_FOR_GPU MIN_FREE_GPU_MEMORY_MB MAX_GPU_UTIL_PERCENT GPU_WAIT_INTERVAL_SECONDS
  export XICM_QWEN25_VL_7B_PATH XICM_VLLM_GPU_MEMORY_UTILIZATION XICM_VLLM_MAX_MODEL_LEN XICM_VL_MAX_IMAGES XICM_GA_REVIEW_BUNDLE
  nohup "$RUN_SCRIPT" > "$REMOTE_RUNNER_LOG_ROOT/rerank_top50_all23_video_launcher_$(date -u +%Y%m%d_%H%M%S).log" 2>&1 &
  echo $! > "$PID_FILE"
)

echo "Launched rerank top-50 all-task one-episode video run."
echo "PID file: $PID_FILE"
echo "Progress: $PROGRESS_JSON"
echo "Log: $LOG_PATH"
echo "Review root: $REMOTE_REVIEW_ROOT"
echo "Method: $METHOD_NAME"
REMOTE_SCRIPT
