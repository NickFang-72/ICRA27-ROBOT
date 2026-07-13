#!/usr/bin/env bash
set -Eeuo pipefail

# Run a five-demo phase-memory smoke test on CAIR and bundle every
# static input plus every runtime controller input/output into review/.

REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/data/yf23/projects/ICRA27-ROBOT}"
REMOTE_RUN_XICM="${REMOTE_RUN_XICM:-$REMOTE_PROJECT_ROOT/X-ICM-phase-memory}"
REMOTE_REVIEW_ROOT="${REMOTE_REVIEW_ROOT:-$REMOTE_PROJECT_ROOT/review}"
CONDA_ENV="${CONDA_ENV:-/data/yf23/conda/envs/zero-shot}"
RUN_NAME="${RUN_NAME:-phase_memory_io5_$(date -u +%Y%m%d_%H%M%S)}"
TASKS="${TASKS:-phone_on_base,close_microwave,take_lid_off_saucepan,toilet_seat_down,lamp_on}"
EPISODES="${EPISODES:-1}"
SEED="${SEED:-0}"
GPU_ID="${GPU_ID:-0}"
PHASE_REVIEW_ROOT="${PHASE_REVIEW_ROOT:-$REMOTE_PROJECT_ROOT/review/phase_anchor_benchmark_eager_5eps_all23_20260708_213644}"
DISPLAY_ID="${DISPLAY_ID:-:993}"
RECORD_VIDEO="${RECORD_VIDEO:-1}"
RECORD_EVERY_N="${RECORD_EVERY_N:-1}"

export COPPELIASIM_ROOT="$REMOTE_RUN_XICM/CoppeliaSim"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export XKB_CONFIG_ROOT="${XKB_CONFIG_ROOT:-/usr/share/X11/xkb}"
export XICM_PHASE_MEMORY_REVIEW_ROOT="$PHASE_REVIEW_ROOT"
export XICM_PHASE_MEMORY_RETRIEVAL_METHOD="${XICM_PHASE_MEMORY_RETRIEVAL_METHOD:-lang_vis.out.geo.aff_v3.rerank_top50}"
export XICM_PHASE_MEMORY_RETRIEVAL_K="${XICM_PHASE_MEMORY_RETRIEVAL_K:-4}"
export XICM_PHASE_MEMORY_MAX_ACTIONS_PER_PHASE="${XICM_PHASE_MEMORY_MAX_ACTIONS_PER_PHASE:-2}"
export XICM_PHASE_MEMORY_MAX_TOKENS="${XICM_PHASE_MEMORY_MAX_TOKENS:-240}"
export XICM_PHASE_MEMORY_USE_IMAGES="${XICM_PHASE_MEMORY_USE_IMAGES:-0}"
export XICM_VLLM_ENFORCE_EAGER="${XICM_VLLM_ENFORCE_EAGER:-1}"
export XICM_VLLM_GPU_MEMORY_UTILIZATION="${XICM_VLLM_GPU_MEMORY_UTILIZATION:-0.72}"
export XICM_VLLM_MAX_MODEL_LEN="${XICM_VLLM_MAX_MODEL_LEN:-12000}"
export XICM_VL_MAX_IMAGES="${XICM_VL_MAX_IMAGES:-2}"
export XICM_VL_MIN_PIXELS="${XICM_VL_MIN_PIXELS:-3136}"
export XICM_VL_MAX_PIXELS="${XICM_VL_MAX_PIXELS:-200704}"

cd "$REMOTE_RUN_XICM"

METHOD_NAME="phase_memory_Qwen2.5.VL.7B.instruct.${RUN_NAME}"
SIM_LOG_ROOT="$REMOTE_RUN_XICM/logs/phase_memory_${RUN_NAME}"
PACKAGED_REVIEW="$REMOTE_REVIEW_ROOT/$RUN_NAME"
PROGRESS_FILE="$REMOTE_REVIEW_ROOT/${RUN_NAME}_progress.json"
mkdir -p "$SIM_LOG_ROOT" "$REMOTE_REVIEW_ROOT"

write_progress() {
  "$CONDA_ENV/bin/python3" - "$PROGRESS_FILE" "$1" "$2" <<'PY'
import json, sys, time
path, status, detail = sys.argv[1:4]
data = {
    "status": status,
    "detail": detail,
    "updated_unix": time.time(),
}
with open(path, "w") as handle:
    json.dump(data, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

cleanup_xvfb() {
  if [[ -n "${XVFB_PID:-}" ]] && kill -0 "$XVFB_PID" >/dev/null 2>&1; then
    kill "$XVFB_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup_xvfb EXIT

write_progress "starting_xvfb" "run=$RUN_NAME tasks=$TASKS"
if pgrep -f "Xvfb $DISPLAY_ID " >/dev/null 2>&1; then
  DISPLAY_ID=":$((900 + RANDOM % 90))"
fi
"$CONDA_ENV/bin/Xvfb" "$DISPLAY_ID" -screen 0 1280x720x24 -ac -nolisten tcp > "/tmp/${RUN_NAME}_xvfb.log" 2>&1 &
XVFB_PID=$!
sleep 2
export DISPLAY="$DISPLAY_ID"

tasks_hydra="[${TASKS}]"
if [[ "$RECORD_VIDEO" == "1" ]]; then
  cinematic_enabled=True
else
  cinematic_enabled=False
fi

write_progress "running_simulation" "logs=$SIM_LOG_ROOT"
set +e
timeout "${RUN_TIMEOUT_SECONDS:-7200}" "$CONDA_ENV/bin/python3" main.py \
  method.name="$METHOD_NAME" \
  "rlbench.tasks=$tasks_hydra" \
  rlbench.demo_path=data/unseen_tasks/test \
  framework.start_seed="$SEED" \
  framework.eval_episodes="$EPISODES" \
  rlbench.episode_length=25 \
  framework.demo_num_per_icl="${XICM_PHASE_MEMORY_RETRIEVAL_K}" \
  framework.ranking_method=phase_memory \
  framework.logdir=logs/phase_memory_"$RUN_NAME" \
  framework.record_every_n="$RECORD_EVERY_N" \
  cinematic_recorder.enabled="$cinematic_enabled" \
  framework.eval_save_metrics=True
status=$?
set -e
echo "$status" > "$SIM_LOG_ROOT/simulation_exit_status.txt"

write_progress "collecting_review" "simulation_exit_status=$status"
"$CONDA_ENV/bin/python3" scripts/collect_phase_memory_runtime_review.py \
  --log-root "$SIM_LOG_ROOT" \
  --output-root "$REMOTE_REVIEW_ROOT" \
  --name "$RUN_NAME" \
  --tasks "$TASKS"

"$CONDA_ENV/bin/python3" - "$PACKAGED_REVIEW" "$SIM_LOG_ROOT" "$PROGRESS_FILE" "$status" "$TASKS" <<'PY'
import json, pathlib, sys, time
review = pathlib.Path(sys.argv[1])
log_root = pathlib.Path(sys.argv[2])
progress = pathlib.Path(sys.argv[3])
status = int(sys.argv[4])
tasks = sys.argv[5].split(",")
summary_path = review / "run_manifest.json"
videos = sorted(str(p) for p in review.rglob("*.mp4"))
data = {
    "review_root": str(review),
    "sim_log_root": str(log_root),
    "simulation_exit_status": status,
    "tasks": tasks,
    "video_count": len(videos),
    "videos": videos,
    "finished_unix": time.time(),
}
summary_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
progress.write_text(json.dumps({"status": "done" if status == 0 else "failed", "detail": str(summary_path), "updated_unix": time.time()}, indent=2, sort_keys=True) + "\n")
PY

echo "Phase-memory review: $PACKAGED_REVIEW"
echo "Simulation logs: $SIM_LOG_ROOT"
echo "Progress file: $PROGRESS_FILE"
echo "Simulation exit status: $status"
exit "$status"
