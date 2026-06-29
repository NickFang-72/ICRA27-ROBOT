#!/usr/bin/env bash
set -Eeuo pipefail

# Launch a clean vanilla X-ICM 7B baseline rerun on CAIR.
#
# This intentionally runs ranking_method=lang_vis.out with the original X-ICM
# prompt/retrieval files copied into an isolated experiment directory. It does
# not use any geometry/affordance descriptors or v2/v3/v4 prompt code.

CAIR_HOST="${CAIR_HOST:-cair}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/data/yf23/projects/ICRA27-ROBOT}"
REMOTE_SOURCE_XICM="${REMOTE_SOURCE_XICM:-$REMOTE_PROJECT_ROOT/X-ICM}"
REMOTE_BASELINE_ROOT="${REMOTE_BASELINE_ROOT:-$REMOTE_PROJECT_ROOT/experiments/vanilla_baseline_rerun_3seed_20260622}"
REMOTE_RUN_XICM="${REMOTE_RUN_XICM:-$REMOTE_BASELINE_ROOT/X-ICM_vanilla}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-$REMOTE_BASELINE_ROOT/logs}"
CONDA_ENV="${CONDA_ENV:-/data/yf23/conda/envs/zero-shot}"
SEEDS="${SEEDS:-0,50,99}"
EPISODES="${EPISODES:-25}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5.7B.instruct}"
DEMO_NUM_PER_ICL="${DEMO_NUM_PER_ICL:-18}"
GPU_ID="${GPU_ID:-0}"
RANKING_METHOD="${RANKING_METHOD:-lang_vis.out}"
TOTAL_TASKS="${TOTAL_TASKS:-23}"

ssh "$CAIR_HOST" \
  "REMOTE_SOURCE_XICM='$REMOTE_SOURCE_XICM' REMOTE_BASELINE_ROOT='$REMOTE_BASELINE_ROOT' REMOTE_RUN_XICM='$REMOTE_RUN_XICM' REMOTE_LOG_ROOT='$REMOTE_LOG_ROOT' CONDA_ENV='$CONDA_ENV' SEEDS='$SEEDS' EPISODES='$EPISODES' MODEL_NAME='$MODEL_NAME' DEMO_NUM_PER_ICL='$DEMO_NUM_PER_ICL' GPU_ID='$GPU_ID' RANKING_METHOD='$RANKING_METHOD' TOTAL_TASKS='$TOTAL_TASKS' bash -s" <<'REMOTE_SCRIPT'
set -Eeuo pipefail

METHOD="XICM_Cross.ZS_Ranking.${RANKING_METHOD}_${MODEL_NAME}_icl.${DEMO_NUM_PER_ICL}_test"
PROGRESS_JSON="$REMOTE_BASELINE_ROOT/progress_baseline.json"
RUN_SCRIPT="$REMOTE_BASELINE_ROOT/run_vanilla_baseline.sh"
IFS=',' read -r -a SEED_LIST <<< "$SEEDS"
TOTAL_SEED_TASKS=$(( ${#SEED_LIST[@]} * TOTAL_TASKS ))

mkdir -p "$REMOTE_BASELINE_ROOT" "$REMOTE_LOG_ROOT"

write_progress() {
  local status="$1"
  local completed="$2"
  local log_path="$3"
  local message="$4"
  cat > "$PROGRESS_JSON" <<JSON
{
  "status": "$status",
  "condition": "vanilla_xicm_baseline_3seed_rerun",
  "method": "$METHOD",
  "ranking_method": "$RANKING_METHOD",
  "seed": "$SEEDS",
  "episodes": $EPISODES,
  "demo_num_per_icl": $DEMO_NUM_PER_ICL,
  "model_name": "$MODEL_NAME",
  "gpu_id": "$GPU_ID",
  "completed_seed_task_csvs": $completed,
  "total_tasks": $TOTAL_TASKS,
  "total_seed_task_csvs": $TOTAL_SEED_TASKS,
  "run_xicm_root": "$REMOTE_RUN_XICM",
  "log_path": "$log_path",
  "message": "$message",
  "updated_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON
}

count_completed_tasks() {
  local method_dir="$REMOTE_RUN_XICM/logs/$METHOD"
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

if pgrep -af "vanilla_baseline_rerun_20260622|X-ICM_vanilla|$METHOD" | grep -v pgrep >/dev/null 2>&1; then
  completed="$(count_completed_tasks)"
  write_progress "already_running" "$completed" "" "A vanilla baseline rerun process already appears to be running."
  echo "Baseline rerun already appears to be running."
  exit 0
fi

if [[ ! -d "$REMOTE_RUN_XICM" ]]; then
  echo "Creating isolated vanilla X-ICM tree at $REMOTE_RUN_XICM"
  mkdir -p "$REMOTE_RUN_XICM"
  rsync -a --delete \
    --exclude 'logs' \
    --exclude 'outputs' \
    "$REMOTE_SOURCE_XICM/" "$REMOTE_RUN_XICM/"

  # Restore the algorithmic baseline files from the source repo HEAD inside the
  # isolated tree. Keep current infrastructure-only path fixes in main.py and
  # rlbench_inference_dynamics_diffusion.py so CAIR can load local models.
  git -C "$REMOTE_SOURCE_XICM" show HEAD:crosstask_icl_agent.py > "$REMOTE_RUN_XICM/crosstask_icl_agent.py"
  git -C "$REMOTE_SOURCE_XICM" show HEAD:form_icl_demonstrations_crosstask_ranking.py > "$REMOTE_RUN_XICM/form_icl_demonstrations_crosstask_ranking.py"
fi

mkdir -p "$REMOTE_RUN_XICM/logs"
completed="$(count_completed_tasks)"
if [[ "$completed" -ge "$TOTAL_SEED_TASKS" ]]; then
  write_progress "skipped_completed" "$completed" "" "Vanilla baseline rerun already has all strict task scores."
  echo "Vanilla baseline rerun already complete: $completed/$TOTAL_SEED_TASKS"
  exit 0
fi

cat > "$RUN_SCRIPT" <<'RUNNER'
#!/usr/bin/env bash
set -Eeuo pipefail

count_completed_tasks() {
  local method_dir="$REMOTE_RUN_XICM/logs/$METHOD"
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
  local completed="$2"
  local message="$3"
  cat > "$PROGRESS_JSON" <<JSON
{
  "status": "$status",
  "condition": "vanilla_xicm_baseline_3seed_rerun",
  "method": "$METHOD",
  "ranking_method": "$RANKING_METHOD",
  "seed": "$SEEDS",
  "episodes": $EPISODES,
  "demo_num_per_icl": $DEMO_NUM_PER_ICL,
  "model_name": "$MODEL_NAME",
  "gpu_id": "$GPU_ID",
  "completed_seed_task_csvs": $completed,
  "total_tasks": $TOTAL_TASKS,
  "total_seed_task_csvs": $TOTAL_SEED_TASKS,
  "run_xicm_root": "$REMOTE_RUN_XICM",
  "log_path": "$LOG_PATH",
  "message": "$message",
  "updated_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON
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
export XICM_QWEN_7B_PATH="${XICM_QWEN_7B_PATH:-/data/yf23/models/Qwen2.5-7B-Instruct}"
export XICM_SD2_BASE_PATH="${XICM_SD2_BASE_PATH:-/data/yf23/models/Manojb-stable-diffusion-2-base}"
export MULTIPROCESSING_START_METHOD=spawn

echo "condition=vanilla_xicm_baseline_3seed_rerun"
echo "method=XICM_Cross.ZS_Ranking.${RANKING_METHOD}_${MODEL_NAME}_icl.${DEMO_NUM_PER_ICL}_test"
echo "ranking=${RANKING_METHOD}"
echo "model=${MODEL_NAME}"
echo "seed=${SEEDS}"
echo "episodes=${EPISODES}"
echo "demo_num_per_icl=${DEMO_NUM_PER_ICL}"
echo "gpu_id=${GPU_ID}"
echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
set +e
bash scripts/eval_XICM.sh "$SEEDS" "$EPISODES" "$MODEL_NAME" "$DEMO_NUM_PER_ICL" "$GPU_ID" "$RANKING_METHOD" "true"
status=$?
set -e
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
completed="$(count_completed_tasks)"
if [[ "$status" == "0" ]]; then
  write_progress "completed" "$completed" "Vanilla baseline rerun finished."
else
  write_progress "failed" "$completed" "Vanilla baseline rerun exited with status $status."
fi
exit "$status"
RUNNER
chmod +x "$RUN_SCRIPT"

log_path="$REMOTE_LOG_ROOT/vanilla_baseline_seed${SEEDS}_$(date -u +%Y%m%d_%H%M%S).log"
write_progress "running" "$completed" "$log_path" "Started clean vanilla X-ICM baseline rerun."

nohup env \
  REMOTE_RUN_XICM="$REMOTE_RUN_XICM" \
  CONDA_ENV="$CONDA_ENV" \
  RANKING_METHOD="$RANKING_METHOD" \
  MODEL_NAME="$MODEL_NAME" \
  DEMO_NUM_PER_ICL="$DEMO_NUM_PER_ICL" \
  SEEDS="$SEEDS" \
  EPISODES="$EPISODES" \
  GPU_ID="$GPU_ID" \
  METHOD="$METHOD" \
  TOTAL_TASKS="$TOTAL_TASKS" \
  TOTAL_SEED_TASKS="$TOTAL_SEED_TASKS" \
  PROGRESS_JSON="$PROGRESS_JSON" \
  LOG_PATH="$log_path" \
  "$RUN_SCRIPT" > "$log_path" 2>&1 &

pid="$!"
echo "$pid" > "$REMOTE_BASELINE_ROOT/vanilla_baseline.pid"
echo "Launched vanilla baseline rerun PID $pid"
echo "Progress: $PROGRESS_JSON"
echo "Log: $log_path"
echo "Method dir: $REMOTE_RUN_XICM/logs/$METHOD"
REMOTE_SCRIPT
