#!/usr/bin/env bash
set -Eeuo pipefail

# Run the geometry/affordance X-ICM ablations on CAIR, one condition at a time.
# This script assumes the patched X-ICM prompt/retrieval files have already been
# copied into XICM_ROOT.

XICM_ROOT="${XICM_ROOT:-/data/yf23/projects/ICRA27-ROBOT/X-ICM}"
CACHE_ROOT="${CACHE_ROOT:-/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_full_cache}"
RUN_ROOT="${RUN_ROOT:-/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_ablations}"
CONDA_ENV="${CONDA_ENV:-/data/yf23/conda/envs/zero-shot}"
SEEDS="${SEEDS:-0}"
EPISODES="${EPISODES:-25}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5.7B.instruct}"
DEMO_NUM_PER_ICL="${DEMO_NUM_PER_ICL:-18}"
GPU_ID="${GPU_ID:-1}"
TOTAL_TASKS="${TOTAL_TASKS:-23}"

mkdir -p "$RUN_ROOT/logs"
PROGRESS_JSON="$RUN_ROOT/progress.json"

write_progress() {
    local status="$1"
    local condition="$2"
    local ranking="$3"
    local method="$4"
    local completed="$5"
    local log_path="$6"
    local message="$7"
    cat > "$PROGRESS_JSON" <<JSON
{
  "status": "$status",
  "condition": "$condition",
  "ranking_method": "$ranking",
  "method": "$method",
  "completed_task_csvs": $completed,
  "total_tasks": $TOTAL_TASKS,
  "seed": "$SEEDS",
  "episodes": $EPISODES,
  "demo_num_per_icl": $DEMO_NUM_PER_ICL,
  "gpu_id": "$GPU_ID",
  "log_path": "$log_path",
  "message": "$message",
  "updated_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON
}

count_completed_tasks() {
    local method="$1"
    local method_dir="$XICM_ROOT/logs/$method"
    local count
    if [[ ! -d "$method_dir" ]]; then
        printf "0"
        return 0
    fi
    count=0
    while IFS= read -r score_file; do
        if grep -Eq "Finished .*Final Score: [0-9]+(\\.[0-9]+)?" "$score_file"; then
            count=$((count + 1))
        fi
    done < <(find "$method_dir" -path "*/seed0/test_data.csv" -type f 2>/dev/null)
    printf "%s" "${count:-0}"
}

run_condition() {
    local condition="$1"
    local ranking="$2"
    local alpha="$3"
    local beta="$4"
    local gamma="$5"
    local method="XICM_Cross.ZS_Ranking.${ranking}_${MODEL_NAME}_icl.${DEMO_NUM_PER_ICL}_test"
    local log_path="$RUN_ROOT/logs/${condition}_$(date -u +%Y%m%d_%H%M%S).log"
    local completed

    completed=$(count_completed_tasks "$method")
    if [[ "$completed" -ge "$TOTAL_TASKS" ]]; then
        write_progress "skipped_completed" "$condition" "$ranking" "$method" "$completed" "$log_path" "Condition already has all task CSVs."
        return 0
    fi

    write_progress "running" "$condition" "$ranking" "$method" "$completed" "$log_path" "Started condition."

    export XICM_GA_REVIEW_BUNDLE="$CACHE_ROOT/review_bundle.jsonl"
    export XICM_GA_ALPHA="$alpha"
    export XICM_GA_BETA="$beta"
    export XICM_GA_GAMMA="$gamma"

    {
        echo "condition=$condition"
        echo "ranking=$ranking"
        echo "weights alpha=$alpha beta=$beta gamma=$gamma"
        echo "method=$method"
        echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        bash scripts/eval_XICM.sh "$SEEDS" "$EPISODES" "$MODEL_NAME" "$DEMO_NUM_PER_ICL" "$GPU_ID" "$ranking" "true"
        echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } 2>&1 | tee "$log_path"

    completed=$(count_completed_tasks "$method")
    write_progress "completed" "$condition" "$ranking" "$method" "$completed" "$log_path" "Condition finished."
}

source /data/yf23/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$XICM_ROOT"

export COPPELIASIM_ROOT="$XICM_ROOT/CoppeliaSim"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export HF_HOME="${HF_HOME:-/data/yf23/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/data/yf23/huggingface/transformers}"
export HF_HUB_DISABLE_XET=1
export XICM_SD2_BASE_PATH="${XICM_SD2_BASE_PATH:-/data/yf23/models/Manojb-stable-diffusion-2-base}"
export MULTIPROCESSING_START_METHOD=spawn

write_progress "starting" "" "" "" 0 "" "Preparing ablation sequence."

run_condition "geometry" "lang_vis.out.geo" "0.65" "0.35" "0.0"
run_condition "affordance" "lang_vis.out.aff" "0.65" "0.0" "0.35"
run_condition "geometry_affordance" "lang_vis.out.geo_aff" "0.65" "0.30" "0.05"

write_progress "all_completed" "" "" "" "$TOTAL_TASKS" "" "All requested ablations finished."
