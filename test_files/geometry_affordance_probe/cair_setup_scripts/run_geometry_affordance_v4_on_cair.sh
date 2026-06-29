#!/usr/bin/env bash
set -Eeuo pipefail

# Run the v4 geometry+affordance X-ICM ablation on CAIR.
# v4 keeps dynamics-anchored retrieval, adds mechanical filtering, and uses a
# two-stage semantic bottleneck: semantic plan first, then one Stage 2 call that
# emits a relative action sketch plus final 7D actions.

XICM_ROOT="${XICM_ROOT:-/data/yf23/projects/ICRA27-ROBOT/X-ICM}"
CACHE_ROOT="${CACHE_ROOT:-/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_full_cache}"
RUN_ROOT="${RUN_ROOT:-/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_ablations}"
CONDA_ENV="${CONDA_ENV:-/data/yf23/conda/envs/zero-shot}"
SEEDS="${SEEDS:-0}"
EPISODES="${EPISODES:-25}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5.7B.instruct}"
DEMO_NUM_PER_ICL="${DEMO_NUM_PER_ICL:-6}"
GPU_ID="${GPU_ID:-1}"
TOTAL_TASKS="${TOTAL_TASKS:-23}"

RANKING_METHOD="${RANKING_METHOD:-lang_vis.out.geo_aff_v4}"
CONDITION="${CONDITION:-geometry_affordance_v4_k${DEMO_NUM_PER_ICL}}"
ALPHA="${XICM_GA_ALPHA:-0.70}"
BETA="${XICM_GA_BETA:-0.05}"
GAMMA="${XICM_GA_GAMMA:-0.05}"
DELTA="${XICM_GA_DELTA:-0.40}"
PENALTY="${XICM_GA_PENALTY:-0.45}"
MAX_PER_TASK="${XICM_GA_MAX_PER_TASK:-2}"
MAX_PER_FAMILY="${XICM_GA_MAX_PER_FAMILY:-3}"

mkdir -p "$RUN_ROOT/logs"
PROGRESS_JSON="$RUN_ROOT/progress_v4.json"

write_progress() {
    local status="$1"
    local completed="$2"
    local log_path="$3"
    local message="$4"
    cat > "$PROGRESS_JSON" <<JSON
{
  "status": "$status",
  "condition": "$CONDITION",
  "ranking_method": "$RANKING_METHOD",
  "method": "XICM_Cross.ZS_Ranking.${RANKING_METHOD}_${MODEL_NAME}_icl.${DEMO_NUM_PER_ICL}_test",
  "completed_task_csvs": $completed,
  "total_tasks": $TOTAL_TASKS,
  "seed": "$SEEDS",
  "episodes": $EPISODES,
  "demo_num_per_icl": $DEMO_NUM_PER_ICL,
  "gpu_id": "$GPU_ID",
  "weights": {
    "alpha": $ALPHA,
    "beta": $BETA,
    "gamma": $GAMMA,
    "delta_mechanical": $DELTA,
    "penalty": $PENALTY,
    "max_per_task": $MAX_PER_TASK,
    "max_per_family": $MAX_PER_FAMILY
  },
  "prompt_style": "two_stage_semantic_bottleneck_with_relative_action_sketch",
  "log_path": "$log_path",
  "message": "$message",
  "updated_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON
}

count_completed_tasks() {
    local method="$1"
    local method_dir="$XICM_ROOT/logs/$method"
    local count=0
    if [[ ! -d "$method_dir" ]]; then
        printf "0"
        return 0
    fi
    while IFS= read -r score_file; do
        if grep -Eq "Finished .*Final Score: [0-9]+(\\.[0-9]+)?" "$score_file"; then
            count=$((count + 1))
        fi
    done < <(find "$method_dir" -path "*/seed0/test_data.csv" -type f 2>/dev/null)
    printf "%s" "${count:-0}"
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

export XICM_GA_REVIEW_BUNDLE="$CACHE_ROOT/review_bundle.jsonl"
export XICM_GA_ALPHA="$ALPHA"
export XICM_GA_BETA="$BETA"
export XICM_GA_GAMMA="$GAMMA"
export XICM_GA_DELTA="$DELTA"
export XICM_GA_PENALTY="$PENALTY"
export XICM_GA_MAX_PER_TASK="$MAX_PER_TASK"
export XICM_GA_MAX_PER_FAMILY="$MAX_PER_FAMILY"

method="XICM_Cross.ZS_Ranking.${RANKING_METHOD}_${MODEL_NAME}_icl.${DEMO_NUM_PER_ICL}_test"
log_path="$RUN_ROOT/logs/${CONDITION}_$(date -u +%Y%m%d_%H%M%S).log"
completed=$(count_completed_tasks "$method")
if [[ "$completed" -ge "$TOTAL_TASKS" ]]; then
    write_progress "skipped_completed" "$completed" "$log_path" "v4 already has all strict task scores."
    exit 0
fi

write_progress "running" "$completed" "$log_path" "Started v4 semantic bottleneck geometry-affordance condition."
{
    echo "condition=$CONDITION"
    echo "ranking=$RANKING_METHOD"
    echo "weights alpha=$ALPHA beta=$BETA gamma=$GAMMA delta_mechanical=$DELTA penalty=$PENALTY max_per_task=$MAX_PER_TASK max_per_family=$MAX_PER_FAMILY"
    echo "method=$method"
    echo "prompt_style=two_stage_semantic_bottleneck_with_relative_action_sketch"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    bash scripts/eval_XICM.sh "$SEEDS" "$EPISODES" "$MODEL_NAME" "$DEMO_NUM_PER_ICL" "$GPU_ID" "$RANKING_METHOD" "true"
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee "$log_path"

completed=$(count_completed_tasks "$method")
write_progress "completed" "$completed" "$log_path" "v4 semantic bottleneck geometry-affordance condition finished."
