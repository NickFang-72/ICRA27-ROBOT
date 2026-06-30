#!/usr/bin/env bash
set -Eeuo pipefail

# Launch the QwenVL closed-loop no-plan ablation on CAIR:
#   1) closed-loop geometry/target-pose retrieval
#   2) closed-loop geometry/target-pose retrieval + contact points
#
# Closed-loop mode replans from the current observation for the first
# XICM_CLOSED_LOOP_MAX_REPLANS environment steps, executing one primitive per
# observation. The ranking names intentionally avoid geo_plan/.plan so retrieval
# stays prompt-only target-pose geometry, with no plan-guided family blocking.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/data/yf23/projects/ICRA27-ROBOT}"
export REMOTE_COMPONENT_ROOT="${REMOTE_COMPONENT_ROOT:-$REMOTE_PROJECT_ROOT/experiments/qwenvl_closed_loop_no_plan_5eps_seed0_k10_20260628}"
export REMOTE_RUN_XICM="${REMOTE_RUN_XICM:-$REMOTE_COMPONENT_ROOT/X-ICM_qwenvl_closed_loop_no_plan_5eps_k10}"
export REMOTE_RUNNER_LOG_ROOT="${REMOTE_RUNNER_LOG_ROOT:-$REMOTE_COMPONENT_ROOT/runner_logs}"
export REMOTE_CLEAN_GEOMETRY_CACHE="${REMOTE_CLEAN_GEOMETRY_CACHE:-$REMOTE_PROJECT_ROOT/experiments/geometry_target_pose_v2_full_cache_20260626/review_bundle.jsonl}"
export REMOTE_OLD_CONTACT_CACHE="${REMOTE_OLD_CONTACT_CACHE:-$REMOTE_PROJECT_ROOT/experiments/geometry_affordance_full_cache/review_bundle.jsonl}"
export REMOTE_CONTACT_CACHE_DIR="${REMOTE_CONTACT_CACHE_DIR:-$REMOTE_COMPONENT_ROOT/merged_v2_geometry_with_contact_hints}"
export REMOTE_CONTACT_CACHE="${REMOTE_CONTACT_CACHE:-$REMOTE_CONTACT_CACHE_DIR/review_bundle.jsonl}"

export SEEDS="${SEEDS:-0}"
export EPISODES="${EPISODES:-5}"
export MODEL_NAME="${MODEL_NAME:-Qwen2.5.VL.7B.instruct}"
export DEMO_NUM_PER_ICL="${DEMO_NUM_PER_ICL:-10}"
export GPU_ID="${GPU_ID:-1}"
export TOTAL_TASKS="${TOTAL_TASKS:-23}"
export WAIT_FOR_GPU="${WAIT_FOR_GPU:-1}"
export MIN_FREE_GPU_MEMORY_MB="${MIN_FREE_GPU_MEMORY_MB:-38000}"
export MAX_GPU_UTIL_PERCENT="${MAX_GPU_UTIL_PERCENT:-10}"
export GPU_WAIT_INTERVAL_SECONDS="${GPU_WAIT_INTERVAL_SECONDS:-300}"
export XICM_QWEN25_VL_7B_PATH="${XICM_QWEN25_VL_7B_PATH:-/data/yf23/checkpoints/ICRA27-ROBOT/Qwen2.5-VL-7B-Instruct}"
export XICM_VLLM_GPU_MEMORY_UTILIZATION="${XICM_VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
export XICM_VLLM_MAX_MODEL_LEN="${XICM_VLLM_MAX_MODEL_LEN:-24576}"
export XICM_VL_MAX_IMAGES="${XICM_VL_MAX_IMAGES:-2}"

export XICM_GA_DELTA="${XICM_GA_DELTA:-0.25}"
export XICM_GA_PENALTY="${XICM_GA_PENALTY:-0.55}"
export XICM_GA_PLAN_WEIGHT="${XICM_GA_PLAN_WEIGHT:-0.0}"
export XICM_GA_PLAN_BLOCK_CAP="${XICM_GA_PLAN_BLOCK_CAP:-0.15}"
export XICM_GA_PLAN_WEAK_CAP="${XICM_GA_PLAN_WEAK_CAP:-0.55}"
export XICM_GA_PLAN_UNKNOWN_CAP="${XICM_GA_PLAN_UNKNOWN_CAP:-0.45}"
export XICM_GA_MAX_PER_TASK="${XICM_GA_MAX_PER_TASK:-2}"
export XICM_GA_MAX_PER_FAMILY="${XICM_GA_MAX_PER_FAMILY:-3}"
export XICM_CLOSED_LOOP_MAX_REPLANS="${XICM_CLOSED_LOOP_MAX_REPLANS:-4}"

export METHOD_CONFIG_TEXT="${METHOD_CONFIG_TEXT:-$(cat <<'CONFIG'
qwenvl_closed_loop_geo_target_pose_5eps_seed0|lang_vis.out.geo.closed_loop|qwenvl_closed_loop_geo_target_pose|clean|0.70|0.30|0.0
qwenvl_closed_loop_geo_target_pose_contact_5eps_seed0|lang_vis.out.geo.aff.closed_loop|qwenvl_closed_loop_geo_target_pose_contact|contact|0.70|0.30|0.0
CONFIG
)}"

bash "$SCRIPT_DIR/run_xicm_qwenvl_ablation_matrix_on_cair.sh"
