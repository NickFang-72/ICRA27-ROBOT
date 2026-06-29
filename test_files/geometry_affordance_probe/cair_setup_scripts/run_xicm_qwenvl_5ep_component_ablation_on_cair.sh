#!/usr/bin/env bash
set -Eeuo pipefail

# Launch a quick QwenVL component ablation on CAIR:
#   1) QwenVL + geometry retrieval / target-pose prompt
#   2) QwenVL + contact points prompt only
#   3) QwenVL + geometry retrieval / target-pose prompt + contact points prompt
#
# Defaults are intentionally small: seed 0, 5 episodes per task, 23 tasks.
# Contact points remain prompt-only; gamma_contact stays 0.0.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/data/yf23/projects/ICRA27-ROBOT}"
export REMOTE_COMPONENT_ROOT="${REMOTE_COMPONENT_ROOT:-$REMOTE_PROJECT_ROOT/experiments/qwenvl_component_5eps_seed0_20260627}"
export REMOTE_RUN_XICM="${REMOTE_RUN_XICM:-$REMOTE_COMPONENT_ROOT/X-ICM_qwenvl_component_5eps}"
export REMOTE_RUNNER_LOG_ROOT="${REMOTE_RUNNER_LOG_ROOT:-$REMOTE_COMPONENT_ROOT/runner_logs}"
export REMOTE_CLEAN_GEOMETRY_CACHE="${REMOTE_CLEAN_GEOMETRY_CACHE:-$REMOTE_PROJECT_ROOT/experiments/geometry_target_pose_v2_full_cache_20260626/review_bundle.jsonl}"
export REMOTE_OLD_CONTACT_CACHE="${REMOTE_OLD_CONTACT_CACHE:-$REMOTE_PROJECT_ROOT/experiments/geometry_affordance_full_cache/review_bundle.jsonl}"
export REMOTE_CONTACT_CACHE_DIR="${REMOTE_CONTACT_CACHE_DIR:-$REMOTE_COMPONENT_ROOT/merged_v2_geometry_with_contact_hints}"
export REMOTE_CONTACT_CACHE="${REMOTE_CONTACT_CACHE:-$REMOTE_CONTACT_CACHE_DIR/review_bundle.jsonl}"

export SEEDS="${SEEDS:-0}"
export EPISODES="${EPISODES:-5}"
export MODEL_NAME="${MODEL_NAME:-Qwen2.5.VL.7B.instruct}"
export GPU_ID="${GPU_ID:-1}"
export WAIT_FOR_GPU="${WAIT_FOR_GPU:-1}"
export MIN_FREE_GPU_MEMORY_MB="${MIN_FREE_GPU_MEMORY_MB:-38000}"
export MAX_GPU_UTIL_PERCENT="${MAX_GPU_UTIL_PERCENT:-10}"
export GPU_WAIT_INTERVAL_SECONDS="${GPU_WAIT_INTERVAL_SECONDS:-300}"
export XICM_QWEN25_VL_7B_PATH="${XICM_QWEN25_VL_7B_PATH:-/data/yf23/checkpoints/ICRA27-ROBOT/Qwen2.5-VL-7B-Instruct}"
export XICM_VLLM_GPU_MEMORY_UTILIZATION="${XICM_VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
export XICM_VLLM_MAX_MODEL_LEN="${XICM_VLLM_MAX_MODEL_LEN:-24576}"
export XICM_VL_MAX_IMAGES="${XICM_VL_MAX_IMAGES:-2}"

bash "$SCRIPT_DIR/run_xicm_v1_10ep_component_ablation_on_cair.sh"
