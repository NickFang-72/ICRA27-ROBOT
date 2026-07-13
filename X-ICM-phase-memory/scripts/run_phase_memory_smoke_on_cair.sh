#!/usr/bin/env bash
set -Eeuo pipefail

# Smoke runner for the phase-memory VLM controller.
# Assumes phase packets already exist and include 05_normalized_anchored_phase_plan.json.

REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/data/yf23/projects/ICRA27-ROBOT}"
REMOTE_RUN_XICM="${REMOTE_RUN_XICM:-$REMOTE_PROJECT_ROOT/X-ICM-phase-memory}"
CONDA_ENV="${CONDA_ENV:-/data/yf23/conda/envs/zero-shot}"
RUN_NAME="${RUN_NAME:-phase_memory_smoke_$(date -u +%Y%m%d_%H%M%S)}"
TASKS="${TASKS:-phone_on_base,close_microwave,take_lid_off_saucepan}"
EPISODES="${EPISODES:-1}"
SEED="${SEED:-0}"
GPU_ID="${GPU_ID:-0}"
PHASE_REVIEW_ROOT="${PHASE_REVIEW_ROOT:-$REMOTE_PROJECT_ROOT/review/phase_anchor_benchmark_eager_5eps_all23_20260708_213644}"
DISPLAY_ID="${DISPLAY_ID:-:993}"

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

cd "$REMOTE_RUN_XICM"

cleanup_xvfb() {
  if [[ -n "${XVFB_PID:-}" ]] && kill -0 "$XVFB_PID" >/dev/null 2>&1; then
    kill "$XVFB_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup_xvfb EXIT

if pgrep -f "Xvfb $DISPLAY_ID " >/dev/null 2>&1; then
  DISPLAY_ID=":$((900 + RANDOM % 90))"
fi
"$CONDA_ENV/bin/Xvfb" "$DISPLAY_ID" -screen 0 1280x720x24 -ac -nolisten tcp > "/tmp/${RUN_NAME}_xvfb.log" 2>&1 &
XVFB_PID=$!
sleep 2
export DISPLAY="$DISPLAY_ID"

tasks_hydra="[${TASKS}]"
METHOD_NAME="phase_memory_Qwen2.5.VL.7B.instruct.${RUN_NAME}"

"$CONDA_ENV/bin/python3" main.py \
  method.name="$METHOD_NAME" \
  "rlbench.tasks=$tasks_hydra" \
  rlbench.demo_path=data/unseen_tasks/test \
  framework.start_seed="$SEED" \
  framework.eval_episodes="$EPISODES" \
  rlbench.episode_length=25 \
  framework.demo_num_per_icl="${XICM_PHASE_MEMORY_RETRIEVAL_K}" \
  framework.ranking_method=phase_memory \
  framework.logdir=logs/phase_memory_"$RUN_NAME" \
  framework.record_every_n=999999 \
  cinematic_recorder.enabled=False \
  framework.eval_save_metrics=True
