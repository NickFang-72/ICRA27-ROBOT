#!/usr/bin/env bash
set -Eeuo pipefail

# Local launcher for the current phase-memory benchmark smoke.
# It syncs only the phase-memory source files to CAIR, generates fresh
# all-view phase packets for 23 unseen tasks x 5 episodes, then runs the
# phase_memory controller in the background.

CAIR_HOST="${CAIR_HOST:-cair}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/data/yf23/projects/ICRA27-ROBOT}"
REMOTE_RUN_XICM="${REMOTE_RUN_XICM:-$REMOTE_PROJECT_ROOT/X-ICM-phase-memory}"
CONDA_ENV="${CONDA_ENV:-/data/yf23/conda/envs/zero-shot}"
RUN_NAME="${RUN_NAME:-phase_memory_benchmark_5ep_all23_$(date -u +%Y%m%d_%H%M%S)}"
EPISODE_IDS="${EPISODE_IDS:-0,1,2,3,4}"
EPISODES="${EPISODES:-5}"
GPU_ID="${GPU_ID:-0}"
RECORD_VIDEO="${RECORD_VIDEO:-0}"
RECORD_EVERY_N="${RECORD_EVERY_N:-999999}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-10}"

TASKS="${TASKS:-put_toilet_roll_on_stand,put_knife_on_chopping_board,close_fridge,close_microwave,close_laptop_lid,phone_on_base,toilet_seat_down,lamp_off,lamp_on,put_books_on_bookshelf,put_umbrella_in_umbrella_stand,open_grill,put_rubbish_in_bin,take_usb_out_of_computer,take_lid_off_saucepan,take_plate_off_colored_dish_rack,basketball_in_hoop,scoop_with_spatula,straighten_rope,turn_oven_on,beat_the_buzz,water_plants,unplug_charger}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_RUN_XICM="$(cd "$SCRIPT_DIR/.." && pwd)"
RSYNC_SSH="ssh -o BatchMode=yes -o ConnectTimeout=$SSH_CONNECT_TIMEOUT -o ServerAliveInterval=10 -o ServerAliveCountMax=2"

echo "Checking CAIR connection: $CAIR_HOST"
ssh -o BatchMode=yes -o ConnectTimeout="$SSH_CONNECT_TIMEOUT" "$CAIR_HOST" \
  "hostname; whoami; test -d '$REMOTE_RUN_XICM' && echo remote-project-found"

echo "Syncing phase-memory source to $CAIR_HOST:$REMOTE_RUN_XICM"
rsync -e "$RSYNC_SSH" -a \
  "$LOCAL_RUN_XICM/.gitignore" \
  "$LOCAL_RUN_XICM/README.md" \
  "$LOCAL_RUN_XICM/PHASE_MEMORY_PIPELINE.md" \
  "$LOCAL_RUN_XICM/PHASE_ANCHOR_PIPELINE.md" \
  "$LOCAL_RUN_XICM/config.yaml" \
  "$LOCAL_RUN_XICM/main.py" \
  "$LOCAL_RUN_XICM/crosstask_icl_agent.py" \
  "$LOCAL_RUN_XICM/form_icl_demonstrations_crosstask_ranking.py" \
  "$LOCAL_RUN_XICM/phase_interpreter.py" \
  "$LOCAL_RUN_XICM/utils.py" \
  "$CAIR_HOST:$REMOTE_RUN_XICM/"

rsync -e "$RSYNC_SSH" -a \
  "$LOCAL_RUN_XICM/phase_anchor_pipeline/" \
  "$CAIR_HOST:$REMOTE_RUN_XICM/phase_anchor_pipeline/"

rsync -e "$RSYNC_SSH" -a \
  "$LOCAL_RUN_XICM/phase_memory_pipeline/" \
  "$CAIR_HOST:$REMOTE_RUN_XICM/phase_memory_pipeline/"

rsync -e "$RSYNC_SSH" -a \
  "$LOCAL_RUN_XICM/scripts/collect_phase_memory_runtime_review.py" \
  "$LOCAL_RUN_XICM/scripts/generate_phase_anchor_pipeline_review.py" \
  "$LOCAL_RUN_XICM/scripts/generate_phase_benchmark_review.py" \
  "$LOCAL_RUN_XICM/scripts/generate_phase_interpreter_review.py" \
  "$LOCAL_RUN_XICM/scripts/generate_phase_memory_packet_review.py" \
  "$LOCAL_RUN_XICM/scripts/generate_rerank_review_trace.py" \
  "$LOCAL_RUN_XICM/scripts/run_phase_memory_review_on_cair.sh" \
  "$LOCAL_RUN_XICM/scripts/run_phase_memory_smoke_on_cair.sh" \
  "$CAIR_HOST:$REMOTE_RUN_XICM/scripts/"

REMOTE_REVIEW_ROOT="$REMOTE_PROJECT_ROOT/review"
REMOTE_PHASE_ROOT="$REMOTE_REVIEW_ROOT/${RUN_NAME}_phase"
REMOTE_RUNTIME_ROOT="$REMOTE_REVIEW_ROOT/${RUN_NAME}_runtime"
REMOTE_LOG="$REMOTE_REVIEW_ROOT/${RUN_NAME}_launcher.log"
REMOTE_PID="$REMOTE_REVIEW_ROOT/${RUN_NAME}_launcher.pid"
REMOTE_PHASE_PROGRESS="$REMOTE_REVIEW_ROOT/${RUN_NAME}_phase_progress.json"
REMOTE_RUNTIME_PROGRESS="$REMOTE_REVIEW_ROOT/${RUN_NAME}_runtime_progress.json"

echo "Launching background benchmark: $RUN_NAME"
ssh -o BatchMode=yes -o ConnectTimeout="$SSH_CONNECT_TIMEOUT" "$CAIR_HOST" \
  RUN_NAME="$RUN_NAME" \
  TASKS="$TASKS" \
  EPISODE_IDS="$EPISODE_IDS" \
  EPISODES="$EPISODES" \
  GPU_ID="$GPU_ID" \
  RECORD_VIDEO="$RECORD_VIDEO" \
  RECORD_EVERY_N="$RECORD_EVERY_N" \
  REMOTE_PROJECT_ROOT="$REMOTE_PROJECT_ROOT" \
  REMOTE_RUN_XICM="$REMOTE_RUN_XICM" \
  CONDA_ENV="$CONDA_ENV" \
  REMOTE_LOG="$REMOTE_LOG" \
  REMOTE_PID="$REMOTE_PID" \
  REMOTE_PHASE_PROGRESS="$REMOTE_PHASE_PROGRESS" \
  REMOTE_RUNTIME_PROGRESS="$REMOTE_RUNTIME_PROGRESS" \
  'bash -s' <<'REMOTE'
set -Eeuo pipefail
mkdir -p "$REMOTE_PROJECT_ROOT/review"
cd "$REMOTE_RUN_XICM"

nohup bash -lc '
set -Eeuo pipefail
cd "$REMOTE_RUN_XICM"
export COPPELIASIM_ROOT="$REMOTE_RUN_XICM/CoppeliaSim"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export XKB_CONFIG_ROOT="${XKB_CONFIG_ROOT:-/usr/share/X11/xkb}"
export XICM_VLLM_ENFORCE_EAGER="${XICM_VLLM_ENFORCE_EAGER:-1}"
export XICM_VLLM_GPU_MEMORY_UTILIZATION="${XICM_VLLM_GPU_MEMORY_UTILIZATION:-0.72}"
export XICM_VLLM_MAX_MODEL_LEN="${XICM_VLLM_MAX_MODEL_LEN:-12000}"
export XICM_VL_MAX_IMAGES="${XICM_VL_MAX_IMAGES:-2}"
export XICM_VL_MIN_PIXELS="${XICM_VL_MIN_PIXELS:-3136}"
export XICM_VL_MAX_PIXELS="${XICM_VL_MAX_PIXELS:-200704}"
echo "[phase-memory benchmark] run=$RUN_NAME"
echo "[phase-memory benchmark] generating phase packets"
"$CONDA_ENV/bin/python3" scripts/generate_phase_benchmark_review.py \
  --output-root "$REMOTE_PROJECT_ROOT/review" \
  --name "${RUN_NAME}_phase" \
  --tasks "$TASKS" \
  --episode-ids "$EPISODE_IDS" \
  --progress-json "$REMOTE_PHASE_PROGRESS"

echo "[phase-memory benchmark] running runtime smoke"
PHASE_REVIEW_ROOT="$REMOTE_PROJECT_ROOT/review/${RUN_NAME}_phase" \
RUN_NAME="${RUN_NAME}_runtime" \
TASKS="$TASKS" \
EPISODES="$EPISODES" \
GPU_ID="$GPU_ID" \
RECORD_VIDEO="$RECORD_VIDEO" \
RECORD_EVERY_N="$RECORD_EVERY_N" \
XICM_PHASE_MEMORY_RETRIEVAL_K=4 \
XICM_PHASE_MEMORY_USE_IMAGES=0 \
XICM_PHASE_MEMORY_MAX_TOKENS=240 \
bash scripts/run_phase_memory_review_on_cair.sh
' > "$REMOTE_LOG" 2>&1 &
echo "$!" > "$REMOTE_PID"
echo "pid=$(cat "$REMOTE_PID")"
echo "log=$REMOTE_LOG"
echo "phase_progress=$REMOTE_PHASE_PROGRESS"
echo "runtime_progress=$REMOTE_RUNTIME_PROGRESS"
REMOTE

cat <<EOF

Launched $RUN_NAME
Remote phase root:   $REMOTE_PHASE_ROOT
Remote runtime root: $REMOTE_RUNTIME_ROOT
Remote log:          $REMOTE_LOG
Remote PID:          $REMOTE_PID

Watch:
  ssh $CAIR_HOST 'tail -f "$REMOTE_LOG"'
  ssh $CAIR_HOST 'cat "$REMOTE_PHASE_PROGRESS" 2>/dev/null; cat "$REMOTE_RUNTIME_PROGRESS" 2>/dev/null'
EOF
