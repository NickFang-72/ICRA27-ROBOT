#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/Users/nicholas/Documents/ICRA27 ROBOT}"
REMOTE="${REMOTE:-cair}"
REMOTE_CACHE_ROOT="${REMOTE_CACHE_ROOT:-/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_full_cache}"
REMOTE_CONDA="${REMOTE_CONDA:-/data/yf23/conda/envs/icra27-robot}"
REMOTE_TRAIN_JSON="${REMOTE_TRAIN_JSON:-/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/train.json}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/data/yf23/datasets/ICRA27-ROBOT}"
REMOTE_QWEN_MODEL="${REMOTE_QWEN_MODEL:-/data/yf23/checkpoints/ICRA27-ROBOT/Qwen2.5-VL-7B-Instruct}"
REMOTE_ROBOPOINT_MODEL="${REMOTE_ROBOPOINT_MODEL:-/data/yf23/checkpoints/ICRA27-ROBOT/robopoint-v1-vicuna-v1.5-13b}"
REMOTE_HF_HOME="${REMOTE_HF_HOME:-/data/yf23/checkpoints/ICRA27-ROBOT/hf_home}"
STAGE="${STAGE:-all}"
LIMIT="${LIMIT:-}"
FORCE="${FORCE:-0}"

LOCAL_RUNNER="$ROOT_DIR/test_files/geometry_affordance_probe/scripts/cache_all_seen_geometry_affordance.py"
REMOTE_RUNNER="$REMOTE_CACHE_ROOT/scripts/cache_all_seen_geometry_affordance.py"
REMOTE_LOG="$REMOTE_CACHE_ROOT/full_cache_${STAGE}.log"
REMOTE_PID="$REMOTE_CACHE_ROOT/full_cache_${STAGE}.pid"

ssh -o BatchMode=yes "$REMOTE" "mkdir -p '$REMOTE_CACHE_ROOT/scripts'"
scp "$LOCAL_RUNNER" "$REMOTE:$REMOTE_RUNNER"

if [ "$FORCE" != "1" ]; then
  if ssh -o BatchMode=yes "$REMOTE" "test -f '$REMOTE_PID' && kill -0 \$(cat '$REMOTE_PID') 2>/dev/null"; then
    echo "A full-cache job already appears to be running. Set FORCE=1 to launch another one."
    ssh -o BatchMode=yes "$REMOTE" "cat '$REMOTE_PID'"
    exit 0
  fi
fi

limit_arg=""
if [ -n "$LIMIT" ]; then
  limit_arg="--limit $LIMIT"
fi

ssh -o BatchMode=yes "$REMOTE" "cd '$REMOTE_CACHE_ROOT' && \
  (nohup bash -lc 'source /data/yf23/miniconda3/etc/profile.d/conda.sh && \
  conda activate \"$REMOTE_CONDA\" && \
  python \"$REMOTE_RUNNER\" \
    --stage \"$STAGE\" \
    --root \"$REMOTE_CACHE_ROOT\" \
    --train-json \"$REMOTE_TRAIN_JSON\" \
    --data-root \"$REMOTE_DATA_ROOT\" \
    --qwen-model \"$REMOTE_QWEN_MODEL\" \
    --robopoint-model \"$REMOTE_ROBOPOINT_MODEL\" \
    --hf-home \"$REMOTE_HF_HOME\" \
    $limit_arg' > '$REMOTE_LOG' 2>&1 < /dev/null & echo \$! > '$REMOTE_PID')"

echo "Launched full seen geometry/affordance cache on $REMOTE."
echo "Remote root: $REMOTE_CACHE_ROOT"
echo "Remote log:  $REMOTE_LOG"
echo "Remote pid:  $REMOTE_PID"
echo
echo "Watch progress with:"
echo "  $ROOT_DIR/test_files/geometry_affordance_probe/cair_setup_scripts/watch_full_seen_cache_progress.sh"
