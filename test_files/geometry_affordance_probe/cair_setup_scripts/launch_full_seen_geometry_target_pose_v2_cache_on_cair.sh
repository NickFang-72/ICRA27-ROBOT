#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/Users/nicholas/Documents/ICRA27 ROBOT}"
REMOTE="${REMOTE:-cair}"
REMOTE_CACHE_ROOT="${REMOTE_CACHE_ROOT:-/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_target_pose_v2_full_cache_20260626}"
REMOTE_CONDA="${REMOTE_CONDA:-/data/yf23/conda/envs/icra27-robot}"
REMOTE_TRAIN_JSON="${REMOTE_TRAIN_JSON:-/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/train.json}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/data/yf23/datasets/ICRA27-ROBOT}"
REMOTE_QWEN_MODEL="${REMOTE_QWEN_MODEL:-/data/yf23/checkpoints/ICRA27-ROBOT/Qwen2.5-VL-7B-Instruct}"
REMOTE_HF_HOME="${REMOTE_HF_HOME:-/data/yf23/checkpoints/ICRA27-ROBOT/hf_home}"
REMOTE_CUDA_VISIBLE_DEVICES="${REMOTE_CUDA_VISIBLE_DEVICES:-1}"
LIMIT="${LIMIT:-}"
FORCE="${FORCE:-0}"

LOCAL_RUNNER="$ROOT_DIR/test_files/geometry_affordance_probe/scripts/cache_all_seen_geometry_affordance.py"
LOCAL_SCHEMA_HELPER="$ROOT_DIR/test_files/geometry_affordance_probe/scripts/run_qwen_dual_view_geometry_target_pose.py"
REMOTE_RUNNER="$REMOTE_CACHE_ROOT/scripts/cache_all_seen_geometry_affordance.py"
REMOTE_SCHEMA_HELPER="$REMOTE_CACHE_ROOT/scripts/run_qwen_dual_view_geometry_target_pose.py"
REMOTE_LOG="$REMOTE_CACHE_ROOT/full_geometry_target_pose_v2_cache.log"
REMOTE_PID="$REMOTE_CACHE_ROOT/full_geometry_target_pose_v2_cache.pid"

ssh -o BatchMode=yes "$REMOTE" "mkdir -p '$REMOTE_CACHE_ROOT/scripts'"
rsync -a "$LOCAL_RUNNER" "$REMOTE:$REMOTE_RUNNER"
rsync -a "$LOCAL_SCHEMA_HELPER" "$REMOTE:$REMOTE_SCHEMA_HELPER"

if [ "$FORCE" != "1" ]; then
  if ssh -o BatchMode=yes "$REMOTE" "test -f '$REMOTE_PID' && kill -0 \$(cat '$REMOTE_PID') 2>/dev/null"; then
    echo "A clean-v2 geometry/target-pose cache job already appears to be running. Set FORCE=1 to launch another one."
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
  export CUDA_VISIBLE_DEVICES=\"$REMOTE_CUDA_VISIBLE_DEVICES\" && \
  export HF_HOME=\"$REMOTE_HF_HOME\" && \
  python \"$REMOTE_RUNNER\" \
    --stage manifest \
    --root \"$REMOTE_CACHE_ROOT\" \
    --train-json \"$REMOTE_TRAIN_JSON\" \
    --data-root \"$REMOTE_DATA_ROOT\" \
    --qwen-model \"$REMOTE_QWEN_MODEL\" \
    --hf-home \"$REMOTE_HF_HOME\" \
    --geometry-only \
    $limit_arg && \
  python \"$REMOTE_RUNNER\" \
    --stage geometry \
    --root \"$REMOTE_CACHE_ROOT\" \
    --train-json \"$REMOTE_TRAIN_JSON\" \
    --data-root \"$REMOTE_DATA_ROOT\" \
    --qwen-model \"$REMOTE_QWEN_MODEL\" \
    --hf-home \"$REMOTE_HF_HOME\" \
    --geometry-only \
    --max-new-tokens 900 && \
  python \"$REMOTE_RUNNER\" \
    --stage normalize \
    --root \"$REMOTE_CACHE_ROOT\" \
    --train-json \"$REMOTE_TRAIN_JSON\" \
    --data-root \"$REMOTE_DATA_ROOT\" \
    --qwen-model \"$REMOTE_QWEN_MODEL\" \
    --hf-home \"$REMOTE_HF_HOME\" \
    --geometry-only' > '$REMOTE_LOG' 2>&1 < /dev/null & echo \$! > '$REMOTE_PID')"

echo "Launched clean-v2 full seen geometry/target-pose cache on $REMOTE."
echo "Remote root: $REMOTE_CACHE_ROOT"
echo "Remote log:  $REMOTE_LOG"
echo "Remote pid:  $REMOTE_PID"
echo
echo "Watch progress with:"
echo "  $ROOT_DIR/test_files/geometry_affordance_probe/cair_setup_scripts/watch_full_seen_geometry_target_pose_v2_cache_progress.sh"
