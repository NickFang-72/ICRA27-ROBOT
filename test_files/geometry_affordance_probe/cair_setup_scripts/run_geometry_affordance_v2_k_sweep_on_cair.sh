#!/usr/bin/env bash
set -Eeuo pipefail

# Run the compact v2 geometry+affordance X-ICM K sweep on CAIR.
# Each K value is launched sequentially through run_geometry_affordance_v2_on_cair.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K_VALUES="${K_VALUES:-6 8 10}"
RUN_ROOT="${RUN_ROOT:-/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_ablations}"
GPU_ID="${GPU_ID:-1}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5.7B.instruct}"
EPISODES="${EPISODES:-25}"
SEEDS="${SEEDS:-0}"

mkdir -p "$RUN_ROOT/logs"
SWEEP_LOG="$RUN_ROOT/logs/geometry_affordance_v2_k_sweep_$(date -u +%Y%m%d_%H%M%S).log"

{
    echo "sweep=geometry_affordance_v2_k"
    echo "k_values=$K_VALUES"
    echo "gpu_id=$GPU_ID"
    echo "model=$MODEL_NAME"
    echo "episodes=$EPISODES"
    echo "seeds=$SEEDS"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    for k in $K_VALUES; do
        echo
        echo "===== starting geometry_affordance_v2_k${k} at $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
        CONDITION="geometry_affordance_v2_k${k}" \
        DEMO_NUM_PER_ICL="$k" \
        GPU_ID="$GPU_ID" \
        MODEL_NAME="$MODEL_NAME" \
        EPISODES="$EPISODES" \
        SEEDS="$SEEDS" \
        bash "$SCRIPT_DIR/run_geometry_affordance_v2_on_cair.sh"
        echo "===== finished geometry_affordance_v2_k${k} at $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
    done

    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee "$SWEEP_LOG"

echo "Sweep log: $SWEEP_LOG"
