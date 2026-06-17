#!/bin/bash
###############################################################################
# Qwen2-VL Evaluation Script (based on qwen2vl_finetune)
#
# Purpose: Evaluate Qwen2-VL model on RLBench (OOD and WithinTask)
# Dependencies: qwen2vl_finetune/main_qwen_sft.py
#
# Usage: bash scripts/eval_qwen_sft.sh [MODE] [SEEDS] [EPISODES] [GPU_ID] [H_LEN] [T_LEN] [STEPS] [START] [NUM]
#
# Examples:
#   bash scripts/eval_qwen_sft.sh ood 0 25 0 1 1 25 0 23              # OOD evaluation
#   bash scripts/eval_qwen_sft.sh withintask 0 25 0 1 1 25 0 18       # WithinTask evaluation
#
###############################################################################

set -e

# ============================================================================
# Parse Arguments
# ============================================================================

if [ $# -lt 10 ]; then
    echo "Usage: $0 [MODE] [CHECKPOINT] [SEEDS] [EPISODES] [GPU_ID] [H_LEN] [T_LEN] [STEPS] [START] [NUM]"
    echo ""
    echo "Examples:"
    echo "  $0 ood outputs/checkpoint-1860 0 25 0 1 1 25 0 23"
    echo "  $0 withintask outputs/checkpoint-1860 0 25 0 1 1 25 0 18"
    exit 1
fi

MODE="${1,,}"
CHECKPOINT="${2}"
seeds=(${3//,/ })
episodes=$4
gpu_id=$5
history_length=$6
target_length=$7
eval_steps=$8
start_id=$9
task_nums=${10}

# ============================================================================
# Paths and Environment
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QWEN_DIR="${SCRIPT_DIR}/qwen2vl_finetune"
DATA_HOME="${SCRIPT_DIR}/data"

# Verify dependencies
if [ ! -f "$QWEN_DIR/main_qwen_sft.py" ]; then
    echo "❌ Error: main_qwen_sft.py not found"
    echo "Please ensure qwen2vl_finetune is properly cloned"
    exit 1
fi

export PYTHONPATH="${QWEN_DIR}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=$gpu_id

# ============================================================================
# Task Definitions
# ============================================================================

SEEN_TASKS=(
    "open_drawer" "meat_off_grill" "turn_tap" "put_money_in_safe"
    "push_buttons" "sweep_to_dustpan_of_size" "slide_block_to_color_target"
    "close_jar" "light_bulb_in" "place_wine_at_rack_location"
    "reach_and_drag" "stack_blocks" "put_item_in_drawer"
    "place_shape_in_shape_sorter" "insert_onto_square_peg" "stack_cups"
    "put_groceries_in_cupboard" "place_cups"
)

UNSEEN_TASKS=(
    "put_toilet_roll_on_stand" "put_knife_on_chopping_board" "close_fridge"
    "close_microwave" "close_laptop_lid" "phone_on_base" "toilet_seat_down"
    "lamp_off" "lamp_on" "put_books_on_bookshelf"
    "put_umbrella_in_umbrella_stand" "open_grill" "put_rubbish_in_bin"
    "take_usb_out_of_computer" "take_lid_off_saucepan"
    "take_plate_off_colored_dish_rack" "basketball_in_hoop"
    "scoop_with_spatula" "straighten_rope" "turn_oven_on"
    "beat_the_buzz" "water_plants" "unplug_charger"
)

# ============================================================================
# Mode Configuration
# ============================================================================

if [[ "$MODE" == "ood" ]]; then
    eval_tasks=("${UNSEEN_TASKS[@]}")
    demo_path="${DATA_HOME}/unseen_tasks"
    mode_name="Unseen Tasks (OOD)"
elif [[ "$MODE" == "withintask" ]]; then
    eval_tasks=("${SEEN_TASKS[@]}")
    demo_path="${DATA_HOME}/seen_tasks"
    mode_name="Seen Tasks (WithinTask)"
else
    echo "❌ Mode must be 'ood' or 'withintask'"
    exit 1
fi

# ============================================================================
# Prepare Tasks and Paths
# ============================================================================

selected_tasks=("${eval_tasks[@]:$start_id:$task_nums}")
tasks_string=$(printf "%s," "${selected_tasks[@]}")
tasks_string=${tasks_string%,}

logdir="${SCRIPT_DIR}/outputs"
mkdir -p "$logdir"

# Model checkpoint from parameter (relative path)
model_checkpoint="${SCRIPT_DIR}/${CHECKPOINT}"

if [ ! -d "$model_checkpoint" ]; then
    echo "❌ Checkpoint directory not found: $model_checkpoint"
    exit 1
fi

# ============================================================================
# Logging
# ============================================================================

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║        Qwen2-VL Evaluation (based on qwen2vl_finetune)        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Configuration:"
echo "  Mode:       $mode_name"
echo "  Num Tasks:  $task_nums"
echo "  Episodes:   $episodes"
echo "  GPU:        $gpu_id"
echo "  Seeds:      ${seeds[@]}"
echo ""
echo "📂 Paths:"
echo "  Model:      $model_checkpoint"
echo "  Demo Data:  $demo_path"
echo "  Output:     $logdir"
echo ""

# ============================================================================
# Run Evaluation
# ============================================================================

cd "$QWEN_DIR"

for seed in "${seeds[@]}"; do
    echo "🔄 Evaluating with seed $seed..."
    
    xvfb-run -a python main_qwen_sft.py \
        "method.name=${model_checkpoint}" \
        "rlbench.tasks=[$tasks_string]" \
        "framework.start_seed=${seed}" \
        "framework.eval_episodes=${episodes}" \
        "framework.logdir=${logdir}" \
        "rlbench.demo_path=${demo_path}" \
        "rlbench.history_length=${history_length}" \
        "rlbench.target_length=${target_length}" \
        "rlbench.episode_length=${eval_steps}"
    
    if [ $? -eq 0 ]; then
        echo "  ✅ Done"
    else
        echo "  ❌ Failed"
        exit 1
    fi
done

echo ""
echo "✅ Evaluation completed! Results saved in: $logdir"
