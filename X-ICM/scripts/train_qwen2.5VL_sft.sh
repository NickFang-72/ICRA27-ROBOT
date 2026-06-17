#!/bin/bash
###############################################################################
# Qwen2-VL Unified SFT Training Script
#
# Purpose: Support flexible training configuration with DeepSpeed distributed training
# Dependencies: src/train/train_sft.py (from Qwen2-VL-Finetune)
#
# Usage: bash train_qwen_sft_unified.sh [DATASET] [LR_LLM] [LR_VISION] [EPOCHS] [BS] [GPUS]
#
# Examples:
#   - Basic training (default parameters):
#     bash train_qwen_sft_unified.sh
#
#   - Custom parameters:
#     bash train_qwen_sft_unified.sh agnostos 1e-4 2e-5 20 128 0,1,2,3,4,5,6,7
#
#   - LoRA fine-tuning (freeze LLM, tune vision tower):
#     bash train_qwen_sft_unified.sh agnostos 1e-4 2e-5 10 128 0,1,2,3
#
###############################################################################

set -e  # Exit on any error

###############################################################################
# Default Configuration Parameters
###############################################################################

# Learning rate configuration
LR_LLM="${1:-1e-4}"           # LLM learning rate
LR_VISION="${2:-2e-5}"         # Vision tower learning rate
LR_MERGER="${3:-1e-5}"         # MLP merger learning rate

# Training configuration
EPOCHS="${4:-20}"              # Number of epochs
BATCH_SIZE="${5:-128}"         # Batch size per GPU
GPU_IDS="${6:-0,1,2,3,4,5,6,7}"  # Available GPU IDs

# Environment variables
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export PYTHONPATH="src:${PYTHONPATH}"
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-29520}
export CUDA_VISIBLE_DEVICES=$GPU_IDS

###############################################################################
# Path Configuration (Relative Paths)
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_HOME="${SCRIPT_DIR}/data"
OUTPUT_BASE="${SCRIPT_DIR}/outputs"

# Training data from seen_tasks
DATA_FILE="${DATA_HOME}/train.json"
TOTAL_SAMPLES=95055

###############################################################################
# Verify Prerequisites
###############################################################################

if [ ! -f "$DATA_FILE" ]; then
    echo "❌ Data file not found: $DATA_FILE"
    exit 1
fi

###############################################################################
# Calculate Training Parameters
###############################################################################

NUM_DEVICES=$(echo $GPU_IDS | tr ',' '\n' | wc -l)
GRAD_ACCUM_STEPS=1
LOGGING_STEPS=10
SAVE_TOTAL_LIMIT=10

# Calculate total training steps and save steps
TOTAL_STEPS=$((EPOCHS * TOTAL_SAMPLES / BATCH_SIZE / NUM_DEVICES / GRAD_ACCUM_STEPS))
SAVE_STEPS=$((TOTAL_STEPS / SAVE_TOTAL_LIMIT))

# Model and feature freeze configuration
FREEZE_VISION_TOWER=False  # Tune vision tower
FREEZE_LLM=False           # Tune LLM
FREEZE_MERGER=False        # Tune merger

# Model identifier
MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"

###############################################################################
# Output Directory
###############################################################################

OUTPUT_DIR="${OUTPUT_BASE}/qwen2vl_agnostos_ep${EPOCHS}_bs${BATCH_SIZE}_lr${LR_LLM}_vlr${LR_VISION}"
mkdir -p "$OUTPUT_DIR"

###############################################################################
# Logging Output
###############################################################################

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║          Qwen2-VL Unified SFT Training Script                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Training Configuration:"
echo "  • Dataset:       $DATASET"
echo "  • Data File:     $(basename "$DATA_FILE")"
echo "  • Total Samples: $TOTAL_SAMPLES"
echo "  • Epochs:        $EPOCHS"
echo "  • Batch Size/GPU: $BATCH_SIZE"
echo "  • GPU Count:     $NUM_DEVICES"
echo "  • GPU IDs:       $GPU_IDS"
echo ""
echo "🔧 Learning Rate Configuration:"
echo "  • LLM LR:        $LR_LLM"
echo "  • Vision LR:     $LR_VISION"
echo "  • Merger LR:     $LR_MERGER"
echo ""
echo "📈 Computation Parameters:"
echo "  • Total Steps:   $TOTAL_STEPS"
echo "  • Save Steps:    $SAVE_STEPS"
echo "  • Grad Accumulation: $GRAD_ACCUM_STEPS"
echo "  • Logging Steps: $LOGGING_STEPS"
echo ""
echo "💾 Output Path:"
echo "  • Checkpoint Dir: $OUTPUT_DIR"
echo ""

###############################################################################
# Run Training
###############################################################################

cd "${SCRIPT_DIR}/qwen2vl_finetune"

deepspeed src/train/train_sft.py \
    --use_liger True \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_ID \
    --data_path $DATA_FILE \
    --image_folder $DATA_HOME \
    --remove_unused_columns False \
    --freeze_vision_tower $FREEZE_VISION_TOWER \
    --freeze_llm $FREEZE_LLM \
    --freeze_merger $FREEZE_MERGER \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((16*28*28)) \
    --image_max_pixels $((1280*28*28)) \
    --learning_rate $LR_LLM \
    --merger_lr $LR_MERGER \
    --vision_lr $LR_VISION \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps $LOGGING_STEPS \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --dataloader_num_workers 4

if [ $? -eq 0 ]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                Training Completed Successfully! 🎉              ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "✓ Final model saved at: $OUTPUT_DIR"
else
    echo ""
    echo "❌ Training failed, please check the logs"
    exit 1
fi
