#!/bin/bash
# DreamZero YAM LoRA — 1 H200 GPU training script.
#
# Designed for full LoRA finetuning on a single 141 GB H200. Mirrors
# yam_training.sh defaults except:
#   - per_device_train_batch_size=1 (was 4 — won't fit 14B + LoRA + 33 frames * 3 views in 141 GB at bs=4)
#   - gradient_accumulation_steps=4 (recovers effective batch of 4 per step)
#   - max_steps=30000 (scaled for ~52 demos; adjust to taste)
#   - save_steps=5000
#   - report_to=wandb (default; override with REPORT_TO=none)
#
# Usage (workstation):
#   YAM_DATA_ROOT=/iris/u/kewalk/dataset/plate_task_0505_lerobot \
#       bash scripts/train/yam_lora_1gpu.sh
#
# Usage (slurm, single H200): launched by cluster_script/dreamzero/yam_lora_1gpu.sbatch

export HYDRA_FULL_ERROR=1

# ============ CONFIG (override via env) ============
YAM_DATA_ROOT=${YAM_DATA_ROOT:-"/iris/u/kewalk/dataset/plate_task_0505_lerobot"}

# Tag checkpoint dir with $SLURM_JOB_ID when on cluster, else with timestamp
RUN_TAG=${RUN_TAG:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/yam_lora_1gpu_${RUN_TAG}"}

NUM_GPUS=${NUM_GPUS:-1}
PER_DEVICE_BS=${PER_DEVICE_BS:-1}
GRAD_ACCUM=${GRAD_ACCUM:-4}
MAX_STEPS=${MAX_STEPS:-30000}
SAVE_STEPS=${SAVE_STEPS:-1000}
LR=${LR:-1e-5}
REPORT_TO=${REPORT_TO:-wandb}
WANDB_PROJECT=${WANDB_PROJECT:-dreamzero_yam}

WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"./checkpoints/DreamZero-AgiBot"}
# ===================================================

echo "=== YAM LoRA 1-GPU run ==="
echo "  YAM_DATA_ROOT : $YAM_DATA_ROOT"
echo "  OUTPUT_DIR   : $OUTPUT_DIR"
echo "  NUM_GPUS     : $NUM_GPUS"
echo "  per-dev bs   : $PER_DEVICE_BS"
echo "  grad accum   : $GRAD_ACCUM  (effective batch = $((PER_DEVICE_BS * GRAD_ACCUM * NUM_GPUS)))"
echo "  max_steps    : $MAX_STEPS"
echo "  save_steps   : $SAVE_STEPS"
echo "  lr           : $LR"
echo "  report_to    : $REPORT_TO"
echo "=========================="

if [ ! -d "$WAN_CKPT_DIR" ] || [ -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ]; then
    echo "Downloading Wan2.1-I2V-14B-480P → $WAN_CKPT_DIR"
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN_CKPT_DIR"
fi
if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "Downloading umt5-xxl → $TOKENIZER_DIR"
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi
if [ ! -d "$PRETRAINED_MODEL" ] || [ -z "$(ls -A "$PRETRAINED_MODEL" 2>/dev/null)" ]; then
    echo "Downloading DreamZero-AgiBot → $PRETRAINED_MODEL"
    huggingface-cli download GEAR-Dreams/DreamZero-AgiBot --repo-type model --local-dir "$PRETRAINED_MODEL"
fi

if [ ! -f "$YAM_DATA_ROOT/meta/embodiment.json" ]; then
    echo "ERROR: $YAM_DATA_ROOT/meta/embodiment.json missing — run convert_lerobot_to_gear.py first"
    exit 1
fi

torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=$REPORT_TO \
    data=dreamzero/yam_relative \
    wandb_project=$WANDB_PROJECT \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=$LR \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=$SAVE_STEPS \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=$PER_DEVICE_BS \
    gradient_accumulation_steps=$GRAD_ACCUM \
    max_steps=$MAX_STEPS \
    weight_decay=1e-5 \
    save_total_limit=5 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=2 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    yam_data_root=$YAM_DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    pretrained_model_path=$PRETRAINED_MODEL \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true
