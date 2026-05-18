#!/bin/bash
#SBATCH --job-name=dreamzero_yam_lora_4gpu
#SBATCH --output=/iris/u/kewalk/dreamzero/logs/yam/dreamzero_yam-%j.out
#SBATCH --partition=iris-hi
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
#SBATCH --mem=400G
#SBATCH --cpus-per-task=24
#SBATCH --mail-user=kewalk@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --account=iris

# 4-GPU single-node H100 box. Pin to the node you've been allocated via
# `#SBATCH --nodelist=<node>` if scheduler picks the wrong one.

set -e

echo "==== Job $SLURM_JOB_ID on $SLURMD_NODENAME ===="
nvidia-smi
echo "================================================"

cd /iris/u/kewalk/dreamzero
source /iris/u/kewalk/.bashrc.user
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate dreamzero

export HF_HOME=${HF_HOME:-/iris/u/kewalk/.cache/huggingface}
export HF_HUB_ENABLE_HF_TRANSFER=1
export WANDB_DIR=/iris/u/kewalk/dreamzero/wandb
mkdir -p /iris/u/kewalk/dreamzero/logs/yam "$WANDB_DIR"

export YAM_DATA_ROOT=${YAM_DATA_ROOT:-/iris/u/kewalk/dataset/plate_task_0505_lerobot}
export RUN_TAG=${RUN_TAG:-$SLURM_JOB_ID}

# 4 GPUs with bs=1/GPU and grad_accum=1 → effective batch 4 (same as 1-H200 run).
# To scale up: PER_DEVICE_BS=2 GRAD_ACCUM=1 → effective batch 8.
export NUM_GPUS=4
export PER_DEVICE_BS=${PER_DEVICE_BS:-1}
export GRAD_ACCUM=${GRAD_ACCUM:-1}

bash scripts/train/yam_lora_1gpu.sh
