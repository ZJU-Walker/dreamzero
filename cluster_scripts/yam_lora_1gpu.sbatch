#!/bin/bash
#SBATCH --job-name=pi05_human_robot
#SBATCH --output=/iris/u/kewalk/dreamzero/logs/yam/dreamzero_yam-%j.out
#SBATCH --partition=iris-hi
#SBATCH --nodelist=iris-hgx-2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=350G
#SBATCH --cpus-per-task=12
#SBATCH --mail-user=kewalk@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --account=iris

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

bash scripts/train/yam_lora_1gpu.sh
