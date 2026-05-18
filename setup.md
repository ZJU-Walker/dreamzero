# DreamZero — Setup & LoRA Training Guide

This guide walks you through setting up DreamZero from scratch on your cluster
and running a LoRA finetune on the YAM plate-memory dataset (or your own LeRobot-format dataset).

> **Replace any path that starts with `/iris/u/kewalk/...`** with the equivalent
> path on your cluster (your home, your scratch, your dataset dir, etc.).
> SLURM headers (`--partition`, `--account`, `--nodelist`) will also need to
> match your cluster.

---

## 1. Hardware requirements

LoRA finetune of the full 14B Wan2.1 backbone needs:

- **At least 1× H200 (141 GB)** _or_ **4× H100 (80 GB) on a single node** for comfortable training
- ≥ 300 GB host RAM (the optimizer offload and dataloader buffers need it)
- ≥ 200 GB free disk for pretrained weights + checkpoints + dataset
- CUDA 12.9+ on the nodes

Smaller GPUs (A100 80GB, H100 80GB × 1) will OOM. If you only have 80 GB GPUs,
you must use ≥ 2 of them with DeepSpeed ZeRO-3.

---

## 2. Clone the repo

```bash
git clone <YOUR_FORK_URL> dreamzero
cd dreamzero
```

If your friend is collaborating with me, this will be:
```bash
git clone git@github.com:ZJU-Walker/dreamzero.git
cd dreamzero
```

---

## 3. Create the conda environment

```bash
# Create + activate
conda create -n dreamzero python=3.11 -y
conda activate dreamzero

# Install all Python deps + the dreamzero package itself (PyTorch 2.8 + CUDA 12.9)
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129

# Flash attention (build can take 15-30 min; MAX_JOBS controls parallelism)
MAX_JOBS=8 pip install --no-build-isolation flash-attn

# HuggingFace CLI (used to fetch model weights)
pip install "huggingface_hub[cli]"
```

> Skip the Transformer Engine / TensorRT steps from the README — those are only
> needed for GB200 inference, not for H100/H200 training.

### Verify the env

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
# Expect: 2.8.x  True  <num_gpus>

python -c "import flash_attn, deepspeed, transformers; print('ok')"
# Expect: ok
```

---

## 4. Download pretrained weights (~73 GB total, one-time)

These three checkpoints are needed by every training run. Pre-downloading avoids
job timeouts at launch.

```bash
cd <YOUR_REPO_ROOT>            # e.g. ~/dreamzero
mkdir -p checkpoints

# Wan2.1-14B base model (~28 GB)
hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P

# umt5-xxl tokenizer (~10 GB)
hf download google/umt5-xxl --local-dir ./checkpoints/umt5-xxl

# DreamZero-AgiBot LoRA-pretrained checkpoint (~45 GB)
hf download GEAR-Dreams/DreamZero-AgiBot --repo-type model --local-dir ./checkpoints/DreamZero-AgiBot
```

> If the download is gated, set `export HF_TOKEN=<your_token>` first.
> For faster downloads: `export HF_HUB_ENABLE_HF_TRANSFER=1`.

---

## 5. Prepare your dataset

The training pipeline expects data in **LeRobot v2 format** with an extra
**GEAR metadata** layer (`modality.json`, `embodiment.json`, `stats.json`).

### Conversion scripts (all in this repo)

| Stage | Script | What it does |
|---|---|---|
| 1 | `dataset/combine_parts.py` | Merge multiple raw-data folders (e.g. `part1`, `part2`, ...) into one folder with sequentially renamed demos (`demo1` ... `demoN`) |
| 2 | `dataset/convert_data_to_lerobot.py` | Raw demos → LeRobot v2 (parquet + mp4 + `meta/info.json`, `meta/episodes.jsonl`, `meta/tasks.jsonl`) |
| 3 | `scripts/data/convert_lerobot_to_gear.py` | Add GEAR metadata (`meta/modality.json`, `meta/embodiment.json`, `meta/stats.json`) on top of the LeRobot dataset |

### End-to-end example (YAM plate-memory)

```bash
cd <YOUR_REPO_ROOT>

# Step 1: merge raw parts → /path/to/plate_task_0505/demo1..demoN
#   Edit paths at the top of the script first (SRC_DIRS, DST_DIR).
python dataset/combine_parts.py

# Step 2: raw → LeRobot v2 (parquet + mp4 + LeRobot meta/)
#   Edit RAW_ROOT and OUT_ROOT at the top of the script.
python dataset/convert_data_to_lerobot.py

# Step 3: add GEAR metadata layer (modality.json, embodiment.json, stats.json)
python scripts/data/convert_lerobot_to_gear.py \
    --dataset_path /path/to/plate_task_0505_lerobot \
    --embodiment_tag yam
```

After step 3, you should have a directory like this — this is what the training
script reads as `YAM_DATA_ROOT`:

```
plate_task_0505_lerobot/
├── data/chunk-000/episode_<N>.parquet                    # state + action arrays
├── videos/chunk-000/observation.images.<cam>/episode_<N>.mp4
└── meta/
    ├── info.json, episodes.jsonl, tasks.jsonl            # LeRobot metadata
    └── modality.json, embodiment.json, stats.json        # GEAR metadata
```

### Heads up

- **Paths inside `dataset/combine_parts.py` and `dataset/convert_data_to_lerobot.py`
  are hardcoded for the original YAM dataset layout.** Open each script and update
  the top-of-file path constants (`SRC_DIRS`, `DST_DIR`, `RAW_ROOT`, `OUT_ROOT`,
  etc.) to match your data before running them.
- **The state/action layout is YAM-specific** (16-dim bimanual with right arm
  zero-padded: left_joint[0:7], left_gripper[7:8], right_joint=0[8:15], right_gripper=0[15:16]).
  If your robot has a different DoF layout, you'll need to adapt
  `convert_data_to_lerobot.py` and re-generate the GEAR `modality.json` slices to match.
- For a step-by-step walkthrough on adapting to a brand-new embodiment, see
  `docs/DATASET_TO_GEAR_AND_TRAIN.md`.

**Note the absolute path of your final `*_lerobot/` directory — you'll pass it to
the training script as `YAM_DATA_ROOT`.**

---

## 6. Sanity check (20 steps, 1 GPU) — do this FIRST

Before launching a multi-hour training run, validate the pipeline end-to-end.

```bash
conda activate dreamzero
cd <YOUR_REPO_ROOT>

YAM_DATA_ROOT=<absolute/path/to/plate_task_0505_lerobot> \
    bash scripts/train/yam_sanity.sh 2>&1 | tee /tmp/yam_sanity.log
```

**Pass criteria:**
- Data loads without `KeyError` or shape-mismatch errors
- 20 steps complete with finite loss
- A checkpoint appears at `./checkpoints/yam_sanity/checkpoint-20/`

Cold-start (loading 14B Wan2.1 + AgiBot + DeepSpeed init) takes 5-15 minutes
before the first step prints — that's normal, don't kill the job.

---

## 7. Run the full LoRA finetune

### 7a. Interactive run (workstation with 1 H200)

```bash
conda activate dreamzero
cd <YOUR_REPO_ROOT>

YAM_DATA_ROOT=<absolute/path/to/plate_task_0505_lerobot> \
MAX_STEPS=10000 SAVE_STEPS=500 \
    bash scripts/train/yam_lora_1gpu.sh 2>&1 | tee /tmp/yam_lora.log
```

Expected wallclock on 1 H200: **~29 hours** for 3,000 steps.

### 7b. Cluster run via SLURM

Two sbatch wrappers are provided in `cluster_scripts/`:
- `yam_lora_1gpu.sbatch` — 1 GPU (H200 recommended; H100 80GB will OOM)
- `yam_lora_4gpu.sbatch` — 4 GPUs single-node (H100 or H200)

**You must edit the SBATCH headers** at the top of these files to match your
cluster. The current values are for the iris-hi cluster at Stanford:

```bash
#SBATCH --partition=iris-hi              # → your partition
#SBATCH --account=iris                   # → your account
#SBATCH --nodelist=iris-hgx-2            # → either delete this, or change to your node
#SBATCH --mail-user=kewalk@stanford.edu  # → your email
#SBATCH --output=/iris/u/kewalk/dreamzero/logs/yam/dreamzero_yam-%j.out  # → your log dir
```

Also update inside the script body:

```bash
cd /iris/u/kewalk/dreamzero                    # → your repo path
source /iris/u/kewalk/.bashrc.user             # → your bashrc (or delete this line)
source ~/miniconda3/etc/profile.d/conda.sh     # → your conda install path
export YAM_DATA_ROOT=${YAM_DATA_ROOT:-/iris/u/kewalk/dataset/plate_task_0505_lerobot}  # → your dataset
```

**Submit:**

```bash
# 1 H200, 3K steps (~29h)
sbatch --export=ALL,MAX_STEPS=3000,SAVE_STEPS=500 \
    cluster_scripts/yam_lora_1gpu.sbatch

# 4× H100 single node, 3K steps (~10h)
sbatch --export=ALL,MAX_STEPS=3000,SAVE_STEPS=500 \
    cluster_scripts/yam_lora_4gpu.sbatch
```

**Monitor:**
```bash
squeue --me                                  # job state
tail -f logs/yam/dreamzero_yam-<JOBID>.out   # live log
scancel <JOBID>                              # kill if needed
```

**Outputs:**
- Log: `logs/yam/dreamzero_yam-<JOBID>.out`
- Checkpoints: `checkpoints/yam_lora_1gpu_<JOBID>/checkpoint-<step>/`

---

## 8. wandb (optional but recommended)

The full LoRA script logs to wandb by default (project: `dreamzero_yam`).

To enable:
```bash
pip install wandb
wandb login          # paste your API key from https://wandb.ai/authorize
```

Or skip wandb entirely:
```bash
sbatch --export=ALL,MAX_STEPS=3000,REPORT_TO=none \
    cluster_scripts/yam_lora_1gpu.sbatch
```

---

## 9. Tunable training knobs

All overridable at submit time via `--export=ALL,KEY=VAL`:

| Variable | Default | What it does |
|---|---|---|
| `YAM_DATA_ROOT` | `/iris/u/kewalk/dataset/plate_task_0505_lerobot` | Path to LeRobot+GEAR dataset |
| `MAX_STEPS` | 30000 (too many for ~50 demos — set to 3000-5000) | Total training steps |
| `SAVE_STEPS` | 1000 | Save a checkpoint every N steps |
| `PER_DEVICE_BS` | 1 | Microbatch per GPU |
| `GRAD_ACCUM` | 4 (1-GPU script) / 1 (4-GPU script) | Gradient accumulation steps |
| `LR` | 1e-5 | Learning rate |
| `REPORT_TO` | wandb | Set to `none` to disable wandb |
| `OUTPUT_DIR` | `./checkpoints/yam_lora_1gpu_<RUN_TAG>` | Where checkpoints land |

---

## 10. OOM troubleshooting

If the job crashes with CUDA OOM during the first few steps, try in this order
(edit `scripts/train/yam_lora_1gpu.sh`):

1. Reduce `num_frames=33` → `25` (drops activation memory ~25%)
2. Reduce `image_resolution_width=320 image_resolution_height=176` → `256 144`
3. Switch deepspeed to `groot/vla/configs/deepspeed/zero2_offload.json` (CPU offload — slower but ~15 GB less GPU memory)
4. Reduce `max_chunk_size=4` → `2`

---

## File reference cheat sheet

| Purpose | Path in this repo |
|---|---|
| Merge raw demo parts → single folder | `dataset/combine_parts.py` |
| Raw demos → LeRobot v2 | `dataset/convert_data_to_lerobot.py` |
| Add GEAR metadata layer | `scripts/data/convert_lerobot_to_gear.py` |
| Sanity training (20 steps, 1 GPU) | `scripts/train/yam_sanity.sh` |
| Full LoRA training (GPU-agnostic) | `scripts/train/yam_lora_1gpu.sh` |
| SLURM wrapper, 1 GPU | `cluster_scripts/yam_lora_1gpu.sbatch` |
| SLURM wrapper, 4 GPUs single-node | `cluster_scripts/yam_lora_4gpu.sbatch` |
| Detailed runbook (workstation paths) | `steps.md` |
| Embodiment-onboarding guide | `docs/DATASET_TO_GEAR_AND_TRAIN.md` |
| Architecture reference | `docs/WAN22_BACKBONE.md` |

---

## Common issues

**`save_total_limit must be >= 5`** — the experiment harness enforces this.
The training scripts already set it correctly; don't override below 5.

**Job stuck for 10+ min with no output after "Run name: ..."** — this is normal.
The model is loading 14B weights + AgiBot LoRA + DeepSpeed initializing. First
forward pass also triggers Triton autotune. Total cold-start: 5-15 min.

**`ModuleNotFoundError: lerobot`** — you forgot to `conda activate dreamzero`.

**Long pause then crash during first training step** — usually OOM. Check
`nvidia-smi` on the node; you'll see a process holding all GPU memory just
before the crash. Apply the OOM fixes from section 10.
