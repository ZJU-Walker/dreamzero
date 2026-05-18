# DreamZero YAM Finetune — Runbook

End-to-end commands for finetuning DreamZero on the YAM plate-memory dataset.
Workstation has 1 H200; cluster job uses 1 H200 on iris-hi.

---

## 0. One-time setup

### 0a. Activate env

```bash
cd /iris/u/kewalk/dreamzero
conda activate dreamzero
```

### 0b. Pre-download checkpoints (~73 GB, once)

Skip if `./checkpoints/Wan2.1-I2V-14B-480P`, `./checkpoints/umt5-xxl`, and
`./checkpoints/DreamZero-AgiBot` already exist. Otherwise:

```bash
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
huggingface-cli download google/umt5-xxl              --local-dir ./checkpoints/umt5-xxl
huggingface-cli download GEAR-Dreams/DreamZero-AgiBot --local-dir ./checkpoints/DreamZero-AgiBot
```

---

## 1. Data preparation (already done — kept for reference)

### 1a. Combine 3 raw dataset parts → 52 sequentially-named demos

Source: `/iris/u/kewalk/dataset/long_task_1_part{1,2,3}` (27 + 10 + 15 = 52 demos)
Output: `/iris/u/kewalk/dataset/plate_task_0505/demo1 … demo52`

```bash
cd /iris/u/kewalk/dataset
python3 combine_parts.py
```

### 1b. Raw demos → LeRobot v2 (zero-pad right arm to 16-dim bimanual)

Output: `/iris/u/kewalk/dataset/plate_task_0505_lerobot/` (parquet + mp4 + meta/)

```bash
cd /iris/u/kewalk/dataset
python3 convert_data_to_lerobot.py
```

State layout: `[left_joint(7), left_gripper(1), right_joint=0(7), right_gripper=0(1)]` = 16 dims
Action layout: `[left_control(7), left_gripper(1), right=0(8)]` = 16 dims

### 1c. LeRobot v2 → GEAR metadata (modality, stats, episodes, embodiment)

```bash
cd /iris/u/kewalk/dreamzero
python scripts/data/convert_lerobot_to_gear.py \
    --dataset_path /iris/u/kewalk/dataset/plate_task_0505_lerobot \
    --embodiment_tag yam
```

Verify: `/iris/u/kewalk/dataset/plate_task_0505_lerobot/meta/embodiment.json` exists
and `meta/modality.json` has keys `left_joint_pos[0,7]`, `left_gripper_pos[7,8]`,
`right_joint_pos[8,15]`, `right_gripper_pos[15,16]`.

---

## 2. Sanity check — 20 steps, 1 GPU (do this first!)

Runs 20 training steps to validate the data + model pipeline end-to-end.
Expect: first batch loads, loss is finite, `./checkpoints/yam_sanity/checkpoint-20/` appears.

```bash
cd /iris/u/kewalk/dreamzero
conda activate dreamzero
YAM_DATA_ROOT=/iris/u/kewalk/dataset/plate_task_0505_lerobot \
    bash scripts/train/yam_sanity.sh 2>&1 | tee /tmp/yam_sanity.log
```

**Pass criteria:**
- No `KeyError: state.left_joint_pos` (data keys match)
- No `RuntimeError: shape mismatch` (dims match)
- 20 steps complete with finite loss
- `./checkpoints/yam_sanity/checkpoint-20/` contains adapter weights

---

## 3. Full LoRA finetune — 1 H200

### 3a. Workstation run

```bash
cd /iris/u/kewalk/dreamzero
conda activate dreamzero
YAM_DATA_ROOT=/iris/u/kewalk/dataset/plate_task_0505_lerobot \
    bash scripts/train/yam_lora_1gpu.sh 2>&1 | tee /tmp/yam_lora.log
```

Output dir: `./checkpoints/yam_lora_1gpu_<timestamp>/`

Defaults: `max_steps=30000`, `per_device_train_batch_size=1`, `grad_accum=4`
(effective batch 4), `lr=1e-5`, `save_steps=5000`, wandb project `dreamzero_yam`.

Override examples:
```bash
MAX_STEPS=10000 LR=5e-6 REPORT_TO=none \
    bash scripts/train/yam_lora_1gpu.sh
```

### 3b. Cluster run (iris-hi, 1 H200) — via sbatch

Sbatch script lives at: `/iris/u/kewalk/cluster_script/dreamzero/yam_lora_1gpu.sbatch`

**First time only** — confirm the H200 node name and pin it in the sbatch
(edit the `#SBATCH --nodelist=<node>` line near the top):

```bash
sinfo -o "%P %n %G" | grep -i h200
```

**Submit the job:**

```bash
sbatch /iris/u/kewalk/cluster_script/dreamzero/yam_lora_1gpu.sbatch
```

**Monitor:**

```bash
squeue -u $USER                                                # queue status
tail -f /iris/u/kewalk/dreamzero/logs/yam/dreamzero_yam-<JOBID>.out   # live log
scancel <JOBID>                                                # kill if needed
```

**Outputs:**
- Log: `/iris/u/kewalk/dreamzero/logs/yam/dreamzero_yam-<JOBID>.out`
- Checkpoints: `/iris/u/kewalk/dreamzero/checkpoints/yam_lora_1gpu_<JOBID>/`

**Override env vars at submit time:**

```bash
# Shorter run, no wandb
sbatch --export=ALL,MAX_STEPS=10000,REPORT_TO=none \
    /iris/u/kewalk/cluster_script/dreamzero/yam_lora_1gpu.sbatch

# Different dataset
sbatch --export=ALL,YAM_DATA_ROOT=/path/to/other/dataset \
    /iris/u/kewalk/cluster_script/dreamzero/yam_lora_1gpu.sbatch
```

---

## 4. (Optional) Open-loop evaluation on saved checkpoint

**Caveat**: `scripts/open_loop_yam.py:47-58` has stale slice indices `[32,46]`
from YAM-pioneer 6-joint data. Patch to `[0,7][7,8][8,15][15,16]` before running
on the 7-joint plate-memory data.

```bash
cd /iris/u/kewalk/dreamzero
python scripts/open_loop_yam.py \
    --model_path ./checkpoints/yam_lora_1gpu_<JOBID> \
    --dataset_path /iris/u/kewalk/dataset/plate_task_0505_lerobot \
    --num_samples 5
```

---

## OOM fallbacks (if 1 H200 / 141 GB isn't enough)

In `scripts/train/yam_lora_1gpu.sh`, try in order:
1. Drop `num_frames=33` → `25`
2. Drop `image_resolution_width=320 image_resolution_height=176` → `256 144`
3. Switch deepspeed to `groot/vla/configs/deepspeed/zero2_offload.json` (CPU offload, slower)

---

## Quick reference: file locations

| Purpose | Path |
|---|---|
| Raw YAM data (3 parts) | `/iris/u/kewalk/dataset/long_task_1_part{1,2,3}/` |
| Combined raw (52 demos) | `/iris/u/kewalk/dataset/plate_task_0505/` |
| LeRobot v2 + GEAR meta | `/iris/u/kewalk/dataset/plate_task_0505_lerobot/` |
| Combine script | `/iris/u/kewalk/dataset/combine_parts.py` |
| LeRobot conversion | `/iris/u/kewalk/dataset/convert_data_to_lerobot.py` |
| GEAR metadata script | `dreamzero/scripts/data/convert_lerobot_to_gear.py` |
| Sanity training | `dreamzero/scripts/train/yam_sanity.sh` |
| 1-H200 LoRA training | `dreamzero/scripts/train/yam_lora_1gpu.sh` |
| Slurm wrapper | `/iris/u/kewalk/cluster_script/dreamzero/yam_lora_1gpu.sbatch` |
