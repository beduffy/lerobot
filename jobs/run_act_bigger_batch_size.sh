#!/usr/bin/env bash
set -euo pipefail

# ---- YOU EDIT ----
TASK="pick_place_one_white_sock_black_out_blinds"
RUN_TAG="l40s_b64"
BATCH=64
STEPS=7500
LR=8e-5
WARMUP=400
DECAY_STEPS=7000
PEAK_LR="$LR"
DECAY_LR=1e-6

# ---- Derived ----
REPO="$HOME/lerobot"
SCRIPT="$REPO/lerobot/scripts/train.py"
RUN_DIR="$REPO/outputs/train/${TASK}_${RUN_TAG}"
CFG_BASE="$REPO/outputs/train/${TASK}_1/train_config.json"

# ---- Sanity ----
[ -f "$SCRIPT" ] || { echo "Missing $SCRIPT"; exit 1; }
[ -f "$CFG_BASE" ] || { echo "Missing base config $CFG_BASE"; exit 1; }
mkdir -p "$RUN_DIR"

# ---- W&B (new run) ----
export WANDB_DIR="$RUN_DIR/wandb"; mkdir -p "$WANDB_DIR"

cd "$REPO"
PYTHONUNBUFFERED=1 python "$SCRIPT" \
  --config_path="$CFG_BASE" \
  --resume=false \
  --output_dir="$RUN_DIR" \
  --job_name="${TASK}_${RUN_TAG}" \
  --wandb.enable=true \
  --wandb.project="lerobot" \
  --wandb.entity="benfduffy-bearcover-gmbh" \
  --policy.device=cuda \
  --policy.use_amp=true \
  --num_workers=8 \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --optimizer.type=adamw \
  --optimizer.lr="$LR" \
  --scheduler.type=cosine_decay_with_warmup \
  --scheduler.num_warmup_steps="$WARMUP" \
  --scheduler.num_decay_steps="$DECAY_STEPS" \
  --scheduler.peak_lr="$PEAK_LR" \
  --scheduler.decay_lr="$DECAY_LR" \
  --save_freq=10000 \
  --dataset.video_backend=pyav


