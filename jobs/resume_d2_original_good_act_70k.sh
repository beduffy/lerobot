#!/usr/bin/env bash
set -euo pipefail

# Resume the D2 run exactly from the 70k checkpoint with the correct W&B run id

CHECKPOINT_DIR="/teamspace/studios/this_studio/lerobot/outputs/train/d2_original_good_act_1/checkpoints/070000"
PRETRAINED_DIR="$CHECKPOINT_DIR/pretrained_model"
CFG="$PRETRAINED_DIR/train_config.json"

# The original run directory (parent of checkpoints)
RUN_DIR="/teamspace/studios/this_studio/lerobot/outputs/train/d2_original_good_act_1"

# W&B settings
WB_ENTITY="benfduffy-bearcover-gmbh"
WB_PROJECT="lerobot"
WB_RUN_ID="rk6ilwga"

# Python path to new train.py
PYTHONPATH="/teamspace/studios/this_studio/lerobot/src"

# Speed-ups for MAE@ep0 during eval
export LEROBOT_VAL_MAE_MAX_FRAMES="800"
export LEROBOT_VAL_MAE_BATCH_SIZE="64"

# Keep W&B files outside RUN_DIR to avoid collisions
export WANDB_RESUME=must
export WANDB_RUN_ID="$WB_RUN_ID"
export WANDB_DIR="/teamspace/studios/this_studio/lerobot/outputs/wandb/d2_original_good_act_1_${WB_RUN_ID}"
mkdir -p "$WANDB_DIR"

# Sanity checks
[ -d "$CHECKPOINT_DIR" ] || { echo "Missing checkpoint dir: $CHECKPOINT_DIR"; exit 1; }
[ -f "$CFG" ] || { echo "Missing train config: $CFG"; exit 1; }

echo "Resuming from:    $CHECKPOINT_DIR"
echo "Using config:     $CFG"
echo "Run directory:    $RUN_DIR"
echo "W&B run id:       $WB_RUN_ID"
echo "WANDB_DIR:        $WANDB_DIR"
echo "MAE frames cap:   $LEROBOT_VAL_MAE_MAX_FRAMES"
echo "MAE batch size:   $LEROBOT_VAL_MAE_BATCH_SIZE"

PYTHONUNBUFFERED=1 PYTHONPATH="$PYTHONPATH" \
python -m lerobot.scripts.train \
  --config_path="$CFG" \
  --resume=true \
  --output_dir="$RUN_DIR" \
  --dataset.video_backend=pyav \
  --wandb.enable=true \
  --wandb.project="$WB_PROJECT" \
  --wandb.entity="$WB_ENTITY" \
  --wandb.run_id="$WB_RUN_ID" \
  --eval_freq=1000


