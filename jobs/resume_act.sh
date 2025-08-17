#!/usr/bin/env bash
set -euo pipefail
TASK="pick_place_one_white_sock_black_out_blinds"
RUN=1
WB_ENTITY="benfduffy-bearcover-gmbh"
WB_PROJECT="lerobot"
WB_RUN_ID="gqrsl05i"

REPO="$HOME/lerobot"
SCRIPT="$REPO/lerobot/scripts/train.py"
RUN_DIR="$REPO/outputs/train/${TASK}_${RUN}"
CFG="$RUN_DIR/train_config.json"
CKPT_LAST="$RUN_DIR/checkpoints/last"
# CKPT_LAST="$RUN_DIR/checkpoints/90000"

[ -f "$SCRIPT" ] || { echo "Missing $SCRIPT"; exit 1; }
[ -d "$RUN_DIR" ] || { echo "Missing $RUN_DIR"; exit 1; }
[ -f "$CFG" ] || { echo "Missing $CFG"; exit 1; }
[ -f "$CKPT_LAST/pretrained_model/model.safetensors" ] || { echo "Missing weights"; exit 1; }

ln -sfn "$CKPT_LAST/pretrained_model/model.safetensors" "$RUN_DIR/model.safetensors" || cp -f "$CKPT_LAST/pretrained_model/model.safetensors" "$RUN_DIR/model.safetensors"
ln -sfn "$CKPT_LAST/pretrained_model/config.json"       "$RUN_DIR/config.json"       || cp -f "$CKPT_LAST/pretrained_model/config.json"       "$RUN_DIR/config.json"

# Ensure the fallback training_state path exists where train.py looks if no explicit arg is provided
FALLBACK_TS="$REPO/outputs/train/training_state"
mkdir -p "$FALLBACK_TS"
for f in optimizer_param_groups.json optimizer_state.safetensors rng_state.safetensors training_step.json; do
  if [ -f "$CKPT_LAST/training_state/$f" ]; then
    ln -sfn "$CKPT_LAST/training_state/$f" "$FALLBACK_TS/$f" || cp -f "$CKPT_LAST/training_state/$f" "$FALLBACK_TS/$f"
  else
    echo "Missing $CKPT_LAST/training_state/$f" && exit 1
  fi
done

export WANDB_RESUME=must
export WANDB_RUN_ID="$WB_RUN_ID"
export WANDB_DIR="$RUN_DIR/wandb"
mkdir -p "$WANDB_DIR"

PYTHONUNBUFFERED=1 python "$SCRIPT" \
  --config_path="$CFG" \
  --resume=true \
  --output_dir="$RUN_DIR" \
  --dataset.video_backend=pyav \
  --wandb.enable=true \
  --wandb.project="$WB_PROJECT" \
  --wandb.entity="$WB_ENTITY" \
  --wandb.run_id="$WB_RUN_ID" \
  --steps=130000 \
  --save_freq=10000
