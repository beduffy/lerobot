#!/usr/bin/env bash
set -euo pipefail
# Allow overriding via environment; provide sensible defaults
TASK="${TASK:-pick_place_one_white_sock_black_out_blinds}"
RUN="${RUN:-3}"
WB_ENTITY="benfduffy-bearcover-gmbh"
WB_PROJECT="lerobot"
# WB_RUN_ID="gqrsl05i"

REPO="${REPO:-/teamspace/studios/this_studio/lerobot}"
# SCRIPT="$REPO/lerobot/scripts/train.py"  # old repo
SCRIPT="$REPO/src/lerobot/scripts/train.py"
RUN_DIR="$REPO/outputs/train/${TASK}_${RUN}"
CFG="$RUN_DIR/train_config.json"
CKPT_LAST="$RUN_DIR/checkpoints/last"
# CKPT_LAST="$RUN_DIR/checkpoints/90000"

[ -f "$SCRIPT" ] || { echo "Missing $SCRIPT"; exit 1; }
# [ -d "$RUN_DIR" ] || { echo "Missing $RUN_DIR"; exit 1; }  # not resuming
# [ -f "$CFG" ] || { echo "Missing $CFG"; exit 1; }
# [ -f "$CKPT_LAST/pretrained_model/model.safetensors" ] || { echo "Missing weights"; exit 1; }

# ln -sfn "$CKPT_LAST/pretrained_model/model.safetensors" "$RUN_DIR/model.safetensors" || cp -f "$CKPT_LAST/pretrained_model/model.safetensors" "$RUN_DIR/model.safetensors"
# ln -sfn "$CKPT_LAST/pretrained_model/config.json"       "$RUN_DIR/config.json"       || cp -f "$CKPT_LAST/pretrained_model/config.json"       "$RUN_DIR/config.json"

# Ensure the fallback training_state path exists where train.py looks if no explicit arg is provided
# FALLBACK_TS="$REPO/outputs/train/training_state"
# mkdir -p "$FALLBACK_TS"
# for f in optimizer_param_groups.json optimizer_state.safetensors rng_state.safetensors training_step.json; do
#   if [ -f "$CKPT_LAST/training_state/$f" ]; then
#     ln -sfn "$CKPT_LAST/training_state/$f" "$FALLBACK_TS/$f" || cp -f "$CKPT_LAST/training_state/$f" "$FALLBACK_TS/$f"
#   else
#     echo "Missing $CKPT_LAST/training_state/$f" && exit 1
  #   fi
  # done

# export WANDB_RESUME=must
# export WANDB_RUN_ID="$WB_RUN_ID"
# Important: do NOT pre-create anything under RUN_DIR to avoid triggering
# train.py's output_dir existence check. Keep WandB logs outside RUN_DIR.
export WANDB_DIR="${REPO}/outputs/wandb/${TASK}_${RUN}"
mkdir -p "$WANDB_DIR"

echo "WANDB_DIR: $WANDB_DIR"
echo "RUN_DIR: $RUN_DIR"
echo "CFG: $CFG"
echo "CKPT_LAST: $CKPT_LAST"
echo "SCRIPT: $SCRIPT"
echo "REPO: $REPO"
echo "TASK: $TASK"
echo "RUN: $RUN"
echo "WB_ENTITY: $WB_ENTITY"
echo "WB_PROJECT: $WB_PROJECT"
# echo "WB_RUN_ID: $WB_RUN_ID"


PYTHONUNBUFFERED=1 python "$SCRIPT" \
  --dataset.repo_id=bearlover365/pick_place_one_white_sock_black_out_blinds \
  --resume=false \
  --policy.type=act \
  --output_dir="$RUN_DIR" \
  --job_name="${TASK}_${RUN}_original_good_act" \
  --policy.repo_id=bearlover365/${TASK}_${RUN}_act_model \
  --dataset.video_backend=pyav \
  --wandb.enable=true \
  --wandb.project="$WB_PROJECT" \
  --wandb.entity="$WB_ENTITY" \
  --steps=130000 \
  --save_freq=10000 \
  --eval_freq=200
