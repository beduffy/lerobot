#!/usr/bin/env bash
set -euo pipefail

# Allow passing VAR=VALUE arguments after the script name by exporting them
for kv in "$@"; do
  case "$kv" in
    *=*) export "$kv" ;;
  esac
done

# ---- Defaults (env override allowed) ----
TASK="${TASK:-d2_single_ep0_act_224x224}"
RUN_TAG="${RUN_TAG:-overfit_cpu_s30}"
BATCH="${BATCH:-1}"
STEPS="${STEPS:-30}"
LOG_FREQ="${LOG_FREQ:-5}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
SAVE_FREQ="${SAVE_FREQ:-10}"
POLICY_DEVICE="${POLICY_DEVICE:-cpu}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"

# Single-episode train/val repos built from dataset 2
DATASET_REPO_ID="${DATASET_REPO_ID:-[\"bearlover365/pick_place_up_to_four_white_socks_black_out_blinds_single_ep0_train\"]}"
VAL_REPO_ID="${VAL_REPO_ID:-bearlover365/pick_place_up_to_four_white_socks_black_out_blinds_single_ep0_val}"

WB_PROJECT="${WB_PROJECT:-lerobot}"
WB_ENTITY="${WB_ENTITY:-benfduffy-bearcover-gmbh}"

REPO="/teamspace/studios/this_studio/lerobot"
SCRIPT="$REPO/src/lerobot/scripts/train.py"
RUN_DIR="$REPO/outputs/train/${TASK}_${RUN_TAG}"

echo "Conda env: ${CONDA_DEFAULT_ENV:-<none>}"
python --version || true

echo "TASK=$TASK"
echo "RUN_TAG=$RUN_TAG"
echo "BATCH=$BATCH"
echo "STEPS=$STEPS"
echo "LOG_FREQ=$LOG_FREQ"
echo "SAVE_FREQ=$SAVE_FREQ"
echo "SAVE_CHECKPOINT=$SAVE_CHECKPOINT"
echo "POLICY_DEVICE=$POLICY_DEVICE"
echo "PUSH_TO_HUB=$PUSH_TO_HUB"
echo "DATASET_REPO_ID=$DATASET_REPO_ID"
echo "VAL_REPO_ID=$VAL_REPO_ID"
echo "REPO=$REPO"
echo "SCRIPT=$SCRIPT"
echo "RUN_DIR=$RUN_DIR"

[ -f "$SCRIPT" ] || { echo "Missing $SCRIPT"; exit 1; }

export WANDB_DIR="$RUN_DIR/wandb"
mkdir -p "$(dirname "$RUN_DIR")"
cd "$REPO"

EXTRA_ARGS=( )
if [ -n "${VAL_REPO_ID}" ]; then
  EXTRA_ARGS+=( --dataset.val_repo_id="${VAL_REPO_ID}" )
fi

if [ "$POLICY_DEVICE" = "cuda" ]; then
  VIDEO_BACKEND="torchcodec"
else
  VIDEO_BACKEND="pyav"
fi

# Reduce workers on CPU for reliability
NUM_WORKERS="${NUM_WORKERS:-2}"

PYTHONUNBUFFERED=1 python "$SCRIPT" \
  --resume=false \
  --output_dir="$RUN_DIR" \
  --job_name="${TASK}_${RUN_TAG}" \
  --wandb.enable=true \
  --wandb.project="$WB_PROJECT" \
  --wandb.entity="$WB_ENTITY" \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --policy.type=act \
  --policy.push_to_hub="$PUSH_TO_HUB" \
  --policy.device="$POLICY_DEVICE" \
  --policy.use_amp=true \
  --use_policy_training_preset=true \
  --num_workers="$NUM_WORKERS" \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --log_freq="$LOG_FREQ" \
  --eval_freq=0 \
  --save_checkpoint="$SAVE_CHECKPOINT" \
  --save_freq="$SAVE_FREQ" \
  --dataset.video_backend="$VIDEO_BACKEND" \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=2 \
  --dataset.image_transforms.random_order=false \
  --dataset.image_transforms.tfs='{"crop":{"type":"CenterCrop","kwargs":{"size":[320,320]}},"resize":{"type":"Resize","kwargs":{"size":[224,224]}}}' \
  --dataset.use_imagenet_stats=true \
  "${EXTRA_ARGS[@]}" \
  |& tee -a "$(dirname "$RUN_DIR")/${TASK}_${RUN_TAG}.log"



