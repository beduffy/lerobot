#!/usr/bin/env bash
set -euo pipefail

# Allow passing VAR=VALUE arguments after the script name by exporting them
for kv in "$@"; do
  case "$kv" in
    *=*) export "$kv" ;;
  esac
done

# TODO need saved params somewhere e.g. train_config.json

# ---- Defaults (env override allowed) ----
TASK="${TASK:-d2_single_ep0_act_224x224}"
RUN_TAG="${RUN_TAG:-overfit_gpu_s30}"
BATCH="${BATCH:-8}"
STEPS="${STEPS:-30000}"
LOG_FREQ="${LOG_FREQ:-200}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
PUSH_TO_HUB="${PUSH_TO_HUB:-true}"
# If you want to push checkpoints, set a base repo id you own (e.g. your_user/act224_overfit_d2_ep0)
POLICY_REPO_ID="${POLICY_REPO_ID:-bearlover365/d1_d2_act224_s300k_b8_ckpt25k}"
# If you want to start from a pretrained checkpoint (Hub repo id or local dir), set POLICY_PATH; leave empty to train from scratch
POLICY_PATH="${POLICY_PATH:-}"
NUM_WORKERS="${NUM_WORKERS:-8}"


# Single-episode train/val repos built from dataset 2
# DATASET_REPO_ID="${DATASET_REPO_ID:-[\"bearlover365/pick_place_up_to_four_white_socks_black_out_blinds_single_ep0_train\"]}"
DATASET_REPO_ID="${DATASET_REPO_ID:-bearlover365/pick_place_up_to_four_white_socks_black_out_blinds_single_ep0_train}"
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
echo "POLICY_REPO_ID=$POLICY_REPO_ID"
echo "POLICY_PATH=$POLICY_PATH"

[ -f "$SCRIPT" ] || { echo "Missing $SCRIPT"; exit 1; }

export WANDB_DIR="$RUN_DIR/wandb"
mkdir -p "$(dirname "$RUN_DIR")"
cd "$REPO"

EXTRA_ARGS=( )
if [ -n "${VAL_REPO_ID}" ]; then
  EXTRA_ARGS+=( --dataset.val_repo_id="${VAL_REPO_ID}" )
fi

  VIDEO_BACKEND="pyav"


# Validate push-to-hub settings
if [ "$PUSH_TO_HUB" = "true" ] && [ -z "$POLICY_REPO_ID" ]; then
  echo "PUSH_TO_HUB=true requires POLICY_REPO_ID to be set (e.g. your_user/act224_overfit_d2_ep0)." >&2
  exit 1
fi

# if i only include
# --policy.path="$POLICY_REPO_ID" \
#     raise FileNotFoundError(
# FileNotFoundError: config.json not found on the HuggingFace Hub in bearlover365/d1_d2_act224_s300k_b8_ckpt25k

# if i only include
  # --policy.type=act \
# raise ValueError(
# ValueError: 'policy.repo_id' argument missing. Please specify it to push the model to the hub.

# if i include both above
#    raise ArgumentError(
# argparse.ArgumentError: Cannot specify both --policy.path and --policy.type


PYTHONUNBUFFERED=1
POLICY_ARGS=( )
if [ -n "$POLICY_PATH" ]; then
  POLICY_ARGS+=( --policy.path="$POLICY_PATH" )
else
  POLICY_ARGS+=( --policy.type=act )
fi

# removed
  # --dataset.image_transforms.enable=true \
  # --dataset.image_transforms.max_num_transforms=2 \
  # --dataset.image_transforms.random_order=false \
  # --dataset.image_transforms.tfs='{"crop":{"type":"CenterCrop","kwargs":{"size":[320,320]}},"resize":{"type":"Resize","kwargs":{"size":[224,224]}}}' \
  # --dataset.use_imagenet_stats=true \  # TODO this one seems very fishy

python "$SCRIPT" \
  --resume=false \
  --output_dir="$RUN_DIR" \
  --job_name="${TASK}_${RUN_TAG}" \
  --wandb.enable=true \
  --wandb.project="$WB_PROJECT" \
  --wandb.entity="$WB_ENTITY" \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --policy.push_to_hub="$PUSH_TO_HUB" \
  --policy.device="$POLICY_DEVICE" \
  --policy.use_amp=true \
  ${POLICY_ARGS[@]} \
  ${PUSH_TO_HUB:+${POLICY_REPO_ID:+--policy.repo_id="$POLICY_REPO_ID"}} \
  --use_policy_training_preset=true \
  --num_workers="$NUM_WORKERS" \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --log_freq="$LOG_FREQ" \
  --eval_freq=0 \
  --save_checkpoint="$SAVE_CHECKPOINT" \
  --save_freq="$SAVE_FREQ" \
  --dataset.video_backend="$VIDEO_BACKEND" \

  "${EXTRA_ARGS[@]}" \
  |& tee -a "$(dirname "$RUN_DIR")/${TASK}_${RUN_TAG}.log"



