#!/usr/bin/env bash
set -euo pipefail

# Allow passing VAR=VALUE arguments after the script name by exporting them
for kv in "$@"; do
  case "$kv" in
    *=*) export "$kv" ;;
  esac
done

# ---- YOU EDIT (env vars override these) ----
# Match the original run naming so RUN_DIR resolves predictably
TASK="${TASK:-d1_and_d2_datasets_act_224x224}"
RUN_TAG="${RUN_TAG:-from_hub_ckpt250k_to400k}"

# Additional steps to train (we initialize from 150k and train +250k => 400k total)
STEPS="${STEPS:-400000}"
# When not using resume=true, we set an initial step for accurate logging
# TODO shouldn't need this
INITIAL_STEP="${INITIAL_STEP:-250000}"

# Device and backend (default to cuda)
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"

# Dataset settings (mirror the original run script)
DATASET_REPO_ID="${DATASET_REPO_ID:-[\"bearlover365/pick_place_one_white_sock_black_out_blinds\",\"bearlover365/pick_place_up_to_four_white_socks_black_out_blinds\"]}"
BATCH="${BATCH:-8}"
SAVE_FREQ="${SAVE_FREQ:-25000}"
LOG_FREQ="${LOG_FREQ:-200}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"

# Hugging Face pretrained checkpoint to initialize from (150k)
# HF_PRETRAINED_REPO_ID="${HF_PRETRAINED_REPO_ID:-bearlover365/d1_d2_act224_s500k_b8_ckpt25k_150000}"
HF_PRETRAINED_REPO_ID="${HF_PRETRAINED_REPO_ID:-bearlover365/d1_d2_act224_s500k_b8_ckpt25k_250000}"
export HF_PRETRAINED_REPO_ID

# W&B (optional). If WB_RUN_ID is set, continue the same run id (not required)
WB_PROJECT="${WB_PROJECT:-lerobot}"
WB_ENTITY="${WB_ENTITY:-benfduffy-bearcover-gmbh}"
WB_RUN_ID="${WB_RUN_ID:-6gbhi09t}"
# WB_RUN_ID="${WB_RUN_ID:-}"

# Repo paths
REPO="/teamspace/studios/this_studio/lerobot"
SCRIPT="$REPO/src/lerobot/scripts/train.py"
RUN_DIR="$REPO/outputs/train/${TASK}_${RUN_TAG}"

echo "TASK=$TASK"
echo "RUN_TAG=$RUN_TAG"
echo "STEPS=$STEPS"
echo "POLICY_DEVICE=$POLICY_DEVICE"
echo "DATASET_REPO_ID=$DATASET_REPO_ID"
echo "BATCH=$BATCH"
echo "SAVE_CHECKPOINT=$SAVE_CHECKPOINT"
echo "SAVE_FREQ=$SAVE_FREQ"
echo "LOG_FREQ=$LOG_FREQ"
echo "HF_PRETRAINED_REPO_ID=$HF_PRETRAINED_REPO_ID"
echo "INITIAL_STEP=$INITIAL_STEP"
echo "WB_PROJECT=$WB_PROJECT"
echo "WB_ENTITY=$WB_ENTITY"
echo "WB_RUN_ID=${WB_RUN_ID:-<none>}"
echo "REPO=$REPO"
echo "SCRIPT=$SCRIPT"
echo "RUN_DIR=$RUN_DIR"

# ---- Sanity ----
[ -f "$SCRIPT" ] || { echo "Missing $SCRIPT"; exit 1; }

echo "Conda env: ${CONDA_DEFAULT_ENV:-<none>}"
python --version || true

# If the run directory already exists and resume=false, avoid overwrite by appending a timestamp suffix
if [ -d "$RUN_DIR" ]; then
  SUFFIX=$(date +%Y%m%d_%H%M%S)
  RUN_TAG="${RUN_TAG}_${SUFFIX}"
  RUN_DIR="$REPO/outputs/train/${TASK}_${RUN_TAG}"
fi

# Download pretrained checkpoint from the Hugging Face Hub
echo "Downloading pretrained checkpoint from HF..."
export HF_HOME="${HF_HOME:-$REPO/.hf_cache}"
PRETRAINED_DIR=$(python - <<'PY'
import os
from huggingface_hub import snapshot_download
repo_id = os.environ.get('HF_PRETRAINED_REPO_ID')
path = snapshot_download(repo_id=repo_id)
print(path)
PY
)
if [ -z "$PRETRAINED_DIR" ] || [ ! -d "$PRETRAINED_DIR" ]; then
  echo "Failed to download pretrained checkpoint from $HF_PRETRAINED_REPO_ID" && exit 1
fi
echo "Pretrained path: $PRETRAINED_DIR"

# W&B configure
# Keep W&B logs outside RUN_DIR to avoid creating it before train.py runs
export WANDB_DIR="$REPO/outputs/wandb/${TASK}_${RUN_TAG}"
mkdir -p "$WANDB_DIR"
if [ -n "$WB_RUN_ID" ]; then
  export WANDB_RESUME=must
  export WANDB_RUN_ID="$WB_RUN_ID"
fi

# # Choose a suitable video backend
# if [ "$POLICY_DEVICE" = "cuda" ]; then
#   VIDEO_BACKEND="torchcodec"
# else
#   VIDEO_BACKEND="pyav"
# fi
VIDEO_BACKEND="pyav"

cd "$REPO"

PYTHONUNBUFFERED=1 python "$SCRIPT" \
  --resume=false \
  --output_dir="$RUN_DIR" \
  --job_name="${TASK}_${RUN_TAG}" \
  --wandb.enable=true \
  --wandb.project="$WB_PROJECT" \
  --wandb.entity="$WB_ENTITY" \
  ${WB_RUN_ID:+--wandb.run_id="$WB_RUN_ID"} \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --policy.path="$PRETRAINED_DIR" \
  --policy.device="$POLICY_DEVICE" \
  --policy.use_amp=true \
  --use_policy_training_preset=true \
  --num_workers=8 \
  --batch_size="$BATCH" \
  --initial_step="$INITIAL_STEP" \
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
  |& tee -a "$(dirname "$RUN_DIR")/${TASK}_${RUN_TAG}.log"


