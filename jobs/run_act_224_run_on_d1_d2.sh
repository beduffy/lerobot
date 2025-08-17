#!/usr/bin/env bash
set -euo pipefail

# ---- YOU EDIT (env vars override these) ----
# TASK="${TASK:-pick_place_one_white_sock_black_out_blinds}"
TASK="${TASK:-pick_place_one_white_sock_black_out_blinds}"
RUN_TAG="${RUN_TAG:-act_224x224_s1k_b64}"
BATCH="${BATCH:-8}"
STEPS="${STEPS:-1000}"
LR="${LR:-8e-5}"
DATASET_REPO_ID="${DATASET_REPO_ID:-bearlover365/${TASK}}"
LOG_FREQ="${LOG_FREQ:-10}"

# ---- Derived ----
REPO="/teamspace/studios/this_studio/lerobot"
SCRIPT="$REPO/src/lerobot/scripts/train.py"
RUN_DIR="$REPO/outputs/train/${TASK}_${RUN_TAG}"

# If the run directory already exists and resume=false, avoid overwrite by appending a timestamp suffix
if [ -d "$RUN_DIR" ]; then
  SUFFIX=$(date +%Y%m%d_%H%M%S)
  RUN_TAG="${RUN_TAG}_${SUFFIX}"
  RUN_DIR="$REPO/outputs/train/${TASK}_${RUN_TAG}"
fi
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"

# ---- Sanity ----
[ -f "$SCRIPT" ] || { echo "Missing $SCRIPT"; exit 1; }

echo "Conda env: ${CONDA_DEFAULT_ENV:-<none>}"
python --version || true

# ---- W&B (new run) ----
export WANDB_DIR="$RUN_DIR/wandb"
# Ensure parent output dir exists for logging
mkdir -p "$(dirname "$RUN_DIR")"
cd "$REPO"

PYTHONUNBUFFERED=1 python "$SCRIPT" \
  --resume=false \
  --output_dir="$RUN_DIR" \
  --job_name="${TASK}_${RUN_TAG}" \
  --wandb.enable=false \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --policy.type=act \
  --policy.push_to_hub=false \
  --policy.device="$POLICY_DEVICE" \
  --policy.use_amp=true \
  --use_policy_training_preset=true \
  --num_workers=8 \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --log_freq="$LOG_FREQ" \
  --eval_freq=0 \
  --save_checkpoint=false \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=2 \
  --dataset.image_transforms.random_order=false \
  --dataset.image_transforms.tfs='{"crop":{"type":"CenterCrop","kwargs":{"size":[320,320]}},"resize":{"type":"Resize","kwargs":{"size":[224,224]}}}' \
  --dataset.use_imagenet_stats=true \
  |& tee -a "$(dirname "$RUN_DIR")/${TASK}_${RUN_TAG}.log"


