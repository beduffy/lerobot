#!/usr/bin/env bash
set -euo pipefail

# Allow passing VAR=VALUE arguments after the script name by exporting them
for kv in "$@"; do
  case "$kv" in
    *=*) export "$kv" ;;
  esac
done 

# ---- YOU EDIT (env vars override these) ----
# TASK="${TASK:-pick_place_one_white_sock_black_out_blinds}"
TASK="${TASK:-d1_and_d2_datasets_act_224x224}"
RUN_TAG="${RUN_TAG:-checkpoint_25k_300k_steps}"
BATCH="${BATCH:-8}"
# STEPS="${STEPS:-5}"
STEPS="${STEPS:-500000}"
# LR="${LR:-8e-5}"  # using default. 
# JSON list of datasets by default (D1 + D2); can be overridden via env
DATASET_REPO_ID="${DATASET_REPO_ID:-[\"bearlover365/pick_place_one_white_sock_black_out_blinds\",\"bearlover365/pick_place_up_to_four_white_socks_black_out_blinds\"]}"
LOG_FREQ="${LOG_FREQ:-200}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
SAVE_FREQ="${SAVE_FREQ:-25000}"
# Hub push parameters (optional)
PUSH_TO_HUB="${PUSH_TO_HUB:-true}"
POLICY_REPO_ID="${POLICY_REPO_ID:-bearlover365/d1_d2_act224_s300k_b8_ckpt25k}"
# Optional offline validation repo (held-out set)
VAL_REPO_ID="${VAL_REPO_ID:-}"
# W&B defaults (enabled by default like other scripts)
WB_PROJECT="${WB_PROJECT:-lerobot}"
WB_ENTITY="${WB_ENTITY:-benfduffy-bearcover-gmbh}"
REPO="/teamspace/studios/this_studio/lerobot"
SCRIPT="$REPO/src/lerobot/scripts/train.py"
RUN_DIR="$REPO/outputs/train/${TASK}_${RUN_TAG}"

# TODO setting of LOG_FREQ in call didn't seem to work?

echo "DATASET_REPO_ID=$DATASET_REPO_ID"
echo "TASK=$TASK"
echo "RUN_TAG=$RUN_TAG"
echo "BATCH=$BATCH"
echo "STEPS=$STEPS"
echo "LOG_FREQ=$LOG_FREQ"
echo "SAVE_FREQ=$SAVE_FREQ"
echo "SAVE_CHECKPOINT=$SAVE_CHECKPOINT"
echo "POLICY_DEVICE=$POLICY_DEVICE"
echo "PUSH_TO_HUB=$PUSH_TO_HUB"
echo "POLICY_REPO_ID=$POLICY_REPO_ID"
echo "VAL_REPO_ID=$VAL_REPO_ID"
echo "WB_PROJECT=$WB_PROJECT"
echo "WB_ENTITY=$WB_ENTITY"
echo "REPO=$REPO"
echo "SCRIPT=$SCRIPT"
echo "RUN_DIR=$RUN_DIR"

# bash /teamspace/studios/this_studio/lerobot/jobs/run_act_224_run_on_d1_d2.sh POLICY_DEVICE=cpu STEPS=1 RUN_TAG=act_224x224_smoke_on_d1_d2 SAVE_CHECKPOINT=false LOG_FREQ=1
# bash /teamspace/studios/this_studio/lerobot/jobs/run_act_224_run_on_d1_d2.sh POLICY_DEVICE=cpu STEPS=5 RUN_TAG=act_224x224_smoke_on_d1_d2_t4 SAVE_CHECKPOINT=false LOG_FREQ=1
# cpu batch 1, try
# bash lerobot/jobs/run_act_224_run_on_d1_d2.sh \
#   POLICY_DEVICE=cpu STEPS=1 BATCH=1 LOG_FREQ=1 SAVE_CHECKPOINT=false \
#   RUN_TAG=smoke_ckpt25k \
#   DATASET_REPO_ID='["bearlover365/pick_place_one_white_sock_black_out_blinds","bearlover365/pick_place_up_to_four_white_socks_black_out_blinds"]'

# good run l40s from mega-rl-experiments
# bash lerobot/jobs/run_act_224_run_on_d1_d2.sh


# ---- Derived ----


# If the run directory already exists and resume=false, avoid overwrite by appending a timestamp suffix
if [ -d "$RUN_DIR" ]; then
# TODO will this work or break things regardless of what value of resume is?
  SUFFIX=$(date +%Y%m%d_%H%M%S)
  RUN_TAG="${RUN_TAG}_${SUFFIX}"
  RUN_DIR="$REPO/outputs/train/${TASK}_${RUN_TAG}"
fi


# ---- Sanity ----
[ -f "$SCRIPT" ] || { echo "Missing $SCRIPT"; exit 1; }

echo "Conda env: ${CONDA_DEFAULT_ENV:-<none>}"
python --version || true

# ---- W&B (new run) ----
export WANDB_DIR="$RUN_DIR/wandb"
# Ensure parent output dir exists for logging
mkdir -p "$(dirname "$RUN_DIR")"
cd "$REPO"

# Build optional args
EXTRA_ARGS=()
if [ -n "$VAL_REPO_ID" ]; then
  EXTRA_ARGS+=( --dataset.val_repo_id="$VAL_REPO_ID" )
  # For a smoke test, validate each checkpoint
  
  # TODO DANGEROUS not correct. 
  EXTRA_ARGS+=( --save_freq=1 --log_freq=1 )
fi
if [ -n "$POLICY_REPO_ID" ]; then
  EXTRA_ARGS+=( --policy.repo_id="$POLICY_REPO_ID" )
fi

if [ "$POLICY_DEVICE" = "cuda" ]; then
  VIDEO_BACKEND="torchcodec"
else
  VIDEO_BACKEND="pyav"
fi

PYTHONUNBUFFERED=1 python "$SCRIPT" \
  --resume=false \
  --output_dir="$RUN_DIR" \
  --job_name="${TASK}_${RUN_TAG}" \
  --wandb.enable=true \
  --wandb.project="$WB_PROJECT" \
  --wandb.entity="$WB_ENTITY" \
  --dataset.repo_id="$DATASET_REPO_ID" \
  "${EXTRA_ARGS[@]}" \
  --policy.type=act \
  --policy.push_to_hub="$PUSH_TO_HUB" \
  --policy.device="$POLICY_DEVICE" \
  --policy.use_amp=true \
  --use_policy_training_preset=true \
  --num_workers=8 \
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
  |& tee -a "$(dirname "$RUN_DIR")/${TASK}_${RUN_TAG}.log"

