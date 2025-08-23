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
# Use merged 100-episode dataset by default
DATASET_REPO_ID="${DATASET_REPO_ID:-bearlover365/pick_place_white_socks_merged_100_episodes}"
# Optional offline validation repo (held-out set)
VAL_REPO_ID="${VAL_REPO_ID:-bearlover365/pick_place_one_white_sock_black_out_blinds_validation_episode0}"
LOG_FREQ="${LOG_FREQ:-200}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
SAVE_FREQ="${SAVE_FREQ:-50000}"
EVAL_FREQ="${EVAL_FREQ:-5000}"
# Hub push parameters (optional)
PUSH_TO_HUB="${PUSH_TO_HUB:-true}"
POLICY_REPO_ID="${POLICY_REPO_ID:-bearlover365/d1_d2_merged_normal_resolution}"
# W&B defaults (enabled by default like other scripts)
WB_PROJECT="${WB_PROJECT:-lerobot}"
WB_ENTITY="${WB_ENTITY:-benfduffy-bearcover-gmbh}"
REPO="/teamspace/studios/this_studio/lerobot"
SCRIPT="$REPO/src/lerobot/scripts/train.py"
PYTHONPATH="${PYTHONPATH:-$REPO/src}"
RUN_DIR="$REPO/outputs/train/${TASK}_${RUN_TAG}"

# Validation evaluation limits (env overridable)
export LEROBOT_VAL_MAX_BATCHES="${LEROBOT_VAL_MAX_BATCHES:-8}"
export LEROBOT_VAL_MAX_FRAMES="${LEROBOT_VAL_MAX_FRAMES:-1024}"
# MAE validation speed knobs (env overridable)
export LEROBOT_VAL_MAE_MAX_FRAMES="${LEROBOT_VAL_MAE_MAX_FRAMES:-800}"
export LEROBOT_VAL_MAE_BATCH_SIZE="${LEROBOT_VAL_MAE_BATCH_SIZE:-64}"
# MAE eval episode indices (train/val)
export LEROBOT_TRAIN_MAE_EP_IDX="${LEROBOT_TRAIN_MAE_EP_IDX:-50}"
export LEROBOT_VAL_MAE_EP_IDX="${LEROBOT_VAL_MAE_EP_IDX:-0}"

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

echo "DATASET_REPO_ID=$DATASET_REPO_ID"
echo "VAL_REPO_ID=$VAL_REPO_ID"
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
echo "WB_PROJECT=$WB_PROJECT"
echo "WB_ENTITY=$WB_ENTITY"
echo "REPO=$REPO"
echo "SCRIPT=$SCRIPT"
echo "RUN_DIR=$RUN_DIR"

echo "LEROBOT_VAL_MAX_BATCHES=$LEROBOT_VAL_MAX_BATCHES"
echo "LEROBOT_VAL_MAX_FRAMES=$LEROBOT_VAL_MAX_FRAMES"
echo "LEROBOT_TRAIN_MAE_EP_IDX=$LEROBOT_TRAIN_MAE_EP_IDX"
echo "LEROBOT_VAL_MAE_EP_IDX=$LEROBOT_VAL_MAE_EP_IDX"

# ---- W&B (new run) ----
# Keep W&B files outside RUN_DIR to avoid collisions
export WANDB_DIR="${WANDB_DIR:-$REPO/outputs/wandb/${TASK}_${RUN_TAG}}"
# Ensure parent output dir exists for logging
mkdir -p "$(dirname "$RUN_DIR")"
cd "$REPO"

# Build optional args
EXTRA_ARGS=()
if [ -n "$VAL_REPO_ID" ]; then
  EXTRA_ARGS+=( --dataset.val_repo_id="$VAL_REPO_ID" )
fi
if [ -n "$POLICY_REPO_ID" ]; then
  EXTRA_ARGS+=( --policy.repo_id="$POLICY_REPO_ID" )
fi

VIDEO_BACKEND="pyav"

# removed
#  --policy.use_amp=true \
#  --use_policy_training_preset=true \
# TODO if T4 8 workers is bad so automatically detect how many works. L40s can have 16 workers. 

PYTHONUNBUFFERED=1 PYTHONPATH="$PYTHONPATH" python "$SCRIPT" \
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
  --num_workers=8 \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --log_freq="$LOG_FREQ" \
  --save_checkpoint="$SAVE_CHECKPOINT" \
  --save_freq="$SAVE_FREQ" \
  --eval_freq=$EVAL_FREQ \
  --dataset.video_backend="$VIDEO_BACKEND" \
  "${EXTRA_ARGS[@]}" \
  |& tee -a "$(dirname "$RUN_DIR")/${TASK}_${RUN_TAG}.log"

