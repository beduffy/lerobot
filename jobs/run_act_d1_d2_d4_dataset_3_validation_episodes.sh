#!/usr/bin/env bash
set -euo pipefail

# Allow passing VAR=VALUE arguments after the script name by exporting them
for kv in "$@"; do
  case "$kv" in
    *=*) export "$kv" ;;
  esac
done 

# ---- YOU EDIT (env vars override these) ----
# what needs to change per run
# Use by 174 episodes
# dataset.num_frames=249369 (249K)
# TASK="${TASK:-pick_place_one_white_sock_black_out_blinds}"
TASK="${TASK:-d1_d2_d4_dataset_only_3_validation_episodes}"
RUN_TAG="${RUN_TAG:-checkpoint_50k_500k_steps}"
DATASET_REPO_ID="${DATASET_REPO_ID:-bearlover365/d1_d2_d4_white_socks_train}"
# Optional offline validation repo (held-out set)
VAL_REPO_ID="${VAL_REPO_ID:-bearlover365/d1_d2_d4_validation_dataset_3_episodes}"
POLICY_REPO_ID="${POLICY_REPO_ID:-bearlover365/d1_d2_d4_dataset_3_validation_episodes}"

# ------ Resume controls (env overridable)
RESUME="${RESUME:-false}"
# CHECKPOINT_DIR="${CHECKPOINT_DIR:-/teamspace/studios/this_studio/lerobot/outputs/train/d4_dataset_only_2_validation_episodes/checkpoints/050000}"
# TODO had to do two pretrained_model
# CHECKPOINT_DIR="${CHECKPOINT_DIR:-/teamspace/studios/this_studio/lerobot/outputs/train/d4_dataset_only_2_validation_episodes/checkpoints/050000/pretrained_model}"
# export WANDB_RUN_ID="${WANDB_RUN_ID:-ms2tqgeo}"

# MORE FIXED -----
BATCH="${BATCH:-8}"
STEPS="${STEPS:-500000}"

LOG_FREQ="${LOG_FREQ:-200}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
SAVE_FREQ="${SAVE_FREQ:-50000}"
EVAL_FREQ="${EVAL_FREQ:-5000}"
# Hub push parameters (optional)
PUSH_TO_HUB="${PUSH_TO_HUB:-true}"
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
export LEROBOT_TRAIN_MAE_EP_IDX="${LEROBOT_TRAIN_MAE_EP_IDX:-0}"
export LEROBOT_VAL_MAE_EP_IDX="${LEROBOT_VAL_MAE_EP_IDX:-1}"

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
echo "RESUME=$RESUME"
# echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
# echo "WANDB_RUN_ID=${WANDB_RUN_ID}"

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
# # Auto-detect GPU to adjust num_workers. T4s have less memory and benefit from fewer workers.
# NUM_WORKERS=8 # Default for beefier GPUs
# if command -v nvidia-smi &> /dev/null && nvidia-smi --query-gpu=gpu_name --format=csv,noheader | grep -q "T4"; then
#   echo "T4 GPU detected. Overriding num_workers to 4."
#   NUM_WORKERS=4
# fi
NUM_WORKERS=4

  # --dataset.val_episodes='[0,72]' \


if [ "$RESUME" = "true" ]; then
  [ -n "$CHECKPOINT_DIR" ] || { echo "RESUME=true but CHECKPOINT_DIR is not set"; exit 1; }
  [ -d "$CHECKPOINT_DIR" ] || { echo "Missing checkpoint dir: $CHECKPOINT_DIR"; exit 1; }
  [ -n "${WANDB_RUN_ID:-}" ] || { echo "RESUME=true but WANDB_RUN_ID is not set"; exit 1; }

  PRETRAINED_DIR="$CHECKPOINT_DIR/pretrained_model"
  CFG="$PRETRAINED_DIR/train_config.json"
  [ -f "$CFG" ] || { echo "Missing train config: $CFG"; exit 1; }

  # Derive original run dir from checkpoint path (…/run_dir/checkpoints/NNNNNN)
  PARENT_1="$(dirname "$CHECKPOINT_DIR")"
  PARENT_2="$(dirname "$PARENT_1")"
  RUN_DIR="$PARENT_2"
  echo "Derived RUN_DIR from checkpoint: $RUN_DIR"

  export WANDB_RESUME=must

  PYTHONUNBUFFERED=1 PYTHONPATH="$PYTHONPATH" \
  python -m lerobot.scripts.train \
    --config_path="$CFG" \
    --resume=true \
    --output_dir="$RUN_DIR" \
    --dataset.video_backend="$VIDEO_BACKEND" \
    --wandb.enable=true \
    --wandb.project="$WB_PROJECT" \
    --wandb.entity="$WB_ENTITY" \
    --wandb.run_id="$WANDB_RUN_ID" \
    --eval_freq=$EVAL_FREQ \
    --save_freq="$SAVE_FREQ" \
    --steps="$STEPS" \
    --dataset.val_episodes='[0,1,2]' \
    --policy.device="$POLICY_DEVICE" \
    "${EXTRA_ARGS[@]}" \
    |& tee -a "$(dirname "$RUN_DIR")/${TASK}_${RUN_TAG}.log"

else

  PYTHONUNBUFFERED=1 PYTHONPATH="$PYTHONPATH" python "$SCRIPT" \
    --resume=false \
    --output_dir="$RUN_DIR" \
    --job_name="${TASK}_${RUN_TAG}" \
    --wandb.enable=true \
    --wandb.project="$WB_PROJECT" \
    --wandb.entity="$WB_ENTITY" \
    --dataset.repo_id="$DATASET_REPO_ID" \
    --dataset.val_episodes='[0,1,2]' \
    --policy.type=act \
    --policy.push_to_hub="$PUSH_TO_HUB" \
    --policy.device="$POLICY_DEVICE" \
    --num_workers="$NUM_WORKERS" \
    --batch_size="$BATCH" \
    --save_checkpoint="$SAVE_CHECKPOINT" \
    --steps="$STEPS" \
    --log_freq="$LOG_FREQ" \
    --save_freq="$SAVE_FREQ" \
    --eval_freq=$EVAL_FREQ \
    --dataset.video_backend="$VIDEO_BACKEND" \
    "${EXTRA_ARGS[@]}" \
    |& tee -a "$(dirname "$RUN_DIR")/${TASK}_${RUN_TAG}.log"
fi

