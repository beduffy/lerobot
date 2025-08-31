#!/usr/bin/env bash
set -euo pipefail

# Allow passing VAR=VALUE arguments after the script name by exporting them
for kv in "$@"; do
  case "$kv" in
    *=*) export "$kv" ;;
  esac
done

# ---- YOU EDIT (env vars override these) ----
TASK="${TASK:-d4_dataset_only_2_validation_episodes}"
RUN_TAG_BASE="${RUN_TAG_BASE:-diffusion_smoke_then_train}"

# Dataset repos
DATASET_REPO_ID="${DATASET_REPO_ID:-bearlover365/pick_place_up_to_four_white_socks_varying_daylight_intensity_train}"
VAL_REPO_ID="${VAL_REPO_ID:-bearlover365/pick_place_up_to_four_white_socks_varying_daylight_intensity_validation_episode_0_and_72}"

# Common training knobs
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"

# Diffusion-specific knobs (safe defaults from configuration_diffusion.py)
POLICY_TYPE="diffusion"
DIFF_N_OBS_STEPS="${DIFF_N_OBS_STEPS:-2}"
DIFF_HORIZON="${DIFF_HORIZON:-16}"   # Must be multiple of 2**len(down_dims) (default 8)
DIFF_N_ACTION_STEPS="${DIFF_N_ACTION_STEPS:-8}"  # Must satisfy n_action_steps <= horizon - n_obs_steps + 1
DIFF_USE_AMP="${DIFF_USE_AMP:-true}"

# Hub push parameters
PUSH_TO_HUB="${PUSH_TO_HUB:-true}"
POLICY_REPO_ID="${POLICY_REPO_ID:-bearlover365/${TASK}_diffusion}"
WB_PROJECT="${WB_PROJECT:-lerobot}"
WB_ENTITY="${WB_ENTITY:-benfduffy-bearcover-gmbh}"

REPO="/teamspace/studios/this_studio/lerobot"
SCRIPT="$REPO/src/lerobot/scripts/train.py"
PYTHONPATH="${PYTHONPATH:-$REPO/src}"

# Validation evaluation limits (env overridable)
export LEROBOT_VAL_MAX_BATCHES="${LEROBOT_VAL_MAX_BATCHES:-8}"
export LEROBOT_VAL_MAX_FRAMES="${LEROBOT_VAL_MAX_FRAMES:-1024}"
# MAE validation speed knobs (env overridable)
export LEROBOT_VAL_MAE_MAX_FRAMES="${LEROBOT_VAL_MAE_MAX_FRAMES:-800}"
export LEROBOT_VAL_MAE_BATCH_SIZE="${LEROBOT_VAL_MAE_BATCH_SIZE:-64}"
# MAE eval episode indices (train/val)
export LEROBOT_TRAIN_MAE_EP_IDX="${LEROBOT_TRAIN_MAE_EP_IDX:-0}"
export LEROBOT_VAL_MAE_EP_IDX="${LEROBOT_VAL_MAE_EP_IDX:-1}"

# ---- Phases (Smoke, then Full) ----
# Toggle each phase via env vars
RUN_SMOKE="${RUN_SMOKE:-true}"
RUN_FULL="${RUN_FULL:-true}"

# Smoke test params: tiny run to catch shape/config issues fast
SMOKE_BATCH="${SMOKE_BATCH:-4}"
SMOKE_STEPS="${SMOKE_STEPS:-300}"
SMOKE_LOG_FREQ="${SMOKE_LOG_FREQ:-50}"
SMOKE_EVAL_FREQ="${SMOKE_EVAL_FREQ:-100}"
SMOKE_SAVE_CHECKPOINT="${SMOKE_SAVE_CHECKPOINT:-false}"

# Full run defaults (adjust as needed)
FULL_BATCH="${FULL_BATCH:-8}"
FULL_STEPS="${FULL_STEPS:-300000}"
FULL_LOG_FREQ="${FULL_LOG_FREQ:-200}"
FULL_EVAL_FREQ="${FULL_EVAL_FREQ:-5000}"
FULL_SAVE_CHECKPOINT="${FULL_SAVE_CHECKPOINT:-true}"
FULL_SAVE_FREQ="${FULL_SAVE_FREQ:-50000}"

# ---- Sanity ----
[ -f "$SCRIPT" ] || { echo "Missing $SCRIPT"; exit 1; }

echo "Conda env: ${CONDA_DEFAULT_ENV:-<none>}"
python --version || true

echo "DATASET_REPO_ID=$DATASET_REPO_ID"
echo "VAL_REPO_ID=$VAL_REPO_ID"
echo "TASK=$TASK"
echo "RUN_TAG_BASE=$RUN_TAG_BASE"
echo "POLICY=$POLICY_TYPE on $POLICY_DEVICE"
echo "DIFF: n_obs=$DIFF_N_OBS_STEPS horizon=$DIFF_HORIZON n_action=$DIFF_N_ACTION_STEPS use_amp=$DIFF_USE_AMP"
echo "PUSH_TO_HUB=$PUSH_TO_HUB POLICY_REPO_ID=$POLICY_REPO_ID"
echo "WB_PROJECT=$WB_PROJECT WB_ENTITY=$WB_ENTITY"
echo "REPO=$REPO SCRIPT=$SCRIPT"

echo "LEROBOT_VAL_MAX_BATCHES=$LEROBOT_VAL_MAX_BATCHES"
echo "LEROBOT_VAL_MAX_FRAMES=$LEROBOT_VAL_MAX_FRAMES"
echo "LEROBOT_TRAIN_MAE_EP_IDX=$LEROBOT_TRAIN_MAE_EP_IDX"
echo "LEROBOT_VAL_MAE_EP_IDX=$LEROBOT_VAL_MAE_EP_IDX"

# ---- W&B dirs (avoid collisions across phases) ----
mkdir -p "$REPO/outputs/wandb"
cd "$REPO"

# Build common args
COMMON_ARGS=(
  --dataset.repo_id="$DATASET_REPO_ID"
  --dataset.val_repo_id="$VAL_REPO_ID"
  --dataset.val_episodes='[0,1]'
  --dataset.video_backend="$VIDEO_BACKEND"
  --policy.type=$POLICY_TYPE
  --policy.device="$POLICY_DEVICE"
  --policy.push_to_hub="$PUSH_TO_HUB"
  --policy.repo_id="$POLICY_REPO_ID"
  --policy.use_amp="$DIFF_USE_AMP"
  --policy.n_obs_steps="$DIFF_N_OBS_STEPS"
  --policy.horizon="$DIFF_HORIZON"
  --policy.n_action_steps="$DIFF_N_ACTION_STEPS"
)

# 1) Smoke test phase
if [ "$RUN_SMOKE" = "true" ]; then
  RUN_TAG="${RUN_TAG_BASE}_smoke"
  RUN_DIR="$REPO/outputs/train/${TASK}_${RUN_TAG}"
  if [ -d "$RUN_DIR" ]; then
    SUFFIX=$(date +%Y%m%d_%H%M%S)
    RUN_TAG="${RUN_TAG}_${SUFFIX}"
    RUN_DIR="$REPO/outputs/train/${TASK}_${RUN_TAG}"
  fi
  mkdir -p "$(dirname "$RUN_DIR")"
  export WANDB_DIR="$REPO/outputs/wandb/${TASK}_${RUN_TAG}"
  export WANDB_RUN_ID="${WANDB_RUN_ID:-diff_smoke_$(date +%s)}"

  echo "[SMOKE] RUN_DIR=$RUN_DIR"
  PYTHONUNBUFFERED=1 PYTHONPATH="$PYTHONPATH" python "$SCRIPT" \
    --resume=false \
    --output_dir="$RUN_DIR" \
    --job_name="${TASK}_${RUN_TAG}" \
    --wandb.enable=true \
    --wandb.project="$WB_PROJECT" \
    --wandb.entity="$WB_ENTITY" \
    --num_workers="$NUM_WORKERS" \
    --batch_size="$SMOKE_BATCH" \
    --save_checkpoint="$SMOKE_SAVE_CHECKPOINT" \
    --steps="$SMOKE_STEPS" \
    --log_freq="$SMOKE_LOG_FREQ" \
    --eval_freq="$SMOKE_EVAL_FREQ" \
    "${COMMON_ARGS[@]}" \
    |& tee -a "$(dirname "$RUN_DIR")/${TASK}_${RUN_TAG}.log"
fi

# 2) Full run phase
if [ "$RUN_FULL" = "true" ]; then
  RUN_TAG="${RUN_TAG_BASE}_full"
  RUN_DIR="$REPO/outputs/train/${TASK}_${RUN_TAG}"
  if [ -d "$RUN_DIR" ]; then
    SUFFIX=$(date +%Y%m%d_%H%M%S)
    RUN_TAG="${RUN_TAG}_${SUFFIX}"
    RUN_DIR="$REPO/outputs/train/${TASK}_${RUN_TAG}"
  fi
  mkdir -p "$(dirname "$RUN_DIR")"
  export WANDB_DIR="$REPO/outputs/wandb/${TASK}_${RUN_TAG}"
  export WANDB_RUN_ID="${WANDB_RUN_ID:-diff_full_$(date +%s)}"

  echo "[FULL] RUN_DIR=$RUN_DIR"
  PYTHONUNBUFFERED=1 PYTHONPATH="$PYTHONPATH" python "$SCRIPT" \
    --resume=false \
    --output_dir="$RUN_DIR" \
    --job_name="${TASK}_${RUN_TAG}" \
    --wandb.enable=true \
    --wandb.project="$WB_PROJECT" \
    --wandb.entity="$WB_ENTITY" \
    --num_workers="$NUM_WORKERS" \
    --batch_size="$FULL_BATCH" \
    --save_checkpoint="$FULL_SAVE_CHECKPOINT" \
    --save_freq="$FULL_SAVE_FREQ" \
    --steps="$FULL_STEPS" \
    --log_freq="$FULL_LOG_FREQ" \
    --eval_freq="$FULL_EVAL_FREQ" \
    "${COMMON_ARGS[@]}" \
    |& tee -a "$(dirname "$RUN_DIR")/${TASK}_${RUN_TAG}.log"
fi


