#!/usr/bin/env bash
set -euo pipefail

# Allow passing VAR=VALUE arguments after the script name by exporting them
for kv in "$@"; do
  case "$kv" in
    *=*) export "$kv" ;;
  esac
done

# ---- YOU EDIT (env vars override these) ----
# Match the original run naming so RUN_DIR resolves to the same folder
TASK="${TASK:-d1_and_d2_datasets_act_224x224}"
RUN_TAG="${RUN_TAG:-checkpoint_25k_300k_steps}"

# Total steps to train to (target, not delta)
STEPS="${STEPS:-400000}"

# Device and backend (default to CPU for a safe smoke test)
POLICY_DEVICE="${POLICY_DEVICE:-cpu}"

# Optional: resume from a specific checkpoint step; default to "last"
CKPT_STEP="${CKPT_STEP:-}"

# I/O and logging defaults
SAVE_FREQ="${SAVE_FREQ:-25000}"
LOG_FREQ="${LOG_FREQ:-200}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"

# W&B (optional resume). If WB_RUN_ID is set, we will force-resume that run
WB_PROJECT="${WB_PROJECT:-lerobot}"
WB_ENTITY="${WB_ENTITY:-benfduffy-bearcover-gmbh}"
WB_RUN_ID="${WB_RUN_ID:-}"

# Repo paths
REPO="/teamspace/studios/this_studio/lerobot"
SCRIPT="$REPO/src/lerobot/scripts/train.py"
RUN_DIR="$REPO/outputs/train/${TASK}_${RUN_TAG}"
CFG="$RUN_DIR/train_config.json"

echo "TASK=$TASK"
echo "RUN_TAG=$RUN_TAG"
echo "STEPS=$STEPS"
echo "POLICY_DEVICE=$POLICY_DEVICE"
echo "CKPT_STEP=${CKPT_STEP:-<last>}"
echo "SAVE_CHECKPOINT=$SAVE_CHECKPOINT"
echo "SAVE_FREQ=$SAVE_FREQ"
echo "LOG_FREQ=$LOG_FREQ"
echo "WB_PROJECT=$WB_PROJECT"
echo "WB_ENTITY=$WB_ENTITY"
echo "WB_RUN_ID=${WB_RUN_ID:-<none>}"
echo "REPO=$REPO"
echo "SCRIPT=$SCRIPT"
echo "RUN_DIR=$RUN_DIR"
echo "CFG=$CFG"

# ---- Sanity ----
[ -f "$SCRIPT" ] || { echo "Missing $SCRIPT"; exit 1; }
[ -d "$RUN_DIR" ] || { echo "Missing $RUN_DIR"; exit 1; }
[ -f "$CFG" ] || { echo "Missing $CFG"; exit 1; }

echo "Conda env: ${CONDA_DEFAULT_ENV:-<none>}"
python --version || true

# Determine checkpoint directory to pull weights+state from
if [ -n "$CKPT_STEP" ]; then
  CKPT_DIR="$RUN_DIR/checkpoints/$CKPT_STEP"
else
  CKPT_DIR="$RUN_DIR/checkpoints/last"
fi

[ -d "$CKPT_DIR" ] || { echo "Missing $CKPT_DIR"; exit 1; }
[ -f "$CKPT_DIR/pretrained_model/model.safetensors" ] || { echo "Missing weights in $CKPT_DIR"; exit 1; }
[ -f "$CKPT_DIR/pretrained_model/config.json" ]       || { echo "Missing config.json in $CKPT_DIR"; exit 1; }

# Link (or copy fallback) the model files into RUN_DIR for train.py discovery
ln -sfn "$CKPT_DIR/pretrained_model/model.safetensors" "$RUN_DIR/model.safetensors" || cp -f "$CKPT_DIR/pretrained_model/model.safetensors" "$RUN_DIR/model.safetensors"
ln -sfn "$CKPT_DIR/pretrained_model/config.json"       "$RUN_DIR/config.json"       || cp -f "$CKPT_DIR/pretrained_model/config.json"       "$RUN_DIR/config.json"

# Ensure the fallback training_state path exists where train.py looks if no explicit arg is provided
FALLBACK_TS="$REPO/outputs/train/training_state"
mkdir -p "$FALLBACK_TS"
for f in optimizer_param_groups.json optimizer_state.safetensors rng_state.safetensors training_step.json; do
  if [ -f "$CKPT_DIR/training_state/$f" ]; then
    ln -sfn "$CKPT_DIR/training_state/$f" "$FALLBACK_TS/$f" || cp -f "$CKPT_DIR/training_state/$f" "$FALLBACK_TS/$f"
  else
    echo "Missing $CKPT_DIR/training_state/$f" && exit 1
  fi
done

# W&B resume if provided
export WANDB_DIR="$RUN_DIR/wandb"
mkdir -p "$WANDB_DIR"
if [ -n "$WB_RUN_ID" ]; then
  export WANDB_RESUME=must
  export WANDB_RUN_ID="$WB_RUN_ID"
fi

# Choose a suitable video backend
if [ "$POLICY_DEVICE" = "cuda" ]; then
  VIDEO_BACKEND="torchcodec"
else
  VIDEO_BACKEND="pyav"
fi

cd "$REPO"

PYTHONUNBUFFERED=1 python "$SCRIPT" \
  --config_path="$CFG" \
  --resume=true \
  --output_dir="$RUN_DIR" \
  --policy.device="$POLICY_DEVICE" \
  --dataset.video_backend="$VIDEO_BACKEND" \
  --wandb.enable=true \
  --wandb.project="$WB_PROJECT" \
  --wandb.entity="$WB_ENTITY" \
  ${WB_RUN_ID:+--wandb.run_id="$WB_RUN_ID"} \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --log_freq="$LOG_FREQ" \
  --save_checkpoint="$SAVE_CHECKPOINT" \
  |& tee -a "$(dirname "$RUN_DIR")/${TASK}_${RUN_TAG}.resume.log"


