#!/usr/bin/env bash

set -euo pipefail

# 1) Create the HF-format folder for MODEL_STEPS_NUMBER
# 2) Download the weights from W&B
# 3) Move weights into the HF folder
# 4) Add configs from your run (step-invariant)
# 5) Upload in the same format you normally do

print_usage() {
  cat <<'USAGE'
Usage:
  convert_wandb_to_hugging_face.sh \
    --steps 90000 \
    --run-dir /teamspace/studios/this_studio/lerobot/outputs/train/pick_place_one_white_sock_black_out_blinds_1 \
    --hf-user <hf_user> --task-name <repo_name> \
    [--artifact <entity/project/artifact_name:version>] \
    [--entity benfduffy-bearcover-gmbh] [--project lerobot] [--policy act] [--seed 1000] [--dataset bearlover365/pick_place_one_white_sock_black_out_blinds] \
    [--tmp-dir /path/to/tmp] [--skip-upload]

Notes:
- If --artifact is NOT provided, the artifact name will be constructed as:
    policy_<policy>-dataset_<dataset_sanitized>-seed_<seed>-<step_id>
  and fetched from <entity>/<project>.
- <step_id> is zero-padded to 6 digits (e.g., 90000 -> 090000).
- The script expects config.json and train_config.json to be present in --run-dir.
USAGE
}


die() { echo "Error: $*" >&2; exit 1; }


# Defaults
STEPS="${MODEL_STEPS_NUMBER:-}"
RUN_DIR=""
HF_USER="${HF_USER:-}"
TASK_NAME="${task_name:-}"
HF_REPO_ID="${HF_REPO_ID:-}"
ARTIFACT_REF=""
ENTITY="${WB_ENTITY:-benfduffy-bearcover-gmbh}"
PROJECT="${WB_PROJECT:-lerobot}"
POLICY="act"
SEED="1000"
DATASET_REPO_ID=""
TMP_DIR=""
SKIP_UPLOAD="false"


while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps|--step)
      STEPS="$2"; shift 2 ;;
    --run-dir)
      RUN_DIR="$2"; shift 2 ;;
    --hf-user)
      HF_USER="$2"; shift 2 ;;
    --task-name)
      TASK_NAME="$2"; shift 2 ;;
    --hf-repo-id)
      HF_REPO_ID="$2"; shift 2 ;;
    --artifact)
      ARTIFACT_REF="$2"; shift 2 ;;
    --entity)
      ENTITY="$2"; shift 2 ;;
    --project)
      PROJECT="$2"; shift 2 ;;
    --policy)
      POLICY="$2"; shift 2 ;;
    --seed)
      SEED="$2"; shift 2 ;;
    --dataset)
      DATASET_REPO_ID="$2"; shift 2 ;;
    --tmp-dir)
      TMP_DIR="$2"; shift 2 ;;
    --skip-upload)
      SKIP_UPLOAD="true"; shift 1 ;;
    -h|--help)
      print_usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2; print_usage; exit 2 ;;
  esac
done


[[ -n "$STEPS" ]] || die "--steps is required (or set MODEL_STEPS_NUMBER env var)"
[[ -n "$RUN_DIR" ]] || die "--run-dir is required"

[[ -d "$RUN_DIR" ]] || die "Run dir not found: $RUN_DIR"
[[ -f "$RUN_DIR/config.json" ]] || die "Missing $RUN_DIR/config.json"
[[ -f "$RUN_DIR/train_config.json" ]] || die "Missing $RUN_DIR/train_config.json"

# Resolve repo id
if [[ -z "${HF_REPO_ID}" ]]; then
  [[ -n "$HF_USER" ]] || die "--hf-user is required when --hf-repo-id is not provided"
  [[ -n "$TASK_NAME" ]] || die "--task-name is required when --hf-repo-id is not provided"
  HF_REPO_ID="${HF_USER}/${TASK_NAME}"
fi

# Validate tools
command -v wandb >/dev/null 2>&1 || die "wandb CLI not found in PATH"
command -v huggingface-cli >/dev/null 2>&1 || echo "Warning: huggingface-cli not found; will skip upload if not installed"

# Step id padded to 6 digits, used by W&B artifacts and local checkpoints
STEP_ID=$(printf "%06d" "$STEPS")

# Build artifact ref if not provided
if [[ -z "$ARTIFACT_REF" ]]; then
  [[ -n "$DATASET_REPO_ID" ]] || die "--dataset is required when --artifact is not provided"
  DATASET_SAFE=${DATASET_REPO_ID//\//_}
  ARTIFACT_NAME="policy_${POLICY}-dataset_${DATASET_SAFE}-seed_${SEED}-${STEP_ID}:v0"
  ARTIFACT_REF="${ENTITY}/${PROJECT}/${ARTIFACT_NAME}"
fi

echo "Using artifact: $ARTIFACT_REF"
echo "Target HF repo: $HF_REPO_ID"

# Create HF-format folder for the checkpoint
TARGET_DIR="$RUN_DIR/checkpoints/${STEP_ID}/pretrained_model"
mkdir -p "$TARGET_DIR"

# Choose temp dir for W&B download
if [[ -z "$TMP_DIR" ]]; then
  TMP_DIR="/teamspace/studios/this_studio/lerobot/tmp_wandb_${STEP_ID}"
fi
mkdir -p "$TMP_DIR"

# Download the weights from W&B
wandb artifact get --type model "$ARTIFACT_REF" --root "$TMP_DIR"

# Move weights into the HF folder
[[ -f "$TMP_DIR/model.safetensors" ]] || die "model.safetensors not found at $TMP_DIR"
cp "$TMP_DIR/model.safetensors" "$TARGET_DIR/model.safetensors"

# Add configs from your run (step-invariant)
cp "$RUN_DIR/config.json" "$TARGET_DIR/config.json"
cp "$RUN_DIR/train_config.json" "$TARGET_DIR/train_config.json"

# Upload in the same format you normally do
if [[ "$SKIP_UPLOAD" == "true" ]]; then
  echo "Skipping upload as requested. Reconstructed folder at: $TARGET_DIR"
  exit 0
fi

if command -v huggingface-cli >/dev/null 2>&1; then
  echo "Uploading $TARGET_DIR to $HF_REPO_ID ..."
  huggingface-cli upload "$HF_REPO_ID" "$TARGET_DIR" --repo-type=model --commit-message "${STEP_ID} checkpoint reconstructed from W&B"
else
  echo "huggingface-cli not found. Reconstruction complete at: $TARGET_DIR"
fi