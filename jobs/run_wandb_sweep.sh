#!/usr/bin/env bash
set -euo pipefail

# Accept VAR=VALUE overrides
for kv in "$@"; do
  case "$kv" in
    *=*) export "$kv" ;;
  esac
done

REPO="/teamspace/studios/this_studio/lerobot"
SWEEP_YAML="${SWEEP_YAML:-$REPO/jobs/wandb_sweep_quick_200.yaml}"
COUNT="${COUNT:-0}"

echo "Conda env: ${CONDA_DEFAULT_ENV:-<none>}"
python --version || true

cd "$REPO"

# Ensure WANDB dir exists per run
export WANDB_DIR="$REPO/outputs/wandb_sweeps"
mkdir -p "$WANDB_DIR"

echo "Creating W&B sweep from: $SWEEP_YAML"
SWEEP_OUT=$(wandb sweep --entity "${WB_ENTITY:-benfduffy-bearcover-gmbh}" --project "${WB_PROJECT:-lerobot}" "$SWEEP_YAML" | tee /dev/stderr)

# Try to extract full sweep path first (entity/project/id)
SWEEP_PATH=$(echo "$SWEEP_OUT" | grep -oE 'wandb agent [A-Za-z0-9_-]+/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+' | awk '{print $3}' | tail -n1)

# Fallback 1: extract just the ID and build the path
if [ -z "${SWEEP_PATH}" ]; then
  SWEEP_ID=$(echo "$SWEEP_OUT" | awk '/Creating sweep with ID:/ {print $NF; exit}')
  if [ -n "${SWEEP_ID:-}" ]; then
    SWEEP_PATH="${WB_ENTITY:-benfduffy-bearcover-gmbh}/${WB_PROJECT:-lerobot}/$SWEEP_ID"
  fi
fi

# Fallback 2: parse from the sweeps URL
if [ -z "${SWEEP_PATH}" ]; then
  SWEEP_ID=$(echo "$SWEEP_OUT" | grep -oE 'sweeps/[A-Za-z0-9_-]+' | head -n1 | sed 's@.*/@@')
  if [ -n "${SWEEP_ID:-}" ]; then
    SWEEP_PATH="${WB_ENTITY:-benfduffy-bearcover-gmbh}/${WB_PROJECT:-lerobot}/$SWEEP_ID"
  fi
fi

if [ -z "${SWEEP_PATH}" ]; then
  echo "Failed to parse sweep ID/path from wandb output" >&2
  exit 1
fi

echo "Sweep path: $SWEEP_PATH"

echo "Launching agent(s) with count=$COUNT"
WANDB_PROJECT="${WB_PROJECT:-lerobot}" WANDB_ENTITY="${WB_ENTITY:-benfduffy-bearcover-gmbh}" \
wandb agent --count "$COUNT" "$SWEEP_PATH"


