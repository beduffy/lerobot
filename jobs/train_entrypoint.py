#!/usr/bin/env python
import argparse
import math
import os
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Translate sweep params to train.py CLI with computed LRs")
    parser.add_argument("--base_lr", type=float, default=3e-5)
    parser.add_argument("--base_bs", type=int, default=8)
    parser.add_argument("--lr_scale", type=float, default=1.0)
    parser.add_argument("--backbone_lr_scale", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--target_samples", type=int, default=None)
    # Parse known args; keep unknown to forward
    known, unknown = parser.parse_known_args()

    # Compute main learning rate using linear scaling rule
    scaled_lr = known.base_lr * (float(known.batch_size) / float(known.base_bs)) * known.lr_scale
    backbone_lr = scaled_lr * known.backbone_lr_scale

    # Build command for train.py
    repo_root = "/teamspace/studios/this_studio/lerobot"
    script = f"{repo_root}/src/lerobot/scripts/train.py"

    cmd = [
        sys.executable,
        script,
        "--resume=false",
        "--wandb.enable=true",
        f"--batch_size={known.batch_size}",
        f"--policy.optimizer_lr={scaled_lr}",
        f"--policy.optimizer_lr_backbone={backbone_lr}",
        "--policy.push_to_hub=false",
    ]

    if known.target_samples is not None:
        cmd.append(f"--target_samples={known.target_samples}")

    # Forward remaining CLI args from sweep (dataset.*, eval/log/save settings, etc.)
    # Ensure they are '--key=value' or '--flag' tokens; pass through verbatim
    for tok in unknown:
        # Normalize quotes in forwarded args for safety
        if isinstance(tok, str):
            tok = tok.strip()
        if not tok:
            continue
        # Drop any explicit LR overrides; we compute them above
        lower = tok.lower()
        if lower.startswith("--policy.optimizer_lr=") or lower.startswith("--policy.optimizer_lr_backbone="):
            continue
        cmd.append(tok)

    os.environ.setdefault("WANDB_SILENT", "true")
    # Print the final command for debugging
    print("Launching:", " ".join(shlex.quote(x) for x in cmd))
    proc = subprocess.Popen(cmd)
    proc.wait()
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()


