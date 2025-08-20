#!/usr/bin/env python

import argparse
import os
from pathlib import Path

import torch

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy
from lerobot.utils.utils import get_safe_torch_device

# Reuse the exact evaluation logic from train.py to avoid drift
from lerobot.scripts.train import _evaluate_mae_on_episode


def main():
    parser = argparse.ArgumentParser(description="Compute MAE on training episode 0 from a checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to step checkpoint dir (e.g., .../checkpoints/030000)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on: cuda|cpu|mps")
    parser.add_argument("--max_frames", type=int, default=1000, help="Max contiguous frames from start of episode 0")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path for the plot")
    parser.add_argument("--dataset_repo_id", type=str, default=None, help="Optionally override dataset repo id")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir).resolve()
    pretrained_dir = ckpt_dir / "pretrained_model"
    if not pretrained_dir.is_dir():
        raise FileNotFoundError(f"Expected pretrained_model under {ckpt_dir}")

    # Load train config that was saved with the checkpoint
    cfg = TrainPipelineConfig.from_pretrained(pretrained_dir)

    # Force episode-0 evaluation and correct device
    if args.dataset_repo_id:
        cfg.dataset.repo_id = args.dataset_repo_id
    cfg.dataset.episodes = [0]
    cfg.policy.device = args.device
    cfg.policy.pretrained_path = str(pretrained_dir)

    device = get_safe_torch_device(args.device, log=True)

    # Make dataset and policy bound to its stats
    dataset = make_dataset(cfg)
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)

    # Control frame budget via the same env var used in train.py
    os.environ["LEROBOT_VAL_MAE_MAX_FRAMES"] = str(args.max_frames)

    out_png = Path(args.output) if args.output else ckpt_dir / f"mae_train_ep0_max{args.max_frames}.png"

    overall_mae, per_joint, _, _ = _evaluate_mae_on_episode(
        policy=policy,
        dataset=dataset,
        episode_idx=0,
        device=device,
        out_path=out_png,
    )

    print({
        "overall_mae": float(overall_mae),
        "per_joint_mae": [float(x) for x in (per_joint or [])],
        "plot_path": str(out_png),
    })


if __name__ == "__main__":
    main()


