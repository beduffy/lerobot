#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any
from copy import deepcopy

try:
    import matplotlib.pyplot as plt
    import numpy as np
except Exception:
    # not optional, hard crash
    raise ImportError("matplotlib and numpy are required for plotting")

import torch
from termcolor import colored
import torch.nn.functional as F  # noqa: N812
import numpy as np
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.scripts.eval import eval_policy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.utils.wandb_utils import WandBLogger
from huggingface_hub import HfApi



def _ensure_minimal_stats(ds_meta) -> None:
    """Ensure required stats entries exist for non-visual features used by normalization.

    Fills missing entries with zero-mean and unit-std tensors of the right shape.
    """
    try:
        features = getattr(ds_meta, "features", {})
        stats = getattr(ds_meta, "stats", {})
        for key, ft in features.items():
            if ft.get("dtype") in ["image", "video"]:
                # Visual stats handled elsewhere (ImageNet injection or dataset-provided)
                continue
            # Keys commonly required by policies: 'observation.state', 'action'
            if key not in stats:
                stats[key] = {}
            shape = ft.get("shape") or ()
            # Normalize shapes to 1D list length when applicable
            if isinstance(shape, (list, tuple)) and len(shape) == 1:
                dim = int(shape[0])
                zeros = np.zeros((dim,), dtype=np.float32)
                ones = np.ones((dim,), dtype=np.float32)
            else:
                zeros = np.zeros((1,), dtype=np.float32)
                ones = np.ones((1,), dtype=np.float32)
            for k in ("mean", "std", "min", "max"):
                if k not in stats[key]:
                    stats[key][k] = zeros if k in ("mean", "min",) else ones
            if "count" not in stats[key]:
                stats[key]["count"] = np.array([1], dtype=np.int64)
        ds_meta.stats = stats
    except Exception:
        # Do not fail hard; normalization code will raise if truly required
        pass

def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


def _compute_offline_val_loss(policy: PreTrainedPolicy, dataset, device: torch.device, batch_size: int, num_workers: int) -> float:
    """Compute average loss on a held-out dataset (no grad). L1 loss so not really comparable to training loss."""
    # TODO understand better
    policy.eval()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    losses = []
    max_batches = int(os.environ.get("LEROBOT_VAL_MAX_BATCHES", "2"))
    max_frames = int(os.environ.get("LEROBOT_VAL_MAX_FRAMES", "256"))
    processed_batches = 0
    processed_frames = 0
    with torch.no_grad(), torch.autocast(device_type=device.type) if policy.config.use_amp else nullcontext():
        for batch in dataloader:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")
            # ACT expects sequence targets during training; compute a simple 1-step L1 on predicted first action
            if getattr(policy, "name", None) == "act":
                with torch.no_grad():
                    pred_chunk = policy.predict_action_chunk(batch)  # (B, S, A)
                pred = pred_chunk[:, :1]  # (B, 1, A)
                gt = batch.get("action")
                if gt is None:
                    processed_batches += 1
                    continue
                # Align target to first-step action shape when sequences are provided
                if gt.ndim == 3:  # (B, S, A) -> (B, 1, A)
                    gt = gt[:, :1, :]
                elif gt.ndim == 2:  # (B, A) -> (B, 1, A)
                    gt = gt.unsqueeze(1)
                l1 = F.l1_loss(pred, gt, reduction="mean")
                losses.append(l1.item())
            else:
                loss, _ = policy.forward(batch)
                losses.append(loss.item())
            processed_batches += 1
            processed_frames += batch_size
            if processed_batches >= max_batches or processed_frames >= max_frames:
                break
    policy.train()
    return float(sum(losses) / max(1, len(losses)))


def _plot_action_state_trajectories(
    policy: PreTrainedPolicy,
    dataset,
    device: torch.device,
    out_path,
    *,
    preds_np: np.ndarray | None = None,
    gt_np: np.ndarray | None = None,
):
    """Plot Pred vs GT for a few dims. If arrays are provided, reuse them to avoid extra decode/forward."""
    if preds_np is None or gt_np is None:
        return

    T, D = gt_np.shape[0], gt_np.shape[-1]

    fig_h = 2.0 * min(6, D)
    fig, axes = plt.subplots(min(6, D), 1, figsize=(10, fig_h), squeeze=False)
    axes = axes.flatten()
    xs = np.arange(T)
    for d in range(min(6, D)):
        ax = axes[d]
        ax.plot(xs, gt_np[:, d], 'r--', label='GT' if d == 0 else None)
        ax.plot(xs, preds_np[:, d], 'b-', label='Pred' if d == 0 else None)
        ax.set_title(f"Dim {d}")
        ax.set_xlabel("Time Step")
    if D > 0:
        axes[0].legend(loc='best')
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _evaluate_mae_on_episode(
    policy: PreTrainedPolicy,
    dataset,
    episode_idx: int,
    device: torch.device,
    out_path,
) -> tuple[float, list[float] | None, np.ndarray | None, np.ndarray | None]:
    """Evaluate MAE on a single episode using 1-step actions and optionally save a plot.

    Returns overall_mae and per_joint_mae (or None if plot disabled).
    """
    # TODO understand better inputs and outputs

    # Build index range for the episode (subsample for speed on CPU), handling multi-dataset
    source_dataset = dataset
    try:
        ep_from = dataset.episode_data_index["from"][episode_idx].item()
        ep_to = dataset.episode_data_index["to"][episode_idx].item()
        full_indices = list(range(ep_from, ep_to))
        get_sample = lambda i: source_dataset[i]
    except Exception:
        if hasattr(dataset, "_datasets") and len(getattr(dataset, "_datasets")) > 0:
            source_dataset = dataset._datasets[0]
            ep_from = source_dataset.episode_data_index["from"][episode_idx].item()
            ep_to = source_dataset.episode_data_index["to"][episode_idx].item()
            full_indices = list(range(ep_from, ep_to))
            get_sample = lambda i: source_dataset[i]
        else:
            raise AttributeError("Dataset does not expose episode indices for MAE computation.")
    max_frames_env = int(os.environ.get("LEROBOT_VAL_MAE_MAX_FRAMES", "200"))
    # Use a contiguous window from the start of the episode to preserve temporal coherence
    # for chunk-based policies. Subsampling with stride breaks action chunk semantics.
    if max_frames_env > 0 and len(full_indices) > max_frames_env:
        indices = full_indices[: max_frames_env]
    else:
        indices = full_indices

    # Batched fast path (when allowed), else frame-by-frame fallback
    preds = []
    gts = []
    policy.eval()
    mae_bs_env = int(os.environ.get("LEROBOT_VAL_MAE_BATCH_SIZE", "0"))
    mae_bs = mae_bs_env if mae_bs_env > 0 else (64 if device.type == "cuda" else 1)
    # Diffusion requires sequential evaluation to warm up internal queues
    if getattr(policy, "name", None) == "diffusion":
        mae_bs = 1

    if mae_bs > 1:
        subset = torch.utils.data.Subset(source_dataset, indices)
        dl = torch.utils.data.DataLoader(
            subset,
            batch_size=mae_bs,
            shuffle=False,
            num_workers=2,
            pin_memory=device.type == "cuda",
        )
        autocast_ctx = torch.amp.autocast(device_type="cuda") if device.type == "cuda" else nullcontext()
        with torch.inference_mode(), autocast_ctx:
            for batch in dl:
                inp = {}
                for key, val in batch.items():
                    if key.startswith("observation.") and isinstance(val, torch.Tensor):
                        inp[key] = val.to(device, non_blocking=device.type == "cuda")
                if not inp:
                    continue
                # For diffusion, always use select_action to leverage internal queues
                if getattr(policy, "name", None) == "diffusion":
                    out = policy.select_action(inp)
                    pred = out[:, 0, :] if out.ndim == 3 else out
                elif hasattr(policy, "predict_action_chunk"):
                    chunk = policy.predict_action_chunk(inp)
                    pred = chunk[:, 0, :]
                else:
                    out = policy.select_action(inp)
                    pred = out[:, 0, :] if out.ndim == 3 else out
                gt = batch.get("action")
                if isinstance(gt, torch.Tensor):
                    # Align GT to first-step action shape when sequences are provided (B, S, A)
                    if gt.ndim == 3:
                        gt = gt[:, 0, :]
                    preds.append(pred.detach().cpu())
                    gts.append(gt.detach().cpu())
    else:
        # Reset policy state for new episode to clear queues
        if hasattr(policy, "reset"):
            try:
                policy.reset()
            except Exception:
                pass
        with torch.no_grad():
            for idx in indices:
                sample = get_sample(idx)
                inp = {}
                for key, val in sample.items():
                    if key.startswith("observation.") and isinstance(val, torch.Tensor):
                        t = val
                        # For diffusion/select_action we need a single observation step (latest)
                        try:
                            if "image" in key:
                                # Handle shapes: (S,C,H,W) or (B,S,C,H,W) -> select last S
                                if t.ndim == 5:  # (B,S,C,H,W)
                                    t = t[:, -1]
                                elif t.ndim == 4:  # (S,C,H,W)
                                    t = t[-1]
                            else:
                                # Handle state/env_state shapes: (S,D) or (B,S,D) -> select last S
                                if t.ndim == 3:  # (B,S,D)
                                    t = t[:, -1]
                                elif t.ndim == 2:  # (S,D)
                                    t = t[-1]
                        except Exception:
                            pass
                        # Ensure batch dimension exists
                        if t.ndim in (1, 3):
                            t = t.unsqueeze(0)
                        inp[key] = t.to(device)
                if not inp:
                    continue
                # For diffusion, always use select_action to ensure queues are handled
                if getattr(policy, "name", None) == "diffusion":
                    pred = policy.select_action(inp)
                elif hasattr(policy, "predict_action_chunk"):
                    pred_chunk = policy.predict_action_chunk(inp)
                    pred = pred_chunk[:, 0, :]
                else:
                    pred = policy.select_action(inp)
                gt = sample.get("action")
                if isinstance(gt, torch.Tensor):
                    if gt.ndim == 1:
                        gt = gt.unsqueeze(0)
                    preds.append(pred[0].detach().cpu())
                    gts.append(gt[0].detach().cpu())

    if len(preds) == 0:
        return float("nan"), None, None, None

    # Concatenate mini-batches when using batched path
    if isinstance(preds[0], torch.Tensor) and preds[0].dim() == 2:
        preds_t = torch.cat(preds, dim=0)
        gts_t = torch.cat(gts, dim=0)
    else:
        preds_t = torch.stack(preds, dim=0)
        gts_t = torch.stack(gts, dim=0)
    per_joint_mae_t = torch.mean(torch.abs(preds_t - gts_t), dim=0)
    overall_mae = float(torch.mean(per_joint_mae_t).item())
    per_joint_mae = per_joint_mae_t.tolist()

    # Plot if matplotlib available
    if plt is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        num_joints = preds_t.shape[1]
        # Share x-axis to keep time aligned; annotate overall MAE in suptitle
        fig, axes = plt.subplots(num_joints, 1, figsize=(11, max(2.5, 2.2 * num_joints)), sharex=True)
        if num_joints == 1:
            axes = [axes]
        xs = np.arange(preds_t.shape[0])
        for j in range(num_joints):
            ax = axes[j]
            ax.plot(xs, gts_t[:, j].numpy(), label="GT", color="#d62728", linestyle="--", linewidth=1.5)
            ax.plot(xs, preds_t[:, j].numpy(), label="Pred", color="#1f77b4", linewidth=1.5)
            ax.set_ylabel(f"Dim {j}")
            ax.grid(True, alpha=0.25)
            # Per-joint MAE in the corner
            ax.text(0.01, 0.92, f"MAE={per_joint_mae[j]:.4f}", transform=ax.transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, linewidth=0.5))
            if j == 0:
                ax.legend(loc="best")
        axes[-1].set_xlabel("Time Step")
        fig.suptitle(f"Pred vs GT (Episode {episode_idx})  Overall MAE={overall_mae:.4f}")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out_path, dpi=160)
        plt.close(fig)

    return overall_mae, per_joint_mae, preds_t.numpy(), gts_t.numpy()


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    # Silence torchvision video deprecation spam
    import warnings as _warnings
    _warnings.filterwarnings(
        "ignore",
        message="The video decoding and encoding capabilities of torchvision are deprecated",
    )

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)
    _ensure_minimal_stats(dataset.meta)
    val_dataset = None
    if getattr(cfg.dataset, "val_episodes", None) or getattr(cfg.dataset, "val_repo_id", None):
        # Make a shallow copy of cfg to reuse dataset factory for val episodes
        val_cfg = deepcopy(cfg)
        if getattr(cfg.dataset, "val_repo_id", None):
            val_cfg.dataset.repo_id = cfg.dataset.val_repo_id
            val_cfg.dataset.episodes = cfg.dataset.val_episodes or [0]
            val_dataset = make_dataset(val_cfg)
        else:
            val_cfg.dataset.episodes = cfg.dataset.val_episodes
            # Ensure transforms/stats match train dataset
            val_dataset = make_dataset(val_cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)
    else:
        # Allow initializing from a pretrained Hub snapshot without optimizer state
        # by starting at an offset step for correct logging/scheduling.
        step = int(getattr(cfg, "initial_step", 0) or 0)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    logging.info("Start offline training on a fixed dataset")
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        # Always log after the very first step for quick smoke tests on slow CPUs
        if step == 1:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                # Log using samples as the custom step key to enable W&B x-axis as "num samples seen"
                wandb_logger.log_dict(wandb_log_dict, custom_step_key="samples")
            train_tracker.reset_averages()
        elif is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                # Log using samples as the custom step key to enable W&B x-axis as "num samples seen"
                wandb_logger.log_dict(wandb_log_dict, custom_step_key="samples")
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            # stopped models to saving to wandb
            # if wandb_logger:
                # wandb_logger.log_policy(checkpoint_dir)

            # Optionally push this checkpoint (weights + training_state) to the Hugging Face Hub
            if cfg.policy.push_to_hub and cfg.policy.repo_id:
                try:
                    step_id = get_step_identifier(step, cfg.steps)
                    repo_id_with_step = f"{cfg.policy.repo_id}_{step_id}"
                    api = HfApi()
                    created = api.create_repo(repo_id=repo_id_with_step, private=cfg.policy.private, exist_ok=True)

                    # Upload the entire checkpoint directory so the repo contains both
                    # 'pretrained_model/' and 'training_state/' at its root.
                    api.upload_folder(
                        repo_id=created.repo_id,
                        # TODO double check that this is still loadable in the same way?? 
                        repo_type="model",
                        folder_path=str(checkpoint_dir),
                        commit_message=f"Upload full checkpoint {step_id}",
                        allow_patterns=["*.safetensors", "*.json"],
                        ignore_patterns=["*.tmp", "*.log"],
                    )
                    # Log the model URL robustly
                    created_url = getattr(created, "repo_url", getattr(created, "url", getattr(created, "repo_id", str(created))))
                    logging.info(colored("Pushed checkpoint to Hub:", "yellow", attrs=["bold"]) + f" {created_url}")
                except Exception as e:
                    logging.warning(f"Failed to push checkpoint {step} to Hugging Face Hub: {e}")
        
        # TODO below is robot eval, have another for val step on validation episode and on episode 0 to see overfitting. or just hack the below
        if is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Validate policy at step {step} ({step/1000.0:.1f}K)")
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
            ):
                # Always compute MAE on training episode 0
                try:
                    t_tr0 = time.perf_counter()
                    train_mae_png = cfg.output_dir / "train_plots" / f"mae_train_ep0_step_{get_step_identifier(step, cfg.steps)}.png"
                    # Select episode index for TRAIN MAE via env (default 0)
                    train_mae_ep_idx = int(os.environ.get("LEROBOT_TRAIN_MAE_EP_IDX", "0"))
                    train_overall_mae, _, _, _ = _evaluate_mae_on_episode(
                        policy, dataset, episode_idx=train_mae_ep_idx, device=device, out_path=train_mae_png
                    )
                    logging.info(colored("Train MAE ep0:", "yellow", attrs=["bold"]) + f" mae={train_overall_mae:.6f}")
                    if wandb_logger:
                        log_dict = {"train/mae_ep0": train_overall_mae}
                        wandb_logger.log_dict(log_dict, step)
                        if train_mae_png.exists():
                            wandb_logger.log_named_image("train_mae_ep0_plot", str(train_mae_png), step, mode="train", caption=f"train mae ep0 step {step}")
                    t_tr1 = time.perf_counter()
                    logging.info(f"timing_s: mae(train_ep0)={t_tr1 - t_tr0:.2f}")
                except Exception as e:
                    logging.warning(f"Failed to compute/log TRAIN MAE ep0: {e}")

                # Offline validation on held-out episodes
                if val_dataset is not None:
                    t0 = time.perf_counter()
                    val_loss = _compute_offline_val_loss(policy, val_dataset, device, cfg.batch_size, cfg.num_workers)
                    logging.info(colored("Validation:", "yellow", attrs=["bold"]) + f" loss={val_loss:.6f}")
                    if wandb_logger:
                        wandb_logger.log_dict({"val_loss": val_loss}, step)
                    # Compute MAE on episode 0 of validation and log + plot
                    try:
                        t1 = time.perf_counter()
                        mae_png = cfg.output_dir / "val_plots" / f"mae_ep0_step_{get_step_identifier(step, cfg.steps)}.png"
                        # Select episode index for VAL MAE via env (default 0)
                        val_mae_ep_idx = int(os.environ.get("LEROBOT_VAL_MAE_EP_IDX", "0"))
                        overall_mae, per_joint_mae, preds_np, gt_np = _evaluate_mae_on_episode(
                            policy, val_dataset, episode_idx=val_mae_ep_idx, device=device, out_path=mae_png
                        )
                        logging.info(colored("Val MAE ep0:", "yellow", attrs=["bold"]) + f" mae={overall_mae:.6f}")
                        if wandb_logger:
                            log_dict = {"val/mae_ep0": overall_mae}
                            if per_joint_mae is not None:
                                for j, v in enumerate(per_joint_mae):
                                    log_dict[f"val/mae_ep0_joint_{j}"] = float(v)
                            wandb_logger.log_dict(log_dict, step)
                            if mae_png.exists():
                                wandb_logger.log_named_image("val_mae_ep0_plot", str(mae_png), step, mode="eval", caption=f"val mae ep0 step {step}")
                        # Only MAE timing printed
                        t2 = time.perf_counter()
                        logging.info(f"val_timing_s: loss={t1 - t0:.2f} mae={t2 - t1:.2f} total={t2 - t0:.2f}")
                    except Exception as e:
                        logging.warning(f"Failed to compute/log MAE ep0: {e}")


    if eval_env:
        eval_env.close()
    logging.info("End of training")

    if cfg.policy.push_to_hub:
        policy.push_model_to_hub(cfg)


def main():
    init_logging()
    train()


if __name__ == "__main__":
    main()
