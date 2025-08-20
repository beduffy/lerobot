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
    """Compute average loss on a held-out dataset (no grad)."""
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
                if gt.ndim == 2:  # (B, A) -> (B, 1, A)
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
    max_frames = max_frames_env  # cap frames for quick MAE
    if len(full_indices) > max_frames:
        stride = max(1, len(full_indices) // max_frames)
        indices = full_indices[::stride][:max_frames]
    else:
        indices = full_indices

    # Iterate frame-by-frame (batch_size=1) for consistent select_action semantics
    preds = []
    gts = []
    policy.eval()
    policy.reset()
    with torch.no_grad():
        for idx in indices:
            sample = get_sample(idx)
            inp = {}
            for key, val in sample.items():
                if key.startswith("observation.") and isinstance(val, torch.Tensor):
                    inp[key] = val.unsqueeze(0).to(device)
            if not inp:
                continue
            pred = policy.select_action(inp)  # (B, A)
            if pred.ndim == 3:  # (B, S, A) -> first step
                pred = pred[:, 0, :]
            gt = sample.get("action")
            if isinstance(gt, torch.Tensor):
                if gt.ndim == 1:
                    gt = gt.unsqueeze(0)
                preds.append(pred[0].detach().cpu())
                gts.append(gt[0].detach().cpu())

    if len(preds) == 0:
        return float("nan"), None, None, None

    preds_t = torch.stack(preds, dim=0)
    gts_t = torch.stack(gts, dim=0)
    per_joint_mae_t = torch.mean(torch.abs(preds_t - gts_t), dim=0)
    overall_mae = float(torch.mean(per_joint_mae_t).item())
    per_joint_mae = per_joint_mae_t.tolist()

    # Plot if matplotlib available
    if plt is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        num_joints = preds_t.shape[1]
        fig, axes = plt.subplots(num_joints, 1, figsize=(10, max(2, 2 * num_joints)))
        if num_joints == 1:
            axes = [axes]
        xs = np.arange(preds_t.shape[0])
        for j in range(num_joints):
            ax = axes[j]
            ax.plot(xs, gts_t[:, j].numpy(), label="GT")
            ax.plot(xs, preds_t[:, j].numpy(), label="Pred")
            ax.set_title(f"Dim {j}  MAE={per_joint_mae[j]:.4f}")
            ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
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
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()
        elif is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
        # TODO below is robot eval, have another for val step on validation episode and on episode 0 to see overfitting. or just hack the below
        if is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Validate policy at step {step}")
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
            ):


                # Always compute MAE on training episode 0
                try:
                    t_tr0 = time.perf_counter()
                    train_mae_png = cfg.output_dir / "train_plots" / f"mae_train_ep0_step_{get_step_identifier(step, cfg.steps)}.png"
                    train_overall_mae, _, _, _ = _evaluate_mae_on_episode(
                        policy, dataset, episode_idx=0, device=device, out_path=train_mae_png
                    )
                    logging.info(colored("Train MAE ep0:", "yellow", attrs=["bold"]) + f" mae={train_overall_mae:.6f}")
                    t_tr1 = time.perf_counter()
                    logging.info(f"timing_s: mae(train_ep0)={t_tr1 - t_tr0:.2f}")
                    if wandb_logger:
                        log_dict = {"train/mae_ep0": train_overall_mae}
                        wandb_logger.log_dict(log_dict, step)
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
                        overall_mae, per_joint_mae, preds_np, gt_np = _evaluate_mae_on_episode(
                            policy, val_dataset, episode_idx=0, device=device, out_path=mae_png
                        )
                        logging.info(colored("Val MAE ep0:", "yellow", attrs=["bold"]) + f" mae={overall_mae:.6f}")
                        if wandb_logger:
                            log_dict = {"val/mae_ep0": overall_mae}
                            if per_joint_mae is not None:
                                for j, v in enumerate(per_joint_mae):
                                    log_dict[f"val/mae_ep0_joint_{j}"] = float(v)
                            wandb_logger.log_dict(log_dict, step)
                            if mae_png.exists():
                                wandb_logger.log_named_image("mae_ep0_plot", str(mae_png), step, mode="eval", caption=f"mae ep0 step {step}")
                        # Only MAE timing printed
                        t2 = time.perf_counter()
                        logging.info(f"val_timing_s: loss={t1 - t0:.2f} mae={t2 - t1:.2f} total={t2 - t0:.2f}")
                    except Exception as e:
                        logging.warning(f"Failed to compute/log MAE ep0: {e}")


                # eval_info = eval_policy(
                #     eval_env,
                #     policy,
                #     cfg.eval.n_episodes,
                #     videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                #     max_episodes_rendered=4,
                #     start_seed=cfg.seed,
                # )

            # eval_metrics = {
            #     "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
            #     "pc_success": AverageMeter("success", ":.1f"),
            #     "eval_s": AverageMeter("eval_s", ":.3f"),
            # }
            # eval_tracker = MetricsTracker(
            #     cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
            # )
            # eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
            # eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
            # eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
            # logging.info(eval_tracker)
            # if wandb_logger:
            #     wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
            #     wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
            #     wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

    if eval_env:
        eval_env.close()
    logging.info("End of training")

    if cfg.policy.push_to_hub:
        policy.push_model_to_hub(cfg)


if __name__ == "__main__":
    init_logging()
    train()
