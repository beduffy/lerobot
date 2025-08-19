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
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
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
    with torch.no_grad(), torch.autocast(device_type=device.type) if policy.config.use_amp else nullcontext():
        for batch in dataloader:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")
            loss, _ = policy.forward(batch)
            losses.append(loss.item())
    policy.train()
    return float(sum(losses) / max(1, len(losses)))


def _plot_action_state_trajectories(policy: PreTrainedPolicy, dataset, device: torch.device, out_path):
    """Generate a simple action vs state plot for the first held-out episode.

    Saves a PNG at out_path. This is lightweight: it samples one episode worth of frames and
    runs a forward pass to compare predicted action against ground-truth action per dimension.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return  # plotting optional

    # Extract the first episode indices
    ep_idx = None
    for i in range(len(dataset)):
        item = dataset[i]
        ep_idx = int(item["episode_index"]) if hasattr(item["episode_index"], "item") else int(item["episode_index"])
        break
    if ep_idx is None:
        return

    indices = []
    for i in range(len(dataset)):
        item = dataset[i]
        this_ep = int(item["episode_index"]) if hasattr(item["episode_index"], "item") else int(item["episode_index"])
        if this_ep == ep_idx:
            indices.append(i)
        elif indices:
            break

    # Build a mini-batch for the whole episode
    batch = dataset.hf_dataset.select(indices)  # raw HF dataset rows
    # Convert via dataset transform for a proper batch
    collated = {}
    for key in dataset.hf_features:
        vals = [dataset[i][key] for i in indices]
        if isinstance(vals[0], torch.Tensor):
            collated[key] = torch.stack(vals)
        else:
            collated[key] = torch.tensor(np.stack([np.array(v) for v in vals]))

    # Add visuals if present
    for cam in getattr(dataset.meta, "camera_keys", []):
        if cam in dataset.hf_features or cam in batch.features:  # loaded as tensors by __getitem__
            vals = [dataset[i][cam] for i in indices]
            if isinstance(vals[0], torch.Tensor):
                collated[cam] = torch.stack(vals)

    # Move to device
    for key in list(collated.keys()):
        if isinstance(collated[key], torch.Tensor):
            collated[key] = collated[key].to(device)

    policy.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type) if policy.config.use_amp else nullcontext():
        loss, out = policy.forward(collated)
        # Try to get predicted action if available in out
        pred = None
        if isinstance(out, dict):
            for k in out:
                if k.startswith("pred_action") or k == "action_pred" or k == "action":
                    pred = out[k]
                    break
        if pred is None and "action" in collated:
            # Some policies only expose loss; skip plotting in that case
            pred = collated["action"] * 0.0

    gt = collated.get("action")
    if gt is None or pred is None:
        return

    gt_np = gt.detach().float().cpu().numpy()
    pred_np = pred.detach().float().cpu().numpy()
    T, D = gt_np.shape[0], gt_np.shape[-1]

    fig_h = 2.0 * min(6, D)
    fig, axes = plt.subplots(min(6, D), 1, figsize=(10, fig_h), squeeze=False)
    axes = axes.flatten()
    xs = np.arange(T)
    for d in range(min(6, D)):
        ax = axes[d]
        ax.plot(xs, gt_np[:, d], 'r--', label='State' if d == 0 else None)
        ax.plot(xs, pred_np[:, d], 'b-', label='Action' if d == 0 else None)
        ax.set_title(f"Dim {d}")
        ax.set_xlabel("Time Step")
    if D > 0:
        axes[0].legend(loc='best')
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

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
        from copy import deepcopy
        val_cfg = deepcopy(cfg)
        if getattr(cfg.dataset, "val_repo_id", None):
            val_cfg.dataset.repo_id = cfg.dataset.val_repo_id
            val_cfg.dataset.episodes = cfg.dataset.val_episodes
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
            # if wandb_logger:
                # wandb_logger.log_policy(checkpoint_dir)

            # Offline validation on held-out episodes
            if val_dataset is not None:
                val_loss = _compute_offline_val_loss(policy, val_dataset, device, cfg.batch_size, cfg.num_workers)
                logging.info(colored("Validation:", "yellow", attrs=["bold"]) + f" loss={val_loss:.6f}")
                if wandb_logger:
                    # Log validation using samples as custom step key for consistent x-axis
                    wandb_logger.log_dict({"val_loss": val_loss, **train_tracker.to_dict(use_avg=True)}, custom_step_key="samples")
                # Plot trajectories once per saving step
                try:
                    png_path = cfg.output_dir / "val_plots" / f"traj_step_{get_step_identifier(step, cfg.steps)}.png"
                    _plot_action_state_trajectories(policy, val_dataset, device, png_path)
                    if wandb_logger and png_path.exists():
                        wandb_logger.log_image(str(png_path), step, mode="eval", caption="val_trajectories")
                except Exception as e:
                    logging.warning(f"Failed to plot validation trajectories: {e}")

            # Optionally push this checkpoint (weights + training_state) to the Hugging Face Hub
            if cfg.policy.push_to_hub and cfg.policy.repo_id:
                try:
                    step_id = get_step_identifier(step, cfg.steps)
                    repo_id_with_step = f"{cfg.policy.repo_id}_{step_id}"
                    api = HfApi()
                    created = api.create_repo(repo_id=repo_id_with_step, private=cfg.policy.private, exist_ok=True)
                    pretrained_dir = checkpoint_dir / "pretrained_model"
                    training_state_dir = checkpoint_dir / "training_state"
                    # Upload pretrained model (weights + config) at repo root for easy loading
                    api.upload_folder(
                        repo_id=created.repo_id,
                        repo_type="model",
                        folder_path=pretrained_dir,
                        commit_message=f"Upload checkpoint {step_id} pretrained_model",
                        allow_patterns=["*.safetensors", "*.json"],
                        ignore_patterns=["*.tmp", "*.log"],
                    )
                    # Also upload training_state to enable true resume across machines
                    if training_state_dir.exists():
                        api.upload_folder(
                            repo_id=created.repo_id,
                            repo_type="model",
                            folder_path=training_state_dir,
                            path_in_repo="training_state",
                            commit_message=f"Upload checkpoint {step_id} training_state",
                            allow_patterns=["*.json", "*.safetensors"],
                            ignore_patterns=["*.tmp", "*.log"],
                        )
                    # TODO does it work for single datasets? gpt5 thought it was fixed but was it?
                    # Log the model URL of the created repo
                    # logging.info(colored("Pushed checkpoint to Hub:", "yellow", attrs=["bold"]) + f" {created.repo_url.url}")
                    logging.info(colored("Pushed checkpoint to Hub:", "yellow", attrs=["bold"]) + f" {created.repo_url}")
                except Exception as e:
                    logging.warning(f"Failed to push checkpoint {step} to Hugging Face Hub: {e}")


        if cfg.env and is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
            ):
                eval_info = eval_policy(
                    eval_env,
                    policy,
                    cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                )

            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(
                cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
            )
            eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
            eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
            eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
            logging.info(eval_tracker)
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                # Log eval using samples as custom step key for consistent x-axis
                wandb_logger.log_dict(wandb_log_dict, mode="eval", custom_step_key="samples")
                wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

    if eval_env:
        eval_env.close()
    logging.info("End of training")

    if cfg.policy.push_to_hub:
        policy.push_model_to_hub(cfg)


if __name__ == "__main__":
    init_logging()
    train()
