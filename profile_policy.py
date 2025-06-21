#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# This code is based on the original eval.py script from LeRobot.
# It has been modified to include profiling capabilities.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import draccus
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from torch.profiler import ProfilerActivity, profile, record_function

from lerobot.common.policies.factory import make_policy
from lerobot.common.robots import Robot, RobotConfig, make_robot_from_config
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import init_logging, move_cursor_up


# TODO have never ran. 

@dataclass
class EvalConfig:
    robot: RobotConfig
    policy_path: Path
    device: str = "cpu"
    fps: int = 30
    # Profiler settings
    profile: bool = True
    profile_wait: int = 2
    profile_warmup: int = 2
    profile_active: int = 5
    profile_output: str = "trace.json"


def eval_loop(
    policy, robot: Robot, fps: int, device: str, profiler: torch.profiler.profile | None
):
    display_len = max(len(key) for key in robot.action_features)
    obs_buffer = {}

    while True:
        with record_function("eval_loop_iteration"):
            loop_start = time.perf_counter()

            with record_function("get_observation"):
                raw_obs = robot.get_observation()

            with record_function("prepare_observation"):
                for key, value in raw_obs.items():
                    if isinstance(value, np.ndarray):
                        obs_buffer[key] = torch.from_numpy(value).unsqueeze(0).to(device)
                    elif isinstance(value, float):
                        if "state" not in obs_buffer:
                            obs_buffer["state"] = []
                        obs_buffer["state"].append(value)
                if "state" in obs_buffer:
                    obs_buffer["state"] = torch.tensor(obs_buffer["state"]).unsqueeze(0).to(device)

            with record_function("select_action"):
                with torch.no_grad():
                    action_tensor = policy.select_action(obs_buffer)

            with record_function("prepare_action"):
                action_array = action_tensor.squeeze().cpu().numpy()
                action_dict = {}
                action_names = list(robot.action_features.keys())
                for i, name in enumerate(action_names):
                    action_dict[name] = action_array[i]

            with record_function("send_action"):
                robot.send_action(action_dict)

            dt_s = time.perf_counter() - loop_start
            busy_wait(1 / fps - dt_s)
            loop_s = time.perf_counter() - loop_start

            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'VALUE':>7}")
            for motor, value in action_dict.items():
                print(f"{motor:<{display_len}} | {value:>7.2f}")
            print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
            move_cursor_up(len(action_dict) + 5)

        if profiler:
            profiler.step()


@draccus.wrap()
def evaluate(cfg: EvalConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)

    policy_path_str = str(cfg.policy_path)
    if Path(policy_path_str).exists():
        policy_dir = Path(policy_path_str)
        config_path = policy_dir / "config.json"
        weights_path = policy_dir / "model.safetensors"
        if not config_path.exists() or not weights_path.exists():
            raise FileNotFoundError(f"Could not find config.json and/or model.safetensors in {policy_dir}")
    else:
        try:
            config_path = hf_hub_download(repo_id=policy_path_str, filename="config.json", repo_type="model")
            weights_path = hf_hub_download(
                repo_id=policy_path_str, filename="model.safetensors", repo_type="model"
            )
        except Exception as e:
            raise FileNotFoundError(f"Could not download policy from hub repo '{policy_path_str}'. Error: {e}")

    policy_cfg = OmegaConf.load(config_path)
    policy = make_policy(policy_cfg, robot_cfg=cfg.robot)
    policy.load_weights(weights_path)
    policy.to(cfg.device)
    policy.eval()

    robot.connect()

    try:
        if cfg.profile:
            with profile(
                activities=[ProfilerActivity.CPU],
                schedule=torch.profiler.schedule(
                    wait=cfg.profile_wait,
                    warmup=cfg.profile_warmup,
                    active=cfg.profile_active,
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs"),
                record_shapes=True,
                with_stack=True,
            ) as prof:
                eval_loop(policy, robot, cfg.fps, cfg.device, profiler=prof)
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
            prof.export_chrome_trace(cfg.profile_output)
            print(f"Detailed trace saved to {cfg.profile_output}")
        else:
            eval_loop(policy, robot, cfg.fps, cfg.device, profiler=None)
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()


if __name__ == "__main__":
    evaluate() 