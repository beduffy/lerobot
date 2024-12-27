"""This scripts demonstrates how to train Diffusion Policy on the PushT environment.

Once you have trained a model with this script, you can try to evaluate it on
examples/2_evaluate_pretrained_policy.py
"""

from pathlib import Path

import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy


# below did not work due to repo id and offline etc so trying here TODO
# MUJOCO_GL=egl python lerobot/scripts/train.py \
#   hydra.job.name=action_change_insertion_ben_hack \
#   hydra.run.dir=outputs/train/action_change_insertion_ben_hack \
#   policy=act \
#   policy.use_vae=true \
#   env=aloha \
#   env.task=AlohaInsertion-v0 \
#   dataset_repo_id=ben/ben_keyboard_aloha_control \
#   training.eval_freq=1000 \
#   training.log_freq=250 \
#   training.offline_steps=100000 \
#   training.save_freq=2000 \
#   eval.n_episodes=50 \
#   eval.batch_size=50 \
#   wandb.enable=true \
#   device=cuda

# Create a directory to store the training checkpoint.
output_directory = Path("outputs/train/ben_hack_action_change_insertion")
output_directory.mkdir(parents=True, exist_ok=True)

# Number of offline training steps (we'll only do offline training for this example.)
# Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
training_steps = 5000
# device = torch.device("cuda")
device = torch.device("cpu")
log_freq = 250

# Set up the dataset.
delta_timestamps = {
    # Load the previous image and state at -0.1 seconds before current frame,
    # then load current image and state corresponding to 0.0 second.
    "observation.image.pixels": [-0.1, 0.0],
    "observation.agent_state": [-0.1, 0.0],
    # Load the previous action (-0.1), the next action to be executed (0.0),
    # and 14 future actions with a 0.1 seconds spacing. All these actions will be
    # used to supervise the policy.
    "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
}
dataset = LeRobotDataset("ben/ben_keyboard_aloha_control", delta_timestamps=delta_timestamps, local_files_only=True)

# Create a wrapper class to rename the keys
class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.meta = dataset.meta  # Pass through the meta attribute
     
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Rename the keys to match what the policy expects
        item["observation.image"] = item.pop("observation.image.pixels")
        item["observation.state"] = item.pop("observation.agent_state")
        return item

# Wrap the dataset
dataset = DatasetWrapper(dataset)

# Now modify the stats dictionary
stats = dataset.meta.stats.copy()
# Rename observation.image.pixels to observation.image
stats['observation.image'] = stats.pop('observation.image.pixels')
# Rename observation.agent_state to observation.state
if 'observation.agent_state' in stats:
    stats['observation.state'] = stats.pop('observation.agent_state')

# Set up the the policy.
# Policies are initialized with a configuration class, in this case `DiffusionConfig`.
# For this example, no arguments need to be passed because the defaults are set up for PushT.
# If you're doing something different, you will likely need to change at least some of the defaults.
cfg = DiffusionConfig(
    # Modify the configuration to match your input/output dimensions
    input_shapes={
        "observation.image": [3, 96, 96],  # Standard image dimensions
        "observation.state": [14],  # Your state dimension is 14
    },
    output_shapes={
        "action": [14],  # Your action dimension is 14
    },
    n_action_steps=16,  # Match the number of future actions in delta_timestamps
    horizon=16  # Should match n_action_steps
)

# TODO problem is 'top'...... in train.py feckin hell.
# 
# and here:
# 
#  
# # Traceback (most recent call last):
#   File "/home/ben/all_projects/lerobot/examples/3_train_policy_ben_hack_action.py", line 64, in <module>
#     policy = DiffusionPolicy(cfg, dataset_stats=dataset.meta.stats)
#   File "/home/ben/all_projects/lerobot/lerobot/common/policies/diffusion/modeling_diffusion.py", line 76, in __init__
#     self.normalize_inputs = Normalize(
#   File "/home/ben/all_projects/lerobot/lerobot/common/policies/normalize.py", line 128, in __init__
#     stats_buffers = create_stats_buffers(shapes, modes, stats)
#   File "/home/ben/all_projects/lerobot/lerobot/common/policies/normalize.py", line 80, in create_stats_buffers
#     buffer["mean"].data = stats[key]["mean"].clone()
# KeyError: 'observation.image'

# import pdb; pdb.set_trace()

policy = DiffusionPolicy(cfg, dataset_stats=stats)
policy.train()
policy.to(device)

optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

# Create dataloader for offline training.
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=4,
    batch_size=64,
    shuffle=True,
    pin_memory=device != torch.device("cpu"),
    drop_last=True,
)

# Run training loop.
step = 0
done = False
while not done:
    for batch in dataloader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        output_dict = policy.forward(batch)
        loss = output_dict["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % log_freq == 0:
            print(f"step: {step} loss: {loss.item():.3f}")
        step += 1
        if step >= training_steps:
            done = True
            break

# Save a policy checkpoint.
policy.save_pretrained(output_directory)
