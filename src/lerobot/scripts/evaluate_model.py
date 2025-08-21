#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Multi-Model Evaluation Script

Evaluates trained models on dataset episodes with visualization.

Usage:
    python evaluate_model.py ./single_episode_model --episode 0
    python evaluate_model.py path/to/model --episode 1 --compare-episodes 0,1,2
    python evaluate_model.py ./single_episode_model --episode 0 --plot --save-plots
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
import re

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader, Subset

# Import all model types
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy

# Import foundation models (VLAs)
try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    SMOLVLA_AVAILABLE = True
except ImportError:
    # Fallback to old structure or not available
    try:
        from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        SMOLVLA_AVAILABLE = True
    except ImportError:
        SMOLVLA_AVAILABLE = False

try:
    from lerobot.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
    PI0FAST_AVAILABLE = True
except ImportError:
    # Fallback to old structure or not available
    try:
        from lerobot.common.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
        PI0FAST_AVAILABLE = True
    except ImportError:
        PI0FAST_AVAILABLE = False


class EpisodeSampler(torch.utils.data.Sampler):
    """Custom sampler to iterate through a specific episode."""
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def get_episode_indices(dataset, episode_idx):
    """Get indices for a specific episode."""
    from_idx = dataset.episode_data_index["from"][episode_idx].item()
    to_idx = dataset.episode_data_index["to"][episode_idx].item()
    return list(range(from_idx, to_idx))


def detect_model_type(model_path):
    """Detect model type from model directory or checkpoint file."""
    model_path = Path(model_path)
    
    # If it's a checkpoint file, we need to rely on the model_type argument
    # since we can't easily detect from the checkpoint content
    if str(model_path).endswith('.pt'):
        print("   📁 Checkpoint file detected - model type should be specified with --model-type")
        return None  # Let the caller use the specified model_type
    
    # Check for training_info.json first
    info_path = model_path / "training_info.json"
    if info_path.exists():
        try:
            with open(info_path, 'r') as f:
                info = json.load(f)
                if 'model_type' in info:
                    return info['model_type']
        except:
            pass
    
    # Check for config.json
    config_path = model_path / "config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Check for model-specific identifiers
                if 'model_type' in config:
                    return config['model_type']
                elif 'architectures' in config:
                    arch = config['architectures'][0] if config['architectures'] else ""
                    if 'ACT' in arch:
                        return 'act'
                    elif 'Diffusion' in arch:
                        return 'diffusion'
                    elif 'VQBeT' in arch:
                        return 'vqbet'
                    elif 'SmolVLA' in arch:
                        return 'smolvla'
                    elif 'PI0FAST' in arch:
                        return 'pi0fast'
        except:
            pass
    
    # Fallback: try to load each model type and see which one works
    model_types = ['act', 'diffusion', 'vqbet']
    if SMOLVLA_AVAILABLE:
        model_types.append('smolvla')
    if PI0FAST_AVAILABLE:
        model_types.append('pi0fast')
    
    for model_type in model_types:
        try:
            if model_type == 'act':
                ACTPolicy.from_pretrained(model_path)
                return 'act'
            elif model_type == 'diffusion':
                DiffusionPolicy.from_pretrained(model_path)
                return 'diffusion'
            elif model_type == 'vqbet':
                VQBeTPolicy.from_pretrained(model_path)
                return 'vqbet'
            elif model_type == 'smolvla' and SMOLVLA_AVAILABLE:
                SmolVLAPolicy.from_pretrained(model_path)
                return 'smolvla'
            elif model_type == 'pi0fast' and PI0FAST_AVAILABLE:
                PI0FASTPolicy.from_pretrained(model_path)
                return 'pi0fast'
        except:
            continue
    
    # Default to ACT if nothing works
    print("⚠️  Could not detect model type, defaulting to ACT")
    return 'act'


def load_policy(model_path, model_type, camera_remap=None):
    """Load policy based on model type."""
    print(f"🧠 Loading {model_type.upper()} policy from {model_path}...")
    
    # Check if this is a checkpoint file
    if str(model_path).endswith('.pt'):
        # Load from checkpoint file
        print(f"   📁 Loading from checkpoint file: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f"   ✅ Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
        
        # Create policy based on model type using direct initialization
        if model_type == 'act':
            # For ACT, we need to create with minimal config
            from lerobot.policies.act.configuration_act import ACTConfig
            config = ACTConfig()
            policy = ACTPolicy(config)
        elif model_type == 'diffusion':
            # For Diffusion, we need to create with minimal config
            from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
            config = DiffusionConfig()
            policy = DiffusionPolicy(config)
        elif model_type == 'vqbet':
            # For VQBeT, we need to create with minimal config
            from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig
            config = VQBeTConfig()
            policy = VQBeTPolicy(config)
        elif model_type == 'smolvla':
            if not SMOLVLA_AVAILABLE:
                raise RuntimeError("SmolVLA not available. Install with updated LeRobot version.")
            # For SmolVLA, create with proper config and dataset stats
            from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
            from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
            from lerobot.datasets.utils import dataset_to_policy_features
            
            # Create proper config with dataset features
            metadata = LeRobotDatasetMetadata("bearlover365/red_cube_always_in_same_place")
            features = dataset_to_policy_features(metadata.features)
            
            # Apply camera remapping to features if needed
            if camera_remap:
                remapped_features = {}
                for key, value in features.items():
                    if key in camera_remap:
                        new_key = camera_remap[key]
                        remapped_features[new_key] = value
                    else:
                        remapped_features[key] = value
                features = remapped_features
            
            config = SmolVLAConfig(
                input_features=features,
                output_features={'action': features['action']}
            )
            
            policy = SmolVLAPolicy(config, dataset_stats=metadata.stats)
        elif model_type == 'pi0fast':
            if not PI0FAST_AVAILABLE:
                raise RuntimeError("π0-FAST not available. Install with updated LeRobot version.")
            # For π0-FAST, create with minimal config
            from lerobot.policies.pi0fast.configuration_pi0fast import PI0FASTConfig
            config = PI0FASTConfig()
            policy = PI0FASTPolicy(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Load state dict with strict=False to ignore normalization buffers
        if 'model_state_dict' in checkpoint:
            missing_keys, unexpected_keys = policy.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"   ✅ Loaded state dict successfully")
            if missing_keys:
                print(f"   ⚠️  Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"   ⚠️  Ignored unexpected keys: {unexpected_keys}")
        else:
            print("   ⚠️  No model_state_dict found in checkpoint")
        
        return policy
    
    else:
        # Load from model directory (original behavior)
        if model_type == 'act':
            policy = ACTPolicy.from_pretrained(model_path)
        elif model_type == 'diffusion':
            policy = DiffusionPolicy.from_pretrained(model_path)
        elif model_type == 'vqbet':
            policy = VQBeTPolicy.from_pretrained(model_path)
        elif model_type == 'smolvla':
            if not SMOLVLA_AVAILABLE:
                raise RuntimeError("SmolVLA not available. Install with updated LeRobot version.")
            policy = SmolVLAPolicy.from_pretrained(model_path)
        elif model_type == 'pi0fast':
            if not PI0FAST_AVAILABLE:
                raise RuntimeError("π0-FAST not available. Install with updated LeRobot version.")
            policy = PI0FASTPolicy.from_pretrained(model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return policy


def evaluate_model_on_episode(model_path, dataset_name, episode_idx, device, use_dataloader=True, camera_remap=None, model_type=None, max_frames: int | None = None, eval_batch_size: int = 1):
    """Evaluate model on a specific episode and return predictions/ground truth for plotting."""
    print(f"🔬 Evaluating model on episode {episode_idx}...")
    
    # Detect model type (or use provided one)
    if model_type is None:
        detected_type = detect_model_type(model_path)
        if detected_type is None:
            print("   ❌ Could not detect model type and none specified. Use --model-type argument.")
            return None, None, None, None, None, None
        model_type = detected_type
    
    print(f"   Detected model type: {model_type.upper()}")
    
    # Load model
    policy = load_policy(model_path, model_type, camera_remap)
    policy.to(device)
    policy.eval()
    policy.reset()
    
    # Load dataset  
    dataset = LeRobotDataset(dataset_name, video_backend="pyav")
    
    # Apply camera remapping if specified
    if camera_remap:
        print(f"   📷 Applying camera remapping: {camera_remap}")
        
        class CameraRemapDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, camera_remap):
                self.dataset = dataset
                self.camera_remap = camera_remap
                # Preserve important attributes from the original dataset
                if hasattr(dataset, 'episode_data_index'):
                    self.episode_data_index = dataset.episode_data_index
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                batch = self.dataset[idx]
                remapped_batch = {}
                
                for key, value in batch.items():
                    if key in self.camera_remap:
                        # Remap camera keys
                        new_key = self.camera_remap[key]
                        remapped_batch[new_key] = value
                    else:
                        remapped_batch[key] = value
                
                return remapped_batch
        
        dataset = CameraRemapDataset(dataset, camera_remap)
    
    if use_dataloader:
        # Build contiguous indices and optionally cap frames for speed
        episode_indices = get_episode_indices(dataset, episode_idx)
        if isinstance(max_frames, int) and max_frames > 0:
            episode_indices = episode_indices[: max_frames]

        dataset_subset = Subset(dataset, episode_indices)
        test_dataloader = DataLoader(
            dataset_subset,
            num_workers=4,
            batch_size=max(1, int(eval_batch_size)),
            shuffle=False,
            pin_memory=device.type != "cpu",
        )

        print(f"   Episode {episode_idx}: {len(episode_indices)} steps (batch_size={max(1, int(eval_batch_size))})")

        predictions = []
        ground_truths = []

        # Faster/safer eval context
        autocast_ctx = torch.amp.autocast(device_type="cuda") if device.type == "cuda" else torch.no_grad()
        with torch.inference_mode(), autocast_ctx:
            total = 0
            for i, batch in enumerate(test_dataloader):
                try:
                    # Prepare inputs: ONLY observations
                    inp_batch = {}
                    for key, value in batch.items():
                        if key.startswith("observation.") and isinstance(value, torch.Tensor):
                            inp_batch[key] = value.to(device, non_blocking=True)

                    if not inp_batch:
                        continue

                    # Add language task for VLA models (SmolVLA, π0-FAST)
                    if model_type in ["smolvla", "pi0fast"]:
                        inp_batch["task"] = "grab red cube and put to left"

                    # Predict in batch; prefer chunk predictor when available
                    if hasattr(policy, "predict_action_chunk"):
                        chunk = policy.predict_action_chunk(inp_batch)  # (B, S, A)
                        pred_action = chunk[:, 0, :]
                    else:
                        out = policy.select_action(inp_batch)
                        pred_action = out[:, 0, :] if out.dim() == 3 else out

                    gt_action = batch["action"]  # (B, A)

                    predictions.append(pred_action.detach().cpu())
                    ground_truths.append(gt_action.detach().cpu())

                    total += pred_action.shape[0]
                    if i % 10 == 0:
                        print(f"   Processed {total}/{len(episode_indices)} steps...")

                except Exception as e:
                    print(f"   Warning: Skipped mini-batch {i}: {e}")
                    continue
        
    else:
        # Fallback to original approach
        episode_indices = get_episode_indices(dataset, episode_idx)
        print(f"   Episode {episode_idx}: {len(episode_indices)} steps")
        
        # Debug: Check policy configuration before the loop
        print(f"   🔍 Policy input features: {list(policy.config.input_features.keys())}")
        image_features = {k: v for k, v in policy.config.input_features.items() if 'image' in k.lower()}
        print(f"   📷 Policy expects image features: {list(image_features.keys())}")
        
        predictions = []
        ground_truths = []
        
        with torch.no_grad():
            for i, idx in enumerate(episode_indices):
                try:
                    sample = dataset[idx]
                    
                    # Debug: Print available keys in sample
                    if i == 0:
                        print(f"   🔍 Available keys in sample: {list(sample.keys())}")
                        image_keys = [k for k in sample.keys() if 'image' in k.lower()]
                        print(f"   📷 Image keys found: {image_keys}")
                        
                        # Debug: Check what the policy expects
                        print(f"   🔍 Policy input features: {list(policy.config.input_features.keys())}")
                        image_features = {k: v for k, v in policy.config.input_features.items() if 'image' in k.lower()}
                        print(f"   📷 Policy expects image features: {list(image_features.keys())}")
                    
                    # Prepare input - ONLY OBSERVATIONS FOR INFERENCE
                    batch = {}
                    for key, value in sample.items():
                        if key.startswith("observation.") and isinstance(value, torch.Tensor):
                            # For SmolVLA, ensure images are properly formatted
                            if "image" in key.lower() and value.dim() == 3:
                                # Ensure image is in correct format [C, H, W]
                                if value.shape[0] == 3:  # Already [C, H, W]
                                    batch[key] = value.unsqueeze(0).to(device)
                                else:  # Assume [H, W, C] and transpose
                                    batch[key] = value.permute(2, 0, 1).unsqueeze(0).to(device)
                            else:
                                batch[key] = value.unsqueeze(0).to(device)
                    
                    # Debug: Print what we're sending to the model
                    if i == 0:
                        print(f"   📤 Sending to model: {list(batch.keys())}")
                        # Check image shapes
                        for k, v in batch.items():
                            if "image" in k.lower():
                                print(f"      {k}: shape {v.shape}, dtype {v.dtype}")
                        
                        # Debug: Check what the policy expects
                        print(f"   🔍 Policy input features: {list(policy.config.input_features.keys())}")
                        image_features = {k: v for k, v in policy.config.input_features.items() if 'image' in k.lower()}
                        print(f"   📷 Policy expects image features: {list(image_features.keys())}")
                    
                    # Add language task for VLA models (SmolVLA, π0-FAST)
                    if model_type in ["smolvla", "pi0fast"]:
                        batch["task"] = "grab red cube and put to left"
                    
                    # Debug: Check if we have the expected image features
                    if i == 0:
                        expected_image_keys = [k for k in policy.config.input_features.keys() if 'image' in k.lower()]
                        actual_image_keys = [k for k in batch.keys() if 'image' in k.lower()]
                        print(f"   🔍 Expected image keys: {expected_image_keys}")
                        print(f"   📤 Actual image keys: {actual_image_keys}")
                        
                        # If we don't have the expected keys, try to map them
                        if expected_image_keys and not any(k in batch for k in expected_image_keys):
                            print(f"   ⚠️  Missing expected image keys, attempting to map...")
                            # Try to map our image key to the expected one
                            for expected_key in expected_image_keys:
                                if 'wrist' in expected_key.lower():
                                    # Map our wrist image to the expected key
                                    if 'observation.images.wrist' in batch:
                                        batch[expected_key] = batch['observation.images.wrist']
                                        print(f"      Mapped observation.images.wrist -> {expected_key}")
                                        break
                    
                    # Get prediction
                    pred_action_chunk = policy.select_action(batch)
                    
                    # Handle both chunked and non-chunked outputs
                    if pred_action_chunk.dim() == 3:  # [batch, chunk_size, action_dim]
                        pred_action = pred_action_chunk[0, 0, :]  # Take first action from chunk
                    else:  # [batch, action_dim]
                        pred_action = pred_action_chunk[0, :]  # Take first action
                    
                    gt_action = sample["action"]  # Ground truth action
                    
                    # Ensure both tensors are 1D before appending
                    if pred_action.dim() > 1:
                        pred_action = pred_action.squeeze()
                    if gt_action.dim() > 1:
                        gt_action = gt_action.squeeze()
                    
                    predictions.append(pred_action.cpu())
                    ground_truths.append(gt_action.cpu())
                    
                except Exception as e:
                    print(f"   Warning: Skipped step {idx}: {e}")
                    continue
    
    if predictions:
        # predictions/ground_truths may be a list of [B, A] mini-batches; concatenate across batches
        if isinstance(predictions[0], torch.Tensor) and predictions[0].dim() == 2:
            predictions = torch.cat(predictions, dim=0)
            ground_truths = torch.cat(ground_truths, dim=0)
        else:
            predictions = torch.stack(predictions, dim=0)
            ground_truths = torch.stack(ground_truths, dim=0)
        
        # Calculate metrics
        mae = torch.mean(torch.abs(predictions - ground_truths)).item()
        mse = torch.mean((predictions - ground_truths) ** 2).item()
        max_error = torch.max(torch.abs(predictions - ground_truths)).item()
        
        print(f"   📊 Results:")
        print(f"      Mean Absolute Error: {mae:.6f}")
        print(f"      Mean Squared Error: {mse:.6f}")
        print(f"      Max Absolute Error: {max_error:.6f}")
        
        if mae < 0.01:
            print("      🎉 Excellent performance!")
        elif mae < 0.1:
            print("      ✅ Good performance")
        elif mae < 1.0:
            print("      ⚠️  Moderate performance")
        else:
            print("      ❌ Poor performance")
        
        # Per-joint analysis
        action_dim = predictions.shape[1]
        joint_names = [f'Joint {i+1}' for i in range(action_dim-1)] + ['Gripper'] if action_dim == 7 else [f'Action {i+1}' for i in range(action_dim)]
        print(f"\n🔧 Per-joint errors:")
        for i, joint_name in enumerate(joint_names):
            joint_error = torch.mean(torch.abs(predictions[:, i] - ground_truths[:, i])).item()
            print(f"      {joint_name}: {joint_error:.6f}")
            
        return mae, mse, max_error, predictions, ground_truths, joint_names
    else:
        print("   ❌ No valid predictions generated")
        return None, None, None, None, None, None


def plot_predictions_vs_ground_truth(predictions, ground_truths, joint_names, episode_idx, model_type, save_plots=False, output_dir="./plots"):
    """Create detailed plots comparing predictions vs ground truth."""
    print(f"📈 Creating plots for episode {episode_idx}...")
    
    action_dim = predictions.shape[1]
    
    # Create output directory if saving plots
    if save_plots:
        Path(output_dir).mkdir(exist_ok=True)
    
    # Create subplots for each joint
    fig, axes = plt.subplots(action_dim, 1, figsize=(15, 3*action_dim))
    if action_dim == 1:
        axes = [axes]

    for i in range(action_dim):
        gt_values = ground_truths[:, i].cpu().numpy()
        pred_values = predictions[:, i].cpu().numpy()
        
        axes[i].plot(gt_values, label='Ground Truth', linewidth=2, alpha=0.8, color='blue')
        axes[i].plot(pred_values, label='Predicted', linewidth=2, alpha=0.8, linestyle='--', color='red')
        axes[i].set_title(f'{joint_names[i]} - Predicted vs Ground Truth')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Action Value')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Calculate and show error for this joint
        joint_error = torch.mean(torch.abs(predictions[:, i] - ground_truths[:, i])).item()
        axes[i].text(0.02, 0.98, f'MAE: {joint_error:.4f}', 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.suptitle(f'🎯 {model_type.upper()} Model Performance: Predicted vs Ground Truth Actions (Episode {episode_idx})', 
                 fontsize=16, y=1.02)
    
    if save_plots:
        plot_path = Path(output_dir) / f"{model_type}_episode_{episode_idx}_predictions.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Saved plot to: {plot_path}")
    
    plt.show()
    
    # Create a summary error plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Calculate per-timestep error
    per_step_error = torch.mean(torch.abs(predictions - ground_truths), dim=1).cpu().numpy()
    
    ax.plot(per_step_error, linewidth=2, color='orange')
    ax.set_title(f'Per-Timestep Mean Absolute Error - {model_type.upper()} (Episode {episode_idx})')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Mean Absolute Error')
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line for overall MAE
    overall_mae = np.mean(per_step_error)
    ax.axhline(y=overall_mae, color='red', linestyle='--', alpha=0.7, label=f'Overall MAE: {overall_mae:.4f}')
    ax.legend()
    
    if save_plots:
        error_plot_path = Path(output_dir) / f"{model_type}_episode_{episode_idx}_error_over_time.png"
        plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Saved error plot to: {error_plot_path}")
    
    plt.show()
    
    # Print summary statistics
    print(f"📊 Summary Statistics for Episode {episode_idx}:")
    print(f"   Dataset contains {len(predictions)} action steps")
    print(f"   Overall Mean Absolute Error: {overall_mae:.4f}")
    best_joint_idx = torch.argmin(torch.mean(torch.abs(predictions - ground_truths), dim=0))
    worst_joint_idx = torch.argmax(torch.mean(torch.abs(predictions - ground_truths), dim=0))
    print(f"   Best performing joint: {joint_names[best_joint_idx]}")
    print(f"   Worst performing joint: {joint_names[worst_joint_idx]}")

    if overall_mae < 0.01:
        print(f"\n🎉 Excellent! The {model_type.upper()} model has learned the demonstration very well.")
        print(f"   Next steps: Collect more diverse demonstrations for better generalization!")
    elif overall_mae < 0.1:
        print(f"\n✅ Good performance! The {model_type.upper()} model learned the general trajectory well.")
    else:
        print(f"\n🔧 Consider: More training steps, different learning rate, or data quality issues.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model (supports both model directories and checkpoint files)")
    parser.add_argument("model_path", help="Path to trained model directory or checkpoint file (.pt)")
    parser.add_argument("--dataset", default="bearlover365/red_cube_always_in_same_place")
    parser.add_argument("--episode", type=int, default=0, help="Episode to evaluate on")
    parser.add_argument("--compare-episodes", help="Comma-separated episode indices to compare")
    parser.add_argument("--plot", action="store_true", help="Generate plots showing predictions vs ground truth")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to disk")
    parser.add_argument("--output-dir", default="./plots", help="Directory to save plots")
    parser.add_argument("--use-simple-loader", action="store_true", help="Use simple data loading instead of DataLoader")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit to first N contiguous frames for speed")
    parser.add_argument("--eval-batch-size", type=int, default=64, help="Batch size for evaluation inference")
    parser.add_argument("--camera-remap", type=str, default=None,
                       help="Camera remapping (e.g., 'observation.images.front:observation.images.wrist')")
    parser.add_argument("--model-type", type=str, default=None, 
                       help="Force model type (act, diffusion, vqbet, smolvla, pi0fast). Auto-detected if not specified.")
    # WandB logging
    parser.add_argument("--wandb.enable", action="store_true", dest="wandb_enable")
    parser.add_argument("--wandb.project", type=str, default="lerobot")
    parser.add_argument("--wandb.entity", type=str, default=None)
    parser.add_argument("--wandb.run_id", type=str, default=None)
    parser.add_argument("--step", type=int, default=None, help="Training step for logging alignment")
    
    args = parser.parse_args()
    
    # Parse camera remapping
    camera_remap = None
    if args.camera_remap:
        camera_remap = {}
        for mapping in args.camera_remap.split(','):
            if ':' in mapping:
                old_key, new_key = mapping.strip().split(':')
                camera_remap[old_key.strip()] = new_key.strip()
        print(f"📷 Camera remapping: {camera_remap}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    print(f"📂 Model: {args.model_path}")
    print(f"📊 Dataset: {args.dataset}")

    # Infer step from model path if not provided
    step_num = args.step
    if step_num is None:
        m = re.findall(r"(\d{3,})", str(args.model_path))
        if m:
            try:
                step_num = int(m[-1])
            except ValueError:
                step_num = None
    
    if not Path(args.model_path).exists():
        print(f"❌ Model path does not exist: {args.model_path}")
        return 1
    
    # Detect or use specified model type
    model_type = args.model_type if args.model_type else detect_model_type(args.model_path)
    print(f"🧠 Model type: {model_type.upper()}")
    
    try:
        # Optionally force saving plots if wandb enabled with images
        force_save = False
        if args.wandb_enable and not args.save_plots:
            force_save = True
            args.save_plots = True
            print("🟡 Enabling --save-plots to log images to WandB")

        # Optionally init WandB
        wb = None
        if args.wandb_enable:
            import wandb
            wb = wandb
            wb.init(project=args.wandb_project, entity=args.wandb_entity, id=args.wandb_run_id, resume="allow")
        if args.compare_episodes:
            # Compare performance across multiple episodes
            episodes = [int(x.strip()) for x in args.compare_episodes.split(",")]
            print(f"\n🔄 Comparing {model_type.upper()} performance across episodes: {episodes}")
            
            results = {}
            for ep in episodes:
                mae, mse, max_err, predictions, ground_truths, joint_names = evaluate_model_on_episode(
                    args.model_path, args.dataset, ep, device,
                    use_dataloader=not args.use_simple_loader, camera_remap=camera_remap, model_type=model_type,
                    max_frames=args.max_frames, eval_batch_size=args.eval_batch_size,
                )
                if mae is not None:
                    results[ep] = {"mae": mae, "mse": mse, "max_error": max_err}
                    
                    # Plot if requested
                    if args.plot and predictions is not None:
                        plot_predictions_vs_ground_truth(
                            predictions, ground_truths, joint_names, ep, model_type,
                            save_plots=args.save_plots, output_dir=args.output_dir
                        )
            
            # Summary
            print(f"\n📈 Summary:")
            print(f"{'Episode':<8} {'MAE':<12} {'MSE':<12} {'Max Error':<12}")
            print("-" * 48)
            for ep, metrics in results.items():
                print(f"{ep:<8} {metrics['mae']:<12.6f} {metrics['mse']:<12.6f} {metrics['max_error']:<12.6f}")
            
            if results:
                best_ep = min(results.keys(), key=lambda x: results[x]["mae"])
                worst_ep = max(results.keys(), key=lambda x: results[x]["mae"])
                print(f"\n🏆 Best episode: {best_ep} (MAE: {results[best_ep]['mae']:.6f})")
                print(f"🤔 Worst episode: {worst_ep} (MAE: {results[worst_ep]['mae']:.6f})")
                
                if best_ep == args.episode:
                    print("✅ Model performs best on its training episode (expected)")
                else:
                    print("🤯 Model generalizes! Performs better on different episode")
        
        else:
            # Single episode evaluation
            mae, mse, max_err, predictions, ground_truths, joint_names = evaluate_model_on_episode(
                args.model_path, args.dataset, args.episode, device,
                use_dataloader=not args.use_simple_loader, camera_remap=camera_remap, model_type=model_type,
                max_frames=args.max_frames, eval_batch_size=args.eval_batch_size,
            )
            
            # Plot if requested and we have data
            if args.plot and predictions is not None:
                plot_predictions_vs_ground_truth(
                    predictions, ground_truths, joint_names, args.episode, model_type,
                    save_plots=args.save_plots, output_dir=args.output_dir
                )

            # Log to WandB
            if wb is not None and predictions is not None:
                # Scalars
                log_data = {
                    "eval/mae_ep0": float(mae) if mae is not None else None,
                    "eval/mse_ep0": float(mse) if mse is not None else None,
                    "eval/maxerr_ep0": float(max_err) if max_err is not None else None,
                }
                if joint_names is not None and predictions is not None and ground_truths is not None:
                    per_joint_mae = torch.mean(torch.abs(predictions - ground_truths), dim=0).tolist()
                    for j, v in enumerate(per_joint_mae):
                        log_data[f"eval/mae_ep0_joint_{j}"] = float(v)
                wb.log(log_data, step=step_num)

                # Images if saved
                pred_plot = Path(args.output_dir) / f"{model_type}_episode_{args.episode}_predictions.png"
                err_plot = Path(args.output_dir) / f"{model_type}_episode_{args.episode}_error_over_time.png"
                if pred_plot.exists():
                    wb.log({"eval/pred_vs_gt": wb.Image(str(pred_plot), caption=f"step {step_num}")}, step=step_num)
                if err_plot.exists():
                    wb.log({"eval/error_over_time": wb.Image(str(err_plot), caption=f"step {step_num}")}, step=step_num)
        
        return 0
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 