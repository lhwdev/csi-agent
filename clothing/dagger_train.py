import os
import sys
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Callable
import torch

from lerobot.datasets import LeRobotDataset
from train import train_dagger, TrainMiscArgs
from split_dataset import split_dataset


def train_dagger_round(
    train_params,
    dataset: LeRobotDataset,
    current_training_round: int,
    new_epochs: int,
    pretrained_path: str | Path | None = None,
    val_ratio: float = 0.1,
    seed: int = 28,
    policy = None,
    cfg = None,
    device: torch.device | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    optimizer_cache = None,
):
    out_dir = Path(train_params.output_directory)
    checkpoints_out_dir = out_dir / "checkpoints"
    
    # 1. Setup output directory and copy files if first training run
    has_checkpoints = False
    if checkpoints_out_dir.exists():
        for p in checkpoints_out_dir.iterdir():
            if p.is_dir() and p.name.isdigit():
                has_checkpoints = True
                break
                
    if not has_checkpoints and pretrained_path is not None:
        src_chk_dir = Path(pretrained_path).resolve()
        if src_chk_dir.exists():
            chk_name = src_chk_dir.name  # e.g., "007449"
            target_chk_dir = checkpoints_out_dir / chk_name
            
            print(f"First training run: copying original checkpoint {chk_name} to {target_chk_dir}...")
            checkpoints_out_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy the checkpoint directory as-is
            shutil.copytree(src_chk_dir, target_chk_dir, dirs_exist_ok=True)
            
            # Create 'last' symlink pointing to the copied step folder
            last_link = checkpoints_out_dir / "last"
            if last_link.is_symlink() or last_link.exists():
                last_link.unlink()
            last_link.symlink_to(chk_name)
            
            # Also copy loss_history.json if it exists in the original model directory (grandparent)
            if "checkpoints" in src_chk_dir.parts:
                idx = src_chk_dir.parts.index("checkpoints")
                original_path = Path(*src_chk_dir.parts[:idx])
                src_loss = original_path / "loss_history.json"
                if src_loss.exists():
                    shutil.copy2(src_loss, out_dir / "loss_history.json")
                    print("Copied loss history to output directory.")
        else:
            print(f"Warning: Pretrained path {pretrained_path} does not exist. Cannot copy.")
            
    # 2. Extract episodes that have round > 0 (new corrections) to force them into training set
    hf_dataset = dataset.hf_dataset
    force_train_episodes = set()
    if "round" in hf_dataset.column_names and "episode_index" in hf_dataset.column_names:
        rounds = hf_dataset["round"]
        episode_indices = hf_dataset["episode_index"]
        for r, ep in zip(rounds, episode_indices):
            if r is None:
                continue
            r_val = r[0] if (isinstance(r, list) or isinstance(r, tuple)) else r
            if r_val is None:
                continue
            r_val = int(r_val)
            if r_val > 0:
                if ep is None:
                    continue
                ep_val = ep[0] if (isinstance(ep, list) or isinstance(ep, tuple)) else ep
                if ep_val is None:
                    continue
                ep_val = int(ep_val)
                force_train_episodes.add(ep_val)
                
    # 3. Split dataset
    train_ds, val_ds = split_dataset(dataset, val_ratio=val_ratio, seed=seed, force_train_episodes=force_train_episodes)
    
    # 4. Dynamic training steps based on new frames
    beta = getattr(train_params, "dagger_rehearsal_beta", 0.3)
    
    hf_dataset = train_ds.hf_dataset
    newest_round_frames = 0
    if "round" in hf_dataset.column_names:
        rounds = hf_dataset["round"]
        for r in rounds:
            if r is None:
                continue
            val = r[0] if (isinstance(r, list) or isinstance(r, tuple)) else r
            if val is None:
                continue
            if int(val) == current_training_round:
                newest_round_frames += 1

    if newest_round_frames > 0:
        computed_steps = int(((new_epochs * newest_round_frames) / beta) / train_params.batch_size)
        train_params.training_steps = max(getattr(train_params, "training_steps", 100), computed_steps)
        print(f"Dynamic steps: {train_params.training_steps} for {newest_round_frames} new frames in train set.")
    else:
        print(f"No new frames in round {current_training_round} in train set. Using default steps: {train_params.training_steps}")
        
    # 5. Load policy/config if not passed
    if policy is None:
        last_checkpoint_dir = checkpoints_out_dir / "last"
        if not last_checkpoint_dir.exists():
            raise ValueError(f"No checkpoint found to load policy from at {last_checkpoint_dir}.")
            
        print(f"Loading policy from {last_checkpoint_dir}...")
        from lerobot.configs.policies import PreTrainedConfig
        p_cfg = PreTrainedConfig.from_pretrained(last_checkpoint_dir, local_files_only=True)
        policy_type = getattr(p_cfg, "policy_type", None)
        
        if policy_type == "act":
            from lerobot.policies.act.modeling_act import ACTPolicy
            policy = ACTPolicy.from_pretrained(last_checkpoint_dir)
            cfg = policy.config
        elif policy_type == "smolvla":
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            policy = SmolVLAPolicy.from_pretrained(last_checkpoint_dir)
            cfg = policy.config
        else:
            raise ValueError(f"Unsupported policy type: {policy_type}")
            
    if cfg is None:
        cfg = policy.config
        
    # 6. Configure batch_transform if ACTPolicy
    batch_transform = None
    if policy.__class__.__name__ == "ACTPolicy":
        def act_batch_transform(batch):
            for k in list(batch.keys()):
                if k.startswith("observation.") and isinstance(batch[k], torch.Tensor) and batch[k].ndim == (5 if "images" in k else 3):
                    batch[k] = batch[k].squeeze(1)
            return batch
        batch_transform = act_batch_transform
        
    # 7. Set current training round inside policy config so it is saved in config.json
    policy.config.current_training_round = current_training_round
    
    # 8. Train the policy
    if device is None:
        device = torch.device("xpu" if torch.xpu.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        
    misc_args = TrainMiscArgs(
        device=device,
        batch_transform=batch_transform,
        optimizer=optimizer_cache,
        progress_callback=progress_callback
    )
    
    train_dagger(
        policy=policy,
        cfg=cfg,
        train_params=train_params,
        dataset=train_ds,
        current_training_round=current_training_round,
        val_dataset=val_ds,
        misc_args=misc_args,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Standalone DAgger Training")
    parser.add_argument("--repo_id", type=str, required=True, help="Dataset repo ID (e.g., records/rollout_hil_towel_fold01_step0)")
    parser.add_argument("--root_dir", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output training directory")
    parser.add_argument("--current_round", type=int, default=1, help="Current training round index (1-based)")
    parser.add_argument("--new_epochs", type=int, default=5, help="Number of epochs to train on the new data")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Path to pretrained checkpoint (for initialization)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--training_steps", type=int, default=100, help="Default/minimum training steps")
    parser.add_argument("--save_freq", type=int, default=100, help="Checkpoint save frequency")
    parser.add_argument("--log_freq", type=int, default=10, help="Logging frequency")
    parser.add_argument("--val_freq", type=int, default=200, help="Validation frequency")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=28, help="Random seed for splitting")
    parser.add_argument("--device", type=str, default=None, help="Device (e.g., cuda, cpu, xpu)")
    parser.add_argument("--dagger_rehearsal_alpha", type=float, default=0.5, help="Rehearsal alpha")
    parser.add_argument("--dagger_rehearsal_beta", type=float, default=0.3, help="Rehearsal beta")
    
    args = parser.parse_args()
    
    torch.set_float32_matmul_precision("high")
    
    # Load dataset
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.root_dir,
    )
    
    # Build train_params
    train_params = SimpleNamespace(
        output_directory=args.output_dir,
        batch_size=args.batch_size,
        training_steps=args.training_steps,
        save_freq=args.save_freq,
        log_freq=args.log_freq,
        val_freq=args.val_freq,
        dagger_rehearsal_alpha=args.dagger_rehearsal_alpha,
        dagger_rehearsal_beta=args.dagger_rehearsal_beta,
    )
    
    device = torch.device(args.device) if args.device else None
    
    train_dagger_round(
        train_params=train_params,
        dataset=dataset,
        current_training_round=args.current_round,
        new_epochs=args.new_epochs,
        pretrained_path=args.pretrained_path,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device=device,
    )
