import os
import sys
import json
from pathlib import Path
from typing import Callable
import torch
from tqdm.auto import tqdm
try:
    import torchvision.transforms.v2 as T
except ImportError:
    import torchvision.transforms as T

from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.feature_utils import dataset_to_policy_features


def train_generic(
    policy,
    cfg,
    train_params,
    dataset: LeRobotDataset,
    val_dataset: LeRobotDataset | None = None,
    device: torch.device = None,
    batch_transform: Callable[[dict], dict] | None = None,
):
    # 1. Output directory
    output_directory = Path(train_params.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")

    # Ensure correct device is set on config and policy
    cfg.device = str(device)
    policy.to(device)
    policy.train()

    # Create processors
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset.meta.stats)

    # Dynamically build delta_timestamps
    delta_timestamps = {}
    for key, feature in dataset.features.items():
        if key.startswith("observation.images.") or key == "observation.state":
            delta_timestamps[key] = [0.0]
        elif key == "action":
            delta_timestamps[key] = [i / dataset.fps for i in range(cfg.chunk_size)]

    # Instantiate dataset and dataloader
    dataset = LeRobotDataset(
        dataset.repo_id,
        root=dataset.root,
        episodes=dataset.episodes,
        delta_timestamps=delta_timestamps,
        tolerance_s=dataset.tolerance_s,
        revision=dataset.revision,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=train_params.batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Instantiate validation dataset and dataloader if provided
    val_dataloader = None
    if val_dataset is not None:
        val_dataset = LeRobotDataset(
            val_dataset.repo_id,
            root=val_dataset.root,
            episodes=val_dataset.episodes,
            delta_timestamps=delta_timestamps,
            tolerance_s=val_dataset.tolerance_s,
            revision=val_dataset.revision,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            num_workers=4,
            batch_size=train_params.batch_size,
            shuffle=False,
            pin_memory=device.type != "cpu",
            drop_last=False,
        )

    # Build task mapping for datasets without language columns
    task_index_to_text = {}
    if getattr(dataset.meta, "tasks", None) is not None:
        try:
            for task_text, row in dataset.meta.tasks.iterrows():
                task_index_to_text[int(row["task_index"])] = task_text
        except Exception as e:
            print(f"Error parsing tasks from dataset meta: {e}")

    # Determine starting step from checkpoint or 0
    last_checkpoint_dir = output_directory / "checkpoints" / "last"
    start_step = 0
    if (last_checkpoint_dir / "state.json").exists():
        with open(last_checkpoint_dir / "state.json", "r") as f:
            state = json.load(f)
        start_step = state["step"]

    # Create optimizer
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.optimizer_lr,
        weight_decay=cfg.optimizer_weight_decay,
        betas=getattr(cfg, "optimizer_betas", (0.9, 0.999)),
        eps=getattr(cfg, "optimizer_eps", 1e-8),
    )
    if start_step > 0:
        optimizer.load_state_dict(torch.load(last_checkpoint_dir / "optimizer.pt", map_location=device))

    # Run training loop
    step = start_step
    done = False
    print(f"Staging training from step {step} to {train_params.training_steps}...")

    # Load existing loss history if resuming
    loss_history_path = output_directory / "loss_history.json"
    loss_history = []
    if start_step > 0 and loss_history_path.exists():
        try:
            with open(loss_history_path, "r") as f:
                loss_history = json.load(f)
            print(f"Loaded {len(loss_history)} previous loss entries.")
        except Exception as e:
            print(f"Could not load previous loss history: {e}")

    # Set up image data augmentations
    color_jitter = T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)
    affine = T.RandomAffine(degrees=2, translate=(0.03, 0.03))

    progress_bar = tqdm(initial=step, total=train_params.training_steps, desc="Training")

    while not done:
        for batch in dataloader:
            # Move tensors to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Populate 'task' for VLA if not present
            if "task" not in batch and "task_index" in batch:
                task_indices = batch["task_index"].cpu().numpy()
                batch["task"] = [task_index_to_text.get(int(idx), "Fold the clothes.") for idx in task_indices]

            # Apply batch transform callback if provided
            if batch_transform is not None:
                batch = batch_transform(batch)

            # Apply Image Data Augmentation (overfitting solution) to all models
            for k in list(batch.keys()):
                if k.startswith("observation.images.") and not k.endswith("_is_pad") and isinstance(batch[k], torch.Tensor):
                    tensor = batch[k]
                    if tensor.ndim == 4:  # [B, C, H, W]
                        augmented_imgs = []
                        for img in tensor:
                            img = color_jitter(img)
                            img = affine(img)
                            augmented_imgs.append(img)
                        batch[k] = torch.stack(augmented_imgs)
                    elif tensor.ndim == 5:  # [B, T, C, H, W]
                        B, T_dim, C, H, W = tensor.shape
                        flat_tensor = tensor.view(B * T_dim, C, H, W)
                        augmented_imgs = []
                        for img in flat_tensor:
                            img = color_jitter(img)
                            img = affine(img)
                            augmented_imgs.append(img)
                        batch[k] = torch.stack(augmented_imgs).view(B, T_dim, C, H, W)

            # Preprocess batch
            batch = preprocessor(batch)
            
            # Forward and backward pass
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Logging and periodic validation
            is_log_step = (step % train_params.log_freq == 0)
            val_freq = getattr(train_params, "val_freq", 200)
            is_val_step = (val_dataloader is not None and step % val_freq == 0)
            
            if is_log_step or is_val_step:
                loss_val = loss.item()
                postfix = {"loss": f"{loss_val:.5f}"}
                
                val_loss_avg = None
                if is_val_step:
                    policy.eval()
                    val_losses = []
                    with torch.no_grad():
                        val_steps = getattr(train_params, "val_steps", 20)
                        for val_idx, val_batch in enumerate(val_dataloader):
                            if val_idx >= val_steps:
                                break
                            # Move tensors to device
                            val_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in val_batch.items()}
                            
                            # Populate 'task' for VLA
                            if "task" not in val_batch and "task_index" in val_batch:
                                task_indices = val_batch["task_index"].cpu().numpy()
                                val_batch["task"] = [task_index_to_text.get(int(idx), "Fold the clothes.") for idx in task_indices]

                            # Apply batch transform callback if provided
                            if batch_transform is not None:
                                val_batch = batch_transform(val_batch)
                                    
                            # Preprocess
                            val_batch = preprocessor(val_batch)
                            
                            # Forward pass (without augmentation)
                            val_loss, _ = policy.forward(val_batch)
                            val_losses.append(val_loss.item())
                    
                    val_loss_avg = sum(val_losses) / len(val_losses) if val_losses else 0.0
                    progress_bar.write(f"Step {step}: Validation Loss = {val_loss_avg:.5f}")
                    postfix["val_loss"] = f"{val_loss_avg:.5f}"
                    policy.train()
                
                progress_bar.set_postfix(**postfix)
                
                # Append to history
                entry = {"step": step, "loss": loss_val}
                if val_loss_avg is not None:
                    entry["val_loss"] = val_loss_avg
                loss_history.append(entry)
                
                # Save loss history to JSON
                try:
                    with open(loss_history_path, "w") as f:
                        json.dump(loss_history, f, indent=2)
                except Exception as e:
                    pass
                
            step += 1
            progress_bar.update(1)
            
            # Periodic saving
            if step % train_params.save_freq == 0 or step >= train_params.training_steps:
                checkpoint_dir = output_directory / "checkpoints" / f"{step:06d}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                # Save policy, processors, and optimizer state
                policy.save_pretrained(checkpoint_dir)
                preprocessor.save_pretrained(checkpoint_dir)
                postprocessor.save_pretrained(checkpoint_dir)
                
                torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
                with open(checkpoint_dir / "state.json", "w") as f:
                    json.dump({"step": step}, f)
                    
                # Update 'last' symlink
                last_link = output_directory / "checkpoints" / "last"
                if last_link.is_symlink() or last_link.exists():
                    last_link.unlink()
                last_link.symlink_to(f"{step:06d}")
                progress_bar.write(f"Saved checkpoint to {checkpoint_dir}")
                
            if step >= train_params.training_steps:
                done = True
                break

    progress_bar.close()


def train_simple_act(train_params, dataset: LeRobotDataset, val_dataset: LeRobotDataset | None = None):
    # Determine if resuming or starting from scratch
    output_directory = Path(train_params.output_directory)
    last_checkpoint_dir = output_directory / "checkpoints" / "last"
    device = torch.device("xpu" if torch.xpu.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    if (last_checkpoint_dir / "state.json").exists():
        print(f"Found existing checkpoint at {last_checkpoint_dir.resolve()}. Resuming training...")
        policy = ACTPolicy.from_pretrained(last_checkpoint_dir)
        cfg = policy.config
    else:
        print("No existing checkpoint found. Starting training from scratch...")
        # Map dataset features to policy features
        features = dataset_to_policy_features(dataset.features)
        output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        input_features = {key: ft for key, ft in features.items() if key not in output_features}
        
        # Create configuration and policy
        cfg = ACTConfig(
            input_features=input_features,
            output_features=output_features,
            temporal_ensemble_coeff=0.01,
            n_action_steps=1,
        )
        policy = ACTPolicy(cfg)

    def act_batch_transform(batch):
        # Squeeze the sequence/time dimension (T=1) for all observation tensors,
        # as ACTPolicy expects 2D states [B, D] and 4D images [B, C, H, W] rather than sequence tensors.
        for k in list(batch.keys()):
            if k.startswith("observation.") and isinstance(batch[k], torch.Tensor) and batch[k].ndim == (5 if "images" in k else 3):
                batch[k] = batch[k].squeeze(1)
        return batch

    train_generic(
        policy=policy,
        cfg=cfg,
        train_params=train_params,
        dataset=dataset,
        val_dataset=val_dataset,
        device=device,
        batch_transform=act_batch_transform,
    )


def train_simple_smolvla(train_params, dataset: LeRobotDataset, val_dataset: LeRobotDataset | None = None):
    # Determine if resuming or starting from scratch
    output_directory = Path(train_params.output_directory)
    last_checkpoint_dir = output_directory / "checkpoints" / "last"
    device = torch.device("xpu" if torch.xpu.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    if (last_checkpoint_dir / "state.json").exists():
        print(f"Found existing checkpoint at {last_checkpoint_dir.resolve()}. Resuming training...")
        policy = SmolVLAPolicy.from_pretrained(last_checkpoint_dir)
        cfg = policy.config
    else:
        print("No existing checkpoint found. Starting training from scratch...")
        # Map dataset features to policy features
        features = dataset_to_policy_features(dataset.features)
        output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        input_features = {key: ft for key, ft in features.items() if key not in output_features}
        
        # Create configuration and policy
        cfg = SmolVLAConfig(
            input_features=input_features,
            output_features=output_features,
        )
        policy = SmolVLAPolicy(cfg)

    train_generic(
        policy=policy,
        cfg=cfg,
        train_params=train_params,
        dataset=dataset,
        val_dataset=val_dataset,
        device=device,
        batch_transform=None,
    )
