import os
import sys
import json
from pathlib import Path
import torch
from tqdm.auto import tqdm
from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.feature_utils import dataset_to_policy_features

def train_simple(train_params, dataset_metadata: LeRobotDatasetMetadata):
    # 1. Output directory
    output_directory = Path(train_params.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # 2. Select device
    device = torch.device("xpu" if torch.xpu.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # 3. Load dataset metadata

    # 4. Determine if resuming or starting from scratch
    last_checkpoint_dir = output_directory / "checkpoints" / "last"

    if (last_checkpoint_dir / "state.json").exists():
        print(f"Found existing checkpoint at {last_checkpoint_dir.resolve()}. Resuming training...")
        
        # Load state
        with open(last_checkpoint_dir / "state.json", "r") as f:
            state = json.load(f)
        start_step = state["step"]
        
        # Load policy from checkpoint
        policy = ACTPolicy.from_pretrained(last_checkpoint_dir)
        cfg = policy.config
    else:
        print("No existing checkpoint found. Starting training from scratch...")
        start_step = 0
        
        # Map dataset features to policy features
        features = dataset_to_policy_features(dataset_metadata.features)
        output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        input_features = {key: ft for key, ft in features.items() if key not in output_features}
        
        # Create configuration and policy
        cfg = ACTConfig(input_features=input_features, output_features=output_features)
        policy = ACTPolicy(cfg)

    # Ensure correct device is set on config and policy
    cfg.device = str(device)
    policy.to(device)
    policy.train()

    # 6. Create processors
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    # 7. Dynamically build delta_timestamps
    delta_timestamps = {}
    for key, feature in dataset_metadata.features.items():
        if key.startswith("observation.images.") or key == "observation.state":
            delta_timestamps[key] = [0.0]
        elif key == "action":
            delta_timestamps[key] = [i / dataset_metadata.fps for i in range(cfg.chunk_size)]

    # 8. Instantiate dataset and dataloader
    dataset = LeRobotDataset(
        dataset_metadata.repo_id,
        root=dataset_metadata.root,
        delta_timestamps=delta_timestamps
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=train_params.batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # 9. Create optimizer and load state if resuming
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.optimizer_lr, weight_decay=cfg.optimizer_weight_decay)
    if start_step > 0:
        optimizer.load_state_dict(torch.load(last_checkpoint_dir / "optimizer.pt", map_location=device))

    # 10. Run training loop
    step = start_step
    done = False
    print(f"Staging training from step {step} to {train_params.training_steps}...")

    progress_bar = tqdm(initial=step, total=train_params.training_steps, desc="Training")

    while not done:
        for batch in dataloader:
            # Move tensors to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Squeeze the sequence/time dimension (T=1) for all observation tensors,
            # as ACTPolicy expects 2D states [B, D] and 4D images [B, C, H, W] rather than sequence tensors.
            for k in list(batch.keys()):
                if k.startswith("observation.") and isinstance(batch[k], torch.Tensor) and batch[k].ndim == (5 if "images" in k else 3):
                    batch[k] = batch[k].squeeze(1)
            
            # Preprocess batch
            batch = preprocessor(batch)
            
            # Forward and backward pass
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if step % train_params.log_freq == 0:
                progress_bar.set_postfix(loss=f"{loss.item():.5f}")
                
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
