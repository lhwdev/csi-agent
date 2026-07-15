import torch

torch.set_float32_matmul_precision("high")

clothing_name = "towel_fold01"

steps = [
  {
    "repo_id": f"lhwdev/{clothing_name}_step0",
    "root_dir": f"/home/lerobot/lerobot2/csi-agent/lhwdev/records/{clothing_name}_step0",
    "task": "Unfold the clothes."
  },
  {
    "repo_id": f"lhwdev/{clothing_name}_step1",
    "root_dir": f"/home/lerobot/lerobot2/csi-agent/lhwdev/records/{clothing_name}_step1",
    "task": "Fold the clothes."
  },
  {
    "repo_id": f"lhwdev/{clothing_name}_step2",
    "root_dir": f"/home/lerobot/lerobot2/csi-agent/lhwdev/records/{clothing_name}_step2",
    "task": "Rotate the clothes."
  },
  {
    "repo_id": f"lhwdev/{clothing_name}_step3",
    "root_dir": f"/home/lerobot/lerobot2/csi-agent/lhwdev/records/{clothing_name}_step3",
    "task": "Fold the clothes."
  }
]

batch_size = 16
epoch = 5
save_freq = 1000

from lerobot.datasets import LeRobotDataset
from types import SimpleNamespace
import math
from split_dataset import split_dataset
import train

for i, step_config in enumerate(steps):
    repo_id = step_config["repo_id"]
    root_dir = step_config["root_dir"]

    print(f"===== Train for {repo_id} ==========")
    
    dataset = LeRobotDataset(
        repo_id,
        root=root_dir,
    )

    train_dataset, val_dataset = split_dataset(dataset, val_ratio=0.1, seed=28)
    
    total_frames = train_dataset.num_frames
    total_per_epoch = math.ceil(total_frames / batch_size)

    training_steps = epoch * total_per_epoch

    train_params = SimpleNamespace(
        output_directory = f"../train/{clothing_name}_step{i}",
        batch_size = batch_size,

        training_steps = training_steps,
        save_freq = save_freq,
        val_freq = 200,
        log_freq = 100,
    )

    train.train_simple_act(train_params, train_dataset, val_dataset)
