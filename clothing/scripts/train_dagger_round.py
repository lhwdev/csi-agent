# 이 코드 아직 오류가 있슴다... 뭐가 문제지

batch_size = 4
current_round=5
epoch = 2.5
save_freq = 1000

import torch
from steps import steps

torch.set_float32_matmul_precision("high")

from lerobot.datasets import LeRobotDataset
from types import SimpleNamespace
import math
from split_dataset import split_dataset
import dagger_train

step_config = steps[0]

repo_id = "lhwdev/rollout_hil_towel_fold01_step0"
root_dir = "../records/rollout_hil_towel_fold01_step0"

print(f"===== Train for {repo_id} ==========")

dataset = LeRobotDataset(
    repo_id,
    root=root_dir,
)

train_params = SimpleNamespace(
    output_directory = f"../train/rollout_hil_towel_fold01_step0",
    batch_size = batch_size,

    save_freq = save_freq,
    val_freq = 200,
    log_freq = 100,
)

dagger_train.train_dagger_round(train_params, dataset, current_training_round=current_round, new_epochs=epoch)
