i = 0 # 단계: 수정해주세요

from lerobot.configs.policies import PreTrainedConfig
from pathlib import Path

from rollout import rollout
from steps import steps
step_config = steps[i]

pretrained_path = f"HyeonseokE/smolvla_towel_fold01_step{i}"
policy_config = PreTrainedConfig.from_pretrained(pretrained_path)

policy_config.pretrained_path = pretrained_path
policies = [policy_config]
rollout_tasks = step_config["task"]
pretrained_rename_map = {
    "observation.images.top": "observation.images.camera1",
    "observation.images.left_cam": "observation.images.camera2",
    "observation.images.right_cam": "observation.images.camera3",
}

rollout(robot, policies, rollout_tasks, 30.0, asynchronous=True, compile=False, rename_map=pretrained_rename_map)
