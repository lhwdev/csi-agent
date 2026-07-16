i = 0 # 단계: 수정해주세요

from lerobot.configs.policies import PreTrainedConfig
from pathlib import Path

from rollout import rollout
from steps import clothing_name, steps
step_config = steps[i]

pretrained_path = Path(f"../train/{clothing_name}_step{i}") / "checkpoints" / "last"
pretrained_path = pretrained_path.resolve()
policy_config = PreTrainedConfig.from_pretrained(pretrained_path, local_files_only=True)
policy_config.pretrained_path = pretrained_path

policies = [policy_config]
rollout_tasks = [step_config["task"]]

rollout(robot, policies, rollout_tasks, 20.0, asynchronous=False, compile=False)
