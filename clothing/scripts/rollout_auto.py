from lerobot.configs import PreTrainedConfig
from pathlib import Path

from rollout import rollout
from steps import steps

policies = []
rollout_tasks = []
for i, step_config in enumerate(steps):
    pretrained_path = Path(f"../train/rollout_hil_{clothing_name}_step{i}") / "checkpoints" / "last"
    pretrained_path = pretrained_path.resolve()
    policy_config = PreTrainedConfig.from_pretrained(pretrained_path, local_files_only=True)
    policy_config.pretrained_path = pretrained_path
    policies.append(policy_config)
    rollout_tasks.append(step_config["task"])

rollout(
    robot, policies, rollout_tasks,
    fps=30.0,
    asynchronous=False,
    compile=False,
    classifier_path="../train/towel_fold01_nextlevel",
)
