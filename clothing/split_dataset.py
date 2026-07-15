import hashlib
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def split_dataset(
    dataset: LeRobotDataset,
    val_ratio: float = 0.1,
    seed: int = 42,
    force_train_episodes: set[int] | None = None,
) -> tuple[LeRobotDataset, LeRobotDataset | None]:
    """
    Splits a LeRobotDataset on the fly into training and validation datasets
    without writing any new files to disk.

    This split is deterministic-random based on a hash of each episode index,
    meaning the assignment of existing episodes does not change when the total
    episode count increases.

    Args:
        dataset: The source LeRobotDataset to split.
        val_ratio: The fraction of episodes to allocate to the validation set.
        seed: The random seed/prefix for hashing to ensure reproducibility and randomness.

    Returns:
        A tuple of (train_dataset, val_dataset). If val_dataset has no episodes,
        returns (train_dataset, None).
    """
    # Get the list of episode indices in the current dataset
    if dataset.episodes is not None:
        episodes = list(dataset.episodes)
    else:
        episodes = list(range(dataset.meta.total_episodes))

    if len(episodes) == 0:
        raise ValueError("Cannot split an empty dataset (no episodes found).")

    train_episodes = []
    val_episodes = []

    for ep_idx in episodes:
        if force_train_episodes is not None and ep_idx in force_train_episodes:
            train_episodes.append(ep_idx)
            continue
            
        # Generate a deterministic hash of the seed and the episode index
        hash_input = f"{seed}_{ep_idx}".encode("utf-8")
        hash_val = int(hashlib.md5(hash_input).hexdigest(), 16)
        # Compute a pseudo-random float between 0 and 1
        fraction = (hash_val % 10000) / 10000.0
        
        if fraction < val_ratio:
            val_episodes.append(ep_idx)
        else:
            train_episodes.append(ep_idx)

    # Sort episode lists to maintain numerical order
    train_episodes = sorted(train_episodes)
    val_episodes = sorted(val_episodes)

    # Instantiate train and validation LeRobotDatasets on the fly
    train_dataset = LeRobotDataset(
        repo_id=dataset.repo_id,
        root=dataset.root,
        episodes=train_episodes,
        tolerance_s=dataset.tolerance_s,
        revision=dataset.revision,
    )

    val_dataset = None
    if len(val_episodes) > 0:
        val_dataset = LeRobotDataset(
            repo_id=dataset.repo_id,
            root=dataset.root,
            episodes=val_episodes,
            tolerance_s=dataset.tolerance_s,
            revision=dataset.revision,
        )

    return train_dataset, val_dataset
