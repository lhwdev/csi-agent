#!/usr/bin/env python3
"""
Script to change the 'round' feature of a specific episode in a LeRobot dataset to an arbitrary number.
"""

import argparse
import glob
import json
import shutil
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

def change_episode_round(dataset_dir: str, episode_index: int, new_round: int):
    dataset_path = Path(dataset_dir).resolve()
    if not dataset_path.exists():
        print(f"Error: Dataset directory {dataset_path} does not exist.")
        return False

    # 1. Validate meta/info.json
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        print(f"Error: info.json not found at {info_path}")
        return False

    with open(info_path, "r") as f:
        info = json.load(f)

    # Ensure "features" section exists
    if "features" not in info:
        info["features"] = {}

    # Check and add "round" to features if missing
    if "round" not in info["features"]:
        info["features"]["round"] = {"dtype": "int32", "shape": [1]}
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)
        print(f"Added 'round' feature to info.json features list.")

    # 2. Glob all data parquet files
    data_dir = dataset_path / "data"
    parquet_files = sorted(data_dir.glob("**/*.parquet"))
    if not parquet_files:
        print(f"Error: No parquet files found in {data_dir}")
        return False

    modified_files_count = 0
    total_rows_updated = 0

    print(f"Scanning parquet files in {data_dir} for episode_index = {episode_index}...")

    for file_path in parquet_files:
        try:
            pf = pq.ParquetFile(file_path)
            # Read only the episode_index column to check if this file contains the target episode
            episode_col = pf.read(columns=["episode_index"]).column("episode_index").to_pylist()
        except Exception as e:
            print(f"Warning: Failed to read episode_index from {file_path.name}: {e}")
            continue

        if episode_index not in episode_col:
            continue

        print(f"Found episode {episode_index} in {file_path.relative_to(dataset_path)}")

        try:
            table = pq.read_table(file_path)
        except Exception as e:
            print(f"Error: Failed to read {file_path.name}: {e}")
            return False

        # Read or initialize columns
        episodes = table.column("episode_index").to_pylist()
        
        if "round" in table.schema.names:
            round_field = table.schema.field("round")
            round_type = round_field.type
            rounds = table.column("round").to_pylist()
        else:
            round_type = pa.int32()
            rounds = [0] * len(episodes)

        new_rounds = []
        rows_updated_in_file = 0
        for ep, r in zip(episodes, rounds):
            if ep == episode_index:
                rows_updated_in_file += 1
                if pa.types.is_list(round_type):
                    new_rounds.append([new_round])
                else:
                    new_rounds.append(new_round)
            else:
                new_rounds.append(r)

        # Reconstruct the table with the modified column
        if "round" in table.schema.names:
            round_idx = table.schema.names.index("round")
            table = table.set_column(round_idx, "round", pa.array(new_rounds, type=round_type))
        else:
            table = table.append_column("round", pa.array(new_rounds, type=round_type))

        # Write the table back
        try:
            pq.write_table(table, file_path, compression="snappy")
            modified_files_count += 1
            total_rows_updated += rows_updated_in_file
            print(f"Successfully updated {rows_updated_in_file} frames in {file_path.name}")
        except Exception as e:
            print(f"Error: Failed to write {file_path.name}: {e}")
            return False

    if modified_files_count > 0:
        print(f"\nSuccessfully modified {total_rows_updated} frames across {modified_files_count} file(s).")
        # Clean local cache directory to force HuggingFace to reload the modified parquet files
        cache_dir = dataset_path / ".cache"
        if cache_dir.exists():
            print(f"Clearing cache directory: {cache_dir.relative_to(dataset_path)}")
            shutil.rmtree(cache_dir, ignore_errors=True)
        return True
    else:
        print(f"Warning: Episode index {episode_index} was not found in any parquet files.")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Change the 'round' feature of a specific episode in a LeRobot dataset."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the LeRobot dataset directory (e.g. records/rollout_hil_towel_fold01_step0)"
    )
    parser.add_argument(
        "--episode_index",
        type=int,
        required=True,
        help="Index of the episode to modify (e.g. 100)"
    )
    parser.add_argument(
        "--round",
        type=int,
        required=True,
        help="The new round value to set for the episode (e.g. 9)"
    )
    
    args = parser.parse_args()
    success = change_episode_round(args.dataset_dir, args.episode_index, args.round)
    if success:
        print("Done!")
    else:
        print("Failed.")

if __name__ == "__main__":
    main()
