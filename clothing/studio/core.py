import time
import asyncio
from pathlib import Path
import shutil
import json

from lerobot.robots import Robot
from lerobot.teleoperators import Teleoperator
from lerobot.datasets import LeRobotDataset, aggregate_pipeline_dataset_features, create_initial_features
from lerobot.utils.feature_utils import combine_feature_dicts
from lerobot.processor import make_default_processors

from lerobot_streamer import CameraStreamer
from studio.patches import JointPlotter
from studio.ui import StudioUIMixin

_active_studio_cleanup = None


class BaseInteractiveStudio(StudioUIMixin):
    """
    Base class for interactive studios.
    Handles communication with devices, dataset initialization, streamer setups, and main loops.
    """
    def __init__(self, robot: Robot, leader: Teleoperator, params):
        global _active_studio_cleanup
        if _active_studio_cleanup is not None:
            try:
                _active_studio_cleanup()
            except Exception:
                pass
            _active_studio_cleanup = None

        self.robot = robot
        self.leader = leader
        self.params = params

        self.datasets = []
        self.current_dataset_step_idx = 0
        
        if params is None:
            raise ValueError("The 'params' argument is required and cannot be None.")

        self.steps_val = getattr(params, "steps", None)
        if self.steps_val is None:
            self.steps_val = [{
                "repo_id": getattr(params, "repo_id", None),
                "root_dir": getattr(params, "root_dir", None),
                "task": getattr(params, "task", None)
            }]
        
        for i, step in enumerate(self.steps_val):
            if not step.get("repo_id") or not step.get("root_dir") or not step.get("task"):
                raise ValueError(f"Step {i} in params must contain 'repo_id', 'root_dir', and 'task'.")

        self.total_episodes = getattr(params, "episodes", 30)
        self.telemetry_fps = getattr(params, "telemetry_fps", 15)

        self.active_bg_task = None
        self.recording_state = "IDLE"
        self.current_episode_idx = 0
        self.frames_in_episode = 0
        self.keep_running = True

        self.episode_start_time = 0
        self.accumulated_time = 0

        self.streamer = CameraStreamer()
        self.streamer.stop_all_captures()
        self.streamer_port = self.streamer.start_server()
        print(f"Camera Stream URL: {self.streamer.get_server_url()}")

        if not self.robot.is_connected:
            self.robot.connect()
        if self.leader is not None and not getattr(self.leader, "is_connected", False):
            self.leader.connect()

        try:
            self.teleop_action_processor
        except AttributeError:
            self.teleop_action_processor, self.robot_action_processor, self.robot_observation_processor = make_default_processors()

        self.plotter = JointPlotter()
        self.plotter_is_updating = False

        self.setup_ui()
        _active_studio_cleanup = self.cleanup_studio_sync

        self.initialize_dataset()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        loop.create_task(self.main_loop())
        self.add_log("Studio started in background. Cell execution finished.")

    def schedule_background_task(self, coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        return loop.create_task(coro)

    def initialize_dataset(self):
        self.add_log(f"Initializing {len(self.steps_val)} datasets...")
        self.recording_state = "SAVING"
        self.update_status_card()
        
        try:
            dataset_features = self.get_dataset_features()
            
            self.datasets.clear()
            for step_cfg in self.steps_val:
                repo_id = step_cfg["repo_id"]
                root_dir = step_cfg["root_dir"]
                
                path = Path(root_dir)
                ds = None
                if path.exists():
                    try:
                        ds = LeRobotDataset.resume(repo_id=repo_id, root=root_dir, streaming_encoding=True)
                        self.current_episode_idx = ds.num_episodes
                    except Exception as e:
                        is_empty = False
                        if path.is_dir():
                            if not any(path.iterdir()):
                                is_empty = True
                            else:
                                info_path = path / "meta" / "info.json"
                                if info_path.exists():
                                    try:
                                        with open(info_path, "r") as f:
                                            info = json.load(f)
                                            if info.get("total_episodes", 0) == 0:
                                                is_empty = True
                                    except Exception:
                                        pass
                                        
                        if not is_empty:
                            raise RuntimeError(f"Directory {root_dir} exists and is not empty, but could not be resumed. Error: {e}")
                            
                        self.add_log(f"Warning: could not resume {root_dir} ({e}). Recreating it.")
                        try:
                            if path.is_dir():
                                shutil.rmtree(path)
                            else:
                                path.unlink()
                        except Exception as e2:
                            self.add_log(f"Failed to clear directory {root_dir}: {e2}")

                if ds is None:
                    ds = LeRobotDataset.create(
                        repo_id=repo_id, fps=30, root=root_dir, robot_type=self.robot.name,
                        features=dataset_features, use_videos=True, streaming_encoding=True
                    )
                    self.current_episode_idx = 0
                self.datasets.append(ds)
                
            self.add_log("All datasets initialized successfully.")
            self.episode_progress.max = self.total_episodes
            self.episode_progress.value = self.current_episode_idx
            
            self.on_datasets_initialized()
        except Exception as e:
            self.add_log(f"Error initializing dataset: {str(e)}")
            import traceback
            traceback.print_exc()
            self.recording_state = "IDLE"
            self.update_status_card()

    def on_datasets_initialized(self):
        pass

    def get_dataset_features(self):
        return combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=self.teleop_action_processor,
                initial_features=create_initial_features(action=self.robot.action_features),
                use_videos=True,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=self.robot_observation_processor,
                initial_features=create_initial_features(observation=self.robot.observation_features),
                use_videos=True,
            ),
        )

    def flush_dataset(self, ds, refresh_reader=True):
        """Forces the dataset writer to increment its file chunk index so that the next episode starts in a fresh file."""
        if hasattr(ds, "writer") and ds.writer is not None:
            w = ds.writer
            if hasattr(w, "close_writer"):
                w.close_writer()
            if hasattr(w, "_meta") and hasattr(w._meta, "_close_writer"):
                w._meta._close_writer()
            
            # Reload episodes in metadata to ensure we don't use stale episode arrays on next save
            if hasattr(w, "_meta") and w._meta is not None:
                try:
                    from lerobot.datasets.dataset_metadata import load_episodes
                    w._meta.episodes = load_episodes(w._root)
                except Exception as e:
                    print(f"Warning: failed to reload metadata episodes: {e}")
                w._meta.latest_episode = None
                
            w._latest_episode = None
            
            # Refresh the reader so that any access to ds.hf_dataset gets the new episodes
            if refresh_reader and hasattr(ds, "reader") and ds.reader is not None:
                try:
                    ds.reader.load_and_activate()
                except Exception as e:
                    print(f"Warning: failed to refresh dataset reader: {e}")

    async def main_loop(self):
        pass
        
    def cleanup_studio_sync(self):
        if not getattr(self, "studio_cleaned_up", False):
            self.studio_cleaned_up = True
            self.keep_running = False
            try:
                self.plotter.cleanup()
                self.streamer.stop_all_captures()
            except Exception: pass
            
            if self.datasets:
                for ds in self.datasets:
                    if ds and not getattr(ds, "_is_finalized", False):
                        try: ds.finalize()
                        except Exception: pass
