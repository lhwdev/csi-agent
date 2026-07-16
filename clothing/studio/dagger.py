import time
import asyncio
from pathlib import Path
import cv2
import shutil
import traceback
import copy
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch

from lerobot.robots import Robot
from lerobot.teleoperators import Teleoperator
from lerobot.datasets import LeRobotDataset
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.common.control_utils import (
    teleop_supports_feedback,
    teleop_smooth_move_to,
    follower_smooth_move_to,
)
from lerobot.configs.policies import PreTrainedConfig
from lerobot.rollout.strategies.core import ActionInterpolator

from rollout import create_rollout_context
from dagger_train import train_dagger_round
from studio.core import BaseInteractiveStudio
from studio.dagger_ui import DAggerUIMixin

SMOOTH_MOVE_DURATION = 1.0


class DAggerInteractiveStudio(DAggerUIMixin, BaseInteractiveStudio):
    def __init__(self, robot, leader, params):
        self.dagger_phase = "IDLE"
        self.current_training_round = 1
        self.record_corrections_only = getattr(params, "record_corrections_only", True)
        self.autonomous_ui_fps = 2.0
        self.autonomous_plot_fps = 1.0
        
        # Clone datasets logic
        steps_val = getattr(params, "steps", None)
        if steps_val is None:
            steps_val = [{
                "repo_id": getattr(params, "repo_id", None),
                "root_dir": getattr(params, "root_dir", None),
                "task": getattr(params, "task", None)
            }]
        steps_val = copy.deepcopy(steps_val)
            
        for step in steps_val:
            original_repo = step["repo_id"]
            original_root = step["root_dir"]
            
            # Change name to rollout_hil_...
            original_repo_separator = original_repo.index("/")
            new_repo = f"{original_repo[:original_repo_separator]}/rollout_hil_{original_repo[original_repo_separator+1:]}"
            original_root_path = Path(original_root)
            new_root_name = f"rollout_hil_{original_root_path.name}"
            new_root_path = original_root_path.parent / new_root_name
            
            # clone if not exists
            if not new_root_path.exists() and original_root_path.exists():
                print(f"Cloning {original_root_path} to {new_root_path}")
                shutil.copytree(original_root_path, new_root_path)
            
            step["repo_id"] = new_repo
            step["root_dir"] = str(new_root_path)
            step["original_root"] = str(original_root_path)
            
        params.steps = steps_val
        super().__init__(robot, leader, params)

        self.optimizer_cache = None

    def get_dataset_features(self):
        features = super().get_dataset_features()
        if "intervention" not in features:
            features["intervention"] = {"dtype": "bool", "shape": (1,)}
        if "round" not in features:
            features["round"] = {"dtype": "int32", "shape": (1,)}
        return features

    def get_online_training_params(self):
        output_dir = getattr(self.params, "output_dir", None)
        if output_dir is None:
            # Try to resolve it from the policy
            pretrained_path = None
            if getattr(self.params, "policies", None) and len(self.params.policies) > 0:
                p_cfg = self.params.policies[0]
                if hasattr(p_cfg, "pretrained_path"):
                    pretrained_path = p_cfg.pretrained_path
            
            if pretrained_path is not None:
                p = Path(pretrained_path).resolve()
                if "checkpoints" in p.parts:
                    idx = p.parts.index("checkpoints")
                    original_path = Path(*p.parts[:idx])
                else:
                    original_path = p
                output_dir = str(original_path.parent / f"rollout_hil_{original_path.name}")
            else:
                output_dir = "./outputs/dagger_online"
                
        return {
            "training_steps": getattr(self.params, "dagger_train_steps", 100),
            "batch_size": getattr(self.params, "dagger_train_batch_size", 4),
            "log_freq": 10,
            "save_freq": 100,
            "output_directory": output_dir,
        }

    def load_episode_rounds_cache(self):
        self.episode_rounds_cache = {}
        if not self.datasets or self.current_dataset_step_idx >= len(self.datasets):
            return
        ds = self.datasets[self.current_dataset_step_idx]
        try:
            if ds.num_episodes == 0:
                return
            hf_dataset = ds.hf_dataset
            if hf_dataset is not None and "episode_index" in hf_dataset.column_names:
                episodes = hf_dataset["episode_index"]
                rounds = hf_dataset["round"] if "round" in hf_dataset.column_names else None
                
                ep_arr = np.array(episodes)
                if rounds is not None:
                    r_arr = np.array(rounds)
                else:
                    r_arr = np.zeros_like(ep_arr)
                
                unique_eps, unique_indices = np.unique(ep_arr, return_index=True)
                for ep_val, idx in zip(unique_eps, unique_indices):
                    if ep_val is None or (isinstance(ep_val, float) and np.isnan(ep_val)):
                        continue
                    r_val = r_arr[idx]
                    if isinstance(r_val, (list, tuple)):
                        r_val = r_val[0] if len(r_val) > 0 else 0
                    elif hasattr(r_val, "ndim") and r_val.ndim > 0:
                        try:
                            r_val = r_val[0]
                        except Exception:
                            pass
                    
                    if hasattr(r_val, "item"):
                        try:
                            r_val = r_val.item()
                        except Exception:
                            pass
                    
                    if r_val is None or (isinstance(r_val, float) and np.isnan(r_val)):
                        r_val = 0
                    
                    try:
                        self.episode_rounds_cache[int(ep_val)] = int(r_val)
                    except (ValueError, TypeError):
                        pass
        except Exception as e:
            self.add_log(f"Warning: Failed to load episode rounds cache: {e}")

    def on_datasets_initialized(self):
        # Resolve policy_path
        policy_path = getattr(self.params, "policy_path", None)
        if policy_path is None:
            # Fallback for backward compatibility
            policies = getattr(self.params, "policies", [])
            if len(policies) > 0 and hasattr(policies[0], "pretrained_path"):
                policy_path = str(policies[0].pretrained_path)
                
        # Resolve original_path from policy_path
        original_path = None
        if policy_path is not None:
            p = Path(policy_path).resolve()
            if "checkpoints" in p.parts:
                idx = p.parts.index("checkpoints")
                original_path = Path(*p.parts[:idx])
            else:
                original_path = p
                
        # Resolve dagger_policy_path (acts as output_dir)
        dagger_policy_path = getattr(self.params, "dagger_policy_path", None)
        if dagger_policy_path is None:
            if original_path is not None:
                dagger_policy_path = str(original_path.parent / f"rollout_hil_{original_path.name}")
            else:
                dagger_policy_path = "./outputs/dagger_online"
                
        self.params.output_dir = dagger_policy_path
        self.params.dagger_policy_path = dagger_policy_path
        self.params.policy_path = policy_path
        
        # Determine actual policy loading path: dagger_policy_path if exists and has checkpoints, else policy_path
        actual_policy_load_path = policy_path
        if dagger_policy_path is not None:
            last_chk = Path(dagger_policy_path) / "checkpoints" / "last"
            if last_chk.exists():
                actual_policy_load_path = str(last_chk)
                self.add_log(f"Resuming from online policy: {actual_policy_load_path}")
            else:
                if policy_path is not None:
                    self.add_log(f"Loading original policy: {policy_path}")
                else:
                    self.add_log("Warning: No policy path provided to load.")
                    
        # Load config and override self.params.policies
        if actual_policy_load_path is not None:
            try:
                policy_cfg = PreTrainedConfig.from_pretrained(actual_policy_load_path, local_files_only=True)
                policy_cfg.pretrained_path = actual_policy_load_path
                self.params.policies = [policy_cfg]
            except Exception as e:
                self.add_log(f"Error loading policy configuration from {actual_policy_load_path}: {e}")
                traceback.print_exc()

        self.add_log("Preloading policies... This may take a few seconds.")
        policies = getattr(self.params, "policies", [])
        if getattr(self.params, "_rollout_contexts", None) is None:
            self.params._rollout_contexts = {}
            
        try:
            for idx, policy_cfg in enumerate(policies):
                if policy_cfg is not None and idx not in self.params._rollout_contexts:
                    ctx = create_rollout_context(
                        robot=self.robot, policy=policy_cfg, task=getattr(self.params, "task", ""),
                        fps=30.0, asynchronous=getattr(self.params, "asynchronous", False),
                        compile=getattr(self.params, "compile", False), rename_map=getattr(self.params, "rename_map", None),
                        teleop_action_processor=self.teleop_action_processor, robot_action_processor=self.robot_action_processor,
                        robot_observation_processor=self.robot_observation_processor
                    )
                    ctx._interpolator = ActionInterpolator(multiplier=2)
                    ctx.policy.inference.start()
                    self.params._rollout_contexts[idx] = ctx
                    self.add_log(f"Preloaded policy {idx+1}/{len(policies)}.")
        except Exception as e:
            self.add_log(f"Warning: Failed to preload policies: {e}")
            
        self.start_btn.disabled = False
        self.prev_step_btn.disabled = True
        self.next_step_btn.disabled = True
        self.update_navigation_buttons()
        self.recording_state = "IDLE"
        self.dagger_phase = "AUTONOMOUS"
        self.current_dataset_step_idx = 0
        
        # Initialize training round from policy config and dataset cache
        policy_round = 0
        if policies and policies[0] is not None:
            policy_round = getattr(policies[0], "current_training_round", 0)
            
        self.load_episode_rounds_cache()
        self._initial_episode_idx = self.current_episode_idx
        max_round = 0
        if getattr(self, "episode_rounds_cache", None):
            max_round = max(self.episode_rounds_cache.values())
            
        self.current_training_round = max(1, policy_round + 1, max_round)
        if hasattr(self, "round_counter"):
            self.round_counter.value = self.current_training_round
        self.add_log(f"Initialized training round to {self.current_training_round} (Policy round: {policy_round}, Max dataset round: {max_round}).")
        
        # Build and append the Training tab panel
        self.build_training_panel()
        self.update_training_info()

    async def start_recording_async(self):
        if self.recording_state != "IDLE": return
        
        self.recording_state = "RECORDING"
        self.dagger_phase = "AUTONOMOUS"
        self.frames_in_episode = 0
        setattr(self.params, "_needs_reset", True)
        
        self.start_btn.disabled = True
        self.pause_btn.disabled = False
        self.correction_btn.disabled = False
        self.prev_step_btn.disabled = True
        self.next_step_btn.disabled = True
        self.stop_btn.disabled = False
        self.discard_btn.disabled = False
        self.train_btn.disabled = True
        
        self.add_log(f"Started HIL Rollout for round {self.current_training_round}...")
        self.update_status_card()

    def on_start_clicked(self, b):
        self.active_bg_task = self.schedule_background_task(self.start_recording_async())

    def on_pause_clicked(self, b):
        if self.recording_state == "RECORDING":
            self.recording_state = "PAUSED"
            self.add_log("HIL execution paused. Motors locked.")
        elif self.recording_state == "PAUSED":
            self.recording_state = "RECORDING"
            self.add_log("HIL execution resumed.")
        self.update_status_card()

    def enable_leader_torque(self):
        try:
            self.leader.enable_torque()
        except Exception as e:
            self.add_log(f"Warning: Failed to enable leader torque: {e}")

    def disable_leader_torque(self):
        try:
            self.leader.disable_torque()
        except Exception as e:
            self.add_log(f"Warning: Failed to disable leader torque: {e}")

    def smooth_move_leader_to(self, target_pos: dict, duration_s: float = SMOOTH_MOVE_DURATION):
        try:
            self.add_log(f"Moving leader to target in {duration_s}s...")
            self.enable_leader_torque()
            teleop_smooth_move_to(self.leader, target_pos, duration_s)
            self.add_log("Leader alignment completed.")
        except Exception as e:
            self.add_log(f"Warning: smooth leader move failed: {e}")

    async def smooth_move_follower_to_async(self, target_action: dict, duration_s: float = SMOOTH_MOVE_DURATION):
        try:
            self.add_log(f"Smoothing follower movement to target ({duration_s}s)...")
            current_action = getattr(self, "current_robot_action", None)
            if current_action is None:
                current_action = {}
                obs = await asyncio.to_thread(self.robot.get_observation)
                for k in target_action.keys():
                    if k in obs:
                        current_action[k] = obs[k]
                    else:
                        current_action[k] = target_action[k]
            async with self.robot_lock:
                await asyncio.to_thread(follower_smooth_move_to, self.robot, current_action, target_action, duration_s)
            self.add_log("Follower alignment completed.")
        except Exception as e:
            self.add_log(f"Warning: smooth follower move failed: {e}")

    async def _move_leader_to_follower_state_async(self):
        """Smoothly move leader to current follower (inference) state when entering CORRECTING_LOCKED."""
        try:
            async with self.robot_lock:
                obs = await asyncio.to_thread(self.robot.get_observation)
            target_dict = {k: v for k, v in obs.items() if k.endswith(".pos")}
            if target_dict:
                await asyncio.to_thread(self.smooth_move_leader_to, target_dict)
                self.add_log("Leader aligned to follower state. Ready for correction.")
            else:
                self.add_log("Warning: No joint positions found in observation for leader alignment.")
        except Exception as e:
            self.add_log(f"Warning: Failed to move leader to follower state: {e}")

    def on_correction_clicked(self, b):
        if self.recording_state == "RECORDING":
            if self.dagger_phase == "AUTONOMOUS":
                self.dagger_phase = "CORRECTING_LOCKED"
                self.correction_btn.description = "Engage (C)"
                self.correction_btn.button_style = "danger"
                self.add_log("Intervention triggered! Follower frozen. Moving leader to follower state...")
                # Smoothly move leader to current follower (inference) state
                self.active_bg_task = self.schedule_background_task(
                    self._move_leader_to_follower_state_async()
                )
            elif self.dagger_phase == "CORRECTING_LOCKED":
                self.dagger_phase = "CORRECTING_ACTIVE"
                self.correction_btn.description = "Correction (C)"
                self.correction_btn.button_style = "warning"
                self.disable_leader_torque()
                self.add_log("Engaged! Human correction active. Leader controls follower.")
            elif self.dagger_phase == "CORRECTING_ACTIVE":
                self.dagger_phase = "AUTONOMOUS"
                self.correction_btn.description = "Correction (C)"
                self.correction_btn.button_style = "warning"
                self.disable_leader_torque()
                setattr(self.params, "_needs_reset", True)
                self.add_log("Switched back to AUTONOMOUS control mode.")
        elif self.recording_state == "PAUSED":
            self.add_log("Cannot intervene while paused.")

    def on_next_step_clicked(self, b):
        pass

    def on_prev_step_clicked(self, b):
        pass

    async def return_to_initial_posture_async(self):
        self.add_log("Aligning teleoperator for autonomous return...")
        self.dagger_phase = "RETURNING"
        
        # Read current follower observation
        async with self.robot_lock:
            obs = await asyncio.to_thread(self.robot.get_observation)
        
        # Prepare target posture from start of current round/dataset if possible
        target_dict = {}
        for k in obs.keys():
            if k.endswith(".pos"):
                # Align leader directly with current follower pose to allow smooth handoff
                target_dict[k] = obs[k]
                
        if target_dict:
            await asyncio.gather(
                asyncio.to_thread(self.smooth_move_leader_to, target_dict),
                self.smooth_move_follower_to_async(target_dict)
            )
            self.disable_leader_torque()

    async def save_current_episode_async(self):
        if self.recording_state == "SAVING": return
        self.recording_state = "SAVING"
        self.update_status_card()
        
        self.start_btn.disabled = True
        self.pause_btn.disabled = True
        self.correction_btn.disabled = True
        self.stop_btn.disabled = True
        self.discard_btn.disabled = True
        self.train_btn.disabled = True
        
        self.add_log(f"Finalizing episode {self.current_episode_idx}...")
        try:
            if self.frames_in_episode > 0:
                # Save episode
                for ds in self.datasets:
                    await asyncio.to_thread(ds.save_episode)
                    await asyncio.to_thread(self.flush_dataset, ds)
                    
                # Update round cache
                self.episode_rounds_cache[self.current_episode_idx] = self.current_training_round
                self.current_episode_idx += 1
                self.episode_progress.value = self.current_episode_idx
                
                # Update newest round frames cache
                round_idx = self.current_training_round
                if not hasattr(self, "_newest_round_frames_cache"):
                    self._newest_round_frames_cache = {}
                self._newest_round_frames_cache[round_idx] = self._newest_round_frames_cache.get(round_idx, 0) + getattr(self, "frames_in_episode", 0)
                
                self.update_training_info()
                self.add_log(f"Episode {self.current_episode_idx - 1} saved successfully.")
            else:
                self.add_log("Episode has 0 frames. Discarding instead of saving.")
                for ds in self.datasets:
                    await asyncio.to_thread(ds.clear_episode_buffer, delete_images=True)
        except Exception as e:
            self.add_log(f"Error saving episode: {e}")
            traceback.print_exc()
        finally:
            try:
                await self.return_to_initial_posture_async()
            except Exception as e:
                self.add_log(f"Error in return_to_initial_posture_async: {e}")
            self.recording_state = "IDLE"
            self.dagger_phase = "AUTONOMOUS"
            self.current_dataset_step_idx = 0
            self.update_status_card()
            
            self.start_btn.disabled = False
            self.pause_btn.disabled = True
            self.correction_btn.disabled = True
            self.stop_btn.disabled = True
            self.discard_btn.disabled = True
            self.train_btn.disabled = False

    def on_train_clicked(self, b):
        self.active_bg_task = self.schedule_background_task(self.train_model_async())

    async def train_model_async(self):
        if self.recording_state not in ["IDLE", "PAUSED"]: 
            self.add_log("Must stop recording before training.")
            return

        self.recording_state = "TRAINING"
        self.update_status_card()
        
        self.start_btn.disabled = True
        self.train_btn.disabled = True
        self.add_log("Starting online training...")

        # Auto Tab Switching
        self.prev_tab_index = self.studio_tab.selected_index
        self.studio_tab.selected_index = self.training_tab_index
        self.train_status_widget.value = "<div class='train-status' style='font-size: 11px; color: #495057; font-family: monospace; border-top: 1px solid #dee2e6; margin-top: 4px; padding-top: 4px;'>Status: Starting...</div>"

        try:
            # Refresh dataset reader so it contains all newly saved episodes
            for ds in self.datasets:
                if hasattr(ds, "reader") and ds.reader is not None:
                    self.add_log("Refreshing dataset reader...")
                    await asyncio.to_thread(ds.reader.load_and_activate)

            # Train the first policy for now, assuming 1 policy setup
            if getattr(self.params, "_rollout_contexts", None) and 0 in self.params._rollout_contexts:
                ctx = self.params._rollout_contexts[0]
                policy_ctx = ctx.policy
                policy = policy_ctx.policy
                
                # Make sure the policy engine isn't running inference
                if hasattr(policy_ctx, "inference"):
                    policy_ctx.inference.stop()
                
                device = torch.device("xpu" if torch.xpu.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
                
                # Fetch dataset
                ds = self.datasets[0]
                
                # Prepare arguments for train_generic
                train_params = SimpleNamespace(**self.get_online_training_params())
                train_params.dagger_rehearsal_beta = getattr(self.params, "dagger_rehearsal_beta", 0.3)
                train_params.dagger_rehearsal_alpha = getattr(self.params, "dagger_rehearsal_alpha", 0.5)

                pretrained_path = None
                if getattr(self.params, "policies", None) and len(self.params.policies) > 0:
                    p_cfg = self.params.policies[0]
                    if hasattr(p_cfg, "pretrained_path"):
                        pretrained_path = p_cfg.pretrained_path

                def run_training():
                    train_dagger_round(
                        train_params=train_params,
                        dataset=ds,
                        current_training_round=getattr(self, "current_training_round", 1),
                        new_epochs=self.epochs_slider.value,
                        pretrained_path=pretrained_path,
                        policy=policy,
                        cfg=policy.config,
                        device=device,
                        progress_callback=self.on_train_progress,
                        optimizer_cache=self.optimizer_cache,
                    )
                    
                await asyncio.to_thread(run_training)
                self.add_log("Training completed successfully.")

                current_round = getattr(self, "current_training_round", 0) + 1
                self.current_training_round = current_round
                if hasattr(self, "round_counter"):
                    self.round_counter.value = self.current_training_round
                self.update_training_info()
                self.add_log(f"Training round {current_round} completed. Datasets are aggregated.")

                # Restart inference engine
                if hasattr(policy_ctx, "inference"):
                    policy_ctx.inference.start()
                    
            else:
                self.add_log("Error: No rollout context found to train.")
                
        except Exception as e:
            self.add_log(f"Training failed: {e}")
            traceback.print_exc()
        finally:
            if hasattr(self, "_newest_round_frames_cache"):
                self._newest_round_frames_cache.clear()
            self.recording_state = "IDLE"
            self.dagger_phase = "AUTONOMOUS"
            self.update_status_card()

            # Switch focus back
            if hasattr(self, "prev_tab_index") and self.prev_tab_index is not None:
                self.studio_tab.selected_index = self.prev_tab_index

            self.train_status_widget.value = "<div class='train-status' style='font-size: 11px; color: #495057; font-family: monospace; border-top: 1px solid #dee2e6; margin-top: 4px; padding-top: 4px;'>Status: Completed</div>"

            self.start_btn.disabled = False
            self.train_btn.disabled = False

    def on_stop_clicked(self, b):
        self.active_bg_task = self.schedule_background_task(self.save_current_episode_async())

    async def on_discard_clicked_async(self):
        if self.recording_state == "SAVING": return
        self.recording_state = "SAVING"
        self.update_status_card()
        
        self.add_log("Discarding current episode buffer...")
        try:
            for ds in self.datasets:
                await asyncio.to_thread(ds.clear_episode_buffer, delete_images=True)
            self.add_log("Episode buffer discarded.")
        except Exception as e:
            self.add_log(f"Error discarding episode: {e}")
        finally:
            try:
                await self.return_to_initial_posture_async()
            except Exception as e:
                self.add_log(f"Error in return_to_initial_posture_async: {e}")
            self.recording_state = "IDLE"
            self.dagger_phase = "AUTONOMOUS"
            self.current_dataset_step_idx = 0
            self.update_status_card()
            
            self.start_btn.disabled = False
            self.pause_btn.disabled = True
            self.correction_btn.disabled = True
            self.stop_btn.disabled = True
            self.discard_btn.disabled = True
            self.train_btn.disabled = False

    def on_discard_clicked(self, b):
        self.active_bg_task = self.schedule_background_task(self.on_discard_clicked_async())

    async def main_loop(self):
        self.robot_lock = asyncio.Lock()
        robot_lock = self.robot_lock
        loop_fps = 30.0
        last_widget_update_time = 0.0
        last_telemetry_update_time = 0.0
        
        try:
            obs = await asyncio.to_thread(self.robot.get_observation)
            raw_action = await asyncio.to_thread(self.leader.get_action)
            obs_processed = self.robot_observation_processor(obs)
            teleop_action = self.teleop_action_processor((raw_action, obs))
            robot_action = self.robot_action_processor((teleop_action, obs))
        except Exception as e:
            self.add_log(f"CRITICAL ERROR: Failed to fetch initial state: {e}")
            self.cleanup_studio_sync()
            return

        active_encode_tasks = {}

        async def encode_and_update_bg(key, img, widget, update_widget, update_stream):
            try:
                bgr_img = await asyncio.to_thread(cv2.cvtColor, img, cv2.COLOR_RGB2BGR)
                _, jpeg_encoded = await asyncio.to_thread(cv2.imencode, '.jpg', bgr_img)
                jpeg_bytes = jpeg_encoded.tobytes()
                if update_widget: widget.value = jpeg_bytes
                if update_stream: self.streamer.update_frame(key, jpeg_bytes)
            except Exception: pass
            finally: active_encode_tasks.pop(key, None)

        async def teleop_loop():
            nonlocal raw_action, teleop_action, robot_action, loop_fps
            last_sent_action = None

            while self.keep_running:
                if self.recording_state == "TRAINING":
                    await asyncio.sleep(0.1)
                    continue
                start_loop = time.perf_counter()
                try:
                    if (self.dagger_phase == "AUTONOMOUS" and self.recording_state != "IDLE") or self.dagger_phase in ["CORRECTING_LOCKED", "RETURNING"]:
                        new_raw_action = raw_action
                        new_teleop_action = teleop_action
                    else:
                        try:
                            new_raw_action = await asyncio.to_thread(self.leader.get_action)
                            new_teleop_action = self.teleop_action_processor((new_raw_action, obs))
                        except ConnectionError:
                            new_raw_action = raw_action
                            new_teleop_action = teleop_action
                    
                    self.last_obs = obs

                    if self.dagger_phase == "RETURNING":
                        new_robot_action = None
                        self.current_teleop_action = new_teleop_action
                    elif self.recording_state == "IDLE" or self.dagger_phase == "CORRECTING_ACTIVE":
                        new_robot_action = self.robot_action_processor((new_teleop_action, obs))
                        self.current_teleop_action = new_teleop_action
                    elif self.dagger_phase == "CORRECTING_LOCKED" or self.recording_state in ["PAUSED", "TRAINING", "SAVING"]:
                        new_robot_action = last_sent_action
                        self.current_teleop_action = new_teleop_action
                    elif self.dagger_phase == "AUTONOMOUS":
                        policies = getattr(self.params, "policies", [])
                        if self.current_dataset_step_idx < len(policies):
                            policy_cfg = policies[self.current_dataset_step_idx]
                            if policy_cfg is not None:
                                if getattr(self.params, "_rollout_contexts", None) is None:
                                    self.params._rollout_contexts = {}
                                if self.current_dataset_step_idx not in self.params._rollout_contexts:
                                    ctx = create_rollout_context(
                                        robot=self.robot, policy=policy_cfg, task=getattr(self.params, "task", ""),
                                        fps=loop_fps, asynchronous=getattr(self.params, "asynchronous", False),
                                        compile=getattr(self.params, "compile", False), rename_map=getattr(self.params, "rename_map", None),
                                        teleop_action_processor=self.teleop_action_processor, robot_action_processor=self.robot_action_processor,
                                        robot_observation_processor=self.robot_observation_processor
                                    )
                                    ctx._interpolator = ActionInterpolator(multiplier=2)
                                    ctx.policy.inference.start()
                                    self.params._rollout_contexts[self.current_dataset_step_idx] = ctx
                                
                                ctx = self.params._rollout_contexts[self.current_dataset_step_idx]
                                engine = ctx.policy.inference
                                interpolator = ctx._interpolator
                                
                                if getattr(self.params, "_needs_reset", False):
                                    engine.reset()
                                    interpolator.reset()
                                    setattr(self.params, "_needs_reset", False)
                                    engine.resume()
                                
                                if getattr(engine, "ready", True):
                                    if interpolator.needs_new_action():
                                        engine.notify_observation(obs_processed)
                                        obs_frame = build_dataset_frame(ctx.data.dataset_features, obs_processed, prefix="observation")
                                        action_tensor = engine.get_action(obs_frame)
                                        if action_tensor is not None:
                                            if hasattr(action_tensor, "cpu"): action_tensor = action_tensor.cpu()
                                            interpolator.add(action_tensor)
                                    interp = interpolator.get()
                                    if interp is not None:
                                        ordered_keys = ctx.data.ordered_action_keys
                                        action_dict = {k: interp[i].item() for i, k in enumerate(ordered_keys)}
                                        new_robot_action = self.robot_action_processor((action_dict, obs))
                                        self.current_teleop_action = action_dict
                                    else:
                                        new_robot_action = last_sent_action
                                        self.current_teleop_action = new_teleop_action
                            else:
                                new_robot_action = last_sent_action
                                self.current_teleop_action = new_teleop_action
                        else:
                            new_robot_action = last_sent_action
                            self.current_teleop_action = new_teleop_action

                    self.current_robot_action = new_robot_action

                    async with robot_lock:
                        if new_robot_action is not None:
                            await asyncio.to_thread(self.robot.send_action, new_robot_action)
                            last_sent_action = new_robot_action
                    
                    raw_action = new_raw_action
                    teleop_action = new_teleop_action
                    robot_action = new_robot_action

                    target_fps = 30.0 if self.dagger_phase in ["CORRECTING_ACTIVE", "IDLE"] else float(self.fps_slider.value)
                    actual_dt = time.perf_counter() - start_loop
                    sleep_time = max(0.0001, (1.0 / target_fps) - actual_dt)
                    await asyncio.sleep(sleep_time)
                    
                    actual_dt_after_sleep = time.perf_counter() - start_loop
                    if actual_dt_after_sleep > 0: loop_fps = 0.9 * loop_fps + 0.1 * (1.0 / actual_dt_after_sleep)
                except Exception as e:
                    self.add_log(f"teleop loop error: {e}")
                    traceback.print_exc()
                    await asyncio.sleep(0.01)

        async def observation_loop():
            nonlocal obs, obs_processed, last_widget_update_time, last_telemetry_update_time
            while self.keep_running:
                if self.recording_state == "TRAINING":
                    await asyncio.sleep(0.1)
                    continue
                start_loop = time.perf_counter()
                try:
                    async with robot_lock:
                        new_obs = await asyncio.to_thread(self.robot.get_observation)
                    obs = new_obs
                    obs_processed = self.robot_observation_processor(obs)
                except Exception:
                    await asyncio.sleep(0.01)
                    continue

                self.streamer.update_telemetry(obs, raw_action)

                if self.recording_state == "RECORDING":
                    if self.dagger_phase in ["AUTONOMOUS", "CORRECTING", "CORRECTING_ACTIVE"]:
                        if self.datasets and self.current_dataset_step_idx < len(self.datasets):
                            ds = self.datasets[self.current_dataset_step_idx]
                            
                            if "intervention" not in ds.features:
                                ds.features["intervention"] = {"dtype": "bool", "shape": (1,)}
                                if hasattr(ds, "writer") and hasattr(ds.writer, "episode_buffer") and ds.writer.episode_buffer is not None:
                                    if "intervention" not in ds.writer.episode_buffer:
                                        ds.writer.episode_buffer["intervention"] = []
                            if "round" not in ds.features:
                                ds.features["round"] = {"dtype": "int32", "shape": (1,)}
                                if hasattr(ds, "writer") and hasattr(ds.writer, "episode_buffer") and ds.writer.episode_buffer is not None:
                                    if "round" not in ds.writer.episode_buffer:
                                        ds.writer.episode_buffer["round"] = []

                            for key, feat in ds.features.items():
                                if key.startswith(f"{OBS_STR}.images."):
                                    cam_key = key.replace(f"{OBS_STR}.images.", "")
                                    if cam_key in obs_processed:
                                        img = obs_processed[cam_key]
                                        expected_shape = feat["shape"]
                                        target_h, target_w = (expected_shape[0], expected_shape[1]) if expected_shape[-1] == 3 else (expected_shape[1], expected_shape[2])
                                        if img.shape[:2] != (target_h, target_w):
                                            obs_processed[cam_key] = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

                            observation_frame = build_dataset_frame(ds.features, obs_processed, prefix=OBS_STR)
                            
                            intervention = False
                            if self.dagger_phase == "CORRECTING_ACTIVE":
                                action_frame = build_dataset_frame(ds.features, getattr(self, "current_teleop_action", teleop_action), prefix=ACTION)
                                intervention = True
                            else:
                                action_frame = build_dataset_frame(ds.features, getattr(self, "current_teleop_action", teleop_action), prefix=ACTION)
                                
                            task_str = self.steps_val[self.current_dataset_step_idx]["task"]
                            
                            frame_data = {
                                **observation_frame,
                                **action_frame,
                                "task": task_str,
                                "intervention": np.array([intervention], dtype=bool),
                                "round": np.array([getattr(self, "current_training_round", 0)], dtype=np.int32)
                            }
                            if action_frame and len(action_frame) > 0:
                                record_this_frame = True
                                if getattr(self, "record_corrections_only", False):
                                    if not intervention:
                                        record_this_frame = False
                                        
                                if record_this_frame:
                                    ds.add_frame(frame_data)
                                    self.frames_in_episode += 1

                now = time.perf_counter()
                ui_fps = self.fps_control.value
                if self.dagger_phase == "AUTONOMOUS":
                    ui_fps = self.autonomous_ui_fps
                should_update_widgets = (now - last_widget_update_time >= 1.0 / ui_fps)
                
                for key, widget in [("left_cam", self.left_camera_widget), ("top", self.top_camera_widget), ("right_cam", self.right_camera_widget)]:
                    if key in obs:
                        has_stream_client = self.streamer.active_clients.get(key, 0) > 0
                        if has_stream_client or should_update_widgets:
                            if key not in active_encode_tasks:
                                task = asyncio.create_task(
                                    encode_and_update_bg(key, obs[key].copy(), widget, should_update_widgets, has_stream_client)
                                )
                                active_encode_tasks[key] = task
                                
                if should_update_widgets: last_widget_update_time = now
                
                now = time.perf_counter()
                if now - last_telemetry_update_time >= 1.0 / ui_fps:
                    self.update_telemetry(obs, raw_action, loop_fps, getattr(self, "current_robot_action", None))
                    last_telemetry_update_time = now
                    if self.recording_state == "RECORDING":
                        self.update_status_card()

                target_fps = 30.0 if self.dagger_phase in ["CORRECTING_ACTIVE", "IDLE"] else float(self.fps_slider.value)
                dt = time.perf_counter() - start_loop
                sleep_time = max(0.0, (1.0 / target_fps) - dt)
                await asyncio.sleep(sleep_time)

        try:
            await asyncio.gather(teleop_loop(), observation_loop())
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.add_log(f"Error in studio loop: {e}")
            traceback.print_exc()
        finally:
            self.cleanup_studio_sync()


def rollout_interactive_dagger(robot, leader, params):
    DAggerInteractiveStudio(robot, leader, params)
