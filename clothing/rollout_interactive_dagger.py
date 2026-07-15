from studio_core import BaseInteractiveStudio

import torch
import copy
from types import SimpleNamespace
import shutil
from pathlib import Path
import asyncio
import time
import cv2
import traceback
import numpy as np
import ipywidgets as widgets
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.common.control_utils import teleop_supports_feedback

from rollout import create_rollout_context
from lerobot.rollout.strategies.core import ActionInterpolator
from lerobot.datasets import LeRobotDataset

# Monkey patch compute_episode_stats to avoid writing stats for intervention and round
try:
    import lerobot.datasets.compute_stats
    import lerobot.datasets.dataset_writer
    import lerobot.datasets.dataset_tools

    # Check if we have already saved the original function to prevent recursion on module reload
    if not hasattr(lerobot.datasets.compute_stats, "_original_compute_episode_stats"):
        lerobot.datasets.compute_stats._original_compute_episode_stats = lerobot.datasets.compute_stats.compute_episode_stats

    _orig_compute_episode_stats = lerobot.datasets.compute_stats._original_compute_episode_stats

    def _patched_compute_episode_stats(episode_data, features, quantile_list=None):
        filtered_episode_data = {k: v for k, v in episode_data.items() if k not in {"intervention", "round"}} if episode_data else episode_data
        filtered_features = {k: v for k, v in features.items() if k not in {"intervention", "round"}} if features else features
        return _orig_compute_episode_stats(filtered_episode_data, filtered_features, quantile_list)

    lerobot.datasets.compute_stats.compute_episode_stats = _patched_compute_episode_stats
    lerobot.datasets.dataset_writer.compute_episode_stats = _patched_compute_episode_stats
    lerobot.datasets.dataset_tools.compute_episode_stats = _patched_compute_episode_stats
except Exception as e:
    print(f"Warning: Failed to monkey patch compute_episode_stats: {e}")

class DAggerInteractiveStudio(BaseInteractiveStudio):
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
            
            # Clean up any existing episodes metadata files that might have stats for intervention or round
            meta_episodes_dir = new_root_path / "meta" / "episodes"
            if meta_episodes_dir.exists():
                import pandas as pd
                for p in meta_episodes_dir.glob("**/*.parquet"):
                    try:
                        df = pd.read_parquet(p)
                        cols_to_drop = [c for c in df.columns if "intervention" in c or "round" in c]
                        if cols_to_drop:
                            print(f"Cleaning up metadata file {p}: dropping {cols_to_drop}")
                            df.drop(columns=cols_to_drop).to_parquet(p)
                    except Exception as e:
                        print(f"Warning: failed to clean up {p}: {e}")
            
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
        return {
            "training_steps": getattr(self.params, "dagger_train_steps", 100),
            "batch_size": getattr(self.params, "dagger_train_batch_size", 4),
            "log_freq": 10,
            "save_freq": 100,
            "output_directory": getattr(self.params, "output_dir", "./outputs/dagger_online"),
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
                
                import numpy as np
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
        self.dagger_phase = "AUTONOMOUS" # or IDLE before started
        self.current_dataset_step_idx = 0
        
        # Initialize training round from policy config and dataset cache
        policy_round = 0
        if policies and policies[0] is not None:
            policy_round = getattr(policies[0], "current_training_round", 0)
            
        self.load_episode_rounds_cache()
        max_round = 0
        if getattr(self, "episode_rounds_cache", None):
            max_round = max(self.episode_rounds_cache.values())
            
        self.current_training_round = max(1, policy_round + 1, max_round)
        if hasattr(self, "round_counter"):
            self.round_counter.value = self.current_training_round
        self.add_log(f"Initialized training round to {self.current_training_round} (Policy round: {policy_round}, Max dataset round: {max_round}).")
        
        self.update_status_card()
        self.add_log("DAgger studio ready. Start episode to begin AUTONOMOUS mode.")

    def build_controls(self):
        super().build_controls()
        # Repurpose standard buttons
        self.pause_btn.description = "Pause/Resume (P)"
        self.correction_btn = widgets.Button(description="Correction (C)", icon="edit", button_style="warning", disabled=True)
        self.train_btn = widgets.Button(description="Train (T)", icon="cogs", button_style="info", disabled=False)
        self.fps_slider = widgets.IntSlider(value=30, min=1, max=30, step=1, description='Policy FPS:', continuous_update=False)
        
        self.round_counter = widgets.IntText(value=getattr(self, "current_training_round", 1), description='Round:', layout=widgets.Layout(width='140px'))
        def on_round_change(change):
            self.current_training_round = change['new']
            self.update_status_card()
        self.round_counter.observe(on_round_change, names='value')
        
        self.start_btn.on_click(self.on_start_clicked)
        self.pause_btn.on_click(self.on_pause_clicked)
        self.correction_btn.on_click(self.on_correction_clicked)
        self.train_btn.on_click(self.on_train_clicked)
        self.next_step_btn.on_click(self.on_next_step_clicked)
        self.prev_step_btn.on_click(self.on_prev_step_clicked)
        self.stop_btn.on_click(self.on_stop_clicked)
        self.discard_btn.on_click(self.on_discard_clicked)

        self.control_row_1.children = [self.start_btn, self.pause_btn, self.correction_btn, self.prev_step_btn, self.next_step_btn]
        self.control_row_2.children = [self.stop_btn, self.discard_btn, self.train_btn, self.fps_slider, self.round_counter]

    def build_shortcuts(self):
        super().build_shortcuts()
        self.shortcut_legend.value = "<b>Shortcuts:</b> Space/R: Start, P: Pause/Resume, C: Correction, S: Save, D: Discard, T: Train, N: Next Step, B: Prev Step, Q: Exit"
        def on_shortcut_change(change):
            key = change['new']
            if not key: return
            key = key.lower()
            self.shortcut_input.value = ""
            if not self.shortcut_toggle.value: return
            
            if key == 'n' and not self.next_step_btn.disabled: self.on_next_step_clicked(None)
            elif key == 'b' and not self.prev_step_btn.disabled: self.on_prev_step_clicked(None)
            elif key in (' ', 'space', 'r'):
                if self.recording_state == "IDLE" and not self.start_btn.disabled: self.on_start_clicked(None)
                elif self.dagger_phase in ["AUTONOMOUS", "CORRECTING_LOCKED", "CORRECTING_ACTIVE", "PAUSED"] and not self.correction_btn.disabled: 
                    self.on_correction_clicked(None)
            elif key == 'p' and not self.pause_btn.disabled: self.on_pause_clicked(None)
            elif key == 'c' and not self.correction_btn.disabled: self.on_correction_clicked(None)
            elif key == 's' and not self.stop_btn.disabled: self.on_stop_clicked(None)
            elif key == 'd' and not self.discard_btn.disabled: self.on_discard_clicked(None)
            elif key == 't' and not self.train_btn.disabled: self.on_train_clicked(None)
            elif key == 'q': self.cleanup_studio_sync()

        self.shortcut_input.observe(on_shortcut_change, names='value')

    def update_navigation_buttons(self):
        if self.datasets and self.recording_state == "IDLE":
            self.prev_btn.disabled = (self.current_episode_idx == 0)
            self.next_btn.disabled = (self.current_episode_idx >= self.datasets[0].num_episodes)
        else:
            self.prev_btn.disabled = True
            self.next_btn.disabled = True

    def update_status_card(self):
        self.update_header()
        state_class = "status-idle"
        sub = "Ready to record"
        if self.recording_state == "RECORDING":
            state_class = "status-recording"
            elapsed = self.accumulated_time + (time.perf_counter() - self.episode_start_time)
            sub = f"[{self.dagger_phase}] Capturing ({self.frames_in_episode} frames, {elapsed:.1f}s)"
        elif self.recording_state == "SAVING":
            state_class = "status-saving"
            sub = "Writing episode to disk..."
        elif self.recording_state == "TRAINING":
            state_class = "status-saving"
            sub = "Training policy..."
            
        fps_text = f" | {self.current_fps:.1f} FPS" if getattr(self, 'current_fps', 0) > 0 else ""
        
        # Calculate DAgger metrics
        round_idx = getattr(self, "current_training_round", 1)
        saved_add_eps = 0
        eps_in_curr_round = 0
        if getattr(self, "episode_rounds_cache", None):
            saved_add_eps = sum(1 for r in self.episode_rounds_cache.values() if r is not None and r > 0)
            eps_in_curr_round = sum(1 for r in self.episode_rounds_cache.values() if r is not None and r == round_idx)
            
        additional_episodes = saved_add_eps
        if self.recording_state == "RECORDING" and round_idx > 0:
            additional_episodes += 1
            
        ep_in_round = eps_in_curr_round + 1
        
        self.status_card.value = f"""
        <div class="status-card {state_class}">
            <div class="status-text">{self.recording_state} - {self.dagger_phase if self.recording_state == 'RECORDING' else ''}</div>
            <div style="font-size: 14px; font-weight: 700; margin-top: 5px; opacity: 0.95;">Episode:
            {self.current_episode_idx} (Step: {self.current_dataset_step_idx + 1}/{len(self.steps_val)}){fps_text} | Rnd: {round_idx} | Add. Eps: {additional_episodes} |
            Ep/Rnd: {ep_in_round}</div>
            <div style="font-size: 12px; margin-top: 3px; opacity: 0.8;">{sub}</div>
        </div>
        """

    async def start_recording_async(self):
        if self.recording_state != "IDLE": return
        
        if not hasattr(self, "initial_robot_action"):
            if getattr(self, "current_robot_action", None) is not None:
                import copy
                self.initial_robot_action = copy.deepcopy(self.current_robot_action)
                self.add_log("Recorded initial posture for return.")
            else:
                self.add_log("Warning: Could not record initial posture, no action available yet.")
                
        self.recording_state = "RECORDING"
        self.dagger_phase = "AUTONOMOUS"
        self.current_dataset_step_idx = 0
        self.episode_start_time = time.perf_counter()
        self.accumulated_time = 0.0
        self.frames_in_episode = 0
        
        self.start_btn.disabled = True
        self.pause_btn.disabled = False
        self.correction_btn.disabled = False
        self.prev_step_btn.disabled = True
        self.next_step_btn.disabled = (len(self.steps_val) <= 1)
        self.stop_btn.disabled = False
        self.discard_btn.disabled = False
        self.train_btn.disabled = True
        self.update_navigation_buttons()
        
        self.add_log(f"Recording episode {self.current_episode_idx}... AUTONOMOUS mode.")
        setattr(self.params, "_needs_reset", True)
        self.update_status_card()

    def on_start_clicked(self, b):
        self.active_bg_task = self.schedule_background_task(self.start_recording_async())

    def update_telemetry(self, obs, raw_action, current_fps, policy_action=None):
        teleop_action = self.teleop_action_processor((raw_action, obs))
        # If in autonomous mode, we plot the policy action instead of human teleop action
        if self.dagger_phase == "AUTONOMOUS" and policy_action is not None:
            robot_action = policy_action
        else:
            robot_action = self.robot_action_processor((teleop_action, obs))
            
        text = f"<b>DAGGER TELEMETRY</b><br/>"
        text += f"Loop frequency: {current_fps:.1f} Hz<br/>"
        text += f"Phase: {self.dagger_phase}<br/>"
        self.telemetry_widget.value = f'<div class="telemetry-card">{text}</div>'
        
        if self.dagger_phase == "AUTONOMOUS":
            now = time.perf_counter()
            if not hasattr(self, "_last_plot_time"):
                self._last_plot_time = 0.0
            target_interval = 1.0 / self.autonomous_plot_fps if self.autonomous_plot_fps > 0 else float('inf')
            if now - self._last_plot_time < target_interval:
                return
            self._last_plot_time = now
            
        if not getattr(self, "plotter_is_updating", False):
            self.plotter_is_updating = True
            def run_plot():
                return self.plotter.update(obs, robot_action)
                
            async def run_plot_async():
                try:
                    chart_bytes = await asyncio.to_thread(run_plot)
                    if chart_bytes:
                        self.charts_holder_widget.value = chart_bytes
                except Exception as e:
                    print(f"Plotter error: {e}")
                finally:
                    self.plotter_is_updating = False
                    
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(run_plot_async())
            except RuntimeError:
                self.plotter_is_updating = False

    def on_pause_clicked(self, b):
        if self.recording_state != "RECORDING": return
        if self.dagger_phase == "AUTONOMOUS":
            self.dagger_phase = "PAUSED"
            self.add_log("Policy PAUSED. Robot holds position.")
            self.correction_btn.disabled = False
            self.update_status_card()
        elif self.dagger_phase == "PAUSED":
            self.dagger_phase = "AUTONOMOUS"
            self.add_log("Resuming AUTONOMOUS mode.")
            self.correction_btn.disabled = False
            setattr(self.params, "_needs_reset", True)
            self.update_status_card()
        elif self.dagger_phase == "CORRECTING":
            self.add_log("You must finish correction using Correction button.")

    def on_correction_clicked(self, b):
        if self.dagger_phase in ["AUTONOMOUS", "PAUSED"]:
            prev_phase = self.dagger_phase
            self.dagger_phase = "CORRECTING_LOCKED"
            self.recording_state = "PAUSED"
            if hasattr(self.leader, "enable_torque"): self.leader.enable_torque()
            
            if prev_phase == "AUTONOMOUS":
                if hasattr(self, "last_obs") and self.last_obs is not None and hasattr(self.leader, "send_feedback"):
                    self.schedule_background_task(asyncio.to_thread(self.leader.send_feedback, self.last_obs))
                self.add_log("Correction Phase 1: Leader locked to follower position.")
            else:
                self.add_log("Correction Phase 1: Leader locked.")
            
        elif self.dagger_phase == "CORRECTING_LOCKED":
            self.dagger_phase = "CORRECTING_ACTIVE"
            if hasattr(self.leader, "disable_torque"): self.leader.disable_torque()
            self.recording_state = "RECORDING"
            self.add_log("Correction Phase 2: Follower following leader. Recording resumed.")
            
        elif self.dagger_phase == "CORRECTING_ACTIVE":
            self.dagger_phase = "AUTONOMOUS"
            self.add_log("Correction ended. Resuming policy rollout.")
            
        self.update_status_card()

    def on_next_step_clicked(self, b):
        if self.recording_state == "RECORDING" and self.current_dataset_step_idx < len(self.steps_val) - 1:
            self.current_dataset_step_idx += 1
            self.frames_in_episode = 0
            if self.current_dataset_step_idx == len(self.steps_val) - 1:
                self.next_step_btn.disabled = True
            self.prev_step_btn.disabled = False
            setattr(self.params, "_needs_reset", True)
            self.update_status_card()

    def on_prev_step_clicked(self, b):
        if self.recording_state == "RECORDING" and self.current_dataset_step_idx > 0:
            self.current_dataset_step_idx -= 1
            self.frames_in_episode = 0
            if self.current_dataset_step_idx == 0:
                self.prev_step_btn.disabled = True
            self.next_step_btn.disabled = False
            setattr(self.params, "_needs_reset", True)
            self.update_status_card()

    async def return_to_initial_posture_async(self):
        if not hasattr(self, "initial_robot_action") or self.initial_robot_action is None:
            return
            
        self.recording_state = "IDLE"
        self.dagger_phase = "RETURNING"
        self.update_status_card()
        
        self.add_log("Returning to initial posture...")
        
        from lerobot.common.control_utils import teleop_supports_feedback, teleop_smooth_move_to, follower_smooth_move_to
        
        target_action = self.initial_robot_action
        current_action = getattr(self, "current_robot_action", None)
        if current_action is None:
            current_action = target_action
            
        tasks = []
        tasks.append(asyncio.to_thread(
            follower_smooth_move_to, self.robot, current_action, target_action, 2.0, 30
        ))
        
        if teleop_supports_feedback(self.leader):
            tasks.append(asyncio.to_thread(
                teleop_smooth_move_to, self.leader, target_action, 2.0, 30
            ))
            
        if tasks:
            try:
                await asyncio.gather(*tasks)
            except Exception as e:
                self.add_log(f"Warning: return to posture failed: {e}")
                import traceback
                traceback.print_exc()
            
        self.add_log("Return complete. Unlocking leader...")
        if hasattr(self.leader, "disable_torque"):
            try:
                self.leader.disable_torque()
            except Exception:
                pass

    async def save_current_episode_async(self):
        if self.recording_state == "SAVING": return
        self.recording_state = "SAVING"
        self.update_status_card()
        
        try:
            for ds in self.datasets:
                await asyncio.to_thread(ds.save_episode)
                
                def flush_ds(ds):
                    if hasattr(ds, "writer") and ds.writer is not None:
                        w = ds.writer
                        if hasattr(w, "close_writer"):
                            w.close_writer()
                        if hasattr(w, "_meta") and hasattr(w._meta, "_close_writer"):
                            w._meta._close_writer()
                        if hasattr(w, "_latest_episode") and w._latest_episode is not None:
                            from lerobot.datasets.utils import update_chunk_file_indices
                            c, f = w._latest_episode["data/chunk_index"], w._latest_episode["data/file_index"]
                            c, f = update_chunk_file_indices(c, f, w._meta.chunks_size)
                            w._latest_episode["data/chunk_index"] = c
                            w._latest_episode["data/file_index"] = f
                            w._current_file_start_frame = w._latest_episode["index"][-1] + 1
                            
                await asyncio.to_thread(flush_ds, ds)
                
            self.current_episode_idx += 1
            self.episode_progress.value = self.current_episode_idx
                
            self.load_episode_rounds_cache()
            self.add_log(f"Episode {self.current_episode_idx - 1} saved successfully.")
        except Exception as e:
            self.add_log(f"Error saving episode: {e}")
            import traceback
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
            self.update_navigation_buttons()

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

        try:
            # Prepare arguments for train_generic
            train_params = SimpleNamespace(**self.get_online_training_params())
            
            from train import train_dagger
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
                
                # Extract episodes that have round > 0 (new corrections) to force them into training set
                hf_dataset = ds.hf_dataset
                force_train_episodes = set()
                if "round" in hf_dataset.column_names and "episode_index" in hf_dataset.column_names:
                    rounds = hf_dataset["round"]
                    episode_indices = hf_dataset["episode_index"]
                    for r, ep in zip(rounds, episode_indices):
                        if r is None:
                            continue
                        r_val = r[0] if (isinstance(r, list) or isinstance(r, tuple)) else r
                        if r_val is None:
                            continue
                        r_val = int(r_val)
                        if r_val > 0:
                            if ep is None:
                                continue
                            ep_val = ep[0] if (isinstance(ep, list) or isinstance(ep, tuple)) else ep
                            if ep_val is None:
                                continue
                            ep_val = int(ep_val)
                            force_train_episodes.add(ep_val)
                            
                # Split dataset
                from split_dataset import split_dataset
                train_ds, val_ds = split_dataset(ds, val_ratio=0.1, seed=28, force_train_episodes=force_train_episodes)
                
                # Dynamic training steps based on new frames
                beta = getattr(train_params, "dagger_rehearsal_beta", 0.3)
                new_epochs = getattr(self.params, "dagger_train_new_epochs", 3.0)
                
                hf_dataset = train_ds.hf_dataset
                newest_round_frames = 0
                if "round" in hf_dataset.column_names:
                    rounds = hf_dataset["round"]
                    current_training_round = getattr(self, "current_training_round", 1)
                    
                    for r in rounds:
                        if r is None:
                            continue
                        val = r[0] if (isinstance(r, list) or isinstance(r, tuple)) else r
                        if val is None:
                            continue
                        if int(val) == current_training_round:
                            newest_round_frames += 1

                if newest_round_frames > 0:
                    computed_steps = int(((new_epochs * newest_round_frames) / beta) / train_params.batch_size)
                    train_params.training_steps = max(getattr(self.params, "dagger_train_steps", 100), computed_steps)
                    self.add_log(f"Dynamic steps: {train_params.training_steps} for {newest_round_frames} new frames in train set.")
                else:
                    self.add_log(f"No new frames in round {getattr(self, 'current_training_round', 1)} in train set. Using default steps.")

                # Retrieve optimizer if cached, or let train_generic create it
                if not hasattr(self, "optimizer_cache"):
                    self.optimizer_cache = None
                    
                def act_batch_transform(batch):
                    import torch
                    for k in list(batch.keys()):
                        if k.startswith("observation.") and isinstance(batch[k], torch.Tensor) and batch[k].ndim == (5 if "images" in k else 3):
                            batch[k] = batch[k].squeeze(1)
                    return batch
                    
                def run_training():
                    nonlocal policy
                    cfg = policy.config
                    
                    batch_transform = None
                    if policy.__class__.__name__ == "ACTPolicy":
                        batch_transform = act_batch_transform
                        
                    # Set current training round inside policy config so it is saved in config.json
                    policy.config.current_training_round = getattr(self, "current_training_round", 1)
                    
                    # Train single model synchronously in thread
                    train_dagger(
                        policy=policy,
                        cfg=cfg,
                        train_params=train_params,
                        dataset=train_ds,
                        current_training_round=getattr(self, "current_training_round", 1),
                        val_dataset=val_ds,
                        device=device,
                        batch_transform=batch_transform,
                        optimizer=self.optimizer_cache,
                    )
                    # We might want to save the optimizer cache here, but AdamW objects modify states in-place.
                    # Actually, we need to extract the optimizer from train_generic. Since we don't return it,
                    # we can create it here and pass it, or just let train_generic recreate it (slower convergence).
                    # For simplicity, we just pass None for now. It will resume from the checkpoint if possible.
                    
                await asyncio.to_thread(run_training)
                self.add_log("Training completed successfully.")

                current_round = getattr(self, "current_training_round", 0) + 1
                self.current_training_round = current_round
                if hasattr(self, "round_counter"):
                    # Temporarily unobserve to avoid double-update
                    self.round_counter.value = self.current_training_round
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
            self.recording_state = "IDLE"
            self.dagger_phase = "AUTONOMOUS"
            self.update_status_card()
            self.start_btn.disabled = False
            self.train_btn.disabled = False

    def on_stop_clicked(self, b):
        self.active_bg_task = self.schedule_background_task(self.save_current_episode_async())

    async def on_discard_clicked_async(self):
        if self.recording_state == "SAVING": return
        self.recording_state = "SAVING"
        self.update_status_card()
        
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
            self.start_btn.disabled = False
            self.pause_btn.disabled = True
            self.correction_btn.disabled = True
            self.stop_btn.disabled = True
            self.discard_btn.disabled = True
            self.train_btn.disabled = False
            self.recording_state = "IDLE"
            self.dagger_phase = "AUTONOMOUS"
            self.update_status_card()
            self.update_navigation_buttons()

    def on_discard_clicked(self, b):
        self.active_bg_task = self.schedule_background_task(self.on_discard_clicked_async())

    async def main_loop(self):
        robot_lock = asyncio.Lock()
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
                start_loop = time.perf_counter()
                try:
                    if self.dagger_phase == "AUTONOMOUS" and self.recording_state != "IDLE":
                        # During autonomous rollout, do not read leader joints.
                        # This saves USB reading latency and serial bus overhead.
                        new_raw_action = raw_action
                        new_teleop_action = teleop_action
                    else:
                        new_raw_action = await asyncio.to_thread(self.leader.get_action)
                        new_teleop_action = self.teleop_action_processor((new_raw_action, obs))
                    
                    self.last_obs = obs

                    if self.dagger_phase == "RETURNING":
                        new_robot_action = None
                        self.current_teleop_action = new_teleop_action
                    elif self.recording_state == "IDLE" or self.dagger_phase == "CORRECTING_ACTIVE":
                        new_robot_action = self.robot_action_processor((new_teleop_action, obs))
                        self.current_teleop_action = new_teleop_action
                    elif self.dagger_phase == "CORRECTING_LOCKED" or self.recording_state in ["PAUSED", "TRAINING", "SAVING"]:
                        new_robot_action = last_sent_action # frozen
                        self.current_teleop_action = new_teleop_action
                    elif self.dagger_phase == "AUTONOMOUS":
                        # Use policy action
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
                                        self.current_teleop_action = action_dict # Plot and save policy action
                                    else:
                                        new_robot_action = last_sent_action # fallback
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

                    # Throttling to reduce lag if not in teleop
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
                    continue

        async def observation_loop():
            nonlocal obs, obs_processed, last_widget_update_time, last_telemetry_update_time
            while self.keep_running:
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
                    # Record during AUTONOMOUS and CORRECTING
                    # PAUSED does not record
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

                            # Resize images in obs_processed if they don't match dataset features
                            for key, feat in ds.features.items():
                                if key.startswith(f"{OBS_STR}.images."):
                                    cam_key = key.replace(f"{OBS_STR}.images.", "")
                                    if cam_key in obs_processed:
                                        img = obs_processed[cam_key]
                                        expected_shape = feat["shape"]
                                        # expected_shape might be (H, W, C) or (C, H, W)
                                        # we assume img is (H, W, C)
                                        target_h, target_w = (expected_shape[0], expected_shape[1]) if expected_shape[-1] == 3 else (expected_shape[1], expected_shape[2])
                                        if img.shape[:2] != (target_h, target_w):
                                            obs_processed[cam_key] = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

                            observation_frame = build_dataset_frame(ds.features, obs_processed, prefix=OBS_STR)
                            
                            # Determine correct action frame and intervention flag
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
                            # Add frame if we have valid action
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
    """
    Interactive recording studio GUI (Rollout & DAgger Mode).
    """
    DAggerInteractiveStudio(robot, leader, params)
