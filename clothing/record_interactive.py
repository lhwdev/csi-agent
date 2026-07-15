# ========================================== 
# LeRobot Interactive Recording Studio GUI
# ========================================== 

import time
import asyncio
from pathlib import Path
import cv2
import ipywidgets as widgets
from IPython.display import display
import datasets
datasets.disable_progress_bar()

import shutil
import tempfile
import traceback
from types import SimpleNamespace
from IPython import get_ipython

# LeRobot imports
from lerobot.robots import Robot
from lerobot.teleoperators import Teleoperator
from lerobot.datasets import LeRobotDataset, aggregate_pipeline_dataset_features, create_initial_features, delete_episodes
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.processor import make_default_processors

from clothing.studio_core import BaseInteractiveStudio

class RecordInteractiveStudio(BaseInteractiveStudio):
    def on_datasets_initialized(self):
        self.start_btn.disabled = False
        self.prev_step_btn.disabled = True
        self.next_step_btn.disabled = True
        self.update_navigation_buttons()
        self.recording_state = "IDLE"
        self.current_dataset_step_idx = 0
        self.update_status_card()
        self.add_log("Studio ready. Position robot and click 'Start Episode' to record.")

    def update_navigation_buttons(self):
        if self.datasets and self.recording_state == "IDLE":
            self.prev_btn.disabled = (self.current_episode_idx == 0)
            self.next_btn.disabled = (self.current_episode_idx >= self.datasets[0].num_episodes)
        else:
            self.prev_btn.disabled = True
            self.next_btn.disabled = True

    def on_prev_clicked(self, b):
        if self.datasets and self.recording_state == "IDLE":
            self.current_episode_idx = max(0, self.current_episode_idx - 1)
            self.episode_progress.value = self.current_episode_idx
            self.update_navigation_buttons()
            self.add_log(f"Selected Episode {self.current_episode_idx} for viewing/recording.")

    def on_next_clicked(self, b):
        if self.datasets and self.recording_state == "IDLE":
            self.current_episode_idx = min(self.datasets[0].num_episodes, self.current_episode_idx + 1)
            self.episode_progress.value = self.current_episode_idx
            self.update_navigation_buttons()
            self.add_log(f"Selected Episode {self.current_episode_idx} for viewing/recording.")

    def build_controls(self):
        super().build_controls()
        self.prev_btn.on_click(self.on_prev_clicked)
        self.next_btn.on_click(self.on_next_clicked)
        self.start_btn.on_click(self.on_start_clicked)
        self.next_step_btn.on_click(self.on_next_step_clicked)
        self.prev_step_btn.on_click(self.on_prev_step_clicked)
        self.pause_btn.on_click(self.on_pause_clicked)
        self.stop_btn.on_click(self.on_stop_clicked)
        self.discard_btn.on_click(self.on_discard_clicked)
        
    def build_shortcuts(self):
        super().build_shortcuts()
        self.shortcut_legend.value = """
        <div style="background: #f8f9fa; border: 1px dashed #ccc; border-radius: 8px; padding: 10px; font-family: sans-serif; font-size: 11px;">
          <b style="color: #333;">Keyboard Shortcuts:</b>
          <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 5px; margin-top: 5px; font-family: monospace;">
            <div><span style="background: #eee; padding: 2px 5px; border-radius: 3px; border: 1px solid #ccc; font-weight: bold;">R</span> Start / Resume</div>
            <div><span style="background: #eee; padding: 2px 5px; border-radius: 3px; border: 1px solid #ccc; font-weight: bold;">Space</span> Next Action</div>
            <div><span style="background: #eee; padding: 2px 5px; border-radius: 3px; border: 1px solid #ccc; font-weight: bold;">B</span> Prev Step</div>
            <div><span style="background: #eee; padding: 2px 5px; border-radius: 3px; border: 1px solid #ccc; font-weight: bold;">N</span> Next Step</div>
            <div><span style="background: #eee; padding: 2px 5px; border-radius: 3px; border: 1px solid #ccc; font-weight: bold;">P</span> Pause</div>
            <div><span style="background: #eee; padding: 2px 5px; border-radius: 3px; border: 1px solid #ccc; font-weight: bold;">S</span> Save Episode</div>
            <div><span style="background: #eee; padding: 2px 5px; border-radius: 3px; border: 1px solid #ccc; font-weight: bold;">D</span> Discard & Redo</div>
            <div><span style="background: #eee; padding: 2px 5px; border-radius: 3px; border: 1px solid #ccc; font-weight: bold;">[</span> Prev Ep</div>
            <div><span style="background: #eee; padding: 2px 5px; border-radius: 3px; border: 1px solid #ccc; font-weight: bold;">]</span> Next Ep</div>
          </div>
          <div style="margin-top: 5px; font-family: monospace;"><span style="background: #eee; padding: 2px 5px; border-radius: 3px; border: 1px solid #ccc; font-weight: bold;">Esc</span> / <span style="background: #eee; padding: 2px 5px; border-radius: 3px; border: 1px solid #ccc; font-weight: bold;">Q</span> Exit Studio</div>
          <div style="margin-top: 8px; font-size: 10px; color: #666; font-style: italic;">
            💡 Click on the input field below to focus and activate shortcuts in VS Code.
          </div>
        </div>
        """
        
        def on_shortcut_change(change):
            key = change['new']
            if not key:
                return
            key = key.lower()
            
            self.shortcut_input.value = ""
            if not self.shortcut_toggle.value:
                return
                
            if key == 'n':
                if not self.next_step_btn.disabled:
                    self.on_next_step_clicked(None)
            elif key == 'b':
                if not self.prev_step_btn.disabled:
                    self.on_prev_step_clicked(None)
            elif key == ' ' or key == 'space':
                if self.recording_state == "IDLE":
                    if not self.start_btn.disabled:
                        self.on_start_clicked(None)
                elif self.recording_state == "RECORDING":
                    if not self.next_step_btn.disabled:
                        self.on_next_step_clicked(None)
                    elif not self.stop_btn.disabled:
                        self.on_stop_clicked(None)
                elif self.recording_state == "PAUSED":
                    if not self.pause_btn.disabled and self.pause_btn.description == "Resume":
                        self.on_pause_clicked(None)
            elif key == 'r':
                if not self.start_btn.disabled:
                    self.on_start_clicked(None)
                elif not self.pause_btn.disabled and self.pause_btn.description == "Resume":
                    self.on_pause_clicked(None)
            elif key == 'p':
                if not self.pause_btn.disabled:
                    self.on_pause_clicked(None)
            elif key == 's':
                if not self.stop_btn.disabled:
                    self.on_stop_clicked(None)
            elif key == 'd':
                if not self.discard_btn.disabled:
                    self.on_discard_clicked(None)
            elif key == '[':
                if not self.prev_btn.disabled:
                    self.on_prev_clicked(None)
            elif key == ']':
                if not self.next_btn.disabled:
                    self.on_next_clicked(None)
            elif key == 'q':
                self.cleanup_studio_sync()
                
        self.shortcut_input.observe(on_shortcut_change, names='value')

    def update_status_card(self):
        self.update_header()
        state_classes = {
            "IDLE": "status-idle",
            "RECORDING": "status-recording",
            "PAUSED": "status-paused",
            "SAVING": "status-saving"
        }
        state_class = state_classes.get(self.recording_state, "status-idle")
        
        if self.recording_state == "RECORDING":
            elapsed = self.accumulated_time + (time.perf_counter() - self.episode_start_time)
            sub = f"Capturing ({self.frames_in_episode} frames, {elapsed:.1f}s)"
            progress_html = ""
        elif self.recording_state == "PAUSED":
            sub = "Paused"
            progress_html = ""
        elif self.recording_state == "SAVING":
            sub = "Writing episode to disk..."
            progress_html = """
            <div style="margin-top: 10px; height: 6px; background-color: rgba(255,255,255,0.2); border-radius: 3px; overflow: hidden; position: relative;">
                <div style="position: absolute; left: 0; top: 0; bottom: 0; width: 30%; background-color: white; border-radius: 3px; animation: loading-bar-shift 1s ease-in-out infinite alternate;"></div>
            </div>
            """
        else:
            sub = "Ready to record"
            progress_html = ""
            
        fps_text = f" | {getattr(self, 'current_fps', 0):.1f} FPS" if getattr(self, 'current_fps', 0) > 0 else ""
            
        self.status_card.value = f"""
        <div class="status-card {state_class}">
            <div class="status-text">{self.recording_state}</div>
            <div style="font-size: 14px; font-weight: 700; margin-top: 5px; opacity: 0.95;">Episode: {self.current_episode_idx} (Step: {self.current_dataset_step_idx + 1}/{len(self.steps_val)}){fps_text}</div>
            <div style="font-size: 12px; margin-top: 3px; opacity: 0.8;">{sub}</div>
            {progress_html}
        </div>
        """

    def update_telemetry(self, robot_obs=None, leader_act=None, loop_fps=0.0):
        super().update_telemetry(robot_obs, leader_act, loop_fps)
        text = f"<b>STUDIO TELEMETRY</b><br/>"
        text += f"Loop frequency: {loop_fps:.1f} Hz<br/>"
        
        if robot_obs is not None:
            pos_keys = sorted([k for k in robot_obs.keys() if k.endswith('.pos')])
            if pos_keys:
                text += f"<b>Follower Joints:</b><br/>"
                left_joints = [k for k in pos_keys if k.startswith('left_')]
                right_joints = [k for k in pos_keys if k.startswith('right_')]
                
                if left_joints:
                    text += "<i>Left Arm:</i><br/>"
                    for k in left_joints:
                        name = k.removeprefix('left_').split('.')[0]
                        text += f" &bull; {name}: {robot_obs[k]:.2f}<br/>"
                if right_joints:
                    text += "<i>Right Arm:</i><br/>"
                    for k in right_joints:
                        name = k.removeprefix('right_').split('.')[0]
                        text += f" &bull; {name}: {robot_obs[k]:.2f}<br/>"
                other_joints = [k for k in pos_keys if not k.startswith('left_') and not k.startswith('right_')]
                for k in other_joints:
                    text += f" &bull; {k.split('.')[0]}: {robot_obs[k]:.2f}<br/>"
                    
        if leader_act is not None:
            pos_keys = sorted([k for k in leader_act.keys() if k.endswith('.pos')])
            if pos_keys:
                text += f"<b>Leader Joints:</b><br/>"
                left_joints = [k for k in pos_keys if k.startswith('left_')]
                right_joints = [k for k in pos_keys if k.startswith('right_')]
                
                if left_joints:
                    text += "<i>Left Arm:</i><br/>"
                    for k in left_joints:
                        name = k.removeprefix('left_').split('.')[0]
                        text += f" &bull; {name}: {leader_act[k]:.2f}<br/>"
                if right_joints:
                    text += "<i>Right Arm:</i><br/>"
                    for k in right_joints:
                        name = k.removeprefix('right_').split('.')[0]
                        text += f" &bull; {name}: {leader_act[k]:.2f}<br/>"
                        
        self.telemetry_widget.value = f'<div class="telemetry-card">{text}</div>'

    async def truncate_dataset_async(self, target_num_episodes):
        self.start_btn.disabled = True
        self.prev_btn.disabled = True
        self.next_btn.disabled = True
        self.recording_state = "SAVING"
        self.update_status_card()
        
        try:
            for idx, ds in enumerate(self.datasets):
                repo_id = self.steps_val[idx]["repo_id"]
                root_dir = Path(self.steps_val[idx]["root_dir"])
                if target_num_episodes == 0:
                    await asyncio.to_thread(ds.finalize)
                    if root_dir.exists():
                        await asyncio.to_thread(shutil.rmtree, root_dir)
                    
                    dataset_features = self.get_dataset_features()
                    self.datasets[idx] = await asyncio.to_thread(LeRobotDataset.create, repo_id=repo_id, fps=30, root=str(root_dir), robot_type=self.robot.name, features=dataset_features, use_videos=True, streaming_encoding=True)
                else:
                    await asyncio.to_thread(ds.finalize)
                    tmp_dir = Path(tempfile.mkdtemp(dir=str(root_dir.parent)))
                    try:
                        await asyncio.to_thread(delete_episodes, dataset=ds, episode_indices=list(range(target_num_episodes, ds.num_episodes)), output_dir=tmp_dir, repo_id=repo_id)
                        if root_dir.exists():
                            await asyncio.to_thread(shutil.rmtree, root_dir)
                        await asyncio.to_thread(shutil.move, str(tmp_dir), str(root_dir))
                        self.datasets[idx] = await asyncio.to_thread(LeRobotDataset.resume, repo_id=repo_id, root=str(root_dir), streaming_encoding=True)
                    finally:
                        if tmp_dir.exists():
                            shutil.rmtree(tmp_dir, ignore_errors=True)
            self.add_log(f"All datasets truncated to {self.datasets[0].num_episodes} episodes.")
        except Exception as e:
            self.add_log(f"Error during truncation: {str(e)}")
        finally:
            self.current_episode_idx = self.datasets[0].num_episodes
            self.episode_progress.value = self.current_episode_idx

    async def start_recording_async(self):
        if self.recording_state != "IDLE":
            return
        
        if self.current_episode_idx < self.datasets[0].num_episodes:
            await self.truncate_dataset_async(self.current_episode_idx)
            if self.current_episode_idx != self.datasets[0].num_episodes:
                self.add_log("Aborting recording start because truncation failed.")
                return
            
        self.recording_state = "RECORDING"
        self.current_dataset_step_idx = 0
        self.episode_start_time = time.perf_counter()
        self.accumulated_time = 0.0
        self.frames_in_episode = 0
        
        self.start_btn.disabled = True
        self.pause_btn.disabled = False
        self.prev_step_btn.disabled = True
        self.next_step_btn.disabled = (len(self.steps_val) <= 1)
        self.stop_btn.disabled = False
        self.discard_btn.disabled = False
        self.prev_btn.disabled = True
        self.next_btn.disabled = True
        
        self.add_log(f"Recording episode {self.current_episode_idx} (Step {self.current_dataset_step_idx+1}/{len(self.steps_val)})...")
        self.update_status_card()

    def on_start_clicked(self, b):
        self.active_bg_task = self.schedule_background_task(self.start_recording_async())

    def on_next_step_clicked(self, b):
        if self.recording_state == "RECORDING" and self.current_dataset_step_idx < len(self.steps_val) - 1:
            self.current_dataset_step_idx += 1
            self.frames_in_episode = 0
            self.add_log(f"Moved to Step {self.current_dataset_step_idx+1}/{len(self.steps_val)}.")
            if self.current_dataset_step_idx == len(self.steps_val) - 1:
                self.next_step_btn.disabled = True
            self.prev_step_btn.disabled = False
            self.update_status_card()

    def on_prev_step_clicked(self, b):
        if self.recording_state == "RECORDING" and self.current_dataset_step_idx > 0:
            self.current_dataset_step_idx -= 1
            self.frames_in_episode = 0
            self.add_log(f"Moved to Step {self.current_dataset_step_idx+1}/{len(self.steps_val)}.")
            if self.current_dataset_step_idx == 0:
                self.prev_step_btn.disabled = True
            self.next_step_btn.disabled = False
            self.update_status_card()

    def on_pause_clicked(self, b):
        if self.recording_state == "RECORDING":
            self.recording_state = "PAUSED"
            self.accumulated_time += time.perf_counter() - self.episode_start_time
            self.pause_btn.description = "Resume"
            self.pause_btn.icon = "play"
            self.add_log("Recording paused.")
        elif self.recording_state == "PAUSED":
            self.recording_state = "RECORDING"
            self.episode_start_time = time.perf_counter()
            self.pause_btn.description = "Pause"
            self.pause_btn.icon = "pause"
            self.add_log("Recording resumed.")
        self.update_status_card()

    async def save_current_episode_async(self):
        if self.recording_state == "SAVING":
            return
        self.recording_state = "SAVING"
        self.update_status_card()
        
        self.start_btn.disabled = True
        self.pause_btn.disabled = True
        self.prev_step_btn.disabled = True
        self.next_step_btn.disabled = True
        self.stop_btn.disabled = True
        self.discard_btn.disabled = True
        
        total_time = self.accumulated_time
        if self.recording_state == "RECORDING":
            total_time += time.perf_counter() - self.episode_start_time
        self.add_log(f"Saving episode {self.current_episode_idx} ({self.frames_in_episode} frames, {total_time:.1f}s) across all datasets...")
        try:
            for idx, ds in enumerate(self.datasets):
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
                
            self.add_log(f"Episode {self.current_episode_idx} saved on disk for all steps.")
            
            self.current_episode_idx += 1
            self.episode_progress.value = self.current_episode_idx
            
            if self.current_episode_idx >= self.total_episodes:
                self.add_log("All target episodes recorded! Finalizing dataset...")
                await self.finalize_dataset_async()
                self.keep_running = False
            else:
                self.recording_state = "IDLE"
                self.current_dataset_step_idx = 0
                self.update_status_card()
                self.start_btn.disabled = False
                self.pause_btn.disabled = True
                self.pause_btn.description = "Pause"
                self.pause_btn.icon = "pause"
                self.stop_btn.disabled = True
                self.discard_btn.disabled = True
                self.update_navigation_buttons()
                self.add_log(f"Automatically moved to Episode {self.current_episode_idx}. Ready to record.")
        except Exception as e:
            self.add_log(f"Error saving episode: {str(e)}")
            self.recording_state = "PAUSED"
            self.update_status_card()
            self.pause_btn.disabled = False
            self.stop_btn.disabled = False
            self.discard_btn.disabled = False
            self.update_navigation_buttons()

    def on_stop_clicked(self, b):
        self.active_bg_task = self.schedule_background_task(self.save_current_episode_async())

    async def on_discard_clicked_async(self):
        if self.recording_state == "SAVING":
            return
        self.recording_state = "SAVING"
        self.update_status_card()
        
        self.add_log(f"Discarding episode {self.current_episode_idx} buffer across all steps...")
        try:
            for ds in self.datasets:
                await asyncio.to_thread(ds.clear_episode_buffer, delete_images=True)
            self.add_log("Episode buffer discarded.")
            
            self.start_btn.disabled = False
            self.pause_btn.disabled = True
            self.pause_btn.description = "Pause"
            self.pause_btn.icon = "pause"
            self.prev_step_btn.disabled = True
            self.next_step_btn.disabled = True
            self.stop_btn.disabled = True
            self.discard_btn.disabled = True
            
            self.recording_state = "IDLE"
            self.current_dataset_step_idx = 0
            self.update_status_card()
            self.update_navigation_buttons()
        except Exception as e:
            self.add_log(f"Error discarding episode: {str(e)}")
            self.recording_state = "PAUSED"
            self.update_status_card()
            self.update_navigation_buttons()

    def on_discard_clicked(self, b):
        self.active_bg_task = self.schedule_background_task(self.on_discard_clicked_async())

    async def finalize_dataset_async(self):
        self.recording_state = "SAVING"
        self.update_status_card()
        self.add_log("Finalizing all datasets on disk...")
        try:
            for ds in self.datasets:
                if ds:
                    await asyncio.to_thread(ds.finalize)
            self.add_log("All datasets finalized successfully! You can now push to hub or train.")
        except Exception as e:
            self.add_log(f"Error during finalization: {str(e)}")
        
        self.start_btn.disabled = True
        self.pause_btn.disabled = True
        self.prev_step_btn.disabled = True
        self.next_step_btn.disabled = True
        self.stop_btn.disabled = True
        self.discard_btn.disabled = True
        
        self.recording_state = "IDLE"
        self.current_dataset_step_idx = 0
        self.update_status_card()

    async def main_loop(self):
        self.keep_running = True
        loop_fps = 30.0
        last_widget_update_time = 0.0
        
        consecutive_failures = 0
        max_consecutive_failures = 10
        obs = None
        raw_action = None

        self.add_log("Studio loop started in background.")

        async def encode_and_update(key, img, widget, update_widget, update_stream):
            bgr_img = await asyncio.to_thread(cv2.cvtColor, img, cv2.COLOR_RGB2BGR)
            _, jpeg_encoded = await asyncio.to_thread(cv2.imencode, '.jpg', bgr_img)
            jpeg_bytes = jpeg_encoded.tobytes()
            if update_widget:
                widget.value = jpeg_bytes
            if update_stream:
                self.streamer.update_frame(key, jpeg_bytes)

        try:
            while self.keep_running:
                start_loop = time.perf_counter()
                
                try:
                    new_obs, new_raw_action = await asyncio.gather(
                        asyncio.to_thread(self.robot.get_observation),
                        asyncio.to_thread(self.leader.get_action)
                    )
                    obs = new_obs
                    raw_action = new_raw_action
                    consecutive_failures = 0
                    
                    self.streamer.update_telemetry(obs, raw_action)
                    
                    obs_processed = self.robot_observation_processor(obs)
                    teleop_action = self.teleop_action_processor((raw_action, obs))
                    robot_action = self.robot_action_processor((teleop_action, obs))
                    
                    await asyncio.to_thread(self.robot.send_action, robot_action)
                except Exception as e:
                    consecutive_failures += 1
                    if consecutive_failures == 1 or consecutive_failures % 5 == 0:
                        self.add_log(f"Warning: Robot loop communication failure ({consecutive_failures}/{max_consecutive_failures}): {e}")
                    if consecutive_failures >= max_consecutive_failures:
                        raise RuntimeError(f"Too many consecutive robot loop failures. Exiting loop. Last error: {e}")
                    if obs is None or raw_action is None:
                        raise e
                    await asyncio.sleep(0.01)
                    continue
                
                if self.recording_state == "RECORDING":
                    if self.datasets and self.current_dataset_step_idx < len(self.datasets):
                        ds = self.datasets[self.current_dataset_step_idx]

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
                        action_frame = build_dataset_frame(ds.features, teleop_action, prefix=ACTION)
                        
                        task_str = self.steps_val[self.current_dataset_step_idx]["task"]
                        
                        frame_data = {
                            **observation_frame,
                            **action_frame,
                            "task": task_str
                        }
                        ds.add_frame(frame_data)
                        self.frames_in_episode += 1
                        
                now = time.perf_counter()
                
                if isinstance(self.telemetry_fps, int):
                    ui_fps_target = self.telemetry_fps
                else:
                    ui_fps_target = self.fps_control.value if getattr(self, 'fps_control', None) else 15
                should_update_widgets = (now - last_widget_update_time >= (1.0 / max(1, ui_fps_target)))
                
                encoding_tasks = []
                for key, widget in [("left_cam", self.left_camera_widget), ("top", self.top_camera_widget), ("right_cam", self.right_camera_widget)]:
                    if key in obs:
                        has_stream_client = self.streamer.active_clients.get(key, 0) > 0
                        if has_stream_client or should_update_widgets:
                            task = encode_and_update(key, obs[key], widget, should_update_widgets, has_stream_client)
                            encoding_tasks.append(task)
                if encoding_tasks:
                    await asyncio.gather(*encoding_tasks)
                                
                if should_update_widgets:
                    last_widget_update_time = now
                
                if getattr(self, 'frames_in_episode', 0) % 6 == 0 or self.recording_state != "RECORDING":
                    self.update_telemetry(obs, raw_action, loop_fps)
                        
                dt = time.perf_counter() - start_loop
                sleep_time = max(0.0, (1.0 / 30.0) - dt)
                await asyncio.sleep(sleep_time)
                
                actual_dt = time.perf_counter() - start_loop
                if actual_dt > 0:
                    loop_fps = 0.9 * loop_fps + 0.1 * (1.0 / actual_dt)
                    
        except asyncio.CancelledError:
            self.add_log("Studio loop was cancelled.")
        except Exception as e:
            err_str = traceback.format_exc()
            self.add_log(f"CRITICAL ERROR in loop: {str(e)}")
            print(f"Error in studio loop:\n{err_str}")
        finally:
            self.add_log("Studio loop stopped.")
            self.cleanup_studio_sync()
            self.add_log("Studio cleanup completed.")

def record_interactive(robot: Robot, leader: Teleoperator, params):
    """
    Interactive recording studio GUI.
    
    Args:
        robot: The follower robot.
        leader: The leader teleoperator robot.
        params: An object (e.g. types.SimpleNamespace) holding configuration parameter attributes.
        
    Example:
        from types import SimpleNamespace
        
        params = SimpleNamespace(
            repo_id="lhwdev/pick_umbrella",
            root_dir="/home/lhwdev/csi-agent/lerobot/lhwdev/records/pick_umbrella",
            task="Pick up the umbrella.",
            episodes=30,
        )
        record_interactive(robot, leader, params)
    """
    if isinstance(params, dict):
        params = SimpleNamespace(**params)
    return RecordInteractiveStudio(robot, leader, params)
