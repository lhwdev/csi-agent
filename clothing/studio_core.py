import time
import asyncio
from pathlib import Path
import cv2
import ipywidgets as widgets
from IPython.display import display
import datasets
datasets.disable_progress_bar()

import traceback
import shutil
import json
import uuid

from lerobot.robots import Robot
from lerobot.teleoperators import Teleoperator
from lerobot.datasets import LeRobotDataset, aggregate_pipeline_dataset_features, create_initial_features
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.processor import make_default_processors
from lerobot_streamer import CameraStreamer

_active_studio_cleanup = None

class JointPlotter:
    def __init__(self, history_limit=50):
        self.history_limit = history_limit
        self.state_history = {}
        self.action_history = {}
        self.fig = None
        self.axes = {}
        self.lines = {}
        self.joint_keys = None
        self.colors = [
            '#e63946', '#f4a261', '#e9c46a', '#2a9d8f', '#457b9d', '#1d3557', '#a8dadc'
        ]

    def update(self, state, action):
        if not state and not action:
            return None
            
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import io
        
        state_keys = sorted([k for k in state.keys() if k.endswith('.pos')]) if state else []
        action_keys = sorted([k for k in action.keys() if k.endswith('.pos')]) if action else []
        all_keys = sorted(list(set(state_keys + action_keys)))
        
        if not all_keys:
            return None
            
        for k in all_keys:
            if k not in self.state_history:
                self.state_history[k] = []
            if k not in self.action_history:
                self.action_history[k] = []
                
            s_val = state.get(k, None) if state else None
            a_val = action.get(k, None) if action else None
            
            if s_val is not None:
                if hasattr(s_val, "item"): s_val = s_val.item()
                self.state_history[k].append(float(s_val))
            else:
                self.state_history[k].append(None)
                
            if a_val is not None:
                if hasattr(a_val, "item"): a_val = a_val.item()
                self.action_history[k].append(float(a_val))
            else:
                self.action_history[k].append(None)
                
            if len(self.state_history[k]) > self.history_limit:
                self.state_history[k].pop(0)
            if len(self.action_history[k]) > self.history_limit:
                self.action_history[k].pop(0)
                
        if self.fig is None:
            self.joint_keys = all_keys
            self.groups = {"left": [], "right": [], "other": []}
            for k in self.joint_keys:
                if k.startswith("left_"): self.groups["left"].append(k)
                elif k.startswith("right_"): self.groups["right"].append(k)
                else: self.groups["other"].append(k)
                    
            active_groups = [g for g, keys in self.groups.items() if len(keys) > 0]
            num_subplots = len(active_groups)
            
            plt.ioff()
            self.fig, axs = plt.subplots(num_subplots, 1, figsize=(6.5, 1.8 * num_subplots), dpi=160)
            if num_subplots == 1: axs = [axs]
                
            self.fig.patch.set_facecolor('#ffffff')
            
            for ax, gname in zip(axs, active_groups):
                ax.set_facecolor('#f8f9fa')
                ax.set_title(f"{gname.capitalize()} Arm (Solid: State, Dashed: Action)", fontsize=8, fontweight='bold', pad=6, color='#212529')
                ax.tick_params(axis='both', labelsize=7, colors='#495057')
                ax.grid(True, color='#dee2e6', linestyle=':', linewidth=0.5)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#ced4da')
                ax.spines['bottom'].set_color('#ced4da')
                
                self.axes[gname] = ax
                
                for idx, k in enumerate(self.groups[gname]):
                    clean_name = k.removeprefix("left_").removeprefix("right_").split(".")[0]
                    color = self.colors[idx % len(self.colors)]
                    s_line, = ax.plot([], [], label=clean_name, color=color, linewidth=1.5, linestyle='-')
                    a_line, = ax.plot([], [], label="_nolegend_", color=color, linewidth=1.2, linestyle='--')
                    self.lines[k] = (s_line, a_line)
                    
                ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7, frameon=True, facecolor='#ffffff', edgecolor='#e9ecef')
                
            self.fig.tight_layout()
            
        for gname, ax in self.axes.items():
            min_y, max_y = float('inf'), float('-inf')
            for k in self.groups[gname]:
                s_data = self.state_history[k]
                a_data = self.action_history[k]
                x_data = list(range(len(s_data)))
                s_line, a_line = self.lines[k]
                
                s_plot_y = [y for y in s_data if y is not None]
                s_plot_x = [x for x, y in zip(x_data, s_data) if y is not None]
                a_plot_y = [y for y in a_data if y is not None]
                a_plot_x = [x for x, y in zip(x_data, a_data) if y is not None]
                
                s_line.set_data(s_plot_x, s_plot_y)
                a_line.set_data(a_plot_x, a_plot_y)
                
                for y in s_plot_y + a_plot_y:
                    if y < min_y: min_y = y
                    if y > max_y: max_y = y
                    
            if min_y != float('inf') and max_y != float('-inf'):
                padding = max((max_y - min_y) * 0.1, 0.1)
                ax.set_ylim(min_y - padding, max_y + padding)
                ax.set_xlim(0, self.history_limit)
                
        buf = io.BytesIO()
        self.fig.savefig(buf, format='jpeg', bbox_inches='tight', dpi=160)
        buf.seek(0)
        return buf.getvalue()

    def cleanup(self):
        if self.fig is not None:
            import matplotlib.pyplot as plt
            try:
                plt.close(self.fig)
            except Exception:
                pass
            self.fig = None

class BaseInteractiveStudio:
    """
    Base class for interactive studios.
    Subclasses should implement specific logic for start, pause, stop, discarding, and loops.
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
        if not self.leader.is_connected:
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

    def setup_ui(self):
        try:
            css_path = Path(__file__).parent / "resources" / "studio_style.css"
            with open(css_path, "r", encoding="utf-8") as f:
                css_content = f.read()
        except Exception as e:
            css_content = ""
            print(f"Warning: Failed to load studio_style.css: {e}")

        self.style_container = widgets.HTML(f"<style>{css_content}</style>")

        self.header_widget = widgets.HTML("""
        <div class="studio-header">
            <div class="studio-title">🎙️ LeRobot Recording Studio</div>
            <div class="studio-subtitle">Interactive Demonstration Capture Suite</div>
        </div>
        """)

        self.status_card = widgets.HTML()
        self.update_status_card()

        self.telemetry_widget = widgets.HTML()
        
        self.log_widget = widgets.HTML()
        self.log_messages = []
        self.add_log("Studio initialized. Connect devices and configure dataset below.")

        self.episode_progress = widgets.IntProgress(value=0, min=0, max=self.total_episodes, description="Episodes:", bar_style="info")

        self.build_controls()
        self.build_shortcuts()
        self.build_cameras()

        left_column = widgets.VBox([
            self.cameras_layout,
            self.episode_progress,
            widgets.HTML("<b>Status:</b>"),
            self.status_card,
            widgets.HTML("<div style='height: 8px;'></div><b>Logs:</b>"),
            self.log_widget
        ], layout=widgets.Layout(width="60%", padding="5px"))

        self.charts_holder_widget = widgets.Image(
            format='jpeg',
            layout=widgets.Layout(border="1px solid #dee2e6", border_radius="12px", width="100%", height="auto")
        )

        self.fps_control = widgets.Dropdown(
            options=[("5 FPS (Low CPU)", 5), ("10 FPS", 10), ("15 FPS (Default)", 15), ("30 FPS (High Performance)", 30)],
            value=self.telemetry_fps if self.telemetry_fps in [5, 10, 15, 30] else 15,
            description='UI FPS:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='100%', margin='8px 0px')
        )

        studio_tab = widgets.Tab()
        studio_tab.children = [self.telemetry_widget, self.charts_holder_widget, self.shortcuts_box]
        studio_tab.set_title(0, "Telemetry")
        studio_tab.set_title(1, "Joint Graph")
        studio_tab.set_title(2, "Shortcuts")
        studio_tab.selected_index = 1

        right_column = widgets.VBox([
            self.header_widget,
            widgets.HTML("<div style='height: 8px;'></div><b>Controls:</b>"),
            self.controls_box,
            widgets.HTML("<div style='height: 8px;'></div>"),
            self.shortcut_input,
            widgets.HTML("<div style='height: 8px;'></div>"),
            studio_tab,
            widgets.HTML("<div style='height: 8px;'></div>"),
            self.fps_control
        ], layout=widgets.Layout(width="40%", padding="5px"))

        self.dashboard_layout = widgets.HBox([left_column, right_column], layout=widgets.Layout(
            border="1px solid #ddd", border_radius="15px", padding="8px", background_color="#ffffff"
        ))

        display(self.style_container)
        display(self.dashboard_layout)

    def build_controls(self):
        # By default, use standard controls. Subclasses can override.
        self.start_btn = widgets.Button(description="Start Episode", icon="circle", button_style="success", disabled=True)
        self.pause_btn = widgets.Button(description="Pause", icon="pause", button_style="warning", disabled=True)
        self.prev_step_btn = widgets.Button(description="Prev Step (B)", icon="step-backward", button_style="info", disabled=True)
        self.next_step_btn = widgets.Button(description="Next Step (N)", icon="step-forward", button_style="info", disabled=True)
        self.stop_btn = widgets.Button(description="Save Episode", icon="save", button_style="primary", disabled=True)
        self.discard_btn = widgets.Button(description="Discard & Redo", icon="trash", button_style="danger", disabled=True)

        self.prev_btn = widgets.Button(description="Prev Ep", icon="chevron-left", disabled=True)
        self.next_btn = widgets.Button(description="Next Ep", icon="chevron-right", disabled=True)
        self.quit_btn = widgets.Button(description="Exit Studio", icon="times-circle", button_style="danger")

        self.control_row_1 = widgets.HBox([self.start_btn, self.pause_btn, self.prev_step_btn, self.next_step_btn])
        self.control_row_2 = widgets.HBox([self.stop_btn, self.discard_btn])
        self.navigation_row = widgets.HBox([self.prev_btn, self.next_btn, self.quit_btn])
        
        self.quit_btn.on_click(lambda b: self.cleanup_studio_sync())

        self.controls_box = widgets.VBox([
            self.control_row_1,
            self.control_row_2,
            self.navigation_row
        ])

    def build_shortcuts(self):
        self.shortcut_input = widgets.Text(value="", placeholder="Click to input shortcuts")
        self.shortcut_toggle = widgets.Checkbox(value=True, description="Enable keyboard shortcuts")
        self.shortcut_legend = widgets.HTML("<b>Keyboard Shortcuts Enabled</b>") # Simplified, subclasses can override
        self.shortcuts_box = widgets.VBox([self.shortcut_toggle, self.shortcut_legend])

    def build_cameras(self):
        self.left_camera_widget = widgets.Image(format='jpeg', width=240, height=180, layout=widgets.Layout(border="1px solid #ccc", border_radius="8px"))
        self.top_camera_widget = widgets.Image(format='jpeg', width=480, height=360, layout=widgets.Layout(border="1px solid #ccc", border_radius="8px"))
        self.right_camera_widget = widgets.Image(format='jpeg', width=240, height=180, layout=widgets.Layout(border="1px solid #ccc", border_radius="8px"))
        
        self.cameras_layout = widgets.HBox([
            widgets.VBox([widgets.Label("Top Cam"), self.top_camera_widget], layout=widgets.Layout(align_items="center")),
            widgets.VBox([
                widgets.VBox([widgets.Label("Left Arm Cam"), self.left_camera_widget], layout=widgets.Layout(align_items="center")),
                widgets.VBox([widgets.Label("Right Arm Cam"), self.right_camera_widget], layout=widgets.Layout(align_items="center")),
            ], layout=widgets.Layout(justify_content="space-around", margin="5px")),
        ], layout=widgets.Layout(justify_content="space-around", margin="5px"))

    def add_log(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        self.log_messages.append(f"[{timestamp}] {msg}")
        if len(self.log_messages) > 20:
            self.log_messages.pop(0)
        log_content = "<br/>".join(reversed(self.log_messages))
        self.log_widget.value = f'<div class="log-area">{log_content}</div>'

    def update_status_card(self):
        # Override in subclasses for specific states
        state_class = "status-idle"
        fps_text = f" | {self.current_fps:.1f} FPS" if getattr(self, 'current_fps', 0) > 0 else ""
        self.status_card.value = f"""
        <div class="status-card {state_class}">
            <div class="status-text">{self.recording_state}</div>
            <div style="font-size: 14px; font-weight: 700; margin-top: 5px; opacity: 0.95;">Episode: {self.current_episode_idx} (Step: {self.current_dataset_step_idx + 1}/{len(self.steps_val)}){fps_text}</div>
        </div>
        """

    def update_telemetry(self, robot_obs=None, leader_act=None, loop_fps=0.0):
        self.current_fps = loop_fps
        self.update_status_card()
        # Standard telemetry updater
        text = f"<b>STUDIO TELEMETRY</b><br/>"
        text += f"Loop frequency: {loop_fps:.1f} Hz<br/>"
        self.telemetry_widget.value = f'<div class="telemetry-card">{text}</div>'
        
        if not self.plotter_is_updating:
            self.plotter_is_updating = True
            def run_plot():
                state_data = self.streamer.latest_telemetry.get("state", {})
                action_data = self.streamer.latest_telemetry.get("action", {})
                return self.plotter.update(state_data, action_data)
                
            async def run_plot_async():
                try:
                    chart_bytes = await asyncio.to_thread(run_plot)
                    if chart_bytes:
                        self.charts_holder_widget.value = chart_bytes
                except Exception:
                    pass
                finally:
                    self.plotter_is_updating = False
                    
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(run_plot_async())
            except RuntimeError:
                self.plotter_is_updating = False

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
            traceback.print_exc()
            self.recording_state = "IDLE"
            self.update_status_card()

    def on_datasets_initialized(self):
        pass # Override in subclasses

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

    async def main_loop(self):
        pass # Override in subclasses
        
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
