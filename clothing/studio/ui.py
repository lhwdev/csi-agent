import time
import asyncio
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display


class StudioUIMixin:
    """
    Mixin for UI layout assembly, widget configuration, telemetry formatting, and logging.
    Designed to separate the front-end Jupyter widgets from the core control loops.
    """

    def setup_ui(self):
        try:
            css_path = Path(__file__).parent.parent / "resources" / "studio_style.css"
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
        ], layout=widgets.Layout(flex="3 1 60%", min_width="500px", max_width="100%", padding="5px", overflow_x="hidden"))
        left_column.add_class("studio-left-column")

        self.charts_holder_widget = widgets.Image(
            format='jpeg',
            layout=widgets.Layout(border="1px solid #dee2e6", border_radius="12px", width="100%", max_width="100%", height="auto")
        )

        self.fps_control = widgets.Dropdown(
            options=[("5 FPS (Low CPU)", 5), ("10 FPS", 10), ("15 FPS (Default)", 15), ("30 FPS (High Performance)", 30)],
            value=self.telemetry_fps if self.telemetry_fps in [5, 10, 15, 30] else 15,
            description='UI FPS:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='100%', margin='8px 0px')
        )

        self.studio_tab = widgets.Tab(layout=widgets.Layout(width='100%', max_width='100%', overflow_x='hidden'))
        self.studio_tab.children = [self.telemetry_widget, self.charts_holder_widget, self.shortcuts_box]
        self.studio_tab.set_title(0, "Telemetry")
        self.studio_tab.set_title(1, "Joint Graph")
        self.studio_tab.set_title(2, "Shortcuts")
        self.studio_tab.selected_index = 1

        self.right_column = widgets.VBox([
            self.header_widget,
            widgets.HTML("<div style='height: 8px;'></div><b>Controls:</b>"),
            self.controls_box,
            widgets.HTML("<div style='height: 8px;'></div>"),
            self.shortcut_input,
            widgets.HTML("<div style='height: 8px;'></div>"),
            self.studio_tab,
            widgets.HTML("<div style='height: 8px;'></div>"),
            self.fps_control
        ], layout=widgets.Layout(flex="2 1 40%", min_width="380px", max_width="100%", padding="5px", overflow_x="hidden"))
        self.right_column.add_class("studio-right-column")

        self.dashboard_layout = widgets.HBox([left_column, self.right_column], layout=widgets.Layout(
            border="1px solid #ddd", border_radius="15px", padding="8px", background_color="#ffffff",
            flex_wrap="wrap", width="100%", max_width="100%", overflow_x="hidden"
        ))
        self.dashboard_layout.add_class("dashboard-layout")

        display(self.style_container)
        display(self.dashboard_layout)

    def build_controls(self):
        self.start_btn = widgets.Button(description="Start Episode", icon="circle", button_style="success", disabled=True)
        self.pause_btn = widgets.Button(description="Pause", icon="pause", button_style="warning", disabled=True)
        self.prev_step_btn = widgets.Button(description="Prev Step (B)", icon="step-backward", button_style="info", disabled=True)
        self.next_step_btn = widgets.Button(description="Next Step (N)", icon="step-forward", button_style="info", disabled=True)
        self.stop_btn = widgets.Button(description="Save Episode", icon="save", button_style="primary", disabled=True)
        self.discard_btn = widgets.Button(description="Discard & Redo", icon="trash", button_style="danger", disabled=True)
        self.snapshot_btn = widgets.Button(description="Snapshot (C)", icon="camera", button_style="info", disabled=True)

        self.prev_btn = widgets.Button(description="Prev Ep", icon="chevron-left", disabled=True)
        self.next_btn = widgets.Button(description="Next Ep", icon="chevron-right", disabled=True)
        self.quit_btn = widgets.Button(description="Exit Studio", icon="times-circle", button_style="danger")

        self.control_row_1 = widgets.HBox([self.start_btn, self.pause_btn, self.prev_step_btn, self.next_step_btn], layout=widgets.Layout(flex_flow='row wrap', width='100%', max_width='100%'))
        self.control_row_2 = widgets.HBox([self.stop_btn, self.discard_btn, self.snapshot_btn], layout=widgets.Layout(flex_flow='row wrap', width='100%', max_width='100%'))
        self.navigation_row = widgets.HBox([self.prev_btn, self.next_btn, self.quit_btn], layout=widgets.Layout(flex_flow='row wrap', width='100%', max_width='100%'))
        
        self.quit_btn.on_click(lambda b: self.cleanup_studio_sync())

        self.controls_box = widgets.VBox([
            self.control_row_1,
            self.control_row_2,
            self.navigation_row
        ], layout=widgets.Layout(width='100%', max_width='100%'))

    def build_shortcuts(self):
        self.shortcut_input = widgets.Text(value="", placeholder="Click to input shortcuts", layout=widgets.Layout(width='100%', max_width='100%'))
        self.shortcut_toggle = widgets.Checkbox(value=True, description="Enable keyboard shortcuts")
        self.shortcut_legend = widgets.HTML("<b>Keyboard Shortcuts Enabled</b>")
        self.shortcuts_box = widgets.VBox([self.shortcut_toggle, self.shortcut_legend])

    def build_cameras(self):
        self.left_camera_widget = widgets.Image(format='jpeg', layout=widgets.Layout(border="1px solid #ccc", border_radius="8px", width="100%", max_width="240px", height="auto", min_width="0"))
        self.top_camera_widget = widgets.Image(format='jpeg', layout=widgets.Layout(border="1px solid #ccc", border_radius="8px", width="100%", max_width="480px", height="auto", min_width="0"))
        self.right_camera_widget = widgets.Image(format='jpeg', layout=widgets.Layout(border="1px solid #ccc", border_radius="8px", width="100%", max_width="240px", height="auto", min_width="0"))
        
        self.cameras_layout = widgets.HBox([
            widgets.VBox([widgets.Label("Top Cam"), self.top_camera_widget], layout=widgets.Layout(align_items="center", flex="2 1 auto", min_width="0")),
            widgets.VBox([
                widgets.VBox([widgets.Label("Left Arm Cam"), self.left_camera_widget], layout=widgets.Layout(align_items="center", min_width="0")),
                widgets.VBox([widgets.Label("Right Arm Cam"), self.right_camera_widget], layout=widgets.Layout(align_items="center", min_width="0")),
            ], layout=widgets.Layout(justify_content="space-around", margin="5px", flex="1 1 auto", min_width="0")),
        ], layout=widgets.Layout(justify_content="space-around", margin="5px", width="100%", max_width="100%", min_width="0"))

    def add_log(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        self.log_messages.append(f"[{timestamp}] {msg}")
        if len(self.log_messages) > 20:
            self.log_messages.pop(0)
        log_content = "<br/>".join(reversed(self.log_messages))
        self.log_widget.value = f'<div class="log-area">{log_content}</div>'

    def update_header(self):
        if hasattr(self, 'header_widget') and self.steps_val and self.current_dataset_step_idx < len(self.steps_val):
            step = self.steps_val[self.current_dataset_step_idx]
            repo_id = step.get("repo_id", "")
            root_dir = step.get("root_dir", "")
            self.header_widget.value = f"""
            <div class="studio-header">
                <div class="studio-title">{repo_id}</div>
                <div class="studio-subtitle">{root_dir}</div>
            </div>
            """

    def update_status_card(self):
        self.update_header()
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
        text = f"<b>STUDIO TELEMETRY</b><br/>"
        text += f"Loop frequency: {loop_fps:.1f} Hz<br/>"
        self.telemetry_widget.value = f'<div class="telemetry-card">{text}</div>'
        
        if robot_obs is not None:
            self.latest_state = robot_obs
        if leader_act is not None:
            self.latest_action = leader_act

        if not self.plotter_is_updating:
            self.plotter_is_updating = True
            def run_plot():
                state_data = getattr(self, "latest_state", {})
                action_data = getattr(self, "latest_action", {})
                return self.plotter.update(state_data, action_data)
                
            async def run_plot_async():
                try:
                    chart_bytes = await asyncio.to_thread(run_plot)
                    if chart_bytes:
                        self.charts_holder_widget.value = chart_bytes
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Plotting error in ui.py: {e}")
                finally:
                    self.plotter_is_updating = False
                    
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(run_plot_async())
            except RuntimeError:
                self.plotter_is_updating = False
