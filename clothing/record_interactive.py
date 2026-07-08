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

# LeRobot imports
from lerobot.robots import Robot
from lerobot.teleoperators import Teleoperator
from lerobot.datasets import LeRobotDataset, aggregate_pipeline_dataset_features, create_initial_features
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.processor import make_default_processors

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
            resume=False
        )
        record_interactive(robot, leader, params)
    """
    datasets = []
    current_dataset_step_idx = 0

    if params is None:
        raise ValueError("The 'params' argument is required and cannot be None.")

    # Extract values from params
    steps_val = getattr(params, "steps", None)
    if steps_val is None:
        steps_val = [{
            "repo_id": getattr(params, "repo_id", None),
            "root_dir": getattr(params, "root_dir", None),
            "task": getattr(params, "task", None)
        }]
    
    # Enforce that all steps have the required keys
    for i, step in enumerate(steps_val):
        if not step.get("repo_id") or not step.get("root_dir") or not step.get("task"):
            raise ValueError(f"Step {i} in params must contain 'repo_id', 'root_dir', and 'task'.")

    total_episodes = getattr(params, "episodes", 30)
    resume = getattr(params, "resume", False)

    # Global state variables for the recording loop
    active_loop_task = None
    active_bg_task = None
    recording_state = "IDLE"  # IDLE, RECORDING, PAUSED, SAVING
    current_episode_idx = 0
    frames_in_episode = 0
    keep_running = True

    episode_start_time = 0
    accumulated_time = 0
    reset_start_time = 0
    last_frame_time = 0
    fps_log = []


    # Stop background camera capture and start the streaming server
    from camera_streamer import CameraStreamer
    streamer = CameraStreamer()
    streamer.stop_all_captures()
    streamer_port = streamer.start_server()
    print(f"Camera Stream URL: {streamer.get_server_url()}")

    # Cancel previous background task if it is still running to avoid duplicate loops
    if active_loop_task is not None and not active_loop_task.done():
        active_loop_task.cancel()

    if not robot.is_connected:
        robot.connect()
    if not leader.is_connected:
        leader.connect()

    # Make sure processors are available
    try:
        teleop_action_processor
    except NameError:
        teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # ----------------- WIDGET DESIGN & STYLING -----------------
    style_container = widgets.HTML("""
    <style>
    .studio-header {
        background: linear-gradient(135deg, hsl(260, 80%, 40%) 0%, hsl(210, 80%, 40%) 100%);
        color: white;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        font-family: 'Outfit', 'Inter', sans-serif;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        margin-bottom: 15px;
    }
    .studio-title {
        font-size: 24px;
        font-weight: 700;
        margin: 0;
    }
    .studio-subtitle {
        font-size: 13px;
        opacity: 0.8;
        margin-top: 5px;
    }
    .status-card {
        border-radius: 12px;
        padding: 15px;
        font-family: 'Inter', sans-serif;
        color: white;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 10px;
        transition: all 0.3s ease;
    }
    .status-idle {
        background: linear-gradient(135deg, hsl(210, 15%, 35%) 0%, hsl(210, 15%, 25%) 100%);
    }
    .status-recording {
        background: linear-gradient(135deg, hsl(0, 80%, 50%) 0%, hsl(340, 80%, 40%) 100%);
        animation: pulse 2s infinite;
    }
    .status-paused {
        background: linear-gradient(135deg, hsl(35, 90%, 50%) 0%, hsl(45, 90%, 40%) 100%);
    }
    .status-resetting {
        background: linear-gradient(135deg, hsl(180, 70%, 45%) 0%, hsl(200, 70%, 35%) 100%);
    }
    .status-saving {
        background: linear-gradient(135deg, hsl(150, 70%, 40%) 0%, hsl(170, 70%, 30%) 100%);
    }
    .status-text {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: 1px;
    }
    .telemetry-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 0px;
        font-family: monospace;
        font-size: 11px;
        line-height: 1.4;
    }
    .log-area {
        background-color: #1e1e1e;
        color: #00ff00;
        font-family: 'Consolas', monospace;
        font-size: 11px;
        padding: 0px;
        border-radius: 8px;
        height: 150px;
        overflow-y: scroll;
        border: 1px solid #333;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(255, 0, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); }
    }
    @keyframes loading-bar-shift {
        0% { left: 0%; width: 30%; }
        100% { left: 70%; width: 30%; }
    }
    .studio-shortcut-input {
        width: 100% !important;
        margin: 8px 0px !important;
    }
    .studio-shortcut-input input {
        background-color: #f8f9fa !important;
        border: 2px solid #dee2e6 !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
        font-family: 'Outfit', 'Inter', sans-serif !important;
        font-size: 13px !important;
        text-align: center !important;
        color: #495057 !important;
        font-weight: 500 !important;
        transition: all 0.2s ease-in-out !important;
    }
    .studio-shortcut-input input:focus {
        border-color: #228be6 !important;
        background-color: #fff !important;
        box-shadow: 0 0 0 3px rgba(34, 139, 230, 0.15) !important;
        color: #212529 !important;
    }
    </style>
    """)

    # 1. Header
    header_widget = widgets.HTML("""
    <div class="studio-header">
        <div class="studio-title">🎙️ LeRobot Recording Studio</div>
        <div class="studio-subtitle">Interactive Demonstration Capture Suite</div>
    </div>
    """)

    # 2. Status Panel
    status_card = widgets.HTML()
    def update_status_card():
        nonlocal recording_state, current_episode_idx, frames_in_episode, episode_start_time, accumulated_time
        state_classes = {
            "IDLE": "status-idle",
            "RECORDING": "status-recording",
            "PAUSED": "status-paused",
            "SAVING": "status-saving"
        }
        state_class = state_classes.get(recording_state, "status-idle")
        
        if recording_state == "RECORDING":
            elapsed = accumulated_time + (time.perf_counter() - episode_start_time)
            sub = f"Capturing ({frames_in_episode} frames, {elapsed:.1f}s)"
            progress_html = ""
        elif recording_state == "PAUSED":
            sub = "Paused"
            progress_html = ""
        elif recording_state == "SAVING":
            sub = "Writing episode to disk..."
            progress_html = """
            <div style="margin-top: 10px; height: 6px; background-color: rgba(255,255,255,0.2); border-radius: 3px; overflow: hidden; position: relative;">
                <div style="position: absolute; left: 0; top: 0; bottom: 0; width: 30%; background-color: white; border-radius: 3px; animation: loading-bar-shift 1s ease-in-out infinite alternate;"></div>
            </div>
            """
        else:
            sub = "Ready to record"
            progress_html = ""
            
        status_card.value = f"""
        <div class="status-card {state_class}">
            <div class="status-text">{recording_state}</div>
            <div style="font-size: 14px; font-weight: 700; margin-top: 5px; opacity: 0.95;">Episode: {current_episode_idx} (Step: {current_dataset_step_idx + 1}/{len(steps_val)})</div>
            <div style="font-size: 12px; margin-top: 3px; opacity: 0.8;">{sub}</div>
            {progress_html}
        </div>
        """

    update_status_card()

    telemetry_widget = widgets.HTML()
    def update_telemetry(robot_obs=None, leader_act=None, loop_fps=0.0):
        text = f"<b>STUDIO TELEMETRY</b><br/>"
        text += f"Loop frequency: {loop_fps:.1f} Hz<br/>"
        
        if robot_obs is not None:
            pos_keys = sorted([k for k in robot_obs.keys() if k.endswith('.pos')])
            if pos_keys:
                text += f"<b>Follower Joints:</b><br/>"
                # Split into left and right
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
                # Any other joints
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
                        
        telemetry_widget.value = f'<div class="telemetry-card">{text}</div>'

    update_telemetry()

    # 4. Interactive Log
    log_widget = widgets.HTML()
    log_messages = []
    def add_log(msg):
        timestamp = time.strftime("%H:%M:%S")
        log_messages.append(f"[{timestamp}] {msg}")
        if len(log_messages) > 20:
            log_messages.pop(0)
        log_content = "<br/>".join(reversed(log_messages))
        log_widget.value = f'<div class="log-area">{log_content}</div>'

    add_log("Studio initialized. Connect devices and configure dataset below.")

    # 6. Session Progress Bars
    episode_progress = widgets.IntProgress(value=0, min=0, max=total_episodes, description="Episodes:", bar_style="info")

    # 7. Action Controls
    start_btn = widgets.Button(description="Start Episode", icon="circle", button_style="success", disabled=True)
    pause_btn = widgets.Button(description="Pause", icon="pause", button_style="warning", disabled=True)
    prev_step_btn = widgets.Button(description="Prev Step (B)", icon="step-backward", button_style="info", disabled=True)
    next_step_btn = widgets.Button(description="Next Step (N)", icon="step-forward", button_style="info", disabled=True)
    stop_btn = widgets.Button(description="Save Episode", icon="save", button_style="primary", disabled=True)
    discard_btn = widgets.Button(description="Discard & Redo", icon="trash", button_style="danger", disabled=True)

    control_row_1 = widgets.HBox([start_btn, pause_btn, prev_step_btn, next_step_btn])
    control_row_2 = widgets.HBox([stop_btn, discard_btn])

    # 8. Navigation Controls
    prev_btn = widgets.Button(description="Prev Ep", icon="chevron-left", disabled=True)
    next_btn = widgets.Button(description="Next Ep", icon="chevron-right", disabled=True)
    quit_btn = widgets.Button(description="Exit Studio", icon="times-circle", button_style="danger")

    navigation_row = widgets.HBox([prev_btn, next_btn, quit_btn])

    # 9. Camera Feed Widgets
    left_camera_widget = widgets.Image(format='jpeg', width=240, height=180, layout=widgets.Layout(border="1px solid #ccc", border_radius="8px"))
    top_camera_widget = widgets.Image(format='jpeg', width=480, height=360, layout=widgets.Layout(border="1px solid #ccc", border_radius="8px"))
    right_camera_widget = widgets.Image(format='jpeg', width=240, height=180, layout=widgets.Layout(border="1px solid #ccc", border_radius="8px"))
    
    cameras_layout = widgets.HBox([
        widgets.VBox([widgets.Label("Top Cam"), top_camera_widget], layout=widgets.Layout(align_items="center")),
        widgets.VBox([
            widgets.VBox([widgets.Label("Left Arm Cam"), left_camera_widget], layout=widgets.Layout(align_items="center")),
            widgets.VBox([widgets.Label("Right Arm Cam"), right_camera_widget], layout=widgets.Layout(align_items="center")),
        ], layout=widgets.Layout(justify_content="space-around", margin="5px")),
    ], layout=widgets.Layout(justify_content="space-around", margin="5px"))

    # Build columns
    left_column = widgets.VBox([
        cameras_layout,
        episode_progress,
        widgets.HTML("<b>Status:</b>"),
        status_card
    ], layout=widgets.Layout(width="60%", padding="5px"))

    telemetry_accordion = widgets.Accordion(children=[telemetry_widget])
    telemetry_accordion.set_title(0, "Telemetry")
    telemetry_accordion.selected_index = None
    telemetry_accordion.layout = widgets.Layout(margin="2px 0px")

    log_accordion = widgets.Accordion(children=[log_widget])
    log_accordion.set_title(0, "Studio Log")
    log_accordion.selected_index = 0
    log_accordion.layout = widgets.Layout(margin="2px 0px")

    # 8.5. Keyboard Shortcuts
    shortcut_input = widgets.Text(
        value="",
        placeholder="Click to input shortcuts"
    )
    shortcut_input.add_class("studio-shortcut-input")
    
    shortcut_toggle = widgets.Checkbox(value=True, description="Enable keyboard shortcuts")
    
    shortcut_legend = widgets.HTML("""
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
    """)
    
    shortcuts_box = widgets.VBox([
        shortcut_toggle,
        shortcut_legend
    ])
    
    shortcuts_accordion = widgets.Accordion(children=[shortcuts_box])
    shortcuts_accordion.set_title(0, "Keyboard Shortcuts")
    shortcuts_accordion.selected_index = None
    shortcuts_accordion.layout = widgets.Layout(margin="2px 0px")

    def on_shortcut_change(change):
        nonlocal current_episode_idx, datasets
        key = change['new']
        if not key:
            return
        key = key.lower()
        
        # Reset value immediately
        shortcut_input.value = ""
        
        if not shortcut_toggle.value:
            return
            
        if key == 'n':
            if not next_step_btn.disabled:
                on_next_step_clicked(None)
        elif key == 'b':
            if not prev_step_btn.disabled:
                on_prev_step_clicked(None)
        elif key == ' ' or key == 'space':
            if recording_state == "IDLE":
                if not start_btn.disabled:
                    on_start_clicked(None)
            elif recording_state == "RECORDING":
                if not next_step_btn.disabled:
                    on_next_step_clicked(None)
                elif not stop_btn.disabled:
                    on_stop_clicked(None)
            elif recording_state == "PAUSED":
                if not pause_btn.disabled and pause_btn.description == "Resume":
                    on_pause_clicked(None)
        elif key == 'r':
            if not start_btn.disabled:
                on_start_clicked(None)
            elif not pause_btn.disabled and pause_btn.description == "Resume":
                on_pause_clicked(None)
        elif key == 'p':
            if not pause_btn.disabled:
                on_pause_clicked(None)
        elif key == 's':
            if not stop_btn.disabled:
                on_stop_clicked(None)
        elif key == 'd':
            if not discard_btn.disabled:
                on_discard_clicked(None)
        elif key == '[':
            if not prev_btn.disabled:
                on_prev_clicked(None)
        elif key == ']':
            if not next_btn.disabled:
                on_next_clicked(None)
        elif key == 'q':
            on_quit_clicked(None)
            
    shortcut_input.observe(on_shortcut_change, names='value')

    controls_box = widgets.VBox([
        control_row_1,
        control_row_2,
        navigation_row,
        shortcut_input
    ])

    right_column = widgets.VBox([
        header_widget,
        widgets.HTML("<div style='height: 8px;'></div><b>Controls:</b>"),
        controls_box,
        widgets.HTML("<div style='height: 8px;'></div>"),
        shortcuts_accordion,
        widgets.HTML("<div style='height: 8px;'></div>"),
        telemetry_accordion,
        widgets.HTML("<div style='height: 8px;'></div>"),
        log_accordion
    ], layout=widgets.Layout(width="40%", padding="5px"))

    dashboard_layout = widgets.HBox([left_column, right_column], layout=widgets.Layout(
        border="1px solid #ddd", border_radius="15px", padding="8px", background_color="#ffffff"
    ))

    # ----------------- CALLBACK IMPLEMENTATION -----------------

    def schedule_background_task(coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        return loop.create_task(coro)

    def initialize_dataset():
        nonlocal datasets
        nonlocal current_episode_idx, total_episodes, recording_state, current_dataset_step_idx
        
        add_log(f"Initializing {len(steps_val)} datasets...")
        recording_state = "SAVING"
        update_status_card()
        
        try:
            # Build features
            dataset_features = combine_feature_dicts(
                aggregate_pipeline_dataset_features(
                    pipeline=teleop_action_processor,
                    initial_features=create_initial_features(action=robot.action_features),
                    use_videos=True,
                ),
                aggregate_pipeline_dataset_features(
                    pipeline=robot_observation_processor,
                    initial_features=create_initial_features(observation=robot.observation_features),
                    use_videos=True,
                ),
            )
            
            datasets.clear()
            for step_cfg in steps_val:
                repo_id = step_cfg["repo_id"]
                root_dir = step_cfg["root_dir"]
                
                if resume:
                    ds = LeRobotDataset.resume(
                        repo_id=repo_id,
                        root=root_dir,
                        streaming_encoding=True
                    )
                    current_episode_idx = ds.num_episodes
                else:
                    path = Path(root_dir)
                    if path.exists():
                        import shutil
                        try:
                            if path.is_dir():
                                shutil.rmtree(path)
                            else:
                                path.unlink()
                            add_log(f"Cleared existing dataset directory at {root_dir}")
                        except Exception as e:
                            add_log(f"Warning: failed to clear directory {root_dir}: {e}")
                    
                    ds = LeRobotDataset.create(
                        repo_id=repo_id,
                        fps=30,
                        root=root_dir,
                        robot_type=robot.name,
                        features=dataset_features,
                        use_videos=True,
                        streaming_encoding=True
                    )
                    current_episode_idx = 0
                datasets.append(ds)
                
            add_log("All datasets initialized successfully.")
                
            episode_progress.max = total_episodes
            episode_progress.value = current_episode_idx
            
            # Enable studio controls
            start_btn.disabled = False
            prev_step_btn.disabled = True
            next_step_btn.disabled = True
            update_navigation_buttons()
            
            recording_state = "IDLE"
            current_dataset_step_idx = 0
            update_status_card()
            add_log("Studio ready. Position robot and click 'Start Episode' to record.")
        except Exception as e:
            add_log(f"Error initializing dataset: {str(e)}")
            recording_state = "IDLE"
            current_dataset_step_idx = 0
            update_status_card()

    def update_navigation_buttons():
        nonlocal current_episode_idx, datasets
        if datasets and recording_state == "IDLE":
            prev_btn.disabled = (current_episode_idx == 0)
            next_btn.disabled = (current_episode_idx >= datasets[0].num_episodes)
        else:
            prev_btn.disabled = True
            next_btn.disabled = True

    def on_prev_clicked(b):
        nonlocal current_episode_idx, datasets
        if datasets and recording_state == "IDLE":
            current_episode_idx = max(0, current_episode_idx - 1)
            episode_progress.value = current_episode_idx
            update_navigation_buttons()
            add_log(f"Selected Episode {current_episode_idx} for viewing/recording.")

    prev_btn.on_click(on_prev_clicked)

    def on_next_clicked(b):
        nonlocal current_episode_idx, datasets
        if datasets and recording_state == "IDLE":
            current_episode_idx = min(dataset.num_episodes, current_episode_idx + 1)
            episode_progress.value = current_episode_idx
            update_navigation_buttons()
            add_log(f"Selected Episode {current_episode_idx} for viewing/recording.")

    next_btn.on_click(on_next_clicked)

    async def truncate_dataset_async(target_num_episodes):
        nonlocal datasets, recording_state, current_episode_idx
        start_btn.disabled = True
        prev_btn.disabled = True
        next_btn.disabled = True
        recording_state = "SAVING"
        update_status_card()
        
        try:
            for idx, ds in enumerate(datasets):
                repo_id = steps_val[idx]["repo_id"]
                root_dir = Path(steps_val[idx]["root_dir"])
                if target_num_episodes == 0:
                    await asyncio.to_thread(ds.finalize)
                    import shutil
                    if root_dir.exists():
                        await asyncio.to_thread(shutil.rmtree, root_dir)
                    
                    dataset_features = combine_feature_dicts(
                        aggregate_pipeline_dataset_features(pipeline=teleop_action_processor, initial_features=create_initial_features(action=robot.action_features), use_videos=True),
                        aggregate_pipeline_dataset_features(pipeline=robot_observation_processor, initial_features=create_initial_features(observation=robot.observation_features), use_videos=True)
                    )
                    datasets[idx] = await asyncio.to_thread(LeRobotDataset.create, repo_id=repo_id, fps=30, root=str(root_dir), robot_type=robot.name, features=dataset_features, use_videos=True, streaming_encoding=True)
                else:
                    await asyncio.to_thread(ds.finalize)
                    episode_indices_to_delete = list(range(target_num_episodes, ds.num_episodes))
                    import tempfile, shutil
                    temp_dir = Path(tempfile.mkdtemp(dir=str(root_dir.parent)))
                    from lerobot.datasets import delete_episodes
                    await asyncio.to_thread(delete_episodes, dataset=ds, episode_indices=episode_indices_to_delete, output_dir=temp_dir, repo_id=repo_id)
                    if root_dir.exists():
                        await asyncio.to_thread(shutil.rmtree, root_dir)
                    await asyncio.to_thread(shutil.move, str(temp_dir), str(root_dir))
                    datasets[idx] = await asyncio.to_thread(LeRobotDataset.resume, repo_id=repo_id, root=str(root_dir), streaming_encoding=True)
            add_log(f"All datasets truncated to {datasets[0].num_episodes} episodes.")
        except Exception as e:
            add_log(f"Error during truncation: {str(e)}")
        finally:
            current_episode_idx = datasets[0].num_episodes
            episode_progress.value = current_episode_idx

    async def start_recording_async():
        nonlocal recording_state, episode_start_time, accumulated_time, frames_in_episode, current_episode_idx, datasets, current_dataset_step_idx
        if recording_state != "IDLE":
            return
        
        if current_episode_idx < datasets[0].num_episodes:
            await truncate_dataset_async(current_episode_idx)
            if current_episode_idx != datasets[0].num_episodes:
                add_log("Aborting recording start because truncation failed.")
                return
            
        recording_state = "RECORDING"
        current_dataset_step_idx = 0
        episode_start_time = time.perf_counter()
        accumulated_time = 0.0
        frames_in_episode = 0
        
        start_btn.disabled = True
        pause_btn.disabled = False
        prev_step_btn.disabled = True
        next_step_btn.disabled = (len(steps_val) <= 1)
        stop_btn.disabled = False
        discard_btn.disabled = False
        prev_btn.disabled = True
        next_btn.disabled = True
        
        add_log(f"Recording episode {current_episode_idx} (Step {current_dataset_step_idx+1}/{len(steps_val)})...")
        update_status_card()

    def on_start_clicked(b):
        nonlocal active_bg_task
        active_bg_task = schedule_background_task(start_recording_async())

    start_btn.on_click(on_start_clicked)

    def on_next_step_clicked(b):
        nonlocal current_dataset_step_idx, frames_in_episode
        if recording_state == "RECORDING" and current_dataset_step_idx < len(steps_val) - 1:
            current_dataset_step_idx += 1
            frames_in_episode = 0
            add_log(f"Moved to Step {current_dataset_step_idx+1}/{len(steps_val)}.")
            if current_dataset_step_idx == len(steps_val) - 1:
                next_step_btn.disabled = True
            prev_step_btn.disabled = False
            update_status_card()

    next_step_btn.on_click(on_next_step_clicked)

    def on_prev_step_clicked(b):
        nonlocal current_dataset_step_idx, frames_in_episode
        if recording_state == "RECORDING" and current_dataset_step_idx > 0:
            current_dataset_step_idx -= 1
            frames_in_episode = 0
            add_log(f"Moved to Step {current_dataset_step_idx+1}/{len(steps_val)}.")
            if current_dataset_step_idx == 0:
                prev_step_btn.disabled = True
            next_step_btn.disabled = False
            update_status_card()

    prev_step_btn.on_click(on_prev_step_clicked)
    

    def on_pause_clicked(b):
        nonlocal recording_state, accumulated_time, episode_start_time
        if recording_state == "RECORDING":
            recording_state = "PAUSED"
            accumulated_time += time.perf_counter() - episode_start_time
            pause_btn.description = "Resume"
            pause_btn.icon = "play"
            add_log("Recording paused.")
        elif recording_state == "PAUSED":
            recording_state = "RECORDING"
            episode_start_time = time.perf_counter()
            pause_btn.description = "Pause"
            pause_btn.icon = "pause"
            add_log("Recording resumed.")
        update_status_card()

    pause_btn.on_click(on_pause_clicked)

    async def save_current_episode_async():
        nonlocal datasets, keep_running
        nonlocal recording_state, current_episode_idx, frames_in_episode, current_dataset_step_idx
        if recording_state == "SAVING":
            return
        recording_state = "SAVING"
        update_status_card()
        
        start_btn.disabled = True
        pause_btn.disabled = True
        prev_step_btn.disabled = True
        next_step_btn.disabled = True
        stop_btn.disabled = True
        discard_btn.disabled = True
        
        total_time = accumulated_time
        if recording_state == "RECORDING":
            total_time += time.perf_counter() - episode_start_time
        add_log(f"Saving episode {current_episode_idx} ({frames_in_episode} frames, {total_time:.1f}s) across all datasets...")
        try:
            for idx, ds in enumerate(datasets):
                # Ensure each dataset saves whatever buffer it has
                await asyncio.to_thread(ds.save_episode)
                await asyncio.to_thread(ds.finalize)
            add_log(f"Episode {current_episode_idx} finalized on disk for all steps.")
            
            current_episode_idx += 1
            episode_progress.value = current_episode_idx
            
            if current_episode_idx >= total_episodes:
                add_log("All target episodes recorded! Finalizing dataset...")
                await finalize_dataset_async()
                keep_running = False
            else:
                add_log("Preparing writer for next episode...")
                for idx, ds in enumerate(datasets):
                    repo_id = steps_val[idx]["repo_id"]
                    root_dir = steps_val[idx]["root_dir"]
                    datasets[idx] = await asyncio.to_thread(
                        LeRobotDataset.resume,
                        repo_id=repo_id,
                        root=root_dir,
                        streaming_encoding=True
                    )
                
                recording_state = "IDLE"
                current_dataset_step_idx = 0
                update_status_card()
                start_btn.disabled = False
                pause_btn.disabled = True
                pause_btn.description = "Pause"
                pause_btn.icon = "pause"
                stop_btn.disabled = True
                discard_btn.disabled = True
                update_navigation_buttons()
                add_log(f"Automatically moved to Episode {current_episode_idx}. Ready to record.")
        except Exception as e:
            add_log(f"Error saving episode: {str(e)}")
            recording_state = "PAUSED"
            update_status_card()
            pause_btn.disabled = False
            stop_btn.disabled = False
            discard_btn.disabled = False
            update_navigation_buttons()

    def on_stop_clicked(b):
        nonlocal active_bg_task
        active_bg_task = schedule_background_task(save_current_episode_async())

    stop_btn.on_click(on_stop_clicked)

    async def on_discard_clicked_async():
        nonlocal datasets
        nonlocal recording_state, current_dataset_step_idx
        if recording_state == "SAVING":
            return
        recording_state = "SAVING"
        update_status_card()
        
        add_log(f"Discarding episode {current_episode_idx} buffer across all steps...")
        try:
            for ds in datasets:
                await asyncio.to_thread(ds.clear_episode_buffer, delete_images=True)
            add_log("Episode buffer discarded.")
            
            start_btn.disabled = False
            pause_btn.disabled = True
            pause_btn.description = "Pause"
            pause_btn.icon = "pause"
            prev_step_btn.disabled = True
            next_step_btn.disabled = True
            stop_btn.disabled = True
            discard_btn.disabled = True
            
            recording_state = "IDLE"
            current_dataset_step_idx = 0
            update_status_card()
            update_navigation_buttons()
        except Exception as e:
            add_log(f"Error discarding episode: {str(e)}")
            recording_state = "PAUSED"
            update_status_card()
            update_navigation_buttons()

    def on_discard_clicked(b):
        nonlocal active_bg_task
        active_bg_task = schedule_background_task(on_discard_clicked_async())

    discard_btn.on_click(on_discard_clicked)

    async def finalize_dataset_async():
        nonlocal datasets
        nonlocal recording_state, current_dataset_step_idx
        recording_state = "SAVING"
        update_status_card()
        add_log("Finalizing all datasets on disk...")
        try:
            for ds in datasets:
                if ds:
                    await asyncio.to_thread(ds.finalize)
            add_log("All datasets finalized successfully! You can now push to hub or train.")
        except Exception as e:
            add_log(f"Error during finalization: {str(e)}")
        
        start_btn.disabled = True
        pause_btn.disabled = True
        prev_step_btn.disabled = True
        next_step_btn.disabled = True
        stop_btn.disabled = True
        discard_btn.disabled = True
        
        recording_state = "IDLE"
        current_dataset_step_idx = 0
        update_status_card()

    def on_quit_clicked(b):
        add_log("Exiting Studio session...")
        cleanup_studio_sync()

    quit_btn.on_click(on_quit_clicked)

    # Render the layout
    display(style_container)
    display(dashboard_layout)

    # ----------------- MAIN STUDIO LOOP -----------------

    # ----------------- MAIN STUDIO LOOP (ASYNC) -----------------

    async def main_loop():
        nonlocal keep_running, recording_state, current_episode_idx, total_episodes
        nonlocal frames_in_episode
        nonlocal episode_start_time, accumulated_time, last_frame_time, fps_log

        # Reset loop parameters
        keep_running = True
        loop_fps = 30.0
        last_widget_update_time = 0.0
        
        # Track communication failure state
        consecutive_failures = 0
        max_consecutive_failures = 10
        obs = None
        raw_action = None

        add_log("Studio loop started in background.")

        async def encode_and_update(key, img, widget, update_widget, update_stream):
            bgr_img = await asyncio.to_thread(cv2.cvtColor, img, cv2.COLOR_RGB2BGR)
            _, jpeg_encoded = await asyncio.to_thread(cv2.imencode, '.jpg', bgr_img)
            jpeg_bytes = jpeg_encoded.tobytes()
            if update_widget:
                widget.value = jpeg_bytes
            if update_stream:
                streamer.update_frame(key, jpeg_bytes)

        try:
            while keep_running:
                start_loop = time.perf_counter()
                
                # 1. Get robot observation and leader action concurrently and send action
                try:
                    new_obs, new_raw_action = await asyncio.gather(
                        asyncio.to_thread(robot.get_observation),
                        asyncio.to_thread(leader.get_action)
                    )
                    obs = new_obs
                    raw_action = new_raw_action
                    consecutive_failures = 0
                    
                    # 2. Always teleoperate so follower mirrors leader
                    obs_processed = robot_observation_processor(obs)
                    teleop_action = teleop_action_processor((raw_action, obs))
                    robot_action = robot_action_processor((teleop_action, obs))
                    
                    # Send to follower
                    await asyncio.to_thread(robot.send_action, robot_action)
                except Exception as e:
                    consecutive_failures += 1
                    if consecutive_failures == 1 or consecutive_failures % 5 == 0:
                        add_log(f"Warning: Robot loop communication failure ({consecutive_failures}/{max_consecutive_failures}): {e}")
                    if consecutive_failures >= max_consecutive_failures:
                        raise RuntimeError(f"Too many consecutive robot loop failures. Exiting loop. Last error: {e}")
                    if obs is None or raw_action is None:
                        raise e
                    await asyncio.sleep(0.01)
                    continue
                
                # 3. Handle states
                if recording_state == "RECORDING":
                    # Write frame to dataset
                    if datasets and current_dataset_step_idx < len(datasets):
                        ds = datasets[current_dataset_step_idx]
                        observation_frame = build_dataset_frame(ds.features, obs_processed, prefix=OBS_STR)
                        action_frame = build_dataset_frame(ds.features, teleop_action, prefix=ACTION)
                        
                        task_str = steps_val[current_dataset_step_idx]["task"]
                        
                        frame_data = {
                            **observation_frame,
                            **action_frame,
                            "task": task_str
                        }
                        ds.add_frame(frame_data)
                        frames_in_episode += 1
                        
                # 4. Update Live Camera feed widgets and HTTP Streamer (Parallelized)
                now = time.perf_counter()
                should_update_widgets = (now - last_widget_update_time >= 0.2)  # Update widgets at 5 Hz
                
                encoding_tasks = []
                for key, widget in [("left_cam", left_camera_widget), ("top", top_camera_widget), ("right_cam", right_camera_widget)]:
                    if key in obs:
                        # Only encode if someone is watching the HTTP stream, or if it is time to update the notebook widgets
                        has_stream_client = streamer.active_clients.get(key, 0) > 0
                        if has_stream_client or should_update_widgets:
                            task = encode_and_update(key, obs[key], widget, should_update_widgets, has_stream_client)
                            encoding_tasks.append(task)
                if encoding_tasks:
                    await asyncio.gather(*encoding_tasks)
                                
                if should_update_widgets:
                    last_widget_update_time = now
                
                # 6. Telemetry updates (throttled to 5Hz to avoid UI lag)
                if frames_in_episode % 6 == 0 or recording_state != "RECORDING":
                    update_telemetry(obs, raw_action, loop_fps)
                    if recording_state == "RECORDING":
                        update_status_card()
                        
                # 7. Sleep to maintain exact loop frequency (30Hz)
                dt = time.perf_counter() - start_loop
                sleep_time = max(0.0, (1.0 / 30.0) - dt)
                await asyncio.sleep(sleep_time)
                
                # Calculate actual FPS
                actual_dt = time.perf_counter() - start_loop
                if actual_dt > 0:
                    loop_fps = 0.9 * loop_fps + 0.1 * (1.0 / actual_dt)
                    
        except asyncio.CancelledError:
            add_log("Studio loop was cancelled.")
        except Exception as e:
            import traceback
            err_str = traceback.format_exc()
            add_log(f"CRITICAL ERROR in loop: {str(e)}")
            print(f"Error in studio loop:\n{err_str}")
        finally:
            add_log("Studio loop stopped.")
            cleanup_studio_sync()
            add_log("Studio cleanup completed.")

    studio_cleaned_up = False
    def cleanup_studio_sync():
        nonlocal keep_running, active_bg_task, datasets, studio_cleaned_up
        if studio_cleaned_up:
            return
        studio_cleaned_up = True
        keep_running = False
        
        try:
            streamer.stop_all_captures()
        except Exception:
            pass

        # Wait for any active background task to finish
        if active_bg_task is not None and not active_bg_task.done():
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    loop.run_until_complete(active_bg_task)
            except Exception as e:
                print(f"Error waiting for background task: {e}")

        # Finalize all datasets that are not finalized
        if datasets:
            for ds in datasets:
                if ds and not getattr(ds, "_is_finalized", False):
                    try:
                        ds.finalize()
                    except Exception as e:
                        print(f"Error finalizing dataset: {e}")

        # Unregister cell listener
        try:
            from IPython import get_ipython
            ip = get_ipython()
            if ip is not None:
                ip.events.unregister('pre_run_cell', stop_studio_on_new_cell)
        except Exception:
            pass

    def stop_studio_on_new_cell(*args, **kwargs):
        cleanup_studio_sync()

    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            ip.events.register('pre_run_cell', stop_studio_on_new_cell)
    except Exception as e:
        pass

    initialize_dataset()

    # Start the task asynchronously in the background so the cell completes immediately.
    # This allows Jupyter/VS Code to render and update widget views in real-time.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()

    loop.create_task(main_loop())
    add_log("Studio started in background. Cell execution finished. You can now use widgets in real-time.")
