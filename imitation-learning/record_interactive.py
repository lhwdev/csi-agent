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

def record_interactive(robot: Robot, leader: Teleoperator, cam: cv2.VideoCapture, params=None):
    """
    Interactive recording studio GUI.
    
    Args:
        robot: The follower robot.
        leader: The leader teleoperator robot.
        cam: OpenCV camera capture object.
        params: An optional object (e.g. types.SimpleNamespace) holding configuration parameter attributes.
        
    Example:
        from types import SimpleNamespace
        
        params = SimpleNamespace(
            repo_id="lhwdev/pick_umbrella",
            root_dir="/home/lhwdev/csi-agent/lerobot/lhwdev/records/pick_umbrella",
            task="Pick up the umbrella.",
            episodes=30,
            episode_time=40.0,
            resume=False
        )
        record_interactive(robot, leader, cam, params)
    """
    dataset = None

    # Extract values from params if provided
    repo_id_val = getattr(params, "repo_id", "lhwdev/pick_umbrella") if params is not None else "lhwdev/pick_umbrella"
    
    # Try multiple common attribute names for output path / root
    root_dir_val = "/home/lhwdev/csi-agent/lerobot/lhwdev/records/pick_umbrella"
    if params is not None:
        if hasattr(params, "root_dir"):
            root_dir_val = params.root_dir
        elif hasattr(params, "root"):
            root_dir_val = params.root
        elif hasattr(params, "output_path"):
            root_dir_val = params.output_path
            
    task_val = "Pick up the umbrella."
    if params is not None:
        if hasattr(params, "task"):
            task_val = params.task
        elif hasattr(params, "task_prompt"):
            task_val = params.task_prompt
        elif hasattr(params, "single_task"):
            task_val = params.single_task
            
    episodes_val = 30
    if params is not None:
        if hasattr(params, "episodes"):
            episodes_val = params.episodes
        elif hasattr(params, "total_episodes"):
            episodes_val = params.total_episodes
        elif hasattr(params, "num_episodes"):
            episodes_val = params.num_episodes
            
    episode_time_val = 40.0
    if params is not None:
        if hasattr(params, "episode_time"):
            episode_time_val = params.episode_time
        elif hasattr(params, "episode_time_limit"):
            episode_time_val = params.episode_time_limit
        elif hasattr(params, "episode_time_s"):
            episode_time_val = params.episode_time_s
            
    resume_val = getattr(params, "resume", False) if params is not None else False

    # Global state variables for the recording loop
    active_loop_task = None
    recording_state = "IDLE"  # IDLE, RECORDING, PAUSED, SAVING
    current_episode_idx = 0
    total_episodes = episodes_val
    frames_in_episode = 0
    episode_time_limit = episode_time_val
    keep_running = True

    episode_start_time = 0
    accumulated_time = 0
    reset_start_time = 0
    last_frame_time = 0
    fps_log = []


    # Cancel previous background task if it is still running to avoid duplicate loops
    if active_loop_task is not None and not active_loop_task.done():
        active_loop_task.cancel()

    if not robot.is_connected:
        robot.connect()
    if not leader.is_connected:
        leader.connect()
    if not cam.isOpened():
        cam.open(2) # reopen if closed

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
        nonlocal recording_state, current_episode_idx, frames_in_episode
        state_classes = {
            "IDLE": "status-idle",
            "RECORDING": "status-recording",
            "PAUSED": "status-paused",
            "SAVING": "status-saving"
        }
        state_class = state_classes.get(recording_state, "status-idle")
        
        if recording_state == "RECORDING":
            sub = f"Capturing ({frames_in_episode} frames)"
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
            <div style="font-size: 14px; font-weight: 700; margin-top: 5px; opacity: 0.95;">Episode: {current_episode_idx}</div>
            <div style="font-size: 12px; margin-top: 3px; opacity: 0.8;">{sub}</div>
            {progress_html}
        </div>
        """

    update_status_card()

    # 3. Telemetry Panel
    telemetry_widget = widgets.HTML()
    def update_telemetry(robot_obs=None, leader_act=None, loop_fps=0.0):
        text = f"<b>STUDIO TELEMETRY</b><br/>"
        text += f"Loop frequency: {loop_fps:.1f} Hz<br/>"
        
        if robot_obs is not None:
            pos_keys = [k for k in robot_obs.keys() if k.endswith('.pos')]
            if pos_keys:
                text += f"<b>Follower Joints:</b><br/>"
                for k in pos_keys:
                    text += f" &bull; {k.split('.')[0]}: {robot_obs[k]:.2f}<br/>"
                    
        if leader_act is not None:
            pos_keys = [k for k in leader_act.keys() if k.endswith('.pos')]
            if pos_keys:
                text += f"<b>Leader Joints:</b><br/>"
                for k in pos_keys:
                    text += f" &bull; {k.split('.')[0]}: {leader_act[k]:.2f}<br/>"
                    
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

    # 5. Configuration Fields (grouped in Accordion)
    dataset_id_input = widgets.Text(value=repo_id_val, description="Repo ID:")
    root_input = widgets.Text(value=root_dir_val, description="Root Dir:")
    task_input = widgets.Text(value=task_val, description="Task Prompt:")
    
    # Safely construct sliders to avoid out-of-range value errors in ipywidgets
    episodes_input = widgets.IntSlider(
        value=episodes_val,
        min=min(1, episodes_val),
        max=max(100, episodes_val),
        description="Episodes:"
    )
    episode_time_input = widgets.FloatSlider(
        value=episode_time_val,
        min=min(5.0, episode_time_val),
        max=max(120.0, episode_time_val),
        step=1.0,
        description="Ep Time (s):"
    )
    resume_checkbox = widgets.Checkbox(value=resume_val, description="Resume existing dataset")
    init_btn = widgets.Button(description="Initialize Dataset", button_style="info", icon="database")

    config_box = widgets.VBox([
        dataset_id_input,
        root_input,
        task_input,
        episodes_input,
        episode_time_input,
        resume_checkbox,
        init_btn
    ])

    config_accordion = widgets.Accordion(children=[config_box])
    config_accordion.set_title(0, "Dataset Configuration")

    # 6. Session Progress Bars
    episode_progress = widgets.IntProgress(value=0, min=0, max=30, description="Episodes:", bar_style="info")
    time_progress = widgets.FloatProgress(value=0.0, min=0.0, max=40.0, description="Ep Time:", bar_style="success")

    # 7. Action Controls
    start_btn = widgets.Button(description="Start Episode", icon="circle", button_style="success", disabled=True)
    pause_btn = widgets.Button(description="Pause", icon="pause", button_style="warning", disabled=True)
    stop_btn = widgets.Button(description="Save Episode", icon="save", button_style="primary", disabled=True)
    discard_btn = widgets.Button(description="Discard & Redo", icon="trash", button_style="danger", disabled=True)

    control_row_1 = widgets.HBox([start_btn, pause_btn])
    control_row_2 = widgets.HBox([stop_btn, discard_btn])

    # 8. Navigation Controls
    prev_btn = widgets.Button(description="Prev Ep", icon="chevron-left", disabled=True)
    next_btn = widgets.Button(description="Next Ep", icon="chevron-right", disabled=True)
    quit_btn = widgets.Button(description="Exit Studio", icon="times-circle", button_style="danger")

    navigation_row = widgets.HBox([prev_btn, next_btn, quit_btn])

    # 9. Camera Feed Widget
    camera_widget = widgets.Image(format='jpeg', width=540, height=400)

    # Build columns
    left_column = widgets.VBox([
        camera_widget,
        episode_progress,
        time_progress,
        widgets.HTML("<b>Status:</b>"),
        status_card
    ], layout=widgets.Layout(width="55%", padding="10px"))

    telemetry_accordion = widgets.Accordion(children=[telemetry_widget])
    telemetry_accordion.set_title(0, "Telemetry")
    telemetry_accordion.selected_index = 0
    telemetry_accordion.layout = widgets.Layout(margin="2px 0px")

    log_accordion = widgets.Accordion(children=[log_widget])
    log_accordion.set_title(0, "Studio Log")
    log_accordion.selected_index = 0
    log_accordion.layout = widgets.Layout(margin="2px 0px")

    right_column = widgets.VBox([
        header_widget,
        config_accordion,
        widgets.HTML("<br/><b>Controls:</b>"),
        control_row_1,
        control_row_2,
        navigation_row,
        widgets.HTML("<br/>"),
        telemetry_accordion,
        widgets.HTML("<br/>"),
        log_accordion
    ], layout=widgets.Layout(width="45%", padding="10px"))

    dashboard_layout = widgets.HBox([left_column, right_column], layout=widgets.Layout(
        border="1px solid #ddd", border_radius="15px", padding="15px", background_color="#ffffff"
    ))

    # ----------------- CALLBACK IMPLEMENTATION -----------------

    def on_init_clicked(b):
        nonlocal dataset
        nonlocal current_episode_idx, total_episodes, episode_time_limit, recording_state
        
        # Freeze config UI
        dataset_id_input.disabled = True
        root_input.disabled = True
        task_input.disabled = True
        episodes_input.disabled = True
        episode_time_input.disabled = True
        resume_checkbox.disabled = True
        init_btn.disabled = True
        
        repo_id = dataset_id_input.value
        root_dir = root_input.value
        total_episodes = episodes_input.value
        episode_time_limit = episode_time_input.value
        resume = resume_checkbox.value
        
        add_log(f"Initializing LeRobotDataset: {repo_id} at {root_dir}...")
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
            
            # Dynamically add the front camera if it is not present
            if "observation.images.front" not in dataset_features:
                # Check camera size dynamically
                h, w = 480, 640
                ret, test_frame = cam.read()
                if ret and test_frame is not None:
                    h, w = test_frame.shape[:2]
                
                dataset_features["observation.images.front"] = {
                    "dtype": "video",
                    "shape": (h, w, 3),
                    "names": ["height", "width", "channels"],
                }
                add_log(f"Configured front camera feature with resolution {w}x{h}")
                
            if resume:
                dataset = LeRobotDataset.resume(
                    repo_id=repo_id,
                    root=root_dir,
                    streaming_encoding=True
                )
                current_episode_idx = dataset.num_episodes
                add_log(f"Resumed dataset. Existing episodes: {current_episode_idx}")
            else:
                path = Path(root_dir)
                if path.exists():
                    if path.is_dir() and not any(path.iterdir()):
                        path.rmdir()  # Safe to delete empty folder so LeRobot can initialize
                    elif path.is_dir():
                        raise FileExistsError(
                            f"Directory '{root_dir}' already exists and is not empty. "
                            "Please check 'Resume existing dataset' or use a different Root Dir."
                        )
                    else:
                        raise FileExistsError(f"A file already exists at '{root_dir}'. Please choose a different path.")

                dataset = LeRobotDataset.create(
                    repo_id=repo_id,
                    fps=30,
                    root=root_dir,
                    robot_type=robot.name,
                    features=dataset_features,
                    use_videos=True,
                    streaming_encoding=True
                )
                current_episode_idx = 0
                add_log("New dataset created successfully.")
                
            episode_progress.max = total_episodes
            episode_progress.value = current_episode_idx
            time_progress.max = episode_time_limit
            
            # Enable studio controls
            start_btn.disabled = False
            prev_btn.disabled = (current_episode_idx == 0)
            
            recording_state = "IDLE"
            update_status_card()
            add_log("Studio ready. Position robot and click 'Start Episode' to record.")
        except Exception as e:
            add_log(f"Error initializing dataset: {str(e)}")
            # Unfreeze config UI
            dataset_id_input.disabled = False
            root_input.disabled = False
            task_input.disabled = False
            episodes_input.disabled = False
            episode_time_input.disabled = False
            resume_checkbox.disabled = False
            init_btn.disabled = False
            recording_state = "IDLE"
            update_status_card()

    init_btn.on_click(on_init_clicked)

    def on_start_clicked(b):
        nonlocal recording_state, episode_start_time, accumulated_time, frames_in_episode
        if recording_state == "IDLE":
            recording_state = "RECORDING"
            episode_start_time = time.perf_counter()
            accumulated_time = 0.0
            frames_in_episode = 0
            
            # Configure buttons
            start_btn.disabled = True
            pause_btn.disabled = False
            stop_btn.disabled = False
            discard_btn.disabled = False
            
            add_log(f"Recording episode {current_episode_idx}...")
            update_status_card()

    start_btn.on_click(on_start_clicked)

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

    def save_current_episode():
        nonlocal dataset, keep_running
        nonlocal recording_state, current_episode_idx, frames_in_episode
        recording_state = "SAVING"
        update_status_card()
        
        # Disable controls during save
        start_btn.disabled = True
        pause_btn.disabled = True
        stop_btn.disabled = True
        discard_btn.disabled = True
        
        add_log(f"Saving episode {current_episode_idx}...")
        try:
            # Save episode to disk
            dataset.save_episode()
            add_log(f"Episode {current_episode_idx} saved successfully! ({frames_in_episode} frames)")
            
            current_episode_idx += 1
            episode_progress.value = current_episode_idx
            
            if current_episode_idx >= total_episodes:
                add_log("All target episodes recorded! Finalizing dataset...")
                finalize_dataset()
                keep_running = False
            else:
                # Automatically move to next episode
                recording_state = "IDLE"
                update_status_card()
                start_btn.disabled = False
                pause_btn.disabled = True
                pause_btn.description = "Pause"
                pause_btn.icon = "pause"
                stop_btn.disabled = True
                discard_btn.disabled = True
                time_progress.value = 0.0
                add_log(f"Automatically moved to Episode {current_episode_idx}. Ready to record.")
        except Exception as e:
            add_log(f"Error saving episode: {str(e)}")
            recording_state = "PAUSED"
            update_status_card()
            pause_btn.disabled = False
            stop_btn.disabled = False
            discard_btn.disabled = False

    stop_btn.on_click(lambda b: save_current_episode())

    def on_discard_clicked(b):
        nonlocal dataset
        nonlocal recording_state
        recording_state = "SAVING"
        update_status_card()
        
        add_log(f"Discarding episode {current_episode_idx} buffer...")
        try:
            dataset.clear_episode_buffer(delete_images=True)
            add_log("Episode buffer discarded.")
            
            # Reset control states
            start_btn.disabled = False
            pause_btn.disabled = True
            pause_btn.description = "Pause"
            pause_btn.icon = "pause"
            stop_btn.disabled = True
            discard_btn.disabled = True
            
            recording_state = "IDLE"
            update_status_card()
        except Exception as e:
            add_log(f"Error discarding episode: {str(e)}")
            recording_state = "PAUSED"
            update_status_card()

    discard_btn.on_click(on_discard_clicked)

    def finalize_dataset():
        nonlocal dataset
        nonlocal recording_state
        recording_state = "SAVING"
        update_status_card()
        add_log("Finalizing dataset on disk...")
        try:
            if dataset:
                dataset.finalize()
            add_log("Dataset finalized successfully! You can now push to hub or train.")
        except Exception as e:
            add_log(f"Error during finalization: {str(e)}")
        
        # Disable recording controls
        start_btn.disabled = True
        pause_btn.disabled = True
        stop_btn.disabled = True
        discard_btn.disabled = True
        
        recording_state = "IDLE"
        update_status_card()

    def on_quit_clicked(b):
        nonlocal keep_running
        add_log("Exiting Studio session...")
        keep_running = False
        try:
            from IPython import get_ipython
            ip = get_ipython()
            if ip is not None:
                ip.events.unregister('pre_run_cell', stop_studio_on_new_cell)
        except Exception as e:
            pass

    quit_btn.on_click(on_quit_clicked)

    # Render the layout
    display(style_container)
    display(dashboard_layout)

    # ----------------- MAIN STUDIO LOOP -----------------

    # ----------------- MAIN STUDIO LOOP (ASYNC) -----------------

    async def main_loop():
        nonlocal keep_running, recording_state, current_episode_idx, total_episodes
        nonlocal frames_in_episode, episode_time_limit
        nonlocal episode_start_time, accumulated_time, last_frame_time, fps_log

        # Reset loop parameters
        keep_running = True
        loop_fps = 30.0

        add_log("Studio loop started in background.")

        try:
            while keep_running:
                start_loop = time.perf_counter()
                
                # 1. Read camera frame
                ret, frame = cam.read()
                if not ret or frame is None:
                    await asyncio.sleep(0.01)
                    continue
                    
                # Convert BGR (OpenCV) to RGB (LeRobot and display)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 2. Get robot observation and leader action
                obs = robot.get_observation()
                raw_action = leader.get_action()
                
                # 3. Always teleoperate so follower mirrors leader
                obs_processed = robot_observation_processor(obs)
                teleop_action = teleop_action_processor((raw_action, obs))
                robot_action = robot_action_processor((teleop_action, obs))
                
                # Send to follower
                robot.send_action(robot_action)
                
                # 4. Handle states
                if recording_state == "RECORDING":
                    # Compute elapsed time
                    elapsed = accumulated_time + (time.perf_counter() - episode_start_time)
                    time_progress.value = min(elapsed, episode_time_limit)
                    
                    # Write frame to dataset
                    if dataset is not None:
                        # Add dynamic camera frame under the expected name in obs_processed
                        obs_processed["front"] = rgb_frame
                        
                        observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
                        action_frame = build_dataset_frame(dataset.features, teleop_action, prefix=ACTION)
                        
                        # Assemble complete frame dictionary
                        frame_data = {
                            **observation_frame,
                            **action_frame,
                            "task": task_input.value
                        }
                        dataset.add_frame(frame_data)
                        frames_in_episode += 1
                        
                        # Check for automatic episode end
                        if elapsed >= episode_time_limit:
                            add_log("Episode time limit reached.")
                            save_current_episode()
                        
                # 5. Update Live Camera feed widget
                # Stream BGR frame directly encoded as JPEG (hardware-accelerated, flicker-free)
                _, jpeg_encoded = cv2.imencode('.jpg', frame)
                camera_widget.value = jpeg_encoded.tobytes()
                
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
            # Safe finalization
            if dataset is not None and not dataset._is_finalized:
                finalize_dataset()
            add_log("Studio cleanup completed.")

    def stop_studio_on_new_cell(*args, **kwargs):
        nonlocal keep_running
        if keep_running:
            add_log("New cell execution started. Automatically stopping studio...")
            keep_running = False
            try:
                from IPython import get_ipython
                ip = get_ipython()
                if ip is not None:
                    ip.events.unregister('pre_run_cell', stop_studio_on_new_cell)
            except Exception as e:
                pass

    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            ip.events.register('pre_run_cell', stop_studio_on_new_cell)
    except Exception as e:
        pass

    if params is not None:
        on_init_clicked(None)

    # Start the task asynchronously in the background so the cell completes immediately.
    # This allows Jupyter/VS Code to render and update widget views in real-time.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()

    loop.create_task(main_loop())
    add_log("Studio started in background. Cell execution finished. You can now use widgets in real-time.")
