from lerobot.configs import PreTrainedConfig
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.rollout import (
    RolloutConfig,
    BaseStrategyConfig,
    RTCInferenceConfig,
    SyncInferenceConfig,
    build_rollout_context,
)
from lerobot.rollout.strategies.base import BaseStrategy
from lerobot.rollout.strategies.core import send_next_action
from lerobot.rollout.context import RolloutContext
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.utils import init_logging
from lerobot.utils.robot_utils import precise_sleep
import time
import logging

logger = logging.getLogger(__name__)

from typing import Optional

_transition_direction: Optional[int] = None

def pump_ipython_events():
    pass

class RolloutTransitionUI:
    def __init__(self, total_steps: int):
        import ipywidgets as widgets
        self.total_steps = total_steps
        
        self.status_label = widgets.HTML(
            value="<b>Initializing...</b>",
            layout=widgets.Layout(margin='5px 0px')
        )
        self.prev_btn = widgets.Button(
            description="Prev Step (B)",
            icon="step-backward",
            button_style="warning",
            layout=widgets.Layout(width='140px')
        )
        self.next_btn = widgets.Button(
            description="Next Step (N)",
            icon="step-forward",
            button_style="success",
            layout=widgets.Layout(width='140px')
        )
        self.terminate_btn = widgets.Button(
            description="Terminate (Q)",
            icon="stop",
            button_style="danger",
            layout=widgets.Layout(width='140px')
        )
        self.shortcut_input = widgets.Text(
            value="",
            placeholder="Click & type 'b' (prev) / 'n' (next) / 'q' (quit)",
            layout=widgets.Layout(width='280px')
        )
        
        self.prev_btn.on_click(self.on_prev_click)
        self.next_btn.on_click(self.on_next_click)
        self.terminate_btn.on_click(self.on_terminate_click)
        self.shortcut_input.observe(self.on_shortcut_change, names='value')
        
        self.controls_row = widgets.HBox([self.prev_btn, self.next_btn, self.terminate_btn])
        self.shortcut_row = widgets.HBox([widgets.Label("Keyboard Input:"), self.shortcut_input])
        self.container = widgets.VBox([
            widgets.HTML("<h3>🔄 Rollout Step Transition Controller</h3>"),
            self.status_label,
            self.controls_row,
            self.shortcut_row
        ], layout=widgets.Layout(
            border='1px solid #ccc',
            padding='15px',
            border_radius='8px',
            background_color='#f9f9f9',
            width='480px'
        ))
        
    def display(self):
        from IPython.display import display
        display(self.container)
        
    def update(self, current_step: int, task: str):
        self.status_label.value = (
            f"<b>Current Step:</b> {current_step} / {self.total_steps - 1}<br/>"
            f"<b>Task:</b> {task}"
        )
        self.prev_btn.disabled = (current_step <= 0)
        self.next_btn.disabled = (current_step >= self.total_steps - 1)
        
    def on_prev_click(self, b):
        global _transition_direction
        _transition_direction = -1
        
    def on_next_click(self, b):
        global _transition_direction
        _transition_direction = 1
        
    def on_terminate_click(self, b):
        global _transition_direction
        _transition_direction = -999
        
    def on_shortcut_change(self, change):
        key = change['new']
        if not key:
            return
        key = key.lower()
        self.shortcut_input.value = ""
        
        if key in ('b', 'p'):
            self.on_prev_click(None)
        elif key == 'n':
            self.on_next_click(None)
        elif key == 'q':
            self.on_terminate_click(None)
            
    def close(self):
        self.container.close()

def check_transition_vlm(obs: dict, current_step: int, task: str) -> Optional[int]:
    """
    Evaluate observation with a VLM to determine if we should transition.
    Returns:
        - None: Continue the current step.
        - int (step_idx): The index of the next step to transition to. 
                          Can be a previous step (e.g. 0) or the next step (current_step + 1).
                          Return -1 to terminate the entire rollout.
    """
    pump_ipython_events()
    
    global _transition_direction
    if _transition_direction is not None:
        direction = _transition_direction
        _transition_direction = None  # Reset after reading
        
        if direction == -999:
            return -1  # Terminate rollout
            
        target = current_step + direction
        return target
        
    return None


class MultiStepStrategy(BaseStrategy):
    def __init__(self, config: BaseStrategyConfig, step_idx: int, task: str):
        super().__init__(config)
        self.step_idx = step_idx
        self.task = task
        self.next_step_idx: Optional[int] = None

    def run(self, ctx: RolloutContext) -> None:
        """Custom run loop that checks for VLM transitions."""
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        interpolator = self._interpolator

        control_interval = interpolator.get_control_interval(cfg.fps)

        start_time = time.perf_counter()
        engine.resume()
        logger.info(f"MultiStepStrategy control loop started for step {self.step_idx}")

        while not ctx.runtime.shutdown_event.is_set():
            loop_start = time.perf_counter()

            if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                logger.info("Duration limit reached (%.0fs)", cfg.duration)
                break

            obs = robot.get_observation()
            
            # --- VLM Transition Check ---
            next_step = check_transition_vlm(obs, self.step_idx, self.task)
            if next_step is not None:
                logger.info(f"VLM signaled transition to step {next_step}")
                self.next_step_idx = next_step
                break
            # ----------------------------

            obs_processed = self._process_observation_and_notify(ctx.processors, obs)

            if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                continue

            action_dict = send_next_action(obs_processed, obs, ctx, interpolator)
            self._log_telemetry(obs_processed, action_dict, ctx.runtime)

            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)
            else:
                logger.warning(
                    f"Record loop is running slower ({1 / dt:.1f} Hz) than the target FPS ({cfg.fps} Hz)."
                )

    async def run_async(self, ctx: RolloutContext) -> None:
        """Custom async run loop that checks for VLM transitions."""
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        interpolator = self._interpolator

        control_interval = interpolator.get_control_interval(cfg.fps)

        start_time = time.perf_counter()
        engine.resume()
        logger.info(f"MultiStepStrategy control loop started for step {self.step_idx}")

        import asyncio

        while not ctx.runtime.shutdown_event.is_set():
            loop_start = time.perf_counter()

            if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                logger.info("Duration limit reached (%.0fs)", cfg.duration)
                break

            obs = robot.get_observation()
            
            # --- VLM Transition Check ---
            next_step = check_transition_vlm(obs, self.step_idx, self.task)
            if next_step is not None:
                logger.info(f"VLM signaled transition to step {next_step}")
                self.next_step_idx = next_step
                break
            # ----------------------------

            obs_processed = self._process_observation_and_notify(ctx.processors, obs)

            if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                await asyncio.sleep(0.001)
                continue

            action_dict = send_next_action(obs_processed, obs, ctx, interpolator)
            self._log_telemetry(obs_processed, action_dict, ctx.runtime)

            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                await asyncio.sleep(sleep_t)
            else:
                logger.warning(
                    f"Record loop is running slower ({1 / dt:.1f} Hz) than the target FPS ({cfg.fps} Hz)."
                )
                await asyncio.sleep(0.001)

    def teardown(self, ctx: RolloutContext) -> None:
        """Stop inference without disconnecting hardware so the next step can reuse the connection."""
        if hasattr(self, "_engine") and self._engine is not None:
            self._engine.stop()
        logger.info(f"MultiStepStrategy teardown complete for step {self.step_idx}")

def create_rollout_context(
    robot: RobotConfig | Robot,
    policy: PreTrainedConfig,
    task: str,
    fps: float,
    asynchronous: bool = True,
    compile: bool = True,
    rename_map: Optional[dict[str, str]] = None,
    shutdown_event = None,
    teleop_action_processor = None,
    robot_action_processor = None,
    robot_observation_processor = None,
) -> RolloutContext:
    inference_config = RTCInferenceConfig() if asynchronous else SyncInferenceConfig()
    
    # IMPROVEMENT: Enable temporal ensembling for ACT policy to prevent shaking
    if policy.type == "act":
        if getattr(policy, "temporal_ensemble_coeff", None) is None:
            policy.temporal_ensemble_coeff = 0.1
            policy.n_action_steps = 1
            
    cfg = RolloutConfig(
        robot=robot,
        policy=policy,
        strategy=BaseStrategyConfig(),
        inference=inference_config,
        fps=fps,
        task=task,
        use_torch_compile=compile,  # Optimizes model execution latency 
        device="xpu",
        interpolation_multiplier=2,
        rename_map=rename_map or {},
    )
    import threading
    event = shutdown_event or threading.Event()
    return build_rollout_context(
        cfg, 
        event,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
    )


def rollout(
    robot: RobotConfig | Robot,
    policies: list[PreTrainedConfig],
    tasks: list[str],
    fps: float,
    asynchronous: bool = True,
    compile: bool = True,
    rename_map: Optional[dict[str, str]] = None,
):
    init_logging()
    register_third_party_plugins()
    
    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    
    # Ensure robot is instantiated and connected once for all steps
    if isinstance(robot, RobotConfig):
        robot = robot.build()
    if not robot.is_connected:
        robot.connect()
 
    # Initialize UI if running in Jupyter
    global _transition_direction
    _transition_direction = None
    
    ui = None
    ipython = None
    try:
        from IPython import get_ipython
        ipython = get_ipython()
    except (ImportError, Exception):
        pass

    # Prepend Step 0 (Idle)
    policies = [None] + policies
    tasks = ["Idle step (press Next Step to start)"] + tasks

    if ipython is not None:
        try:
            ui = RolloutTransitionUI(len(policies))
            ui.display()
        except Exception as e:
            print(f"Warning: Failed to display RolloutTransitionUI: {e}")

    async def run_rollout_loop():
        step_idx = 0
        while 0 <= step_idx < len(policies):
            policy = policies[step_idx]
            task = tasks[step_idx]
            print(f"=== Running Policy Step {step_idx}: {task} ===")
            
            if ui is not None:
                ui.update(step_idx, task)
                
            if policy is None:
                # Step 0: Idle loop
                next_step_idx = None
                while not signal_handler.shutdown_event.is_set():
                    obs = robot.get_observation() if hasattr(robot, "get_observation") else {}
                    next_step = check_transition_vlm(obs, step_idx, task)
                    if next_step is not None:
                        next_step_idx = next_step
                        break
                    await asyncio.sleep(0.05)
                
                if signal_handler.shutdown_event.is_set():
                    break
                    
                if next_step_idx is not None:
                    if next_step_idx == -1:
                        break
                    step_idx = next_step_idx
                else:
                    step_idx += 1
                continue
                
            ctx = create_rollout_context(
                robot=robot,
                policy=policy,
                task=task,
                fps=fps,
                asynchronous=asynchronous,
                compile=compile,
                rename_map=rename_map,
                shutdown_event=signal_handler.shutdown_event
            )
            strategy = MultiStepStrategy(ctx.runtime.cfg.strategy, step_idx, task)
            
            try:
                strategy.setup(ctx)
                await strategy.run_async(ctx)
            except KeyboardInterrupt:
                print("Interrupted by user")
                signal_handler.shutdown_event.set()
                break
            finally:
                strategy.teardown(ctx)
                
            if signal_handler.shutdown_event.is_set():
                break
                
            if strategy.next_step_idx is not None:
                if strategy.next_step_idx == -1:
                    break
                step_idx = strategy.next_step_idx
            else:
                # If the strategy exited for another reason (e.g. duration limit),
                # just proceed to the next step linearly.
                step_idx += 1
                
        if ui is not None:
            ui.close()
                
        print("Multi-step rollout finished. Disconnecting robot...")
        robot.disconnect()
        print("Done")

    if ipython is not None:
        import asyncio
        loop = asyncio.get_event_loop()
        loop.create_task(run_rollout_loop())
        print("Rollout started in background as asyncio Task.")
    else:
        import asyncio
        asyncio.run(run_rollout_loop())
