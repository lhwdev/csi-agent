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

def check_transition_vlm(obs: dict, current_step: int, task: str) -> Optional[int]:
    """
    Evaluate observation with a VLM to determine if we should transition.
    Returns:
        - None: Continue the current step.
        - int (step_idx): The index of the next step to transition to. 
                          Can be a previous step (e.g. 0) or the next step (current_step + 1).
                          Return -1 to terminate the entire rollout.
    """
    # TODO: Implement VLM logic here
    # For now, it never transitions autonomously.
    return None

class MultiStepStrategy(BaseStrategy):
    def __init__(self, step_idx: int, task: str):
        super().__init__()
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

    def teardown(self, ctx: RolloutContext) -> None:
        """Stop inference without disconnecting hardware so the next step can reuse the connection."""
        if hasattr(self, "_engine") and self._engine is not None:
            self._engine.stop()
        logger.info(f"MultiStepStrategy teardown complete for step {self.step_idx}")

def rollout(
    robot: RobotConfig | Robot,
    policies: list[PreTrainedConfig],
    tasks: list[str],
    fps: float,
    asynchronous: bool = True,
    compile: bool = True,
):
    init_logging()
    register_third_party_plugins()
    
    inference_config = RTCInferenceConfig() if asynchronous else SyncInferenceConfig()
    
    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    
    # Ensure robot is instantiated and connected once for all steps
    if isinstance(robot, RobotConfig):
        robot = robot.build()
    if not robot.is_connected:
        robot.connect()

    step_idx = 0
    while 0 <= step_idx < len(policies):
        policy = policies[step_idx]
        task = tasks[step_idx]
        print(f"=== Running Policy Step {step_idx}: {task} ===")
        
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
        )
        
        ctx = build_rollout_context(cfg, signal_handler.shutdown_event)
        strategy = MultiStepStrategy(step_idx, task)
        
        try:
            strategy.setup(ctx)
            strategy.run(ctx)
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
            
    print("Multi-step rollout finished. Disconnecting robot...")
    robot.disconnect()
    print("Done")
