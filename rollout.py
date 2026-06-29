from lerobot.configs import PreTrainedConfig
from lerobot.robots.config import RobotConfig
from lerobot.rollout import (
    RolloutConfig,
    BaseStrategyConfig,
    RTCInferenceConfig,
    SyncInferenceConfig,
    build_rollout_context,
    create_strategy,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.utils import init_logging


def rollout(
    robot: RobotConfig,
    policy: PreTrainedConfig,
    task: str,
    fps: float,
    asynchronous: bool = True,
    width: int | None = None,
    height: int | None = None,
):
    init_logging()
    register_third_party_plugins()

    # Adjust camera resolutions if specified
    if hasattr(robot, "cameras") and robot.cameras:
        for name, camera_cfg in robot.cameras.items():
            if width is not None:
                camera_cfg.width = width
            if height is not None:
                camera_cfg.height = height
    
    # IMPROVEMENT: Enable temporal ensembling for ACT policy to prevent shaking
    if policy.type == "act":
        if getattr(policy, "temporal_ensemble_coeff", None) is None:
            policy.temporal_ensemble_coeff = 0.1
            policy.n_action_steps = 1
        
    # Select between asynchronous RTC (Real-Time Chunking) and synchronous inline inference
    inference_config = RTCInferenceConfig() if asynchronous else SyncInferenceConfig()
        
    cfg = RolloutConfig(
        robot=robot,
        policy=policy,
        strategy=BaseStrategyConfig(),
        inference=inference_config,
        fps=fps,
        task=task,
        # use_torch_compile=True,  # Optimizes model execution latency 
        device="xpu",
    )
    
    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    ctx = build_rollout_context(cfg, signal_handler.shutdown_event)
    
    strategy = create_strategy(cfg.strategy)
    
    try:
        strategy.setup(ctx)
        strategy.run(ctx)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        print("Shutting down and cleaning up strategy...")
        strategy.teardown(ctx)
        
    print("Rollout finished successfully")
