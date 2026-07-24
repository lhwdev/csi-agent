"""
rollout_smoothing.py
====================

기존 `rollout.py` 는 **전혀 수정하지 않고**, action jitter(흔들림) 저감 기법을
on/off 로 얹은 별도 rollout 진입점.

- 기존 `rollout.py` 의 `create_rollout_context`, `MultiStepStrategy` 를 그대로 재사용한다.
- 기법 본체는 `action_smoothing.py` (policy monkey-patch, lerobot 원본 불변).

--------------------------------------------------------------------------------
smoothing 값
--------------------------------------------------------------------------------
    None / "off"    미적용 (기존 rollout 과 완전히 동일)
    "acg"           ACG on.  RTC 는 lerobot 기본값 그대로 둠
                    (=> asynchronous=True 면 RTC 도 기본 설정으로 이미 켜져 있다)
    "rtc"           lerobot 내장 RTC 의 파라미터만 조정 (asynchronous=True 필요)
    "acg+rtc"       둘 다
    SmoothingConfig(...)   세부 파라미터 직접 지정

★ RTC 는 이미 켜져 있다:
  `rollout(..., asynchronous=True)` 는 RTCInferenceConfig 를 만들고,
  lerobot/rollout/context.py:202-205 가 policy.config.rtc_config 를 주입한다.
  즉 기존 노트북 호출에서 RTC 는 이미 동작 중이며, jitter 는 "RTC 를 켠 상태"의 증상이다.
  따라서 새로 추가할 값이 있는 쪽은 ACG 다.  RTC 는 knob 조정만 의미가 있다.

--------------------------------------------------------------------------------
사용법 (노트북에서 기존 rollout 대신 이걸 호출)
--------------------------------------------------------------------------------
    from rollout_smoothing import rollout_smooth

    # 기존 호출과 동일 + ACG
    rollout_smooth(robot, policies, rollout_tasks, 30.0,
                   asynchronous=True, compile=False,
                   rename_map=pretrained_rename_map, smoothing="acg")

    # 세부 튜닝
    from action_smoothing import SmoothingConfig
    rollout_smooth(..., smoothing=SmoothingConfig(acg=True, acg_scale=2.5))

기존 `rollout()` 과 함수명이 다르므로(rollout_smooth) 서로 충돌하지 않는다.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

from lerobot.configs import PreTrainedConfig
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.utils import init_logging

# 기존 rollout.py 재사용 (수정하지 않음)
from rollout import MultiStepStrategy, create_rollout_context

# jitter 저감 기법 (별도 모듈)
from action_smoothing import SmoothingConfig, install_smoothing, uninstall_smoothing

logger = logging.getLogger(__name__)


# =============================================================================
# smoothing 인자 정규화
# =============================================================================
_PRESETS = {
    "acg": dict(acg=True),
    "rtc": dict(rtc=True),
    "acg+rtc": dict(acg=True, rtc=True),
    "rtc+acg": dict(acg=True, rtc=True),
}


def _coerce_smoothing(smoothing) -> Optional[SmoothingConfig]:
    """None / "acg" / "rtc" / "acg+rtc" / SmoothingConfig 를 SmoothingConfig 로 정규화."""
    if smoothing is None:
        return None
    if isinstance(smoothing, SmoothingConfig):
        return smoothing if smoothing.any_enabled else None
    if isinstance(smoothing, str):
        key = smoothing.strip().lower()
        if key in ("", "none", "off"):
            return None
        if key not in _PRESETS:
            raise ValueError(
                f"smoothing 은 None | {' | '.join(repr(k) for k in _PRESETS)} | SmoothingConfig "
                f"여야 합니다: {smoothing!r}"
            )
        return SmoothingConfig(**_PRESETS[key])
    raise TypeError(f"smoothing 은 None | str | SmoothingConfig 여야 합니다: {smoothing!r}")


# =============================================================================
# rollout context 에서 policy 찾기
# =============================================================================
def _locate_policy(ctx):
    """RolloutContext 에서 SmolVLAPolicy 를 꺼낸다.

    확정 경로: `ctx.policy.policy`
      lerobot/rollout/context.py 의 RolloutContext.policy 는 PolicyContext dataclass 이고
      (context.py:143-154), PolicyContext.policy 가 실제 PreTrainedPolicy 다 (context.py:114-121).
      policy 는 build_rollout_context 안에서 이미 로드되어 있으므로
      create_rollout_context 가 반환된 직후부터 접근 가능하다 (context.py:176-209).
    """
    policy = getattr(getattr(ctx, "policy", None), "policy", None)
    if policy is None:
        raise AttributeError(
            "[rollout_smooth] ctx.policy.policy 를 찾지 못했습니다. "
            "lerobot 의 RolloutContext 구조가 바뀐 것 같습니다 "
            "(lerobot/rollout/context.py 의 PolicyContext 확인)."
        )
    return policy


# =============================================================================
# smoothing 지원 rollout (기존 rollout() 과 동일 시그니처 + smoothing 인자)
# =============================================================================
def rollout_smooth(
    robot: Union[RobotConfig, Robot],
    policies: list[PreTrainedConfig],
    tasks: list[str],
    fps: float,
    asynchronous: bool = True,
    compile: bool = True,
    rename_map: Optional[dict[str, str]] = None,
    smoothing: Union[str, SmoothingConfig, None] = None,
):
    """기존 rollout.py 의 rollout() 과 동일하게 동작하되, jitter 저감 기법을 on/off 로 얹는다."""
    smoothing_cfg = _coerce_smoothing(smoothing)

    if smoothing_cfg is not None and smoothing_cfg.acg and compile:
        # context.py:181-182 가 policy_config.compile_model = cfg.use_torch_compile 로 넘기고,
        # VLAFlowMatching.__init__ 이 sample_actions 를 torch.compile 한다.
        # 그 뒤 denoise_step 을 갈아끼우면 dynamo 재컴파일/graph break 가 난다.
        logger.warning(
            "[rollout_smooth] ACG + compile=True 는 torch.compile 재컴파일을 유발할 수 있습니다. "
            "먼저 compile=False 로 검증하세요."
        )

    init_logging()
    register_third_party_plugins()

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)

    # 로봇은 모든 스텝에서 한 번만 연결
    if isinstance(robot, RobotConfig):
        robot = robot.build()
    if not robot.is_connected:
        robot.connect()

    step_idx = 0
    while 0 <= step_idx < len(policies):
        policy = policies[step_idx]
        task = tasks[step_idx]
        print(f"=== Running Policy Step {step_idx}: {task} ===")

        ctx = create_rollout_context(
            robot=robot,
            policy=policy,
            task=task,
            fps=fps,
            asynchronous=asynchronous,
            compile=compile,
            rename_map=rename_map,
            shutdown_event=signal_handler.shutdown_event,
        )
        strategy = MultiStepStrategy(ctx.runtime.cfg.strategy, step_idx, task)

        smoothing_handle = None
        try:
            # ★ setup(ctx) 이전에 설치한다.
            #   setup -> _init_engine -> engine.start() 가 RTC 추론 스레드를 띄우므로
            #   (rollout/strategies/core.py:55-70, rollout/inference/rtc.py:180-194),
            #   그 뒤에 패치하면 스레드가 이미 추론 중일 수 있다.
            #   policy 는 create_rollout_context 시점에 이미 로드되어 있다.
            if smoothing_cfg is not None:
                smoothing_handle = install_smoothing(_locate_policy(ctx), smoothing_cfg)

            strategy.setup(ctx)
            strategy.run(ctx)
        except KeyboardInterrupt:
            print("Interrupted by user")
            signal_handler.shutdown_event.set()
            break
        finally:
            strategy.teardown(ctx)      # 추론 스레드를 먼저 멈추고
            uninstall_smoothing(smoothing_handle)   # 그 다음 패치를 되돌린다

        if signal_handler.shutdown_event.is_set():
            break

        if strategy.next_step_idx is not None:
            if strategy.next_step_idx == -1:
                break
            step_idx = strategy.next_step_idx
        else:
            # duration limit 등 다른 이유로 종료된 경우 다음 스텝으로 선형 진행
            step_idx += 1

    print("Multi-step rollout finished. Disconnecting robot...")
    robot.disconnect()
    print("Done")
