"""
action_smoothing.py
===================

SmolVLA rollout 시 action jitter(동작 흔들림)를 줄이기 위한 test-time 기법을
policy 에 붙였다 뗐다(on/off) 할 수 있게 만든 모듈.  lerobot 원본은 수정하지 않는다.

    1) ACG (Action Coherence Guidance)  -- arXiv:2510.22201  (ICRA 2026, DAVIAN Robotics)
       -> 이 모듈이 monkey-patch 로 "새로 추가" 하는 기법.
    2) RTC (Real-Time Chunking)         -- arXiv:2506.07339  (Physical Intelligence)
       -> lerobot 에 **이미 내장되어 있다**.  이 모듈은 그 knob 만 조정한다.

--------------------------------------------------------------------------------
★ 중요: RTC 는 이미 켜져 있다 (서버 소스 확인 결과)
--------------------------------------------------------------------------------
lerobot 은 RTC 를 자체 구현해 두었다:
    lerobot/policies/rtc/modeling_rtc.py        -> RTCProcessor.denoise_step (guidance 본체)
    lerobot/rollout/inference/rtc.py            -> RTCInferenceEngine (async 추론 스레드)

그리고 `rollout(..., asynchronous=True)` 는 RTCInferenceConfig 를 만들고,
lerobot/rollout/context.py:202-205 에서

    policy.config.rtc_config = cfg.inference.rtc
    policy.init_rtc_processor()

를 실행한다.  즉 **기존 노트북 호출(asynchronous=True)에서 RTC 는 이미 동작 중**이다.
=> 손으로 짠 RTC 를 denoise_step 에 또 얹으면 같은 guidance 가 이중 적용된다.
   그래서 이 모듈에서 자체 RTC 구현은 제거하고, 내장 RTC 의 파라미터만 바꾼다.

RTC 는 async 경로에서만 쓸 수 있다.  sync 엔진은 select_action 을 쓰는데
select_action 은 `assert not self._rtc_enabled()` 로 RTC 를 막는다
(modeling_smolvla.py:335).

--------------------------------------------------------------------------------
ACG 가 하는 일
--------------------------------------------------------------------------------
denoise step 마다 모델을 두 번 돌린다.
    - 정상 forward                                  -> v_t        (coherent)
    - action token 간 시간축 mixing 을 끊은 forward -> v_perturb  (incoherent)
그리고  v_t <- v_t + (scale-1) * (v_t - v_perturb)  로 외삽해
"시간적으로 일관된 성분"을 증폭한다.  (참조 구현: ACG/libs/robomimic/.../guidance/acg.py)

참조 구현은 GR00T DiT 의 `transformer_blocks[i].attn1`(= state+action self-attn)만
value-only 로 바꾸고, vision-language 로 가는 cross-attn(`attn2`)은 건드리지 않는다.
=> conditioning 은 유지하고 "시간 일관성"만 깬다.  SmolVLA 포팅도 이 원칙을 지킨다.

--------------------------------------------------------------------------------
사용법 (on/off)
--------------------------------------------------------------------------------
    from action_smoothing import SmoothingConfig, install_smoothing, uninstall_smoothing

    cfg = SmoothingConfig(acg=True, acg_scale=2.0)
    handle = install_smoothing(policy, cfg)     # policy = 로드된 SmolVLAPolicy
    ...  # rollout 실행
    uninstall_smoothing(handle)                 # 원상복구

보통은 clothing/rollout_smoothing.py 의 `rollout_smooth(..., smoothing="acg")` 를 쓴다.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Callable, Optional

import torch
import torch.nn as nn

from lerobot.configs.types import RTCAttentionSchedule

logger = logging.getLogger(__name__)


# ACG 참조 구현(FlowmatchingActionHead_ACG.get_action)의 기본값 skip_blocks=[7, 9, 11] /
# GR00T DiT 16 blocks 를 "전체 대비 위치" 로 옮긴 것.  SmolVLA 는 ACG 대상 layer 수가
# 다르므로 (index 그대로가 아니라) 같은 상대 위치를 쓴다.
_ACG_REF_FRACTIONS = (7 / 16, 9 / 16, 11 / 16)

_RTC_SCHEDULES = {
    "zeros": RTCAttentionSchedule.ZEROS,
    "ones": RTCAttentionSchedule.ONES,
    "linear": RTCAttentionSchedule.LINEAR,
    "exp": RTCAttentionSchedule.EXP,
}

# install_smoothing 이 in-place 로 덮어쓰는 RTCConfig 필드 (uninstall 시 되돌린다).
_RTC_TUNED_FIELDS = ("enabled", "prefix_attention_schedule", "max_guidance_weight", "execution_horizon")


# =============================================================================
# 설정
# =============================================================================
@dataclasses.dataclass
class SmoothingConfig:
    """ACG on/off + lerobot 내장 RTC knob 조정."""

    # ---------------- ACG (이 모듈이 추가하는 기법) ----------------
    acg: bool = False
    # guidance scale. 1.0 이면 효과 없음(패치해도 no-op). 클수록 강하게 coherence 강조.
    # 참조 구현 기본 3.0. SmolVLA 는 1.5~3.0 에서 튜닝 권장.
    acg_scale: float = 2.0
    # incoherent 로 만들 expert layer 인덱스(VLM layer index 기준).
    # None 이면 _default_acg_layers 가 참조 구현과 같은 상대 위치로 자동 선택.
    acg_layers: Optional[list[int]] = None

    # ---------------- RTC (lerobot 내장. async rollout 에서만 동작) ----------------
    # True 면 아래 값으로 policy.config.rtc_config 를 덮어쓴다.
    # False 면 lerobot 기본값(LINEAR / max_guidance_weight=10.0 / execution_horizon=10)을 그대로 둔다.
    rtc: bool = False
    rtc_enabled: bool = True
    rtc_schedule: str = "exp"
    rtc_max_guidance_weight: float = 5.0
    rtc_execution_horizon: int = 10

    # ---------------- fork 별 이름 hook (보통 그대로 두면 됨) ----------------
    model_attr: str = "model"           # policy.<model_attr> == VLAFlowMatching
    denoise_attr: str = "denoise_step"  # flow 모듈의 velocity 함수 이름
    # ACG 대상 layer 인덱스를 직접 정하고 싶을 때: lambda model -> list[int]
    acg_layers_getter: Optional[Callable[[nn.Module], list[int]]] = None

    def __post_init__(self):
        if self.rtc and self.rtc_schedule.lower() not in _RTC_SCHEDULES:
            raise ValueError(
                f"[action_smoothing] rtc_schedule 은 {sorted(_RTC_SCHEDULES)} 중 하나여야 합니다: "
                f"{self.rtc_schedule!r}"
            )
        if self.rtc and self.rtc_execution_horizon <= 0:
            raise ValueError(
                f"[action_smoothing] rtc_execution_horizon 은 양수여야 합니다: {self.rtc_execution_horizon}"
            )

    @property
    def any_enabled(self) -> bool:
        return self.acg or self.rtc


@dataclasses.dataclass
class _SmoothingHandle:
    """uninstall_smoothing 으로 되돌리기 위한 상태."""

    policy: object
    model: nn.Module
    cfg: SmoothingConfig
    # ACG
    orig_denoise: Optional[Callable] = None
    orig_forward_attn_layer: Optional[Callable] = None
    vlm: Optional[nn.Module] = None
    acg_layers: Optional[list[int]] = None
    # denoise 패치와 forward_attn_layer 패치가 공유하는 플래그.
    # (perturb=True 인 동안에만 attention mask 를 건드린다)
    acg_state: dict = dataclasses.field(default_factory=lambda: {"perturb": False})
    # 되돌릴 인스턴스 속성들: (owner, attr, 원래 인스턴스 __dict__ 에 있었는지, 원래 값)
    patched_attrs: list = dataclasses.field(default_factory=list)
    # RTC
    rtc_config: object = None
    orig_rtc_values: Optional[dict] = None


def _patch_attr(handle: _SmoothingHandle, owner, attr: str, new_value) -> None:
    """인스턴스 속성을 교체하고, 되돌리는 데 필요한 정보를 handle 에 기록한다."""
    had_own = attr in vars(owner)
    handle.patched_attrs.append((owner, attr, had_own, vars(owner).get(attr)))
    setattr(owner, attr, new_value)


def _restore_attrs(handle: _SmoothingHandle) -> None:
    """_patch_attr 로 바꾼 속성을 역순으로 되돌린다.

    원래 인스턴스 __dict__ 에 없던 속성(= 클래스 메서드였던 것)은 값을 되돌리는 게 아니라
    **지운다**.  bound method 를 인스턴스에 남겨두면 클래스 메서드를 영구히 가리고,
    self 를 참조하는 순환 참조가 생긴다.
    """
    for owner, attr, had_own, orig in reversed(handle.patched_attrs):
        if had_own:
            setattr(owner, attr, orig)
        else:
            try:
                delattr(owner, attr)
            except AttributeError:  # 이미 없으면 그만
                pass
    handle.patched_attrs.clear()


# =============================================================================
# 모델 찾기 / 구조 확인
# =============================================================================
def _get_flow_model(policy, cfg: SmoothingConfig) -> nn.Module:
    """policy 에서 VLAFlowMatching 모듈을 꺼낸다."""
    model = getattr(policy, cfg.model_attr, None)
    if model is None or not hasattr(model, cfg.denoise_attr):
        raise AttributeError(
            f"[action_smoothing] policy.{cfg.model_attr}.{cfg.denoise_attr} 를 찾지 못했습니다. "
            f"SmoothingConfig(model_attr=..., denoise_attr=...) 로 지정해 주세요. "
            f"(policy type={getattr(policy, 'name', type(policy).__name__)})"
        )
    return model


def _get_vlm_with_expert(model: nn.Module) -> nn.Module:
    vlm = getattr(model, "vlm_with_expert", None)
    if vlm is None or not hasattr(vlm, "forward_attn_layer"):
        raise AttributeError(
            "[action_smoothing/ACG] model.vlm_with_expert.forward_attn_layer 를 찾지 못했습니다. "
            "ACG 는 SmolVLMWithExpertModel 구조를 전제로 합니다 "
            "(lerobot/policies/smolvla/smolvlm_with_expert.py)."
        )
    return vlm


def _acg_eligible_layers(model: nn.Module) -> list[int]:
    """denoise 중 action token 끼리 시간축 mixing 이 일어나는 layer 인덱스.

    SmolVLMWithExpertModel.forward 의 분기 조건을 그대로 복제한다 (fill_kv_cache=False 기준):

        if fill_kv_cache or "cross" not in attention_mode \
           or (self_attn_every_n_layers > 0 and layer_idx % self_attn_every_n_layers == 0):
            forward_attn_layer(...)     # <- action token 간 self-attn 발생 (ACG 대상)
        else:
            forward_cross_attn_layer(...)   # <- prefix KV 로의 순수 cross-attn (대상 아님)

    towel_fold 체크포인트(attention_mode="cross_attn", self_attn_every_n_layers=2,
    num_vlm_layers=16) 기준으로 짝수 layer 0,2,...,14 (8개)가 대상이 된다.
    """
    vlm = _get_vlm_with_expert(model)
    # expert layer 가 None 인 인덱스(= expert 가 VLM 보다 얕은 경우)는 제외한다.
    expert_layers = vlm.get_model_layers([vlm.get_vlm_model().text_model, vlm.lm_expert])[1]

    eligible = []
    for layer_idx in range(vlm.num_vlm_layers):
        if expert_layers[layer_idx] is None:
            continue
        if "cross" not in vlm.attention_mode or (
            vlm.self_attn_every_n_layers > 0 and layer_idx % vlm.self_attn_every_n_layers == 0
        ):
            eligible.append(layer_idx)

    if not eligible:
        raise RuntimeError(
            "[action_smoothing/ACG] action token 끼리 self-attention 하는 expert layer 가 없습니다 "
            f"(attention_mode={vlm.attention_mode!r}, "
            f"self_attn_every_n_layers={vlm.self_attn_every_n_layers}). "
            "이 설정에서는 chunk 내부 시간 mixing 자체가 없어 ACG 가 의미 없습니다."
        )
    return eligible


def _default_acg_layers(eligible: list[int]) -> list[int]:
    """참조 구현의 skip_blocks=[7,9,11]/16 과 같은 상대 위치의 layer 를 고른다."""
    n = len(eligible)
    picks = sorted({min(n - 1, int(f * n)) for f in _ACG_REF_FRACTIONS})
    return [eligible[i] for i in picks]


# =============================================================================
# ACG: attention mask 수술
# =============================================================================
def _isolate_action_tokens(attention_mask: torch.Tensor) -> torch.Tensor:
    """action token 이 서로를 못 보게 만든다 (prefix conditioning 은 그대로 유지).

    denoise_step 이 만드는 mask 는

        full_att_2d_masks = cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        -> shape [B, suffix_len, prefix_len + suffix_len]   (bool)

    이므로 prefix_len = kv_len - q_len 로 복원할 수 있다.  뒤쪽 suffix_len 개 column
    (= action token) 에 대해서만 대각만 남기면, 각 action token 은
    "prefix 전체 + 자기 자신" 만 보게 된다 = ACG 의 incoherent(value-only) 변형.

    action token 쪽 mask 는 원래 causal 이다 (embed_suffix 가 att_masks=[1]*chunk_size 를
    내보내고, make_att_2d_masks 의 docstring 상 [[1 1 1 ...]] = pure causal).
    따라서 대각은 원래 True 이고, 이 수술 후에도 softmax row 가 전부 -inf 가 되지 않는다.
    """
    q_len, kv_len = attention_mask.shape[1], attention_mask.shape[2]
    if kv_len < q_len:
        # 예상치 못한 형태 -> 건드리지 않는다 (조용한 오작동보다 무효과가 낫다).
        logger.warning(
            "[action_smoothing/ACG] attention_mask 형태가 예상과 다릅니다 "
            "(q_len=%d > kv_len=%d). 이 layer 는 건너뜁니다.",
            q_len,
            kv_len,
        )
        return attention_mask

    prefix_len = kv_len - q_len
    eye = torch.eye(q_len, dtype=torch.bool, device=attention_mask.device)
    out = attention_mask.clone()
    out[:, :, prefix_len:] = out[:, :, prefix_len:] & eye
    return out


def _make_patched_forward_attn_layer(handle: _SmoothingHandle) -> Callable:
    """SmolVLMWithExpertModel.forward_attn_layer 래퍼.

    perturb 모드일 때 대상 layer 의 attention mask 만 바꿔서 넘긴다.
    시그니처는 호출부(smolvlm_with_expert.py:443-454)와 정확히 맞춘다.
    """
    orig = handle.orig_forward_attn_layer
    targets = set(handle.acg_layers)
    state = handle.acg_state

    def patched(
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
        use_cache: bool = True,
        fill_kv_cache: bool = True,
        past_key_values=None,
    ):
        # denoise 중(=prefix KV cache 를 읽기만 하는 pass)에만, 그리고 대상 layer 에만 적용.
        # inputs_embeds == [None, suffix_embs] 인 경우가 denoise pass 다.
        if (
            state["perturb"]
            and layer_idx in targets
            and not fill_kv_cache
            and len(inputs_embeds) == 2
            and inputs_embeds[0] is None
            and inputs_embeds[1] is not None
        ):
            attention_mask = _isolate_action_tokens(attention_mask)

        return orig(
            model_layers,
            inputs_embeds,
            layer_idx,
            position_ids,
            attention_mask,
            batch_size,
            head_dim,
            use_cache=use_cache,
            fill_kv_cache=fill_kv_cache,
            past_key_values=past_key_values,
        )

    return patched


def _make_patched_denoise(handle: _SmoothingHandle) -> Callable:
    """VLAFlowMatching.denoise_step 래퍼 (ACG guidance 적용 지점).

    denoise_step 은 prefix KV cache 를 **읽기만** 하므로(fill_kv_cache=False),
    같은 인자로 두 번 불러도 부작용이 없다 -> perturb forward 를 안전하게 추가할 수 있다.
    """
    cfg = handle.cfg
    orig = handle.orig_denoise
    state = handle.acg_state

    def patched(*args, **kwargs):
        v_t = orig(*args, **kwargs)

        if cfg.acg_scale == 1.0:  # 1.0 = 무효과. 굳이 두 번 돌리지 않는다.
            return v_t

        state["perturb"] = True
        try:
            # perturb 결과는 guidance 방향 계산에만 쓰이므로 그래프가 필요 없다.
            # (내장 RTC 는 denoise_step 을 torch.enable_grad() 안에서 부른다 -> 메모리 절약)
            with torch.no_grad():
                v_perturb = orig(*args, **kwargs)
        finally:
            state["perturb"] = False

        return v_t + (cfg.acg_scale - 1.0) * (v_t - v_perturb.to(v_t.dtype))

    return patched


# =============================================================================
# RTC: lerobot 내장 RTCConfig in-place 조정
# =============================================================================
def _apply_rtc_tuning(policy, cfg: SmoothingConfig) -> tuple[object, dict]:
    """policy.config.rtc_config 를 in-place 로 덮어쓰고, 원래 값을 반환한다.

    in-place 인 이유: 이 RTCConfig 인스턴스 **하나**를 네 곳이 공유하고 있다.
        ctx.runtime.cfg.inference.rtc                       (rollout config)
        policy.config.rtc_config                            (context.py:203, 같은 객체 대입)
        RTCInferenceEngine._rtc_config -> ActionQueue.cfg   (factory.py:118)
        RTCProcessor.rtc_config                             (modeling_smolvla.py:264)
    새 객체를 만들어 대입하면 이 공유가 끊어져 일부만 반영된다.
    """
    rtc_config = getattr(policy.config, "rtc_config", None)
    if rtc_config is None:
        raise RuntimeError(
            "[action_smoothing/RTC] policy.config.rtc_config 가 None 입니다.\n"
            "  lerobot 은 asynchronous=True (RTCInferenceConfig) 인 rollout 에서만 RTC 를 켭니다.\n"
            "  (lerobot/rollout/context.py:202-205 에서 rtc_config 를 주입)\n"
            "  sync 엔진은 select_action 을 쓰는데, select_action 은 RTC 를 assert 로 막습니다\n"
            "  (lerobot/policies/smolvla/modeling_smolvla.py:335).\n"
            "  => rollout_smooth(..., asynchronous=True) 로 호출하거나, smoothing='acg' 만 쓰세요."
        )

    new_values = {
        "enabled": cfg.rtc_enabled,
        "prefix_attention_schedule": _RTC_SCHEDULES[cfg.rtc_schedule.lower()],
        "max_guidance_weight": cfg.rtc_max_guidance_weight,
        "execution_horizon": cfg.rtc_execution_horizon,
    }
    orig_values = {f: getattr(rtc_config, f) for f in _RTC_TUNED_FIELDS}

    for field, value in new_values.items():
        setattr(rtc_config, field, value)

    logger.info(
        "[action_smoothing] RTC tuned | enabled=%s schedule=%s max_w=%.1f execution_horizon=%d",
        new_values["enabled"],
        new_values["prefix_attention_schedule"].value,
        new_values["max_guidance_weight"],
        new_values["execution_horizon"],
    )
    return rtc_config, orig_values


# =============================================================================
# 공개 API
# =============================================================================
def install_smoothing(policy, cfg: Optional[SmoothingConfig]) -> Optional[_SmoothingHandle]:
    """policy 에 smoothing 을 설치하고 되돌리기용 handle 을 반환.

    cfg 가 None 이거나 아무 기법도 켜져 있지 않으면 아무것도 하지 않고 None 을 반환한다.
    반드시 rollout 시작 전(= inference engine start 전)에 호출할 것.
    """
    if cfg is None or not cfg.any_enabled:
        logger.info("[action_smoothing] 켜진 기법 없음 -> 미적용")
        return None

    model = _get_flow_model(policy, cfg)
    handle = _SmoothingHandle(policy=policy, model=model, cfg=cfg)
    try:
        _install(policy, model, handle, cfg)
    except Exception:
        # 중간에 실패하면 이미 적용된 패치를 되돌린다.
        # (handle 을 호출자에게 주지 못한 채 실패하므로, 여기서 정리하지 않으면 영구히 남는다)
        _restore_attrs(handle)
        raise
    return handle


def _install(policy, model: nn.Module, handle: _SmoothingHandle, cfg: SmoothingConfig) -> None:
    # ---- ACG ----
    if cfg.acg:
        if cfg.acg_scale == 1.0:
            logger.warning("[action_smoothing/ACG] acg_scale=1.0 은 무효과입니다 (설치는 하지만 no-op).")

        getter = cfg.acg_layers_getter or _acg_eligible_layers
        eligible = getter(model)
        layers = cfg.acg_layers if cfg.acg_layers is not None else _default_acg_layers(eligible)

        invalid = [layer for layer in layers if layer not in eligible]
        if invalid:
            raise ValueError(
                f"[action_smoothing/ACG] layer {invalid} 는 ACG 대상이 아닙니다. "
                f"denoise 중 action token 간 self-attention 이 일어나는 layer 는 {eligible} 입니다. "
                "(나머지는 prefix 로의 순수 cross-attn 이라 패치해도 효과가 없습니다.)"
            )

        handle.vlm = _get_vlm_with_expert(model)
        handle.acg_layers = list(layers)
        handle.orig_denoise = getattr(model, cfg.denoise_attr)
        handle.orig_forward_attn_layer = handle.vlm.forward_attn_layer

        # 두 곳을 함께 패치한다:
        #   denoise_step         -> guidance 계산 + perturb 플래그 on/off
        #   forward_attn_layer   -> perturb 중 attention mask 수술
        _patch_attr(handle, handle.vlm, "forward_attn_layer", _make_patched_forward_attn_layer(handle))
        _patch_attr(handle, model, cfg.denoise_attr, _make_patched_denoise(handle))

        logger.info(
            "[action_smoothing] ACG on | scale=%.2f | target_layers=%s | eligible=%s",
            cfg.acg_scale,
            handle.acg_layers,
            eligible,
        )

    # ---- RTC (lerobot 내장) ----
    if cfg.rtc:
        handle.rtc_config, handle.orig_rtc_values = _apply_rtc_tuning(policy, cfg)


def uninstall_smoothing(handle: Optional[_SmoothingHandle]) -> None:
    """install_smoothing 이 반환한 handle 로 원상복구."""
    if handle is None:
        return

    _restore_attrs(handle)
    if handle.orig_rtc_values is not None:
        for field, value in handle.orig_rtc_values.items():
            setattr(handle.rtc_config, field, value)

    logger.info("[action_smoothing] uninstalled (acg=%s, rtc=%s).", handle.cfg.acg, handle.cfg.rtc)
