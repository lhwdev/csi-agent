"""
test_action_smoothing.py
========================

`action_smoothing.py` 검증 스크립트.  실제 SmolVLA policy 를 CPU 로 로드해서
install -> uninstall 왕복, shape 보존, 그리고 "ACG 가 정말 동작하는지" 를 확인한다.

실행:
    cd clothing && HF_HUB_OFFLINE=1 python test_action_smoothing.py

로봇/GPU 불필요.  `lerobot/smolvla_base` 를 CPU 로 로드한다(HF 캐시에 있어야 함).
towel_fold 체크포인트가 아니라 base 를 쓰는 이유: 여기서 보는 건 **구조**
(attention_mode, expert layer 배치, denoise_step 시그니처)이고 두 체크포인트가 동일하기 때문.
가중치 값은 이 테스트의 판정 기준이 아니다.

★ 가장 중요한 검사는 "=== 5. ACG install ===" 의
      [PASS] ACG 가 실제로 출력을 바꾼다 (no-op 아님)
  이다.  ACG 를 잘못된 지점(self_attn.forward 등)에 걸면 여기서 max|Δ|=0 이 나오면서
  "2배 연산을 쓰고 아무 효과 없음" 상태가 조용히 만들어진다.  실제로 초기 구현이 그랬다.
"""

import inspect
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching

from action_smoothing import (
    SmoothingConfig,
    _acg_eligible_layers,
    _default_acg_layers,
    _isolate_action_tokens,
    install_smoothing,
    uninstall_smoothing,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
torch.manual_seed(0)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


# ---------------------------------------------------------------- mask 단위 테스트
print("\n=== 0. _isolate_action_tokens 단위 테스트 ===")
# prefix_len=3, q_len=4, action 쪽은 causal (embed_suffix 가 att_masks=[1]*chunk_size 를 내보냄)
prefix = torch.ones(1, 4, 3, dtype=torch.bool)
causal = torch.tril(torch.ones(4, 4, dtype=torch.bool))[None]
mask = torch.cat([prefix, causal], dim=2)
out = _isolate_action_tokens(mask)
check("prefix column 은 그대로 (conditioning 유지)", bool((out[:, :, :3] == prefix).all()))
check("action column 은 대각만 (시간축 mixing 제거)",
      bool((out[:, :, 3:] == torch.eye(4, dtype=torch.bool)[None]).all()))
check("모든 row 가 최소 1개는 볼 수 있음 (softmax 가 전부 -inf 가 되지 않음)", bool(out.any(dim=2).all()))
check("입력 mask 를 in-place 로 바꾸지 않음", bool((mask[:, :, 3:] == causal).all()))

# ---------------------------------------------------------------- policy 로드
print("\n=== 1. policy 로드 (CPU) ===")
cfg = PreTrainedConfig.from_pretrained("lerobot/smolvla_base")
cfg.pretrained_path = "lerobot/smolvla_base"
cfg.device = "cpu"
cfg.num_steps = 2  # 테스트 속도용 (기본 10)
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base", config=cfg)
policy.to("cpu").eval()
model = policy.model
print(f"  policy={type(policy).__name__}  model={type(model).__name__}")

# ---------------------------------------------------------------- 체크리스트 A/B/D/F
print("\n=== 2. 체크리스트 확인 ===")
check("A: policy.model 은 VLAFlowMatching", isinstance(model, VLAFlowMatching))
check("A: sample_actions/denoise_step 존재",
      hasattr(model, "sample_actions") and hasattr(model, "denoise_step"))

sig = inspect.signature(VLAFlowMatching.denoise_step)
print(f"  B: denoise_step{sig}")
check("B: 시그니처 = (prefix_pad_masks, past_key_values, x_t, timestep)",
      list(sig.parameters)[1:] == ["prefix_pad_masks", "past_key_values", "x_t", "timestep"])

vlm = model.vlm_with_expert
print(f"  D: attention_mode={vlm.attention_mode!r} self_attn_every_n_layers={vlm.self_attn_every_n_layers} "
      f"num_vlm_layers={vlm.num_vlm_layers} num_expert_layers={vlm.num_expert_layers}")
eligible = _acg_eligible_layers(model)
default_layers = _default_acg_layers(eligible)
print(f"  D: ACG eligible layers = {eligible}")
print(f"  D: ACG default target  = {default_layers}")
check("D: expert self_attn.forward 는 호출되지 않는다 (=attn.forward 패치는 무의미)",
      "self_attn(" not in inspect.getsource(type(vlm).forward))

print(f"  F: chunk_size={model.config.chunk_size} n_action_steps={model.config.n_action_steps} "
      f"num_steps={model.config.num_steps} rtc_config={model.config.rtc_config}")

# ---------------------------------------------------------------- dummy 입력
print("\n=== 3. dummy 입력 생성 ===")
B = 1
H, W = cfg.resize_imgs_with_padding or (512, 512)
images = [torch.randn(B, 3, H, W) * 0.1]
img_masks = [torch.ones(B, dtype=torch.bool)]
lang_tokens = torch.randint(10, 100, (B, 8))
lang_masks = torch.ones(B, 8, dtype=torch.bool)
state = torch.randn(B, cfg.max_state_dim) * 0.1
noise = torch.randn(B, cfg.chunk_size, cfg.max_action_dim)  # 고정 noise -> 결정론적
print(f"  images={list(images[0].shape)} state={list(state.shape)} noise={list(noise.shape)}")


def run():
    with torch.no_grad():
        return model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise.clone())


print("\n=== 4. baseline (패치 없음) ===")
base = run()
print(f"  sample_actions -> {list(base.shape)}")
check("baseline shape == [B, chunk_size, max_action_dim]",
      list(base.shape) == [B, cfg.chunk_size, cfg.max_action_dim])
base2 = run()
check("고정 noise 에서 결정론적", torch.allclose(base, base2, atol=1e-6))


# bound method 는 접근할 때마다 새 객체가 나오므로 `is` 로 비교하면 안 된다.
# "인스턴스 __dict__ 를 오염시키지 않고 클래스 메서드로 되돌아갔는가" 를 본다.
def is_clean():
    return ("denoise_step" not in vars(model)
            and "forward_attn_layer" not in vars(vlm)
            and model.denoise_step.__func__ is VLAFlowMatching.denoise_step
            and vlm.forward_attn_layer.__func__ is type(vlm).forward_attn_layer)


check("설치 전: 인스턴스 __dict__ 가 깨끗함", is_clean())

# ---------------------------------------------------------------- ACG install
print("\n=== 5. ACG install ===")
handle = install_smoothing(policy, SmoothingConfig(acg=True, acg_scale=2.0))
check("denoise_step 이 교체됨", "denoise_step" in vars(model))
check("forward_attn_layer 가 교체됨", "forward_attn_layer" in vars(vlm))

acg = run()
check("ACG shape 보존", list(acg.shape) == list(base.shape), f"{list(acg.shape)}")
delta = (acg - base).abs().max().item()
check("ACG 가 실제로 출력을 바꾼다 (no-op 아님)", delta > 1e-5, f"max|Δ|={delta:.6f}")
check("ACG 출력이 유한", bool(torch.isfinite(acg).all()))

# ---------------------------------------------------------------- ACG uninstall
print("\n=== 6. uninstall 왕복 ===")
uninstall_smoothing(handle)
check("uninstall 후 인스턴스 __dict__ 오염 없이 클래스 메서드로 복귀", is_clean())
after = run()
check("uninstall 후 baseline 과 비트 동일", torch.equal(after, base),
      f"max|Δ|={(after - base).abs().max().item():.2e}")

# ---------------------------------------------------------------- scale=1.0 sanity
print("\n=== 7. acg_scale=1.0 은 no-op ===")
h1 = install_smoothing(policy, SmoothingConfig(acg=True, acg_scale=1.0))
one = run()
check("scale=1.0 -> baseline 과 동일", torch.equal(one, base))
uninstall_smoothing(h1)

# ---------------------------------------------------------------- 잘못된 layer
print("\n=== 8. ACG 대상 아닌 layer 는 거부 ===")
bad = next(i for i in range(vlm.num_vlm_layers) if i not in eligible)
try:
    install_smoothing(policy, SmoothingConfig(acg=True, acg_layers=[bad]))
    check(f"layer {bad} 거부", False, "에러가 안 났음")
except ValueError as e:
    check(f"layer {bad}(cross-attn 전용) 거부", True, str(e)[:70] + "...")

# ---------------------------------------------------------------- RTC
print("\n=== 9. RTC 튜닝 ===")
check("sync 모드처럼 rtc_config=None 이면 명확한 에러", policy.config.rtc_config is None)
try:
    install_smoothing(policy, SmoothingConfig(rtc=True))
    check("rtc_config=None 에서 에러", False, "에러가 안 났음")
except RuntimeError as e:
    check("rtc_config=None -> RuntimeError", "asynchronous=True" in str(e))

# 설치 도중 실패해도 패치가 남으면 안 된다 (handle 을 못 받으므로 영구 누수)
print("\n=== 9b. install 원자성: acg 성공 + rtc 실패 ===")
try:
    install_smoothing(policy, SmoothingConfig(acg=True, rtc=True))  # rtc_config=None -> RTC 단계에서 실패
    check("실패해야 함", False)
except RuntimeError:
    check("RTC 실패 시 ACG 패치가 누수되지 않음", is_clean())
    check("실패 후 출력이 baseline 과 동일", torch.equal(run(), base))

# async rollout 이 하는 일 재현 (lerobot/rollout/context.py:202-205)
policy.config.rtc_config = RTCConfig()
policy.init_rtc_processor()
shared = policy.config.rtc_config
before = (shared.prefix_attention_schedule, shared.max_guidance_weight, shared.execution_horizon)
print(f"  lerobot 기본값: schedule={before[0].value} max_w={before[1]} execution_horizon={before[2]}")

h2 = install_smoothing(policy, SmoothingConfig(rtc=True, rtc_schedule="exp",
                                               rtc_max_guidance_weight=5.0, rtc_execution_horizon=16))
check("schedule 반영", shared.prefix_attention_schedule.value == "EXP")
check("max_guidance_weight 반영", shared.max_guidance_weight == 5.0)
check("execution_horizon 반영", shared.execution_horizon == 16)
check("RTCProcessor 가 같은 객체를 공유 (in-place 가 전파됨)",
      policy.rtc_processor.rtc_config is shared)

uninstall_smoothing(h2)
check("RTC 값 원복",
      (shared.prefix_attention_schedule, shared.max_guidance_weight, shared.execution_horizon) == before)

# ---------------------------------------------------------------- ACG + RTC 동시
print("\n=== 10. acg+rtc 동시 (RTC 켜진 상태에서 ACG) ===")
h3 = install_smoothing(policy, SmoothingConfig(acg=True, acg_scale=2.0, rtc=True))
both = run()
check("acg+rtc shape 보존", list(both.shape) == list(base.shape))
check("acg+rtc 출력이 유한", bool(torch.isfinite(both).all()))
uninstall_smoothing(h3)
check("전부 원복", is_clean())

# ---------------------------------------------------------------- 결과
print(f"\n{'=' * 60}\nPASS {len(PASS)} / FAIL {len(FAIL)}")
if FAIL:
    print("실패:", FAIL)
sys.exit(1 if FAIL else 0)
