# Action Smoothing (ACG / RTC) — 구현 기록 & 사용 가이드

> 목적: SmolVLA rollout 시 **action jitter(동작 흔들림)** 저감.
> 상태: **서버(실제 lerobot 소스)에서 확정 완료.** 테스트 32/32 통과.
> 이전 버전(Windows 세션)의 "가정" 중 **2개가 틀렸고 둘 다 수정됨** — 아래 3장 참조.

---

## 1. 배경 / 요구사항

- policy 는 **smolvla** (`HyeonseokE/smolvla_towel_fold01_step*`), flow-matching VLA.
- 증상: inference 시 action jitter 가 커서 동작이 크게 흔들림.
- 요청: ACG / RTC 로 흔들림을 줄이는 코드를, **on/off 쉽게**.

참조 원본(리포에 클론되어 있음):
- RTC: `real-time-chunking-kinetix/src/model.py` → `FlowPolicy.realtime_action`, `get_prefix_weights`
- ACG: `ACG/libs/robomimic/robomimic/algo/guidance/acg.py` → `ACGAttnProcessor2_0`, `FlowmatchingActionHead_ACG.get_action`

---

## 2. ★ 결론 먼저: RTC 는 이미 켜져 있다. 새로 얻는 건 ACG 다.

**lerobot 은 RTC 를 자체 구현해 두었다.**

| | 위치 |
|---|---|
| guidance 본체 | `lerobot/policies/rtc/modeling_rtc.py` → `RTCProcessor.denoise_step` |
| async 추론 엔진 | `lerobot/rollout/inference/rtc.py` → `RTCInferenceEngine` |
| queue/leftover | `lerobot/policies/rtc/action_queue.py` → `ActionQueue` |
| SmolVLA 연결 | `modeling_smolvla.py:860-872` (`sample_actions` 안에서 `rtc_processor.denoise_step` 호출) |

그리고 `rollout(..., asynchronous=True)` 는 `RTCInferenceConfig` 를 만들고
`lerobot/rollout/context.py:202-205` 가 이렇게 주입한다:

```python
if is_rtc:
    policy.config.rtc_config = cfg.inference.rtc     # RTCConfig(enabled=True, ...)
    policy.init_rtc_processor()
```

→ **기존 노트북 호출(`asynchronous=True`)에서 RTC 는 이미 동작 중이었다.**
→ 지금 보이는 jitter 는 "RTC 를 켠 상태"의 증상이다.
→ 손으로 짠 RTC 를 `denoise_step` 에 또 얹으면 **같은 guidance 가 이중 적용**된다.

그래서 자체 RTC 구현은 **제거**하고, 내장 RTC 의 knob 만 조정하는 방식으로 바꿨다.
새로 추가되는 기법은 **ACG 뿐**이다.

**RTC 는 async 경로에서만 쓸 수 있다.** sync 엔진은 `select_action` 을 쓰는데
`select_action` 은 `assert not self._rtc_enabled()` 로 RTC 를 막는다 (`modeling_smolvla.py:335`).

---

## 3. 확정 결과 (이전 가정 → 소스 확인)

| | 이전 가정 | 결과 |
|---|---|---|
| **A** | `policy.model`==`VLAFlowMatching`, `sample_actions`/`denoise_step` | ✅ **맞음** (`modeling_smolvla.py:248`, `:812`, `:883`) |
| **B** | `denoise_step` 시그니처 | ✅ **맞음**: `(prefix_pad_masks, past_key_values, x_t, timestep)`. 전부 **kwargs 로** 호출됨(`:852-858`) → shape 추측(`_extract_xt_and_t`) 불필요, 제거 |
| **C** | `x_t=t*noise+(1-t)*a`, t:1→0, `a_hat=x_t-t*v_t`, `v_t-gw*w*(y-a_hat)` | ✅ **부호까지 전부 맞음** — 아래 참조 |
| **D** | `lm_expert.layers[*].self_attn.forward` 패치 | ❌ **틀림 — 조용한 no-op.** 아래 참조 |
| **E** | 후보 경로 + `__dict__` 재귀 탐색 | ✅ **확정: `ctx.policy.policy`.** 재귀 탐색 제거 |
| **F** | RTC 에 `n_action_steps < chunk_size` 필요 | ❌ **틀림.** 아래 참조 |

### C. flow 시간 컨벤션 / RTC 부호 — 가정이 전부 맞았다

`modeling_smolvla.py:785-786` (forward): `x_t = t*noise + (1-t)*actions`, `u_t = noise - actions`
`modeling_smolvla.py:845-876` (sample_actions): `dt = -1.0/num_steps`, `time = 1.0 + step*dt`, `x_t = x_t + dt*v_t` → **t: 1→0, dt<0** ✔

내장 RTC (`modeling_rtc.py:216-229`) 가 문자 그대로 같은 식을 쓴다:
```python
x1_t = x_t - time * v_t                       # a_hat = x_t - t*v_t          <- 가정과 동일
err  = (prev_chunk_left_over - x1_t) * weights
correction = torch.autograd.grad(x1_t, x_t, err)[0]
result = v_t - guidance_weight * correction   # v_t - gw*w*(y - a_hat)       <- 가정과 동일
```
**identity-Jacobian 근사까지 동일하다.** lerobot 도 `x_t.requires_grad_(True)` 를 `v_t` 계산
**뒤에** 부르기 때문에 `∂x1_t/∂x_t = I` 가 되어 `correction == err` 이다.
guidance weight 스케줄(`tau=1-time`, `c*inv_r2`, `min(., max_w)`)도 동일.

→ 즉 자체 RTC 는 **내장 RTC 의 정확한 재구현**이었다. 그래서 지울 수 있었다.

### D. ACG — `self_attn.forward` 패치는 죽은 코드였다

`SmolVLMWithExpertModel.forward` 는 **`layer.self_attn.forward()` 를 한 번도 호출하지 않는다**
(`smolvlm_with_expert.py:209-510`). `self_attn.q_proj/k_proj/v_proj` 서브모듈을 직접 부르고,
자체 `eager_attention_forward` 를 돌린 뒤, 바깥 루프에서 `self_attn.o_proj` 를 적용한다.

→ `attn.forward` 를 갈아끼워도 **아무도 호출하지 않는다.**
→ `v_t + (scale-1)*(v_t - v_t) = v_t` : **2배 연산을 쓰고 효과 0.** 조용히.

**어디를 패치해야 하나 (denoise 중, `fill_kv_cache=False` 기준):**

| layer | 분기 | action↔action mixing | ACG 대상? |
|---|---|---|---|
| 짝수 (`layer_idx % 2 == 0`) | `forward_attn_layer` | **있음** (action query 가 `[prefix_cache_kv ; action_kv]` 를 attend) | ✅ |
| 홀수 | `forward_cross_attn_layer` | **없음** (prefix KV 로의 순수 cross-attn) | ❌ |

분기 조건은 `smolvlm_with_expert.py:437-467`:
`fill_kv_cache or "cross" not in attention_mode or (self_attn_every_n_layers > 0 and layer_idx % self_attn_every_n_layers == 0)`

towel_fold 체크포인트: `attention_mode="cross_attn"`, `self_attn_every_n_layers=2`,
`num_vlm_layers=16`, `num_expert_layers=-1`(→16) → **eligible = [0,2,4,6,8,10,12,14]** (8개).

이건 참조 구현과도 일치한다. ACG 는 GR00T DiT 의 `attn1`(state+action self-attn)만
value-only 로 바꾸고 vision-language 로 가는 `attn2`(cross-attn)는 건드리지 않는다
(`acg.py:187-197`). → **conditioning 은 유지하고 시간 일관성만 깬다.**

**포팅 방식:** `vlm_with_expert.forward_attn_layer` 를 감싸서, 대상 layer 에서만
attention mask 를 고쳐 각 action token 이 **prefix 전체 + 자기 자신**만 보게 한다.
```python
prefix_len = mask.shape[2] - mask.shape[1]      # mask: [B, suffix_len, prefix_len+suffix_len]
mask[:, :, prefix_len:] &= torch.eye(q_len)     # action column 은 대각만
```
action token 쪽 mask 는 원래 **causal** 이므로(`embed_suffix` 가 `att_masks=[1]*chunk_size` 를
내보내고, `make_att_2d_masks` docstring 상 `[[1 1 1 ...]]` = pure causal) 대각은 원래 True 다
→ softmax row 가 전부 -inf 가 되지 않는다.

**기본 대상 layer = `[6, 8, 10]`** — 참조 구현 `skip_blocks=[7,9,11]`/16 blocks 와 **같은 상대 위치**
(`int(f*8) for f in (7/16, 9/16, 11/16)` → eligible[3,4,5]).

### F. RTC 겹침 조건 — `n_action_steps` 는 무관하다

- towel_fold 체크포인트 실측: **`chunk_size=50`, `n_action_steps=50`, `num_steps=10`**.
- 하지만 **RTC 경로에서 `n_action_steps` 는 쓰이지 않는다.** RTC 는 `predict_action_chunk` 만
  쓰고, `n_action_steps` 는 `select_action` 의 queue 길이 전용이다 (`modeling_smolvla.py:348`).
- 실제 겹침 = `ActionQueue` 의 **leftover**(아직 실행 안 된 action). 매 inference 마다
  queue 를 새 chunk 로 교체하고, 남은 걸 다음 chunk 의 `prev_chunk_left_over` 로 넘긴다
  (`rollout/inference/rtc.py:274, 307-309, 324`).
- **RTC 를 쓰려면?** `asynchronous=True` 면 끝. 이미 켜져 있다.
- **핵심 knob 은 `RTCConfig.execution_horizon`** (기본 10). leftover 를 이 길이로 자르고
  (`rtc.py:302-305`), `get_prefix_weights(inference_delay, execution_horizon, chunk_size)` 로
  chunk_size(50) 중 앞 `execution_horizon` 구간에만 prefix guidance 를 건다.
  늘리면 이전 chunk 에 더 길게 정합 → 부드럽지만 반응성 ↓.

---

## 4. 최종 구조

### `clothing/action_smoothing.py`
lerobot 원본을 수정하지 않고 SmolVLA policy 에 얹는 모듈.

- **ACG** (`arXiv:2510.22201`) — chunk *내부* jitter 저감. **이 모듈이 새로 추가하는 기법.**
  - denoise step 마다 모델 2회 forward: 정상 `v_t` vs 대상 layer 의 action↔action attention 을
    끊은 `v_perturb` → `v_t <- v_t + (scale-1)*(v_t - v_perturb)`
  - 패치 지점 2곳: `model.denoise_step`(guidance + perturb 플래그),
    `vlm_with_expert.forward_attn_layer`(perturb 중 mask 수술)
- **RTC** (`arXiv:2506.07339`) — **lerobot 내장.** 이 모듈은 `policy.config.rtc_config` 를
  in-place 로 조정만 한다.
  - in-place 인 이유: RTCConfig 인스턴스 **하나**를 4곳이 공유한다
    (`ctx.runtime.cfg.inference.rtc` / `policy.config.rtc_config` /
    `RTCInferenceEngine._rtc_config`→`ActionQueue.cfg` / `RTCProcessor.rtc_config`).
    새 객체를 대입하면 공유가 끊겨 일부만 반영된다.

공개 API: `SmoothingConfig`, `install_smoothing(policy, cfg) -> handle`, `uninstall_smoothing(handle)`.
`install_smoothing` 은 **원자적**이다(중간 실패 시 이미 건 패치를 되돌린다).

### `clothing/rollout_smoothing.py`
- 기존 `rollout.py` 의 `create_rollout_context`, `MultiStepStrategy` 를 **재사용만** 한다.
- `rollout_smooth(..., smoothing=None|"acg"|"rtc"|"acg+rtc"|SmoothingConfig)` 진입점.
- policy 위치는 **`ctx.policy.policy`** 로 고정.
- **`strategy.setup(ctx)` 이전에** 설치한다. setup → `_init_engine` → `engine.start()` 가
  RTC 추론 스레드를 띄우므로(`strategies/core.py:55-70`, `inference/rtc.py:180-194`),
  그 뒤에 패치하면 스레드가 이미 추론 중일 수 있다.
  policy 는 `create_rollout_context` 시점에 이미 로드되어 있다(`context.py:176-209`).
- teardown(스레드 정지) → uninstall 순서로 정리.

### `clothing/test_action_smoothing.py`
`lerobot/smolvla_base` 를 CPU 로 로드해 검증. 로봇/GPU 불필요.
```bash
cd clothing && HF_HUB_OFFLINE=1 python test_action_smoothing.py     # 32/32 PASS
```
가장 중요한 검사: **"ACG 가 실제로 출력을 바꾼다 (no-op 아님)"** (`max|Δ|=0.378`).
ACG 를 잘못된 지점에 걸면 여기서 `max|Δ|=0` 이 나온다. 초기 구현이 실제로 그랬다.

---

## 5. 사용법

노트북 `clothing.ipynb` cell 35 에서 두 줄만 바꾼다:

```python
from rollout_smoothing import rollout_smooth        # 기존: from rollout import rollout

rollout_smooth(robot, policies, rollout_tasks, 30.0, asynchronous=True, compile=False,
               rename_map=pretrained_rename_map, smoothing="acg")
```

| `smoothing` | 동작 |
|---|---|
| `None` / `"off"` | 미적용 (기존 `rollout()` 과 완전히 동일) |
| `"acg"` | ACG on. RTC 는 lerobot 기본값 그대로 (async 면 이미 켜져 있음) |
| `"rtc"` | 내장 RTC knob 만 조정 (`asynchronous=True` 필요) |
| `"acg+rtc"` | 둘 다 |
| `SmoothingConfig(...)` | 세부 지정 — 예: `SmoothingConfig(acg=True, acg_scale=2.5)` |

기존 `rollout()` 과 함수명이 다르므로 서로 충돌하지 않는다.

---

## 6. 검증 / 튜닝

1. **`smoothing="acg"` 부터.** `acg_scale` 을 **1.0(무효과) → 2.0 → 3.0** 으로 올리며 육안 확인.
2. **추론 지연 주의.** ACG 는 denoise step 마다 모델을 2번 돌리므로 추론 시간이 대략 2배다.
   async 모드에선 그만큼 `inference_delay` 가 커져 RTC 가 이전 chunk 에 더 강하게 정합된다.
   `RTC inference latency=...` 로그를 같이 볼 것.
3. **RTC 튜닝은 `rtc_execution_horizon`.** lerobot 기본은 `LINEAR`/`max_guidance_weight=10.0`/
   `execution_horizon=10`. `smoothing="rtc"` 는 `exp`/`5.0`/`10` 을 적용한다.
4. **jitter 정량화**: 실행된 action 시퀀스의 1차 차분 크기(`mean|a_t - a_{t-1}|`)를
   off/acg 로 비교 로깅.

### 기본값
- ACG: `acg_scale=2.0` (1.5~3.0), `acg_layers=None` (→ 자동 `[6, 8, 10]`)
- RTC: `rtc_schedule="exp"`, `rtc_max_guidance_weight=5.0`, `rtc_execution_horizon=10`

### 주의
- **`compile=True` + ACG**: `context.py:181-182` 가 `policy_config.compile_model = cfg.use_torch_compile`
  로 넘기고 `VLAFlowMatching.__init__` 이 `sample_actions` 를 `torch.compile` 한다. 그 뒤
  `denoise_step` 을 갈아끼우면 dynamo 재컴파일/graph break 가 난다. **먼저 `compile=False` 로 검증할 것.**
  (`rollout_smooth` 가 경고를 낸다.)
- **sync 모드(`asynchronous=False`)**: ACG 는 동작하지만 RTC 는 불가능
  (`smoothing="rtc"` 요청 시 명확한 에러). chunk 경계 불연속이 문제라면 async 를 쓸 것.

---

## 7. 관련 파일

- `clothing/action_smoothing.py` — 핵심 로직 (ACG monkey-patch + 내장 RTC knob 조정)
- `clothing/rollout_smoothing.py` — `rollout_smooth()` 진입점
- `clothing/test_action_smoothing.py` — 검증 스크립트 (32/32)
- `clothing/rollout.py` — **원본 유지, 수정 금지**
- `clothing/clothing.ipynb` — rollout 호출부 (5장 참조)
- lerobot 원본 — **수정 금지.** 이 작업에서 전혀 건드리지 않았다.
- 참조: `real-time-chunking-kinetix/src/model.py`, `ACG/libs/robomimic/robomimic/algo/guidance/acg.py`
