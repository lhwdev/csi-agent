# 수건 개기

## 학습 현황

현재 ACT 모델의 경우 epoch=5까지 학습 완료했습니다.
0/2단계는 DAgger로 fine-tuning 진행 중입니다.

## 장비 설정

다음 위치에 so101 모터 및 cv2 호환 카메라 파일로 링크를 설정해야 합니다.

- `/dev/lerobot/leader_1`, `/dev/lerobot/leader_2`: SO101 bimanual leader의 왼팔/오른팔
- `/dev/lerobot/follower_1`, `/dev/lerobot/follower_2`: SO101 bimanual follower의 왼팔/오른팔

## Quickstart 스크립트

- [rollout_act_step.py](./scripts/rollout_act_step.py): ACT 각 단계 실행 스크립트. `i`를 단계(0~3)으로 수정
  후 실행합니다.
- [rollout_smolvla_step.py](./scripts/rollout_smolvla_step.py): SmolVLA 각 단계 실행 스크립트. `i` 수정 후
  실행합니다.
- [train_all.py](./scripts/train_all.py): ACT 기반 0~3단계를 특정 epoch로 학습합니다. 이렇게 할 경우 DAgger로
  수집한 데이터셋은 제외하고 학습됩니다.
