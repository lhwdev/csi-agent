"""
test_stall_detection.py
========================

Unit tests for ClassifierStateTracker stall detection policies in `rollout.py`:
- Policy 1: Sliding Window Joint Displacement
- Policy 2: Leaky Integrator Accumulator
- Policy 3: Command-vs-Observation Tracking Error (Large Threshold)
- 2-Tier Stall Recovery & Signal Return
"""

import sys
import time
import logging
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rollout import (
    ClassifierStateTracker,
    STALL_TIER1_SIGNAL,
    TRACKING_ERROR_THRESHOLD,
    WINDOW_DISPLACEMENT_THRESHOLD,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

PASS_COUNT = 0
FAIL_COUNT = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS_COUNT, FAIL_COUNT
    if cond:
        PASS_COUNT += 1
        print(f" [PASS] {name} {detail}")
    else:
        FAIL_COUNT += 1
        print(f" [FAIL] {name} {detail}")


def mock_classify_frame(obs):
    # Dummy mock returning (probs_dict, pred_class, confidence)
    probs_dict = {0: 0.1, 1: 0.8, 2: 0.1}
    return probs_dict, 1, 0.8


def test_slow_movement_no_false_positive():
    """Test Policy 1: Slow steady movement (0.008 rad/s) should NOT trigger false positive stall."""
    tracker = ClassifierStateTracker(classifier=True, image_processor=True, stall_timeout=5.0)
    tracker.classify_frame = mock_classify_frame

    dt = 1.0 / 30.0
    joint_angle = 0.0

    # Simulate 6 seconds (180 frames) of slow smooth joint movement (0.008 rad per second -> ~0.00026 rad per frame)
    for _ in range(180):
        joint_angle += 0.008 * dt
        obs = {"joint_1.position": joint_angle, "observation.images.top": torch.zeros((3, 224, 224))}
        action = {"joint_1.position": joint_angle + 0.01}  # tracking error < 0.12
        tracker.update(obs, current_step=1, dt=dt, action=action)

    check(
        "Policy 1: Slow Movement No False Positive",
        tracker.stall_accumulated_time == 0.0,
        f"Accumulated stall time is {tracker.stall_accumulated_time:.2f}s (expected 0.0s)",
    )


def test_brief_pause_and_sustained_rest_stall():
    """Test Policy: Brief stationary posture (2.0s < 5.0s timeout) does not trigger stall, but sustained rest (>= 5.0s) triggers stall timeout."""
    tracker = ClassifierStateTracker(classifier=True, image_processor=True, stall_timeout=5.0)
    tracker.classify_frame = mock_classify_frame

    dt = 1.0 / 30.0
    joint_angle = 1.50

    # 1) Brief rest for 2 seconds (60 frames) -> should accumulate 2.0s but NOT trigger stall recovery
    res = None
    for _ in range(60):
        obs = {"joint_1.position": joint_angle, "observation.images.top": torch.zeros((3, 224, 224))}
        action = {"joint_1.position": joint_angle + 0.03}
        out = tracker.update(obs, current_step=1, dt=dt, action=action)
        if out is not None:
            res = out

    check(
        "Policy: Brief Rest No Stall Trigger",
        res is None and tracker.stall_count == 0,
        f"Brief rest output: res={res}, stall_count={tracker.stall_count}",
    )

    # 2) Sustained rest reaching total >5.0s (160 frames total) -> triggers stall recovery
    res_stall = None
    for _ in range(100):
        obs = {"joint_1.position": joint_angle, "observation.images.top": torch.zeros((3, 224, 224))}
        action = {"joint_1.position": joint_angle + 0.03}
        out = tracker.update(obs, current_step=1, dt=dt, action=action)
        if out is not None:
            res_stall = out

    check(
        "Policy: Sustained Stationary Rest Triggers Stall",
        res_stall == STALL_TIER1_SIGNAL,
        f"Sustained rest output: res={res_stall} (expected {STALL_TIER1_SIGNAL})",
    )


def test_true_physical_jam_2tier_recovery():
    """Test Policy 2 & 3: Stationary robot with large tracking error (>= 0.12) triggers 2-tier recovery."""
    tracker = ClassifierStateTracker(classifier=True, image_processor=True, stall_timeout=2.0)
    tracker.classify_frame = mock_classify_frame

    dt = 1.0 / 30.0
    joint_angle = 1.0

    # 1st Stall (Tier 1 Recovery)
    res_tier1 = None
    for _ in range(100):  # ~3.3 seconds > stall_timeout (2.0s)
        obs = {"joint_1.position": joint_angle, "observation.images.top": torch.zeros((3, 224, 224))}
        action = {"joint_1.position": joint_angle + 0.25}  # tracking error 0.25 >= 0.12
        res = tracker.update(obs, current_step=1, dt=dt, action=action)
        if res is not None:
            res_tier1 = res
            break

    check(
        "Tier 1 Recovery Triggered",
        res_tier1 == STALL_TIER1_SIGNAL,
        f"Returned signal {res_tier1} (expected {STALL_TIER1_SIGNAL})",
    )
    check(
        "Tier 1 Stall Count",
        tracker.stall_count == 1,
        f"Stall count is {tracker.stall_count} (expected 1)",
    )

    # 2nd Stall in same step (Tier 2 Recovery -> Step 0 fallback)
    res_tier2 = None
    for _ in range(100):  # ~3.3 seconds > stall_timeout (2.0s)
        obs = {"joint_1.position": joint_angle, "observation.images.top": torch.zeros((3, 224, 224))}
        action = {"joint_1.position": joint_angle + 0.25}
        res = tracker.update(obs, current_step=1, dt=dt, action=action)
        if res is not None:
            res_tier2 = res
            break

    check(
        "Tier 2 Recovery Fallback to Step0",
        res_tier2 == 1,
        f"Returned step {res_tier2} (expected 1)",
    )
    check(
        "Tier 2 Reset Stall Count",
        tracker.stall_count == 0,
        f"Stall count is {tracker.stall_count} (expected 0)",
    )


def test_leaky_decay():
    """Test Policy 2: Leaky integrator decay when motion resumes after brief blockage."""
    tracker = ClassifierStateTracker(classifier=True, image_processor=True, stall_timeout=5.0)
    tracker.classify_frame = mock_classify_frame

    dt = 1.0 / 30.0
    joint_angle = 1.0

    # 1. Blocked for 1.5 seconds (accumulates 1.5s stall time)
    for _ in range(45):
        obs = {"joint_1.position": joint_angle, "observation.images.top": torch.zeros((3, 224, 224))}
        action = {"joint_1.position": joint_angle + 0.25}
        tracker.update(obs, current_step=1, dt=dt, action=action)

    accumulated_before = tracker.stall_accumulated_time
    check(
        "Accumulator Integration",
        accumulated_before > 1.0,
        f"Accumulated stall time is {accumulated_before:.2f}s",
    )

    # 2. Resumes moving for 1.0 second
    for _ in range(30):
        joint_angle += 0.05 * dt
        obs = {"joint_1.position": joint_angle, "observation.images.top": torch.zeros((3, 224, 224))}
        action = {"joint_1.position": joint_angle + 0.01}
        tracker.update(obs, current_step=1, dt=dt, action=action)

    accumulated_after = tracker.stall_accumulated_time
    check(
        "Leaky Decay Active",
        accumulated_after < accumulated_before,
        f"Decayed stall time from {accumulated_before:.2f}s to {accumulated_after:.2f}s",
    )


if __name__ == "__main__":
    print("=== Running Stall Detection Tests ===")
    test_slow_movement_no_false_positive()
    test_brief_pause_and_sustained_rest_stall()
    test_true_physical_jam_2tier_recovery()
    test_leaky_decay()

    print(f"\nResults: {PASS_COUNT} passed, {FAIL_COUNT} failed.")
    if FAIL_COUNT > 0:
        sys.exit(1)
    else:
        sys.exit(0)
