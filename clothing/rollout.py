import asyncio
from collections import deque
import json
import logging
import math
import os
import random
import threading
import time
from typing import Optional, Tuple, Dict, Any
import traceback

import cv2
import numpy as np
import torch
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

logger = logging.getLogger(__name__)

# Global handle for background or interrupt control
_active_rollout_shutdown_event: Optional[threading.Event] = None
_active_rollout_task: Optional[asyncio.Task] = None


def stop_rollout():
    """Signals the active rollout execution to stop immediately."""
    global _active_rollout_shutdown_event, _active_rollout_task
    if _active_rollout_shutdown_event is not None:
        _active_rollout_shutdown_event.set()
        logger.info("Stop signal sent to active rollout.")
        print("Stop signal sent to active rollout.")
    else:
        logger.info("No active rollout running.")
        print("No active rollout running.")

    if _active_rollout_task is not None and not _active_rollout_task.done():
        _active_rollout_task.cancel()
        logger.info("Active rollout task cancelled.")


# Global configuration
IDLE_ADDITIONAL_DELAY = 10.0        # Added delay (seconds) for transition between idle and step0
MIN_TRANSITION_CONFIDENCE = 0.6     # Minimum confidence required to trigger transition
CONFIDENCE_HIGH = 1.0               # Reference high confidence
DELAY_AT_CONF_HIGH = 1.0            # Delay (seconds) at high confidence
CONFIDENCE_LOW = 0.6                # Reference low confidence
DELAY_AT_CONF_LOW = 5.0             # Delay (seconds) at low confidence
STALL_TIMEOUT_SECONDS = 10.0        # Required motionless stall duration (seconds) before allowing regression fallback
FORWARD_CONFIDENCE_THRESHOLD = 0.20 # Minimum candidate confidence for target forward step

STALL_TIER1_SIGNAL = -999              # Signal indicating Tier 1 stall recovery (position reset + seed perturbation)
WINDOW_DISPLACEMENT_THRESHOLD = 0.003  # Minimum joint displacement (rad) over 0.5s window to count as moving
TRACKING_ERROR_THRESHOLD = 12.0        # Minimum target vs obs tracking error (rad) required to flag a physical stall
STALL_DECAY_RATE = 4.0                 # Leaky decay rate multiplier when robot is moving/holding posture


class ClassifierStateTracker:
    """Tracks classification outputs across frames, applying probability-vector smoothing, confidence delays, motion multipliers, and stall/timeout regression checks."""
    
    def __init__(
        self,
        classifier=None,
        image_processor=None,
        device=None,
        rename_map=None,
        idle_posture=None,
        window_size: int = 15,
        min_confidence: float = MIN_TRANSITION_CONFIDENCE,
        stall_timeout: float = STALL_TIMEOUT_SECONDS,
        tracking_error_threshold: float = TRACKING_ERROR_THRESHOLD,
        window_displacement_threshold: float = WINDOW_DISPLACEMENT_THRESHOLD,
        decay_rate: float = STALL_DECAY_RATE,
    ):
        self.classifier = classifier
        self.image_processor = image_processor
        self.device = device
        self.rename_map = rename_map
        self.idle_posture = idle_posture
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.stall_timeout = stall_timeout
        self.tracking_error_threshold = tracking_error_threshold
        self.window_displacement_threshold = window_displacement_threshold
        self.decay_rate = decay_rate

        self.prob_history: list[dict[int, float]] = []
        self.joint_history: deque = deque(maxlen=15)  # ~0.5s sliding window at 30 FPS
        self.frame_count = 0
        self.transition_target_class: Optional[int] = None
        self.transition_accumulated_time = 0.0
        self.stall_accumulated_time = 0.0
        self.stall_count = 0

        self.prev_joints: Optional[dict[str, float]] = None
        self.prev_time: Optional[float] = None

    @classmethod
    def from_path(cls, classifier_path: Optional[str], rename_map=None, idle_posture=None, window_size: int = 15):
        """Factory method to load classifier and image processor from disk path."""
        if not classifier_path:
            return None
        logger.info(f"Loading state classifier from {classifier_path}...")
        device = "cuda" if torch.cuda.is_available() else ("xpu" if torch.xpu.is_available() else "cpu")
        try:
            from transformers import AutoModelForImageClassification, AutoImageProcessor
            classifier = AutoModelForImageClassification.from_pretrained(classifier_path).to(device)
            classifier.eval()
            processor = AutoImageProcessor.from_pretrained(classifier_path)
            logger.info("Classifier loaded successfully.")
            return cls(classifier, processor, device=device, rename_map=rename_map, idle_posture=idle_posture, window_size=window_size)
        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")
            traceback.print_exc()
            return None

    def reload_classifier(self, classifier_path: str) -> bool:
        """Reloads classifier model and image processor from a new checkpoint directory."""
        if not classifier_path:
            return False
        logger.info(f"Reloading state classifier from {classifier_path}...")
        device = getattr(self, "device", None) or ("cuda" if torch.cuda.is_available() else ("xpu" if torch.xpu.is_available() else "cpu"))
        try:
            from transformers import AutoModelForImageClassification, AutoImageProcessor
            classifier = AutoModelForImageClassification.from_pretrained(classifier_path).to(device)
            classifier.eval()
            processor = AutoImageProcessor.from_pretrained(classifier_path)
            self.classifier = classifier
            self.image_processor = processor
            self.device = device
            logger.info("Classifier reloaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to reload classifier from {classifier_path}: {e}")
            return False

    @staticmethod
    def get_required_delay(confidence: float) -> float:
        """Linear interpolation between LOW and HIGH confidence delay points."""
        if confidence >= CONFIDENCE_HIGH:
            return DELAY_AT_CONF_HIGH
        if confidence <= CONFIDENCE_LOW:
            return DELAY_AT_CONF_LOW
        ratio = (confidence - CONFIDENCE_LOW) / (CONFIDENCE_HIGH - CONFIDENCE_LOW)
        return DELAY_AT_CONF_LOW - ratio * (DELAY_AT_CONF_LOW - DELAY_AT_CONF_HIGH)

    def _extract_joints(self, data: dict) -> dict[str, float]:
        """Extracts numerical joint/state values from observation or action dict."""
        joints = {}
        if not data or not isinstance(data, dict):
            return joints

        def _extract_recursive(d: dict, prefix: str = ""):
            for k, v in d.items():
                full_key = f"{prefix}.{k}" if prefix else str(k)
                key_lower = full_key.lower()

                # Skip image inputs, flags, and non-state entries
                if "image" in key_lower or "img" in key_lower or key_lower.endswith("_is_pad"):
                    continue

                if isinstance(v, dict):
                    _extract_recursive(v, full_key)
                elif isinstance(v, (int, float, np.integer, np.floating)):
                    joints[full_key] = float(v)
                elif isinstance(v, (torch.Tensor, np.ndarray)):
                    size = v.size if isinstance(v, np.ndarray) else v.numel()
                    if size == 1:
                        joints[full_key] = float(v.item())
                    elif size < 100:  # State vectors (e.g. joint positions, velocities)
                        flat_v = v.flatten()
                        for i in range(size):
                            elem = flat_v[i]
                            joints[f"{full_key}_{i}"] = float(elem.item() if hasattr(elem, "item") else elem)
                elif isinstance(v, (list, tuple)) and len(v) < 100:
                    for i, elem in enumerate(v):
                        if isinstance(elem, (int, float, np.integer, np.floating)):
                            joints[f"{full_key}_{i}"] = float(elem)

        _extract_recursive(data)
        return joints

    def _calc_tracking_error(self, current_joints: dict[str, float], target_joints: dict[str, float]) -> float:
        """Calculates maximum absolute joint position tracking error between current and target joints."""
        if not current_joints or not target_joints:
            return 0.0
        common_keys = set(current_joints.keys()) & set(target_joints.keys())
        if common_keys:
            return max(abs(current_joints[k] - target_joints[k]) for k in common_keys)

        curr_vals = list(current_joints.values())
        targ_vals = list(target_joints.values())
        if len(curr_vals) == len(targ_vals) and len(curr_vals) > 0:
            return max(abs(c - t) for c, t in zip(curr_vals, targ_vals))
        return 0.0

    def _get_motion_info(self, obs: dict) -> Tuple[float, float, float]:
        """
        Calculates maximum joint velocity, windowed joint displacement over last 0.5s, and motion delay multiplier.
        Returns (max_joint_vel, max_win_displacement, motion_mult).
        """
        current_joints = self._extract_joints(obs)
        if not current_joints:
            return 0.0, 0.0, 1.0

        self.joint_history.append(current_joints)

        now = time.perf_counter()
        max_vel = 0.0
        if self.prev_joints and self.prev_time is not None:
            time_diff = max(now - self.prev_time, 1e-4)
            vels = [abs(v - self.prev_joints[k]) / time_diff for k, v in current_joints.items() if k in self.prev_joints]
            if vels:
                max_vel = max(vels)

        self.prev_joints = current_joints
        self.prev_time = now

        # Calculate max windowed joint displacement across all joints over joint_history (0.5s)
        max_win_displacement = 0.0
        if len(self.joint_history) >= 2:
            all_keys = set().union(*(frame.keys() for frame in self.joint_history))
            for k in all_keys:
                vals = [frame[k] for frame in self.joint_history if k in frame]
                if vals:
                    disp = max(vals) - min(vals)
                    if disp > max_win_displacement:
                        max_win_displacement = disp

        is_moving = (max_win_displacement > self.window_displacement_threshold) or (max_vel > 0.02)
        moving_score = min(max_vel / 0.1, 2.5) if is_moving else 0.0

        if self.idle_posture:
            errors = [abs(current_joints[k] - self.idle_posture[k]) for k in self.idle_posture.keys() if k in current_joints]
            if errors and max(errors) > 0.05:
                moving_score = max(moving_score, 1.0)

        motion_mult = 1.0 + moving_score
        return max_vel, max_win_displacement, motion_mult

    def classify_frame(self, obs: dict) -> Optional[Tuple[Dict[int, float], int, float]]:
        """Extracts overhead camera image from observation dict and returns (probs_dict, pred_class, max_confidence)."""
        if self.classifier is None or self.image_processor is None:
            return None

        top_key = None
        for k in ["observation.images.top", "top", "camera1", "observation.images.camera1"]:
            if k in obs:
                top_key = k
                break
        if top_key is None and self.rename_map:
            for k in obs.keys():
                if "top" in k or "camera1" in k:
                    top_key = k
                    break
        if top_key is None or obs[top_key] is None:
            return None

        img = obs[top_key]
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()

        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = img.transpose(1, 2, 0)

        img_resized = cv2.resize(img, (224, 224))

        inputs = self.image_processor(images=img_resized, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.classifier(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]

        num_classes = probs.shape[0]
        probs_dict = {c: float(probs[c].item()) for c in range(num_classes)}
        pred_class = torch.argmax(probs, dim=-1).item()
        max_confidence = probs_dict[pred_class]

        return probs_dict, pred_class, max_confidence

    def update(self, obs: dict, current_step: int, dt: float, action: Optional[dict] = None) -> Optional[int]:
        """
        Processes current observation and evaluates full probability vector across classes:
        - Evaluates probability distribution across ALL classes rather than relying solely on argmax.
        - Forward transitions (current_step -> current_step + 1) evaluate target class probability in real time.
        - Physical stall detection applies Policy 1 (windowed displacement), Policy 3 (command-vs-obs tracking error with large threshold),
          and Policy 2 (leaky integrator accumulator).
        """
        if self.classifier is None or self.image_processor is None:
            return None

        res = self.classify_frame(obs)
        if res is None:
            return None

        probs_dict, pred_class, max_confidence = res

        self.prob_history.append(probs_dict)
        if len(self.prob_history) > self.window_size:
            self.prob_history.pop(0)

        # Calculate temporal-average probability vector for all classes over window
        num_classes = len(probs_dict)
        smoothed_probs = {
            c: sum(p[c] for p in self.prob_history) / len(self.prob_history)
            for c in range(num_classes)
        }
        smoothed_class = max(smoothed_probs.keys(), key=lambda c: smoothed_probs[c])
        max_vel, win_disp, motion_mult = self._get_motion_info(obs)

        is_joint_moving = (win_disp > self.window_displacement_threshold)

        current_joints = self._extract_joints(obs)
        target_joints = self._extract_joints(action) if action else {}
        tracking_error = self._calc_tracking_error(current_joints, target_joints) if target_joints else 0.0
        
        is_stall_candidate = (not is_joint_moving) or (bool(target_joints) and tracking_error >= self.tracking_error_threshold)

        if is_stall_candidate:
            if self.frame_count % 30 == 0:
                logger.info(f"[Classifier] Stall {current_step}->{pred_class} moving={win_disp} error={tracking_error:.3f} acc={self.stall_accumulated_time:2f}")
            self.stall_accumulated_time = min(self.stall_timeout, self.stall_accumulated_time + dt)
        else:
            self.stall_accumulated_time = max(0.0, self.stall_accumulated_time - self.decay_rate * dt)

        if self.frame_count % 30 == 0:
            probs_str = ", ".join([f"c{c}:{smoothed_probs[c]:.2f}" for c in sorted(smoothed_probs.keys())])
            logger.info(
                f"[Classifier] Step: {current_step} | Argmax: {pred_class} ({max_confidence:.2f}) | "
                f"Probs: [{probs_str}] | WinDisp: {win_disp:.4f} rad | TrackErr: {tracking_error:.4f} rad | "
                f"StuckTimer: {self.stall_accumulated_time:.1f}s (stall_count: {self.stall_count})"
            )
        self.frame_count += 1

        # 1) Transition Check
        if current_step == 0:
            # From IDLE (step 0), find candidate non-IDLE class c >= 1
            valid_candidates = {c: p for c, p in smoothed_probs.items() if c >= 1 and p >= FORWARD_CONFIDENCE_THRESHOLD}
            if valid_candidates:
                best_class = max(valid_candidates.keys(), key=lambda c: valid_candidates[c])
                best_conf = valid_candidates[best_class]

                if self.transition_target_class is None or self.transition_target_class < 1:
                    self.transition_target_class = best_class
                    self.transition_accumulated_time = 0.0
                else:
                    self.transition_target_class = max(self.transition_target_class, best_class)

                self.transition_accumulated_time += dt
                base_delay = self.get_required_delay(best_conf)
                if best_class == 1:
                    base_delay += IDLE_ADDITIONAL_DELAY

                required_delay = base_delay * motion_mult

                if self.transition_accumulated_time >= required_delay:
                    target = self.transition_target_class
                    logger.info(
                        f"Classifier transition detected from IDLE to state {target} after stable delay {self.transition_accumulated_time:.2f}s (conf: {best_conf:.2f}, motion_mult: {motion_mult:.2f})"
                    )
                    self.transition_target_class = None
                    self.transition_accumulated_time = 0.0
                    return target
            else:
                self.transition_target_class = None
                self.transition_accumulated_time = 0.0
        else:
            # For active steps (current_step >= 1): evaluate target_step = current_step + 1
            target_step = current_step + 1
            target_conf = smoothed_probs.get(target_step, 0.0)

            # Target step is valid if target_conf >= FORWARD_CONFIDENCE_THRESHOLD and target_conf is significant among forward candidates (c >= current_step)
            forward_candidates = [p for c, p in smoothed_probs.items() if c >= current_step]
            is_forward_candidate_highest = bool(forward_candidates and target_conf >= max(forward_candidates) * 0.8)

            is_forward = (target_conf >= FORWARD_CONFIDENCE_THRESHOLD and is_forward_candidate_highest)

            if is_forward:
                if self.transition_target_class != target_step:
                    self.transition_target_class = target_step
                    self.transition_accumulated_time = 0.0

                self.transition_accumulated_time += dt
                base_delay = self.get_required_delay(target_conf)
                required_delay = base_delay * motion_mult

                if self.transition_accumulated_time >= required_delay:
                    logger.info(
                        f"Forward transition detected from step {current_step} to state {target_step} "
                        f"after stable delay {self.transition_accumulated_time:.2f}s (conf: {target_conf:.2f}, required: {required_delay:.2f}s, motion_mult: {motion_mult:.2f})"
                    )
                    self.transition_target_class = None
                    self.transition_accumulated_time = 0.0
                    return target_step
            else:
                self.transition_target_class = None
                self.transition_accumulated_time = 0.0

            # 2) 2-Tier Motion Stall Recovery Check (for current_step >= 1)
            if current_step >= 1 and self.stall_accumulated_time >= self.stall_timeout:
                if self.stall_count == 0:
                    logger.warning(
                        f"[Stall Recovery #1] Motion stuck ({self.stall_accumulated_time:.1f}s) at step {current_step}. "
                        f"Triggering position reset & seed perturbation."
                    )
                    self.stall_accumulated_time = 0.0
                    self.stall_count = 1
                    return STALL_TIER1_SIGNAL
                else:
                    logger.warning(
                        f"[Stall Recovery #2] Motion stuck again ({self.stall_accumulated_time:.1f}s) at step {current_step}. "
                        f"Triggering recovery fallback to step0."
                    )
                    self.stall_accumulated_time = 0.0
                    self.stall_count = 0
                    return 1

        return None


class RobotHoming:
    """Handles loading idle joint configurations and performing joint-space interpolation homing routines."""

    @staticmethod
    def load_idle_posture() -> Optional[dict[str, float]]:
        """Searches and loads idle_posture.json from local directories."""
        for search_dir in [os.path.dirname(__file__), "."]:
            idle_path = os.path.join(search_dir, "idle_posture.json")
            if os.path.exists(idle_path):
                try:
                    with open(idle_path, "r") as f:
                        idle_data = json.load(f)
                    logger.info(f"Loaded idle posture from {idle_path}")
                    return idle_data["joint_positions"]
                except Exception as e:
                    logger.warning(f"Failed to load idle posture from {idle_path}: {e}")
        return None

    @staticmethod
    def interpolate_to_posture(robot, target_posture: dict[str, float], duration_s: float = 2.5, fps: float = 30.0):
        """Interpolates robot joints smoothly to target posture using a cosine profile."""
        current_obs = robot.get_observation()
        start_joints = {k: current_obs[k] for k in target_posture.keys() if k in current_obs}

        for k in target_posture.keys():
            if k not in start_joints:
                logger.warning(f"Homing joint {k} not found in current observation.")
                start_joints[k] = target_posture[k]

        steps = max(int(duration_s * fps), 1)
        control_interval = 1.0 / fps

        for step in range(1, steps + 1):
            loop_start = time.perf_counter()
            t = step / steps
            t_smoothed = (1.0 - math.cos(math.pi * t)) / 2.0

            cmd = {}
            for k in target_posture.keys():
                v_start = start_joints[k]
                v_target = target_posture[k]
                if hasattr(v_start, "item"): v_start = v_start.item()
                if hasattr(v_target, "item"): v_target = v_target.item()
                cmd[k] = v_start * (1 - t_smoothed) + v_target * t_smoothed

            robot.send_action(cmd)

            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                time.sleep(sleep_t)


def classify_frame(obs: dict, classifier, image_processor, device, rename_map=None) -> Optional[Tuple[int, float]]:
    tracker = ClassifierStateTracker(classifier, image_processor, device, rename_map)
    res = tracker.classify_frame(obs)
    if res is None:
        return None
    probs_dict, pred_class, max_conf = res
    return pred_class, max_conf


class MultiStepStrategy(BaseStrategy):
    def __init__(
        self,
        config: BaseStrategyConfig,
        step_idx: int,
        task: str,
        classifier=None,
        image_processor=None,
        device=None,
        rename_map=None,
        idle_posture=None,
    ):
        super().__init__(config)
        self.step_idx = step_idx
        self.task = task
        self.next_step_idx: Optional[int] = None
        self.classifier = classifier
        self.image_processor = image_processor
        self.classifier_device = device
        self.rename_map = rename_map
        self.idle_posture = idle_posture

    def _check_timeout(self, start_time: float) -> Optional[int]:
        timeout_limit = 60.0 if self.step_idx == 1 else 40.0 if self.step_idx == 4 else 30.0
        elapsed = time.perf_counter() - start_time
        if elapsed > timeout_limit:
            logger.warning(f"Step {self.step_idx} timed out after {elapsed:.1f}s.")
            if self.step_idx == 1:
                logger.error("Step0 unfolding timed out! Restarting step0 from idle posture.")
            else:
                logger.warning("Triggering recovery fallback to step0.")
            return 1
        return None

    def run(self, ctx: RolloutContext) -> None:
        """Custom run loop with classifier transitions."""
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        interpolator = self._interpolator

        control_interval = interpolator.get_control_interval(cfg.fps)
        start_time = time.perf_counter()
        engine.resume()
        logger.info(f"MultiStepStrategy control loop started for step {self.step_idx}")

        tracker = ClassifierStateTracker(
            self.classifier,
            self.image_processor,
            self.classifier_device,
            self.rename_map,
            idle_posture=self.idle_posture,
        )

        while not ctx.runtime.shutdown_event.is_set():
            loop_start = time.perf_counter()

            if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                logger.info("Duration limit reached (%.0fs)", cfg.duration)
                break

            obs = robot.get_observation()
            obs_processed = self._process_observation_and_notify(ctx.processors, obs)
            if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                continue

            action_dict = send_next_action(obs_processed, obs, ctx, interpolator)

            if (next_step := tracker.update(obs, self.step_idx, control_interval, action=action_dict)) is not None:
                if next_step == STALL_TIER1_SIGNAL:
                    logger.warning(
                        f"[Stall #1 Action] Performing position reset, random seed perturbation for step {self.step_idx}..."
                    )
                    if self.idle_posture is not None:
                        try:
                            RobotHoming.interpolate_to_posture(robot, self.idle_posture, duration_s=2.0, fps=cfg.fps)
                        except Exception as e:
                            logger.error(f"Error during Tier 1 homing reset: {e}")

                    new_seed = (int(time.perf_counter() * 1000) + tracker.stall_count * 1000) % 2147483647
                    torch.manual_seed(new_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(new_seed)
                    np.random.seed(new_seed % (2**32 - 1))
                    random.seed(new_seed)
                    logger.info(f"[Stall #1 Action] Random seed set to {new_seed}")
                    time.sleep(1.0)
                    continue
                else:
                    self.next_step_idx = next_step
                    break

            if (next_step := self._check_timeout(start_time)) is not None:
                self.next_step_idx = next_step
                break

            self._log_telemetry(obs_processed, action_dict, ctx.runtime)

            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)
            else:
                logger.warning(
                    f"Record loop is running slower ({1 / dt:.1f} Hz) than the target FPS ({cfg.fps} Hz)."
                )

    async def run_async(self, ctx: RolloutContext) -> None:
        """Custom async run loop with classifier transitions."""
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        interpolator = self._interpolator

        control_interval = interpolator.get_control_interval(cfg.fps)
        start_time = time.perf_counter()
        engine.resume()
        logger.info(f"MultiStepStrategy control loop started for step {self.step_idx}")

        tracker = ClassifierStateTracker(
            self.classifier,
            self.image_processor,
            self.classifier_device,
            self.rename_map,
            idle_posture=self.idle_posture,
        )

        while not ctx.runtime.shutdown_event.is_set():
            loop_start = time.perf_counter()

            if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                logger.info("Duration limit reached (%.0fs)", cfg.duration)
                break

            obs = robot.get_observation()
            obs_processed = self._process_observation_and_notify(ctx.processors, obs)
            if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                await asyncio.sleep(0.001)
                continue

            action_dict = send_next_action(obs_processed, obs, ctx, interpolator)

            if (next_step := tracker.update(obs, self.step_idx, control_interval, action=action_dict)) is not None:
                if next_step == STALL_TIER1_SIGNAL:
                    logger.warning(
                        f"[Stall #1 Action] Performing position reset, random seed perturbation for step {self.step_idx}..."
                    )
                    if self.idle_posture is not None:
                        try:
                            RobotHoming.interpolate_to_posture(robot, self.idle_posture, duration_s=2.0, fps=cfg.fps)
                        except Exception as e:
                            logger.error(f"Error during Tier 1 homing reset: {e}")

                    new_seed = (int(time.perf_counter() * 1000) + tracker.stall_count * 1000) % 2147483647
                    torch.manual_seed(new_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(new_seed)
                    np.random.seed(new_seed % (2**32 - 1))
                    random.seed(new_seed)
                    logger.info(f"[Stall #1 Action] Random seed set to {new_seed}")
                    await asyncio.sleep(1.0)
                    continue
                else:
                    self.next_step_idx = next_step
                    break

            if (next_step := self._check_timeout(start_time)) is not None:
                self.next_step_idx = next_step
                break

            self._log_telemetry(obs_processed, action_dict, ctx.runtime)

            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                await asyncio.sleep(sleep_t)
            else:
                await asyncio.sleep(0.001)

    def teardown(self, ctx: RolloutContext) -> None:
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
        use_torch_compile=compile,
        device="xpu",
        interpolation_multiplier=2,
        rename_map=rename_map or {},
    )
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
    classifier_path: Optional[str] = None,
    background: bool = False,
):
    global _active_rollout_shutdown_event, _active_rollout_task

    init_logging()
    register_third_party_plugins()
    
    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    _active_rollout_shutdown_event = signal_handler.shutdown_event
    
    idle_posture = RobotHoming.load_idle_posture()
    tracker_template = ClassifierStateTracker.from_path(classifier_path, rename_map=rename_map, idle_posture=idle_posture)
    if tracker_template:
        classifier = tracker_template.classifier
        image_processor = tracker_template.image_processor
        device = tracker_template.device
    else:
        classifier = None
        image_processor = None
        device = None

    if isinstance(robot, RobotConfig):
        robot = robot.build()
    if not robot.is_connected:
        robot.connect()

    ipython = None
    try:
        from IPython import get_ipython
        ipython = get_ipython()
    except (ImportError, Exception):
        pass

    # Prepend Step 0 (Idle)
    policies = [None] + policies
    tasks = ["Idle step"] + tasks

    async def run_rollout_loop():
        step_idx = 0
        while 0 <= step_idx < len(policies):
            policy = policies[step_idx]
            task = tasks[step_idx]
            print(f"=== Running Policy Step {step_idx}: {task} ===")
            
            if policy is None:
                if classifier is None:
                    logger.info("No classifier provided. Proceeding directly to step 1.")
                    step_idx = 1
                    continue

                logger.info("Idle step: waiting for classifier to detect initial posture / step0...")
                next_step_idx = None
                control_interval = 1.0 / fps
                tracker = ClassifierStateTracker(classifier, image_processor, device, rename_map, idle_posture=idle_posture)

                while not signal_handler.shutdown_event.is_set():
                    loop_start = time.perf_counter()
                    obs = robot.get_observation() if hasattr(robot, "get_observation") else {}
                    if (next_step := tracker.update(obs, current_step=0, dt=control_interval)) is not None:
                        next_step_idx = next_step
                        break

                    dt = time.perf_counter() - loop_start
                    if (sleep_t := control_interval - dt) > 0:
                        await asyncio.sleep(sleep_t)
                    else:
                        await asyncio.sleep(0.001)

                if signal_handler.shutdown_event.is_set():
                    break

                step_idx = next_step_idx if next_step_idx is not None else step_idx + 1
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
            strategy = MultiStepStrategy(
                ctx.runtime.cfg.strategy,
                step_idx,
                task,
                classifier=classifier,
                image_processor=image_processor,
                device=device,
                rename_map=rename_map,
                idle_posture=idle_posture,
            )
            
            try:
                strategy.setup(ctx)
                await strategy.run_async(ctx)
            except (KeyboardInterrupt, asyncio.CancelledError):
                print("\n[Rollout] Interrupted or cancelled inside strategy setup/run. Triggering shutdown...")
                signal_handler.shutdown_event.set()
                break
            finally:
                strategy.teardown(ctx)
                
            if signal_handler.shutdown_event.is_set():
                break
                
            if strategy.next_step_idx is not None:
                next_step = strategy.next_step_idx
                if next_step == -1:
                    break
                
                if next_step <= step_idx and idle_posture is not None:
                    print(f"=== Fallback Detected (from step {step_idx} to step {next_step}) ===")
                    print("Executing safe homing reset routine...")
                    try:
                        RobotHoming.interpolate_to_posture(robot, idle_posture, duration_s=2.5, fps=30.0)
                        print("Safe homing reset complete.")
                    except Exception as e:
                        print(f"Error during safe homing reset: {e}")
                
                step_idx = next_step
            else:
                step_idx += 1
                
        print("Multi-step rollout finished. Disconnecting robot...")
        try:
            robot.disconnect()
        except Exception:
            pass
        print("Done")

    main_task: Optional[asyncio.Task] = None
    try:
        if ipython is not None:
            import nest_asyncio
            nest_asyncio.apply()

            loop = asyncio.get_event_loop()
            main_task = loop.create_task(run_rollout_loop())
            _active_rollout_task = main_task
            if background:
                print("Rollout started in background as asyncio Task. Run `stop_rollout()` to stop.")
            else:
                print("Rollout running in Jupyter (press 'Interrupt Kernel' or Ctrl+C to stop)...")
                loop.run_until_complete(main_task)
        else:
            asyncio.run(run_rollout_loop())
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n[Rollout] Interrupted by user (KeyboardInterrupt / Interrupt Kernel). Safely shutting down...")
        signal_handler.shutdown_event.set()
    finally:
        if main_task is not None and not main_task.done():
            main_task.cancel()
            if ipython is not None:
                try:
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(main_task)
                except (asyncio.CancelledError, KeyboardInterrupt, Exception):
                    pass
        _active_rollout_shutdown_event = None
        _active_rollout_task = None
