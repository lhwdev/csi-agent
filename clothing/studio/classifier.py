import os
import time
import asyncio
import threading
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
from PIL import Image
import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset

try:
    import torchvision.transforms.v2 as T
except ImportError:
    import torchvision.transforms as T

from studio.ui import StudioUIMixin
from studio.classifier_ui import ClassifierUIMixin
from train_classifier import (
    build_or_load_classifier,
    build_classifier_dataloaders,
    build_cotraining_dataloaders,
    train_classifier,
    train_classifier_cotrain,
    save_classifier,
    PyTorchImageFolderDataset,
    DEFAULT_MODEL_NAME,
)

DEFAULT_CLASS_LABELS = [
    "0: IDLE",
    "1: step0",
    "2: step1",
    "3: step2",
    "4: step3",
    "5: FINISH",
]


class ClassifierImageDataset:
    def __init__(
        self,
        dataset_dir: Union[str, Path] = "./data/classifier_images",
        class_labels: Optional[List[str]] = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.class_labels = class_labels or DEFAULT_CLASS_LABELS
        self.label2id = {lbl: idx for idx, lbl in enumerate(self.class_labels)}
        self.id2label = {idx: lbl for idx, lbl in enumerate(self.class_labels)}

        for lbl in self.class_labels:
            clean_name = self._sanitize_folder_name(lbl)
            (self.dataset_dir / clean_name).mkdir(parents=True, exist_ok=True)

        self._sample_counter = 0

    def _sanitize_folder_name(self, name: str) -> str:
        return name.replace(":", "_").replace(" ", "_")

    def save_sample(self, image_np: np.ndarray, class_label: str) -> Optional[Path]:
        if class_label not in self.label2id:
            matched = [lbl for lbl in self.class_labels if str(class_label) in lbl or lbl.startswith(str(class_label))]
            if matched:
                class_label = matched[0]
            else:
                print(f"Warning: Unknown label '{class_label}', saving to first class '{self.class_labels[0]}'")
                class_label = self.class_labels[0]

        folder_name = self._sanitize_folder_name(class_label)
        target_dir = self.dataset_dir / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)

        self._sample_counter += 1
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{self._sample_counter:06d}.jpg"
        file_path = target_dir / filename

        if hasattr(image_np, "cpu"):
            image_np = image_np.cpu().numpy()
        if hasattr(image_np, "numpy") and not isinstance(image_np, np.ndarray):
            image_np = image_np.numpy()
        if isinstance(image_np, Image.Image):
            image_np = np.array(image_np)

        if isinstance(image_np, np.ndarray):
            if image_np.ndim == 3 and image_np.shape[0] in (1, 3):
                image_np = np.transpose(image_np, (1, 2, 0))

            if image_np.dtype == np.float32 or image_np.dtype == np.float64:
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255.0).astype(np.uint8)
                else:
                    image_np = image_np.astype(np.uint8)

            img_pil = Image.fromarray(image_np)
            img_pil.save(file_path, quality=95)
            return file_path
        return None

    def get_class_counts(self) -> Dict[str, int]:
        ds = self.to_torch_dataset()
        counts = {lbl: 0 for lbl in self.class_labels}
        for _, label_idx in ds.samples:
            if 0 <= label_idx < len(self.class_labels):
                counts[self.class_labels[label_idx]] += 1
        return counts

    def get_total_samples(self) -> int:
        return len(self.to_torch_dataset())

    def to_torch_dataset(self) -> PyTorchImageFolderDataset:
        return PyTorchImageFolderDataset(self.dataset_dir, self.class_labels)




class ClassifierInteractiveStudio(ClassifierUIMixin, StudioUIMixin):
    def __init__(
        self,
        classifier_tracker=None,
        camera: Optional[Any] = None,
        dataset_dir: Union[str, Path] = "./data/classifier_images",
        output_dir: Union[str, Path] = "./outputs/classifier_online",
        class_labels: Optional[List[str]] = None,
        prev_dataset: Optional[Dataset] = None,
        cotrain_ratio: float = 0.5,
        telemetry_fps: int = 15,
        total_episodes: int = 100,
    ):
        self.classifier_tracker = classifier_tracker
        self.camera = camera
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.class_labels = class_labels or DEFAULT_CLASS_LABELS
        self.prev_dataset = prev_dataset
        self.cotrain_ratio = cotrain_ratio
        self.telemetry_fps = telemetry_fps
        self.total_episodes = total_episodes

        self.image_dataset = ClassifierImageDataset(self.dataset_dir, self.class_labels)

        self.recording_state = "IDLE"
        self.is_stream_recording = False
        self.current_fps = 0.0
        self.current_episode_idx = 0
        self.current_dataset_step_idx = 0
        self.steps_val = [{"repo_id": "ClassifierStudio", "root_dir": str(self.dataset_dir)}]

        self.latest_top_img = None
        self.latest_probs: Dict[int, float] = {}
        self.latest_pred_class: Optional[int] = None
        self.latest_confidence: float = 0.0

        self.plotter_is_updating = False
        self._is_training_running = False
        self.keep_running = True
        self.camera_thread = None

        self.setup_ui()
        self.build_training_panel()

        def on_fps_change(change):
            new_fps = change["new"]
            self.telemetry_fps = new_fps
            self.add_log(f"Target UI FPS updated to {new_fps} FPS.")

        if hasattr(self, "fps_control"):
            self.fps_control.observe(on_fps_change, names="value")

        self.on_initialized()

        if self.camera is not None:
            self._start_camera_loop()

    def on_initialized(self):
        self.add_log("Classifier Studio initialized.")
        self.add_log(f"Dataset dir: {self.dataset_dir.resolve()}")
        self.add_log(f"Output model dir: {self.output_dir.resolve()}")
        if self.camera is not None:
            self.add_log("Camera stream attached.")
        self.update_status_card()
        self.update_counts_display()

    def cleanup_studio_sync(self):
        self.keep_running = False
        self.add_log("Classifier Studio exited.")

    def _start_camera_loop(self):
        if self.camera is None:
            return

        if hasattr(self.camera, "is_connected") and not getattr(self.camera, "is_connected", False):
            try:
                if hasattr(self.camera, "connect"):
                    self.camera.connect()
            except Exception as e:
                self.add_log(f"Warning: Could not connect camera: {e}")

        def camera_reader_bg():
            self.add_log("Started live camera stream.")
            last_time = time.perf_counter()
            frames_count = 0

            while self.keep_running:
                start_loop = time.perf_counter()
                fps_target = getattr(self, "telemetry_fps", None) or (self.fps_control.value if getattr(self, "fps_control", None) else 15)
                interval = 1.0 / max(1, fps_target)

                try:
                    img = None
                    if hasattr(self.camera, "async_read"):
                        img = self.camera.async_read()
                    elif hasattr(self.camera, "read"):
                        img = self.camera.read()
                    elif hasattr(self.camera, "read_latest"):
                        res = self.camera.read_latest()
                        img = res[0] if isinstance(res, (tuple, list)) else res

                    if img is not None:
                        obs = {"top": img, "observation.images.top": img}
                        self.latest_state = obs
                        frames_count += 1

                        now = time.perf_counter()
                        elapsed = now - last_time
                        loop_fps = frames_count / elapsed if elapsed > 0.5 else self.current_fps
                        if elapsed > 1.0:
                            frames_count = 0
                            last_time = now

                        self.update_telemetry(obs, current_fps=loop_fps)
                except Exception:
                    pass

                dt = time.perf_counter() - start_loop
                sleep_time = max(0.0, interval - dt)
                time.sleep(sleep_time)

            self.add_log("Live camera stream stopped.")

        self.camera_thread = threading.Thread(target=camera_reader_bg, daemon=True)
        self.camera_thread.start()

    def _extract_top_image(self, obs: Any) -> Optional[np.ndarray]:
        if obs is None:
            return None

        if isinstance(obs, np.ndarray):
            return obs
        if hasattr(obs, "cpu") and not isinstance(obs, dict):
            return obs.cpu().numpy()
        if isinstance(obs, Image.Image):
            return np.array(obs)

        if not isinstance(obs, dict):
            return None

        top_key = None
        for k in ["top", "observation.images.top", "top_cam", "camera0", "camera1", "observation.images.top_cam"]:
            if k in obs and obs[k] is not None:
                top_key = k
                break
        if top_key is None:
            for k in obs.keys():
                if "top" in str(k) or "camera" in str(k):
                    if obs[k] is not None:
                        top_key = k
                        break
        if top_key is None or obs[top_key] is None:
            return None

        img = obs[top_key]
        if hasattr(img, "cpu"):
            img = img.cpu().numpy()
        if hasattr(img, "numpy") and not isinstance(img, np.ndarray):
            img = img.numpy()
        if isinstance(img, Image.Image):
            img = np.array(img)
        return img

    def _capture_observation_frame(self, obs: Any) -> Optional[Path]:
        img_np = self._extract_top_image(obs)
        if img_np is not None:
            target_label = self.label_dropdown.value
            saved_path = self.image_dataset.save_sample(img_np, target_label)
            if saved_path:
                self.update_counts_display()
                self.update_status_card()
                return saved_path
            else:
                self.add_log(f"Error: Failed to write image sample for label '{target_label}'")
        else:
            self.add_log("Error: Could not extract valid camera frame.")
        return None

    def on_snapshot_clicked(self, b):
        saved_path = None
        if hasattr(self, "latest_state") and self.latest_state is not None:
            saved_path = self._capture_observation_frame(self.latest_state)

        if not saved_path and self.camera is not None:
            try:
                img = None
                if hasattr(self.camera, "async_read"):
                    img = self.camera.async_read()
                elif hasattr(self.camera, "read"):
                    img = self.camera.read()
                elif hasattr(self.camera, "read_latest"):
                    res = self.camera.read_latest()
                    img = res[0] if isinstance(res, (tuple, list)) else res

                if img is not None:
                    obs = {"top": img}
                    saved_path = self._capture_observation_frame(obs)
            except Exception as e:
                self.add_log(f"Error taking camera snapshot: {e}")

        if saved_path:
            self.add_log(f"📸 Saved snapshot frame under '{self.label_dropdown.value}' -> {saved_path.name}")
        elif not hasattr(self, "latest_state") or self.latest_state is None:
            self.add_log("Warning: No camera observation state available to snapshot.")

    def on_toggle_stream_recording(self, b):
        self.is_stream_recording = not self.is_stream_recording
        if self.is_stream_recording:
            self.start_btn.description = "Pause Stream (R)"
            self.start_btn.button_style = "warning"
            self.add_log(f"Started continuous frame sampling for label '{self.label_dropdown.value}'")
        else:
            self.start_btn.description = "Stream Rec (R)"
            self.start_btn.button_style = "success"
            self.add_log("Stopped stream recording.")
        self.update_status_card()

    def on_train_clicked(self, b):
        if self._is_training_running:
            self.add_log("Training is already in progress.")
            return

        total_samples = self.image_dataset.get_total_samples()
        if total_samples < 4:
            self.add_log(f"Cannot train: dataset has only {total_samples} samples (minimum 4 required).")
            self.train_status_widget.value = f"<div style='color:red;'>Need at least 4 images to train (current: {total_samples}).</div>"
            return

        self._is_training_running = True
        self.train_btn.disabled = True
        self.update_status_card()
        self.add_log("Starting classifier fine-tuning thread...")

        epochs = self.epochs_slider.value
        lr = self.lr_dropdown.value

        def run_training_bg():
            try:
                torch_ds = self.image_dataset.to_torch_dataset()
                num_classes = len(self.class_labels)

                def on_progress(info):
                    ep = info["epoch"]
                    tot = info["total_epochs"]
                    t_loss = info["train_loss"]
                    t_acc = info["train_acc"]
                    status_html = f"""
                    <div style='font-family:monospace; font-size:11px; background:#f5f5f5; padding:6px; border-radius:4px;'>
                        <b>Epoch {ep}/{tot}</b><br/>
                        Train Loss: <span style='color:#d32f2f;'>{t_loss:.4f}</span> | Acc: <span style='color:#388e3c;'>{t_acc:.2%}</span>
                    </div>
                    """
                    self.train_status_widget.value = status_html

                model_source = DEFAULT_MODEL_NAME
                if self.output_dir.exists() and (self.output_dir / "config.json").exists():
                    model_source = str(self.output_dir)

                model, processor = build_or_load_classifier(
                    model_name_or_path=model_source,
                    num_classes=num_classes,
                    id2label={i: lbl for i, lbl in enumerate(self.class_labels)},
                )

                if self.prev_dataset is not None:
                    train_classifier_cotrain(
                        model=model,
                        dataset_prev_train=self.prev_dataset,
                        dataset_new_train=torch_ds,
                        class_labels=self.class_labels,
                        num_classes=num_classes,
                        batch_size=min(32, max(len(torch_ds), 2)),
                        ratio=self.cotrain_ratio,
                        num_epochs=epochs,
                        lr=lr,
                        progress_callback=on_progress,
                    )
                else:
                    train_loader, _, class_weights = build_classifier_dataloaders(
                        dataset_train=torch_ds,
                        num_classes=num_classes,
                        batch_size=min(32, max(len(torch_ds), 2)),
                    )

                    train_classifier(
                        model=model,
                        train_dataloader=train_loader,
                        class_weights=class_weights,
                        num_epochs=epochs,
                        lr=lr,
                        progress_callback=on_progress,
                    )

                save_classifier(model, processor, str(self.output_dir))
                self.add_log(f"Model saved to {self.output_dir}")

                if self.classifier_tracker is not None:
                    reloaded = self.classifier_tracker.reload_classifier(str(self.output_dir))
                    if reloaded:
                        self.add_log("Live ClassifierStateTracker reloaded with new model weights!")

                self.train_status_widget.value = f"<div style='color:green; font-weight:bold;'>Training Complete! Saved to {self.output_dir.name}</div>"

            except Exception as e:
                traceback.print_exc()
                self.add_log(f"Training failed: {e}")
                self.train_status_widget.value = f"<div style='color:red;'>Training error: {e}</div>"
            finally:
                self._is_training_running = False
                self.train_btn.disabled = False
                self.update_status_card()

        thread = threading.Thread(target=run_training_bg, daemon=True)
        thread.start()

def rollout_interactive_classifier(
    classifier_tracker=None,
    camera: Optional[Any] = None,
    dataset_dir: str = "./data/classifier_images",
    output_dir: str = "./outputs/classifier_online",
    class_labels: Optional[List[str]] = None,
    prev_dataset: Optional[Dataset] = None,
    cotrain_ratio: float = 0.5,
    telemetry_fps: int = 15,
) -> ClassifierInteractiveStudio:
    """
    Launches interactive ViT classifier studio for data annotation and incremental training.
    """
    studio = ClassifierInteractiveStudio(
        classifier_tracker=classifier_tracker,
        camera=camera,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        class_labels=class_labels,
        prev_dataset=prev_dataset,
        cotrain_ratio=cotrain_ratio,
        telemetry_fps=telemetry_fps,
    )
    return studio
