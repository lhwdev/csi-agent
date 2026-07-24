import os
import time
import random
from pathlib import Path
from collections import Counter
from typing import Callable, Optional, Dict, Any, Tuple, List, Union
from PIL import Image

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

try:
    import torchvision.transforms.v2 as T
except ImportError:
    import torchvision.transforms as T

from transformers import (
    AutoModelForImageClassification,
    AutoConfig,
    AutoImageProcessor,
)

DEFAULT_MODEL_NAME = "facebook/deit-tiny-patch16-224"


def sample_step_dataset(dataset, label, target_hz=1.0, fps=30.0):
    """
    Samples frames uniformly with random jitter from a dataset for a single label.
    """
    if dataset is None:
        return []
    episode_indices = np.array(dataset.hf_dataset["episode_index"])
    diff = np.diff(episode_indices)
    split_indices = np.where(diff != 0)[0] + 1
    episode_starts = [0] + list(split_indices)
    episode_ends = list(split_indices) + [len(episode_indices)]

    step_size = int(fps / target_hz)
    sampled_indices = []

    for start, end in zip(episode_starts, episode_ends):
        for step_start in range(start, end, step_size):
            offset = random.randrange(step_size)
            idx = min(step_start + offset, end - 1)
            sampled_indices.append(idx)

    print(f"Sampled {len(sampled_indices)} frames from {dataset.repo_id} with label {label}")
    return [(dataset, idx, label) for idx in sampled_indices]


def sample_split_step_dataset(dataset, current_label, next_label, transition_duration_s=1.5, target_hz=1.0, fps=30.0):
    """
    Samples frames from a step dataset, splitting active frames (current_label) and transition frames (next_label).
    """
    if dataset is None:
        return []
    episode_indices = np.array(dataset.hf_dataset["episode_index"])
    diff = np.diff(episode_indices)
    split_indices = np.where(diff != 0)[0] + 1
    episode_starts = [0] + list(split_indices)
    episode_ends = list(split_indices) + [len(episode_indices)]

    transition_frames_count = int(transition_duration_s * fps)
    samples = []
    step_size = int(fps / target_hz)

    for start, end in zip(episode_starts, episode_ends):
        episode_len = end - start
        if episode_len <= transition_frames_count:
            for idx in range(start, end):
                samples.append((dataset, idx, next_label))
            continue

        for idx in range(end - transition_frames_count, end):
            samples.append((dataset, idx, next_label))

        active_end = end - transition_frames_count
        for step_start in range(start, active_end, step_size):
            offset = random.randrange(step_size)
            idx = min(step_start + offset, active_end - 1)
            samples.append((dataset, idx, current_label))

    print(f"Sampled {dataset.repo_id}: labeled current ({current_label}), next ({next_label})")
    return samples


def get_classifier_transforms(is_train: bool = True, augment: bool = True) -> Any:
    if is_train and augment:
        return T.Compose([
            T.Resize((224, 224)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.RandomAffine(degrees=(-5, 5), translate=(0.03, 0.03), scale=(0.95, 1.05)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


class TowelStateDataset(Dataset):
    """
    PyTorch Dataset wrapper for LeRobotDataset frame samples tuple list (dataset, frame_idx, label).
    """

    def __init__(
        self,
        samples: List[Tuple[Any, int, int]],
        transform: Optional[Any] = None,
        is_train: bool = False,
        augment: bool = True,
    ):
        if transform is None:
            self.transform = get_classifier_transforms(is_train=is_train, augment=augment)
        else:
            self.transform = transform
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        dataset, frame_idx, label = self.samples[idx]
        img_tensor = dataset[frame_idx]["observation.images.top"]
        if img_tensor.dtype == torch.uint8:
            img_tensor = img_tensor.float() / 255.0

        img_pil = T.ToPILImage()(img_tensor)
        img_t = self.transform(img_pil)
        return img_t, label


class PyTorchImageFolderDataset(Dataset):
    """
    PyTorch Dataset wrapper for loading image folder samples collected from studio classifier.
    Matches folder names against class_labels.
    """

    def __init__(
        self,
        dataset_dir: Union[str, Path],
        class_labels: Optional[List[str]] = None,
        transform: Optional[Any] = None,
        is_train: bool = False,
        augment: bool = True,
    ):
        self.dataset_dir = Path(dataset_dir)
        if class_labels is None:
            self.class_labels = [
                "0: IDLE", "1: step0", "2: step1", "3: step2", "4: step3", "5: FINISH"
            ]
        else:
            self.class_labels = class_labels
        self.label2id = {lbl: idx for idx, lbl in enumerate(self.class_labels)}

        if transform is None:
            self.transform = get_classifier_transforms(is_train=is_train, augment=augment)
        else:
            self.transform = transform

        self.samples: List[Tuple[Path, int]] = []
        if self.dataset_dir.exists():
            for idx, lbl in enumerate(self.class_labels):
                sanitized_name = lbl.replace(":", "_").replace(" ", "_")
                clean_name = lbl.split(":")[-1].strip().replace(" ", "_")
                possible_names = [
                    sanitized_name,
                    lbl.replace(":", "").replace(" ", "_"),
                    clean_name,
                    f"{idx}_{clean_name}",
                    f"{idx}",
                ]
                matched_folder = None
                for name in possible_names:
                    folder_path = self.dataset_dir / name
                    if folder_path.exists():
                        matched_folder = folder_path
                        break

                if matched_folder is not None:
                    for ext in ("*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"):
                        for img_path in sorted(list(matched_folder.glob(ext))):
                            self.samples.append((img_path, idx))

            unique_samples = []
            seen = set()
            for img_path, idx in self.samples:
                if img_path not in seen:
                    seen.add(img_path)
                    unique_samples.append((img_path, idx))
            self.samples = unique_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img_pil)
        return img_tensor, label


def build_or_load_classifier(
    model_name_or_path: str = DEFAULT_MODEL_NAME,
    num_classes: int = 6,
    id2label: Optional[Dict[int, str]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("xpu" if torch.xpu.is_available() else "cpu"))

    path = Path(model_name_or_path)
    if path.exists():
        model = AutoModelForImageClassification.from_pretrained(str(path)).to(device)
        try:
            processor = AutoImageProcessor.from_pretrained(str(path))
        except Exception:
            processor = AutoImageProcessor.from_pretrained(DEFAULT_MODEL_NAME)
    else:
        config = AutoConfig.from_pretrained(model_name_or_path)
        config.num_labels = num_classes
        if id2label is not None:
            config.id2label = id2label
            config.label2id = {v: k for k, v in id2label.items()}

        model = AutoModelForImageClassification.from_pretrained(
            model_name_or_path,
            config=config,
            ignore_mismatched_sizes=True,
        ).to(device)
        processor = AutoImageProcessor.from_pretrained(model_name_or_path)

    return model, processor


def build_classifier_dataloaders(
    dataset_train: Dataset,
    dataset_val: Optional[Dataset] = None,
    num_classes: int = 6,
    batch_size: int = 32,
    num_workers: int = 2,
    device: Optional[torch.device] = None,
) -> Tuple[DataLoader, Optional[DataLoader], torch.Tensor]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("xpu" if torch.xpu.is_available() else "cpu"))

    if hasattr(dataset_train, "samples"):
        labels_train = [s[-1] if isinstance(s, (tuple, list)) else s for s in dataset_train.samples]
    elif hasattr(dataset_train, "labels"):
        labels_train = list(dataset_train.labels)
    else:
        labels_train = [dataset_train[i][1] for i in range(len(dataset_train))]

    class_counts = Counter(labels_train)
    total_samples = len(labels_train)

    class_weights = torch.zeros(num_classes, dtype=torch.float32)
    for cls in range(num_classes):
        cnt = class_counts.get(cls, 0)
        if cnt > 0:
            class_weights[cls] = total_samples / (cnt * num_classes)
        else:
            class_weights[cls] = 1.0
    class_weights = class_weights.to(device)

    sample_weights = [1.0 / max(class_counts.get(lbl, 1), 1) for lbl in labels_train]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_dataloader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_dataloader = None
    if dataset_val is not None and len(dataset_val) > 0:
        val_dataloader = DataLoader(
            dataset_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

    return train_dataloader, val_dataloader, class_weights


def train_classifier(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    class_weights: Optional[torch.Tensor] = None,
    optimizer: Optional[optim.Optimizer] = None,
    num_epochs: int = 3,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    device: Optional[torch.device] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("xpu" if torch.xpu.is_available() else "cpu"))

    model = model.to(device)

    if optimizer is None:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in tqdm(train_dataloader, desc=f"Train Epoch {epoch+1}/{num_epochs}", leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # --- Validation Phase ---
        val_loss, val_acc = 0.0, 0.0
        if val_dataloader is not None and len(val_dataloader) > 0:
            model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for imgs, labels in val_dataloader:
                    imgs = imgs.to(device)
                    labels = labels.to(device)

                    outputs = model(imgs)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    loss = criterion(logits, labels)

                    val_running_loss += loss.item() * imgs.size(0)
                    preds = torch.argmax(logits, dim=-1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_loss = val_running_loss / max(val_total, 1)
            val_acc = val_correct / max(val_total, 1)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_info = {
            "epoch": epoch + 1,
            "total_epochs": num_epochs,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "status": f"Epoch {epoch+1}/{num_epochs}: Loss={train_loss:.4f}, Acc={train_acc:.4f}",
        }

        if progress_callback is not None:
            try:
                progress_callback(epoch_info)
            except Exception as e:
                print(f"Progress callback exception: {e}")

    return history


def save_classifier(model: nn.Module, processor: Any, output_dir: str) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(out_path))
    else:
        torch.save(model.state_dict(), out_path / "pytorch_model.bin")

    if processor is not None and hasattr(processor, "save_pretrained"):
        processor.save_pretrained(str(out_path))
    print(f"Classifier saved successfully to {output_dir}")


class CoTrainingDataLoader:
    """
    DataLoader wrapper for 50:50 (or custom ratio) co-training.
    Iterates over previous dataset loader and new dataset loader simultaneously.
    """

    def __init__(
        self,
        dataloader_prev: DataLoader,
        dataloader_new: DataLoader,
        steps_per_epoch: Optional[int] = None,
    ):
        self.dataloader_prev = dataloader_prev
        self.dataloader_new = dataloader_new
        if steps_per_epoch is None:
            self.steps_per_epoch = max(len(dataloader_prev), len(dataloader_new))
        else:
            self.steps_per_epoch = steps_per_epoch

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        iter_prev = iter(self.dataloader_prev)
        iter_new = iter(self.dataloader_new)

        for _ in range(self.steps_per_epoch):
            try:
                imgs_prev, labels_prev = next(iter_prev)
            except StopIteration:
                iter_prev = iter(self.dataloader_prev)
                imgs_prev, labels_prev = next(iter_prev)

            try:
                imgs_new, labels_new = next(iter_new)
            except StopIteration:
                iter_new = iter(self.dataloader_new)
                imgs_new, labels_new = next(iter_new)

            imgs = torch.cat([imgs_prev, imgs_new], dim=0)
            labels = torch.cat([labels_prev, labels_new], dim=0)
            yield imgs, labels


def build_cotraining_dataloaders(
    dataset_prev_train: Dataset,
    dataset_new_train: Union[Dataset, str, Path],
    dataset_prev_val: Optional[Dataset] = None,
    dataset_new_val: Optional[Union[Dataset, str, Path]] = None,
    class_labels: Optional[List[str]] = None,
    num_classes: int = 6,
    batch_size: int = 32,
    dataset_weight_prev: float = 1.0,
    dataset_weight_new: float = 1.0,
    ratio: Optional[float] = None,
    mode: str = "dataset_weights",
    num_workers: int = 2,
    device: Optional[torch.device] = None,
) -> Tuple[Any, Optional[DataLoader], torch.Tensor]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("xpu" if torch.xpu.is_available() else "cpu"))

    if isinstance(dataset_new_train, (str, Path)):
        dataset_new_train = PyTorchImageFolderDataset(
            dataset_new_train, class_labels=class_labels, is_train=True, augment=True
        )

    if dataset_new_val is not None and isinstance(dataset_new_val, (str, Path)):
        dataset_new_val = PyTorchImageFolderDataset(
            dataset_new_val, class_labels=class_labels, is_train=False, augment=False
        )

    if len(dataset_new_train) == 0:
        print("[Co-Training] Warning: New dataset has 0 samples. Falling back to standard dataloader on previous dataset.")
        return build_classifier_dataloaders(
            dataset_train=dataset_prev_train,
            dataset_val=dataset_prev_val,
            num_classes=num_classes,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )

    def extract_labels(ds):
        if hasattr(ds, "samples"):
            return [s[-1] if isinstance(s, (tuple, list)) else s for s in ds.samples]
        elif hasattr(ds, "labels"):
            return list(ds.labels)
        else:
            return [ds[i][1] for i in range(len(ds))]

    labels_prev = extract_labels(dataset_prev_train)
    labels_new = extract_labels(dataset_new_train)

    if ratio is not None:
        mode = "batch_split"

    if mode == "dataset_weights":
        counts_prev = Counter(labels_prev)
        raw_w_prev = [1.0 / max(counts_prev.get(lbl, 1), 1) for lbl in labels_prev]
        sum_w_prev = max(sum(raw_w_prev), 1e-8)
        weights_prev = [w / sum_w_prev * dataset_weight_prev for w in raw_w_prev]

        counts_new = Counter(labels_new)
        raw_w_new = [1.0 / max(counts_new.get(lbl, 1), 1) for lbl in labels_new]
        sum_w_new = max(sum(raw_w_new), 1e-8)
        weights_new = [w / sum_w_new * dataset_weight_new for w in raw_w_new]

        combined_sample_weights = weights_prev + weights_new
        combined_dataset = torch.utils.data.ConcatDataset([dataset_prev_train, dataset_new_train])

        sampler = WeightedRandomSampler(
            weights=combined_sample_weights,
            num_samples=len(combined_sample_weights),
            replacement=True,
        )

        train_dataloader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        total_w = dataset_weight_prev + dataset_weight_new
        total_len = len(combined_sample_weights)
        eff_prev = int(total_len * (dataset_weight_prev / total_w))
        eff_new = total_len - eff_prev
        pct_prev = (dataset_weight_prev / total_w) * 100.0
        pct_new = (dataset_weight_new / total_w) * 100.0

        print(
            f"[Co-Training] Built dataset-weighted dataloader:\n"
            f"  - Previous Dataset (video):  raw={len(dataset_prev_train)} files | weight={dataset_weight_prev:.2f} -> effective ~{eff_prev} samples/epoch ({pct_prev:.1f}%)\n"
            f"  - New Dataset (studio pics): raw={len(dataset_new_train)} files | weight={dataset_weight_new:.2f} -> effective ~{eff_new} samples/epoch ({pct_new:.1f}%)"
        )
    else:
        split_ratio = ratio if ratio is not None else 0.5
        batch_size_prev = max(1, int(batch_size * split_ratio))
        batch_size_new = max(1, batch_size - batch_size_prev)

        counts_prev = Counter(labels_prev)
        weights_prev = [1.0 / max(counts_prev.get(lbl, 1), 1) for lbl in labels_prev]
        sampler_prev = WeightedRandomSampler(
            weights=weights_prev,
            num_samples=len(weights_prev),
            replacement=True,
        )
        loader_prev = DataLoader(
            dataset_prev_train,
            batch_size=batch_size_prev,
            sampler=sampler_prev,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        counts_new = Counter(labels_new)
        weights_new = [1.0 / max(counts_new.get(lbl, 1), 1) for lbl in labels_new]
        sampler_new = WeightedRandomSampler(
            weights=weights_new,
            num_samples=len(weights_new),
            replacement=True,
        )
        loader_new = DataLoader(
            dataset_new_train,
            batch_size=batch_size_new,
            sampler=sampler_new,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        train_dataloader = CoTrainingDataLoader(loader_prev, loader_new)

        print(
            f"[Co-Training] Built batch-split dataloader: "
            f"prev batch_size={batch_size_prev} ({len(dataset_prev_train)} samples), "
            f"new batch_size={batch_size_new} ({len(dataset_new_train)} samples)"
        )

    all_labels = labels_prev + labels_new
    counts_all = Counter(all_labels)
    total_samples = len(all_labels)

    class_weights = torch.zeros(num_classes, dtype=torch.float32)
    for cls in range(num_classes):
        cnt = counts_all.get(cls, 0)
        if cnt > 0:
            class_weights[cls] = total_samples / (cnt * num_classes)
        else:
            class_weights[cls] = 1.0
    class_weights = class_weights.to(device)

    val_datasets = [ds for ds in [dataset_prev_val, dataset_new_val] if ds is not None and len(ds) > 0]
    val_dataloader = None
    if val_datasets:
        if len(val_datasets) == 1:
            val_dataset = val_datasets[0]
        else:
            val_dataset = torch.utils.data.ConcatDataset(val_datasets)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

    return train_dataloader, val_dataloader, class_weights


def train_classifier_cotrain(
    model: nn.Module,
    dataset_prev_train: Dataset,
    dataset_new_train: Union[Dataset, str, Path],
    dataset_prev_val: Optional[Dataset] = None,
    dataset_new_val: Optional[Union[Dataset, str, Path]] = None,
    class_labels: Optional[List[str]] = None,
    num_classes: int = 6,
    batch_size: int = 32,
    dataset_weight_prev: float = 1.0,
    dataset_weight_new: float = 1.0,
    mode: str = "dataset_weights",
    ratio: Optional[float] = None,
    num_epochs: int = 3,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    num_workers: int = 2,
    device: Optional[torch.device] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Trains a ViT image classifier using dataset weighting co-training (or batch-split ratio)
    combining the previous dataset (offline video steps) and new dataset (online studio pictures).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("xpu" if torch.xpu.is_available() else "cpu"))

    train_dataloader, val_dataloader, class_weights = build_cotraining_dataloaders(
        dataset_prev_train=dataset_prev_train,
        dataset_new_train=dataset_new_train,
        dataset_prev_val=dataset_prev_val,
        dataset_new_val=dataset_new_val,
        class_labels=class_labels,
        num_classes=num_classes,
        batch_size=batch_size,
        dataset_weight_prev=dataset_weight_prev,
        dataset_weight_new=dataset_weight_new,
        mode=mode,
        ratio=ratio,
        num_workers=num_workers,
        device=device,
    )

    return train_classifier(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        class_weights=class_weights,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        progress_callback=progress_callback,
    )
