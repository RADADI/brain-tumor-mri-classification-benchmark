#!/usr/bin/env python3
"""
Publication-grade brain tumor MRI classification experiments.

Built from the user's final Lamar University project pipeline. The purpose of
this script is to generate a clean, auditable experiment record for a new
research manuscript. It does NOT reuse old reported numbers as publication
results; it reruns controlled experiments from the dataset.

Main experiment modes
---------------------
1) benchmark
   Fair six-model benchmark under one common training/evaluation pipeline:
   ResNet50, InceptionV3, EfficientNetB0, DenseNet121, VGG16, VGG19.

2) ablation
   Sequential DenseNet121 ablation study:
   D0 baseline
   D1 + CLAHE/median preprocessing
   D2 + weighted sampling
   D3 + label smoothing
   D4 + AdamW + cosine annealing
   D5 + test-time augmentation (TTA; original + horizontal flip)

3) all
   Runs benchmark then ablation.

Outputs include:
- exact JSON configuration
- dataset counts and split manifests
- training history CSV/PNG
- model weights selected by validation macro AUC
- test predictions CSV
- classification report JSON/TXT
- confusion matrices
- ROC curves
- accuracy, macro/weighted precision/recall/F1, macro AUC
- parameter count and inference timing
- per-run summary and aggregate mean/std across seeds

Example usage
-------------
# Quick initial benchmark with one seed
python brain_tumor_publication_train.py --mode benchmark --seeds 42

# DenseNet ablation with one seed
python brain_tumor_publication_train.py --mode ablation --seeds 42

# Final manuscript runs with three seeds
python brain_tumor_publication_train.py --mode all --seeds 42 123 2026

Dataset layout expected
-----------------------
DatasetNew/
  Training/
    glioma/
    meningioma/
    no_tumor/
    pituitary/
  Testing/
    glioma/
    meningioma/
    no_tumor/
    pituitary/
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import cv2
except ImportError as exc:
    raise ImportError(
        "OpenCV is required for the CLAHE ablation. Install with: pip install opencv-python"
    ) from exc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BENCHMARK_MODELS = [
    "resnet50",
    "inceptionv3",
    "efficientnetb0",
    "densenet121",
    "vgg16",
    "vgg19",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model_name: str

    # Data/training
    val_split: float = 0.20
    batch_size: int = 16
    num_workers: int = 2
    epochs_phase1: int = 15
    epochs_phase2: int = 10
    lr_phase1: float = 1e-3
    lr_phase2: float = 1e-5
    early_stop_patience: int = 6

    # Ablation switches
    use_clahe_median: bool = False
    use_weighted_sampler: bool = False
    label_smoothing: float = 0.0
    optimizer_name: str = "adam"          # adam | adamw
    scheduler_name: str = "plateau"       # plateau | cosine
    use_tta: bool = False

    # DenseNet-specific classifier option. Kept common by default.
    classifier_hidden: int = 512
    dropout1: float = 0.40
    dropout2: float = 0.30

    # Augmentation
    rotation_degrees: int = 15
    jitter_brightness: float = 0.15
    jitter_contrast: float = 0.15


def benchmark_configs(args) -> List[ExperimentConfig]:
    """Common settings for all six architectures."""
    configs = []
    for model_name in BENCHMARK_MODELS:
        configs.append(
            ExperimentConfig(
                name=f"benchmark_{model_name}",
                model_name=model_name,
                val_split=args.val_split,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                epochs_phase1=args.epochs_phase1,
                epochs_phase2=args.epochs_phase2,
                lr_phase1=args.lr_phase1,
                lr_phase2=args.lr_phase2,
                # Intentionally OFF for the fair baseline benchmark.
                use_clahe_median=False,
                use_weighted_sampler=False,
                label_smoothing=0.0,
                optimizer_name="adam",
                scheduler_name="plateau",
                use_tta=False,
            )
        )
    return configs


def ablation_configs(args) -> List[ExperimentConfig]:
    """Sequential DenseNet121 ablations so each row adds one component."""
    base = ExperimentConfig(
        name="D0_baseline",
        model_name="densenet121",
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs_phase1=args.epochs_phase1,
        epochs_phase2=args.epochs_phase2,
        lr_phase1=args.lr_phase1,
        lr_phase2=args.lr_phase2,
        use_clahe_median=False,
        use_weighted_sampler=False,
        label_smoothing=0.0,
        optimizer_name="adam",
        scheduler_name="plateau",
        use_tta=False,
    )
    return [
        base,
        replace(base, name="D1_plus_clahe_median", use_clahe_median=True),
        replace(
            base,
            name="D2_plus_weighted_sampler",
            use_clahe_median=True,
            use_weighted_sampler=True,
        ),
        replace(
            base,
            name="D3_plus_label_smoothing",
            use_clahe_median=True,
            use_weighted_sampler=True,
            label_smoothing=0.10,
        ),
        replace(
            base,
            name="D4_plus_adamw_cosine",
            use_clahe_median=True,
            use_weighted_sampler=True,
            label_smoothing=0.10,
            optimizer_name="adamw",
            scheduler_name="cosine",
        ),
        replace(
            base,
            name="D5_full_plus_tta",
            use_clahe_median=True,
            use_weighted_sampler=True,
            label_smoothing=0.10,
            optimizer_name="adamw",
            scheduler_name="cosine",
            use_tta=True,
        ),
    ]


# -----------------------------------------------------------------------------
# Reproducibility / device
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Publication-oriented dataset integrity checks
# -----------------------------------------------------------------------------

def iter_images(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def dataset_integrity_check(train_dir: Path, test_dir: Path, out_dir: Path) -> Dict:
    """Detect exact-file duplicates across train/test and record class counts."""
    result: Dict = {
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "generated_at": datetime.now().isoformat(),
    }

    def counts(root: Path) -> Dict[str, int]:
        d = {}
        for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            d[class_dir.name] = sum(1 for _ in iter_images(class_dir))
        return d

    result["train_folder_counts"] = counts(train_dir)
    result["test_folder_counts"] = counts(test_dir)

    train_hashes: Dict[str, List[str]] = {}
    test_hashes: Dict[str, List[str]] = {}

    for p in iter_images(train_dir):
        train_hashes.setdefault(sha256_file(p), []).append(str(p))
    for p in iter_images(test_dir):
        test_hashes.setdefault(sha256_file(p), []).append(str(p))

    overlap_hashes = sorted(set(train_hashes).intersection(test_hashes))
    cross_split_duplicates = []
    for h in overlap_hashes:
        cross_split_duplicates.append(
            {
                "sha256": h,
                "training_files": train_hashes[h],
                "testing_files": test_hashes[h],
            }
        )

    result["exact_cross_split_duplicate_count"] = len(cross_split_duplicates)
    result["exact_cross_split_duplicates"] = cross_split_duplicates

    with (out_dir / "dataset_integrity.json").open("w") as f:
        json.dump(result, f, indent=2)

    return result


# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------

class CLAHEMedian:
    """Apply CLAHE on luminance + median filtering, returning RGB PIL image."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8), median_ksize: int = 3):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        self.median_ksize = median_ksize

    def __call__(self, image: Image.Image) -> Image.Image:
        rgb = np.array(image.convert("RGB"))
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        enhanced = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        enhanced = cv2.medianBlur(enhanced, self.median_ksize)
        return Image.fromarray(enhanced)


def build_transforms(input_size: int, cfg: ExperimentConfig):
    preprocess = [transforms.Resize((input_size, input_size))]
    if cfg.use_clahe_median:
        preprocess.append(CLAHEMedian())

    train_transform = transforms.Compose(
        preprocess
        + [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(cfg.rotation_degrees),
            transforms.ColorJitter(
                brightness=cfg.jitter_brightness,
                contrast=cfg.jitter_contrast,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    eval_transform = transforms.Compose(
        preprocess
        + [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def denormalize_batch(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def renormalize_batch(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

def build_model(model_name: str, num_classes: int, cfg: ExperimentConfig, pretrained: bool = True):
    model_name = model_name.lower()

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(cfg.dropout1),
            nn.Linear(in_features, cfg.classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout2),
            nn.Linear(cfg.classifier_hidden, num_classes),
        )
        input_size = 224
        head_attr = "fc"

    elif model_name == "inceptionv3":
        weights = models.Inception_V3_Weights.DEFAULT if pretrained else None
        model = models.inception_v3(weights=weights, aux_logits=True)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(cfg.dropout1),
            nn.Linear(in_features, cfg.classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout2),
            nn.Linear(cfg.classifier_hidden, num_classes),
        )
        if model.AuxLogits is not None:
            aux_in = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(aux_in, num_classes)
        input_size = 299
        head_attr = "fc"

    elif model_name == "efficientnetb0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(cfg.dropout1),
            nn.Linear(in_features, cfg.classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout2),
            nn.Linear(cfg.classifier_hidden, num_classes),
        )
        input_size = 224
        head_attr = "classifier"

    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(cfg.dropout1),
            nn.Linear(in_features, cfg.classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout2),
            nn.Linear(cfg.classifier_hidden, num_classes),
        )
        input_size = 224
        head_attr = "classifier"

    elif model_name == "vgg16":
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        model = models.vgg16(weights=weights)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        input_size = 224
        head_attr = "classifier"

    elif model_name == "vgg19":
        weights = models.VGG19_Weights.DEFAULT if pretrained else None
        model = models.vgg19(weights=weights)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        input_size = 224
        head_attr = "classifier"

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model, input_size, head_attr


def freeze_for_phase1(model: nn.Module, model_name: str) -> None:
    for p in model.parameters():
        p.requires_grad = False

    if model_name == "resnet50":
        modules = [model.fc]
    elif model_name == "inceptionv3":
        modules = [model.fc]
        if model.AuxLogits is not None:
            modules.append(model.AuxLogits)
    else:
        modules = [model.classifier]

    for module in modules:
        for p in module.parameters():
            p.requires_grad = True


def unfreeze_for_phase2(model: nn.Module, model_name: str) -> None:
    """Unfreeze a reproducible top portion plus the classifier head."""
    for p in model.parameters():
        p.requires_grad = False

    if model_name == "resnet50":
        modules = [model.layer4, model.fc]
    elif model_name == "inceptionv3":
        modules = [model.Mixed_7a, model.Mixed_7b, model.Mixed_7c, model.fc]
        if model.AuxLogits is not None:
            modules.append(model.AuxLogits)
    elif model_name == "efficientnetb0":
        modules = [model.features[-2], model.features[-1], model.classifier]
    elif model_name == "densenet121":
        modules = [model.features.denseblock4, model.features.norm5, model.classifier]
    elif model_name in {"vgg16", "vgg19"}:
        modules = [model.features[-10:], model.classifier]
    else:
        raise ValueError(model_name)

    for module in modules:
        for p in module.parameters():
            p.requires_grad = True


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}


# -----------------------------------------------------------------------------
# Data loading and split manifests
# -----------------------------------------------------------------------------

def build_dataloaders(
    train_dir: Path,
    test_dir: Path,
    input_size: int,
    cfg: ExperimentConfig,
    seed: int,
    output_dir: Path,
):
    train_transform, eval_transform = build_transforms(input_size, cfg)

    base_dataset = datasets.ImageFolder(str(train_dir))
    class_names = base_dataset.classes
    targets = np.asarray(base_dataset.targets)
    indices = np.arange(len(base_dataset))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=cfg.val_split,
        random_state=seed,
        stratify=targets,
    )

    train_full = datasets.ImageFolder(str(train_dir), transform=train_transform)
    val_full = datasets.ImageFolder(str(train_dir), transform=eval_transform)
    test_dataset = datasets.ImageFolder(str(test_dir), transform=eval_transform)

    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)

    # Save exact split manifest so the paper's split is reproducible.
    manifest_rows = []
    for split_name, split_indices in [("train", train_indices), ("val", val_indices)]:
        for idx in split_indices:
            path, label = base_dataset.samples[int(idx)]
            manifest_rows.append(
                {
                    "split": split_name,
                    "path": str(Path(path).resolve()),
                    "class_index": int(label),
                    "class_name": class_names[int(label)],
                }
            )
    test_base = datasets.ImageFolder(str(test_dir))
    for path, label in test_base.samples:
        manifest_rows.append(
            {
                "split": "test",
                "path": str(Path(path).resolve()),
                "class_index": int(label),
                "class_name": test_base.classes[int(label)],
            }
        )
    pd.DataFrame(manifest_rows).to_csv(output_dir / "split_manifest.csv", index=False)

    # Dataset count summary.
    split_count_df = (
        pd.DataFrame(manifest_rows)
        .groupby(["split", "class_name"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    split_count_df.to_csv(output_dir / "split_counts.csv", index=False)

    if cfg.use_weighted_sampler:
        train_targets = targets[train_indices]
        class_counts = np.bincount(train_targets, minlength=len(class_names))
        class_weights = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
        sample_weights = [class_weights[t] for t in train_targets]
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    pin_memory = torch.cuda.is_available()
    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        generator=generator if sampler is None else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, class_names


# -----------------------------------------------------------------------------
# Forward helpers / metrics
# -----------------------------------------------------------------------------

def unpack_inception_output(output):
    if hasattr(output, "logits"):
        return output.logits, getattr(output, "aux_logits", None)
    if isinstance(output, (tuple, list)):
        return output[0], output[1] if len(output) > 1 else None
    return output, None


def multiclass_auc(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> float:
    try:
        y_true_bin = np.eye(n_classes)[y_true]
        return float(roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


def one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_name: str,
    optimizer: Optional[optim.Optimizer] = None,
):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total = 0
    all_labels: List[int] = []
    all_probs: List[np.ndarray] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            output = model(inputs)
            logits, aux = unpack_inception_output(output)
            loss = criterion(logits, labels)
            if is_train and model_name == "inceptionv3" and aux is not None:
                loss = loss + 0.4 * criterion(aux, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            probs = torch.softmax(logits, dim=1)
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            all_labels.extend(labels.detach().cpu().tolist())
            all_probs.extend(probs.detach().cpu().numpy())

    y_true = np.asarray(all_labels)
    y_prob = np.asarray(all_probs)
    y_pred = np.argmax(y_prob, axis=1)

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_auc": multiclass_auc(y_true, y_prob, y_prob.shape[1]),
    }


def build_optimizer(model: nn.Module, cfg: ExperimentConfig, learning_rate: float):
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer_name == "adamw":
        return optim.AdamW(params, lr=learning_rate, weight_decay=1e-4)
    if cfg.optimizer_name == "adam":
        return optim.Adam(params, lr=learning_rate)
    raise ValueError(cfg.optimizer_name)


def build_scheduler(optimizer, cfg: ExperimentConfig, epochs: int):
    if cfg.scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-6)
    if cfg.scheduler_name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    raise ValueError(cfg.scheduler_name)


def scheduler_step(scheduler, cfg: ExperimentConfig, val_loss: float):
    if cfg.scheduler_name == "plateau":
        scheduler.step(val_loss)
    else:
        scheduler.step()


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def fit_phase(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: ExperimentConfig,
    device: torch.device,
    output_dir: Path,
    phase_name: str,
    epochs: int,
    learning_rate: float,
):
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = build_optimizer(model, cfg, learning_rate)
    scheduler = build_scheduler(optimizer, cfg, epochs)

    best_weights = copy.deepcopy(model.state_dict())
    best_val_auc = -math.inf
    no_improve = 0
    history_rows = []

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_metrics = one_epoch(
            model, train_loader, criterion, device, cfg.model_name, optimizer=optimizer
        )
        val_metrics = one_epoch(
            model, val_loader, criterion, device, cfg.model_name, optimizer=None
        )
        scheduler_step(scheduler, cfg, val_metrics["loss"])

        row = {
            "phase": phase_name,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_macro_auc": train_metrics["macro_auc"],
            "val_macro_auc": val_metrics["macro_auc"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - start,
        }
        history_rows.append(row)

        print(
            f"[{cfg.name} | {phase_name}] epoch {epoch:02d}/{epochs} "
            f"train_acc={row['train_accuracy']:.4f} "
            f"val_acc={row['val_accuracy']:.4f} "
            f"val_auc={row['val_macro_auc']:.4f} "
            f"val_loss={row['val_loss']:.4f}"
        )

        val_auc = row["val_macro_auc"]
        score = -math.inf if np.isnan(val_auc) else val_auc
        if score > best_val_auc:
            best_val_auc = score
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, output_dir / f"best_{phase_name}.pth")
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= cfg.early_stop_patience:
            print(f"Early stopping: {cfg.name} {phase_name}")
            break

    model.load_state_dict(best_weights)
    return model, history_rows


def plot_training_history(history_df: pd.DataFrame, path: Path) -> None:
    if history_df.empty:
        return
    x = np.arange(1, len(history_df) + 1)
    fig = plt.figure(figsize=(10, 6))
    plt.plot(x, history_df["train_accuracy"], label="Train accuracy")
    plt.plot(x, history_df["val_accuracy"], label="Validation accuracy")
    plt.plot(x, history_df["train_macro_auc"], label="Train macro AUC")
    plt.plot(x, history_df["val_macro_auc"], label="Validation macro AUC")
    plt.xlabel("Epoch (phase 1 followed by phase 2)")
    plt.ylabel("Score")
    plt.title("Training History")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Evaluation with optional TTA
# -----------------------------------------------------------------------------

def tta_probabilities(model: nn.Module, inputs: torch.Tensor, use_tta: bool) -> torch.Tensor:
    output = model(inputs)
    logits, _ = unpack_inception_output(output)
    probs = torch.softmax(logits, dim=1)

    if not use_tta:
        return probs

    # Left-right flip only. A vertical flip is intentionally avoided because it
    # inverts superior/inferior anatomy and is difficult to justify clinically.
    # This assumes tumor class is independent of left/right laterality.
    hflip = torch.flip(inputs, dims=[3])

    logits_h, _ = unpack_inception_output(model(hflip))
    probs_h = torch.softmax(logits_h, dim=1)
    return (probs + probs_h) / 2.0


@torch.no_grad()
def evaluate_test(
    model: nn.Module,
    loader: DataLoader,
    class_names: Sequence[str],
    device: torch.device,
    cfg: ExperimentConfig,
    output_dir: Path,
) -> Dict:
    model.eval()
    all_true: List[int] = []
    all_pred: List[int] = []
    all_prob: List[np.ndarray] = []

    # Warm-up (where possible) before timing.
    try:
        first_inputs, _ = next(iter(loader))
        first_inputs = first_inputs.to(device)
        for _ in range(2):
            _ = tta_probabilities(model, first_inputs, cfg.use_tta)
        if device.type == "cuda":
            torch.cuda.synchronize()
    except Exception:
        pass

    start = time.perf_counter()
    sample_count = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        probs = tta_probabilities(model, inputs, cfg.use_tta)
        preds = probs.argmax(dim=1)

        all_true.extend(labels.numpy().tolist())
        all_pred.extend(preds.cpu().numpy().tolist())
        all_prob.extend(probs.cpu().numpy())
        sample_count += labels.size(0)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    y_true = np.asarray(all_true)
    y_pred = np.asarray(all_pred)
    y_prob = np.asarray(all_prob)
    n_classes = len(class_names)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "auc_macro_ovr": multiclass_auc(y_true, y_prob, n_classes),
        "test_samples": int(sample_count),
        "test_seconds": float(elapsed),
        "milliseconds_per_image": float(1000.0 * elapsed / max(sample_count, 1)),
        "tta": bool(cfg.use_tta),
    }

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        digits=6,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    with (output_dir / "classification_report.json").open("w") as f:
        json.dump(report_dict, f, indent=2)
    (output_dir / "classification_report.txt").write_text(report_text)

    pred_df = pd.DataFrame({
        "true_index": y_true,
        "true_class": [class_names[i] for i in y_true],
        "pred_index": y_pred,
        "pred_class": [class_names[i] for i in y_pred],
    })
    for i, class_name in enumerate(class_names):
        pred_df[f"prob_{class_name}"] = y_prob[:, i]
    pred_df.to_csv(output_dir / "test_predictions.csv", index=False)

    save_confusion_matrices(y_true, y_pred, class_names, output_dir)
    save_roc_curves(y_true, y_prob, class_names, output_dir)

    with (output_dir / "test_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def save_confusion_matrices(y_true, y_pred, class_names, output_dir: Path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    for matrix, filename, fmt in [
        (cm, "confusion_matrix.png", "d"),
        (cm_norm, "confusion_matrix_normalized.png", ".2f"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(matrix, interpolation="nearest")
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            xlabel="Predicted",
            ylabel="True",
            title="Confusion Matrix" if "normalized" not in filename else "Normalized Confusion Matrix",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        threshold = matrix.max() / 2.0 if matrix.size else 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = format(matrix[i, j], fmt)
                ax.text(j, i, value, ha="center", va="center",
                        color="white" if matrix[i, j] > threshold else "black")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)


def save_roc_curves(y_true, y_prob, class_names, output_dir: Path):
    y_bin = np.eye(len(class_names))[y_true]
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, class_name in enumerate(class_names):
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            score = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{class_name} (AUC={score:.3f})")
        except Exception:
            continue
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("One-vs-Rest ROC Curves")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# One experiment run
# -----------------------------------------------------------------------------

def run_experiment(
    cfg: ExperimentConfig,
    seed: int,
    train_dir: Path,
    test_dir: Path,
    root_output: Path,
    device: torch.device,
) -> Dict:
    set_seed(seed)
    run_dir = root_output / cfg.name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build a temporary model to obtain input size.
    temp, input_size, _ = build_model(cfg.model_name, num_classes=4, cfg=cfg, pretrained=True)
    del temp

    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        train_dir, test_dir, input_size, cfg, seed, run_dir
    )
    num_classes = len(class_names)

    model, input_size, _ = build_model(cfg.model_name, num_classes, cfg, pretrained=True)
    model = model.to(device)

    run_config = asdict(cfg)
    run_config.update({
        "seed": seed,
        "input_size": input_size,
        "class_names": class_names,
        "device": str(device),
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "started_at": datetime.now().isoformat(),
    })
    with (run_dir / "config.json").open("w") as f:
        json.dump(run_config, f, indent=2)

    print("\n" + "=" * 90)
    print(f"EXPERIMENT: {cfg.name} | model={cfg.model_name} | seed={seed}")
    print("=" * 90)

    freeze_for_phase1(model, cfg.model_name)
    phase1_params = count_parameters(model)
    model, h1 = fit_phase(
        model, train_loader, val_loader, cfg, device, run_dir,
        "phase1", cfg.epochs_phase1, cfg.lr_phase1
    )

    unfreeze_for_phase2(model, cfg.model_name)
    phase2_params = count_parameters(model)
    model, h2 = fit_phase(
        model, train_loader, val_loader, cfg, device, run_dir,
        "phase2", cfg.epochs_phase2, cfg.lr_phase2
    )

    torch.save(model.state_dict(), run_dir / "final_model.pth")

    history_df = pd.DataFrame(h1 + h2)
    history_df.to_csv(run_dir / "training_history.csv", index=False)
    plot_training_history(history_df, run_dir / "training_history.png")

    metrics = evaluate_test(model, test_loader, class_names, device, cfg, run_dir)
    metrics.update({
        "experiment": cfg.name,
        "model": cfg.model_name,
        "seed": seed,
        **{f"phase1_{k}": v for k, v in phase1_params.items()},
        **{f"phase2_{k}": v for k, v in phase2_params.items()},
    })

    with (run_dir / "run_summary.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------

def aggregate_results(rows: List[Dict], output_dir: Path) -> None:
    if not rows:
        return
    raw = pd.DataFrame(rows)
    raw.to_csv(output_dir / "all_runs_summary.csv", index=False)

    numeric_cols = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "auc_macro_ovr",
        "milliseconds_per_image",
    ]

    grouped_rows = []
    for (experiment, model), group in raw.groupby(["experiment", "model"]):
        out = {
            "experiment": experiment,
            "model": model,
            "n_seeds": len(group),
        }
        for col in numeric_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            out[f"{col}_mean"] = values.mean()
            out[f"{col}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
        grouped_rows.append(out)

    agg = pd.DataFrame(grouped_rows)
    agg.to_csv(output_dir / "aggregate_mean_std.csv", index=False)

    # Easy-to-paste manuscript table.
    manuscript = agg[[
        "experiment",
        "model",
        "n_seeds",
        "accuracy_mean",
        "accuracy_std",
        "f1_macro_mean",
        "f1_macro_std",
        "f1_weighted_mean",
        "f1_weighted_std",
        "auc_macro_ovr_mean",
        "auc_macro_ovr_std",
    ]].copy()
    manuscript.to_csv(output_dir / "manuscript_results_table.csv", index=False)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Publication-grade brain tumor experiments")
    parser.add_argument("--mode", choices=["benchmark", "ablation", "all"], default="benchmark")
    parser.add_argument("--train-dir", default="DatasetNew/Training")
    parser.add_argument("--test-dir", default="DatasetNew/Testing")
    parser.add_argument("--output-dir", default="publication_outputs")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-split", type=float, default=0.20)
    parser.add_argument("--epochs-phase1", type=int, default=15)
    parser.add_argument("--epochs-phase2", type=int, default=10)
    parser.add_argument("--lr-phase1", type=float, default=1e-3)
    parser.add_argument("--lr-phase2", type=float, default=1e-5)
    parser.add_argument(
        "--skip-integrity-check",
        action="store_true",
        help="Skip SHA-256 cross-split duplicate check (faster, but not recommended for final paper).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Dataset folders not found. Expected:\n  {train_dir}\n  {test_dir}\n"
            "Run this script from the project directory or pass --train-dir/--test-dir."
        )

    device = get_device()
    print(f"Device: {device}")
    print(f"Seeds: {args.seeds}")

    if not args.skip_integrity_check:
        print("Running exact train/test duplicate check...")
        integrity = dataset_integrity_check(train_dir, test_dir, output_dir)
        n_dup = integrity["exact_cross_split_duplicate_count"]
        print(f"Exact cross-split duplicate groups: {n_dup}")
        if n_dup > 0:
            print(
                "WARNING: Exact duplicates were found across Training and Testing. "
                "Do not use final manuscript numbers until these are reviewed and the split is corrected."
            )

    configs: List[ExperimentConfig] = []
    if args.mode in {"benchmark", "all"}:
        configs.extend(benchmark_configs(args))
    if args.mode in {"ablation", "all"}:
        configs.extend(ablation_configs(args))

    all_rows: List[Dict] = []
    for cfg in configs:
        for seed in args.seeds:
            row = run_experiment(cfg, seed, train_dir, test_dir, output_dir, device)
            all_rows.append(row)
            aggregate_results(all_rows, output_dir)

    print("\nCompleted experiments.")
    print(f"Results: {output_dir.resolve()}")
    print("Key files:")
    print("  all_runs_summary.csv")
    print("  aggregate_mean_std.csv")
    print("  manuscript_results_table.csv")


if __name__ == "__main__":
    main()
