"""
Brain Tumor Classification using Transfer Learning

This script trains multiple deep learning models:
- ResNet50
- InceptionV3
- EfficientNetB0
- DenseNet121
- VGG16
- VGG19

Main ideas used in this script:
1. Transfer learning using ImageNet-pretrained models
2. Two-phase training:
   - Phase 1: freeze pretrained backbone, train classifier head
   - Phase 2: unfreeze top layers, fine-tune carefully
3. Data augmentation to improve generalization
4. Weighted sampling to handle class imbalance
5. Learning-rate scheduling to stabilize training
6. Early stopping to prevent unnecessary training
7. Evaluation using accuracy, AUC, confusion matrix, and classification report

Goal:
To compare multiple pretrained CNN architectures under the same
pipeline and identify the best model for 4-class brain tumor classification.
"""

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import os
# Used for:
# - creating folders
# - joining file paths
# - saving model outputs, config files, and plots

import json
# Used for:
# - saving experiment configuration
# - saving final test metrics in JSON format

import time
# Used to measure training time per epoch

import copy
# Used to make a deep copy of model weights
# so best weights can be stored safely

import random
# Used for reproducibility of random processes

from datetime import datetime
# Used to add timestamps when saving config files

import numpy as np
# Used for:
# - array operations
# - class counting
# - label encoding
# - AUC preparation

import matplotlib
matplotlib.use("Agg")
# "Agg" backend allows saving figures to files
# without opening any display window

import matplotlib.pyplot as plt
# Used to plot training history and confusion matrices

import seaborn as sns
# Used to make confusion matrix plots more readable

from PIL import Image, ImageFile
# PIL is used to load image files

ImageFile.LOAD_TRUNCATED_IMAGES = True
# Prevents crashes if some images are partially corrupted

# ---------------- PyTorch core ----------------
import torch
# Main PyTorch library

import torch.nn as nn
# Contains neural network layers and loss functions

import torch.optim as optim
# Contains optimizers like Adam

from torch.optim.lr_scheduler import ReduceLROnPlateau
# Automatically reduces learning rate if validation loss stops improving

from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
# DataLoader = loads batches
# Subset = builds train/validation subsets from dataset
# WeightedRandomSampler = helps reduce class imbalance bias

from torchvision import datasets, transforms, models
# datasets -> folder-based image datasets
# transforms -> preprocessing and augmentation
# models -> pretrained CNN architectures

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
# classification_report -> precision/recall/F1 per class
# confusion_matrix -> raw class prediction table
# roc_auc_score -> AUC metric for multi-class classification

from sklearn.model_selection import train_test_split
# Used to split training set into train and validation sets


# =========================================================
# USER SETTINGS
# =========================================================

MODEL_LIST = [
    "resnet50",
    "inceptionv3",
    "efficientnetb0",
    "densenet121",
    "vgg16",
    "vgg19"
]
# These are the models to benchmark under the same pipeline.
# VGG16 and VGG19 are added based on the reference paper to strengthen
# comparison and report contribution.

TRAIN_DIR = "DatasetNew/Training"
# Folder containing training images.
# Expected format:
# Training/
#   glioma/
#   meningioma/
#   no_tumor/
#   pituitary/

TEST_DIR = "DatasetNew/Testing"
# Folder containing final unseen test images

BATCH_SIZE = 16
# Number of images processed in one batch.
# Smaller batch = less memory, often more stable
# Larger batch = faster, but needs more memory

EPOCHS_PHASE1 = 15
# Phase 1 trains only the classifier head.
# This is called feature extraction stage in transfer learning.

EPOCHS_PHASE2 = 10
# Phase 2 fine-tunes top backbone layers.
# Fewer epochs are used because pretrained layers already contain useful features.

LR_PHASE1 = 1e-3
# Learning rate for training new classifier layers.
# Higher LR is acceptable because new head starts from scratch.

LR_PHASE2 = 1e-5
# Learning rate for fine-tuning pretrained backbone layers.
# Very small LR avoids destroying useful pretrained weights.

VAL_SPLIT = 0.2
# 20% of training dataset is reserved for validation.

NUM_WORKERS = 2
# Number of subprocesses for loading data.
# Adjust depending on CPU capability.

RANDOM_SEED = 42
# Fixed random seed for reproducibility


# =========================================================
# SET RANDOM SEED
# =========================================================

def set_seed(seed: int = 42) -> None:
    """
    Make experiments more reproducible by fixing random seeds.
    """
    random.seed(seed)          # Python random seed
    np.random.seed(seed)       # NumPy random seed
    torch.manual_seed(seed)    # PyTorch CPU seed

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # GPU seed if using CUDA


# =========================================================
# DEVICE SELECTION
# =========================================================

def get_device() -> torch.device:
    """
    Automatically choose the best available device:
    1. Apple MPS (Apple Silicon GPU)
    2. CUDA (NVIDIA GPU)
    3. CPU
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# =========================================================
# MODEL CREATION
# =========================================================

def get_model(model_name: str, num_classes: int, pretrained: bool = True):
    """
    Create the selected pretrained model and replace its original
    classification head so it matches the number of tumor classes.
    """
    model_name = model_name.lower()

    # -----------------------------------------------------
    # RESNET50
    # -----------------------------------------------------
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        # Load pretrained ImageNet weights if requested

        model = models.resnet50(weights=weights)
        # ResNet50 backbone

        in_features = model.fc.in_features
        # Number of features coming from ResNet backbone into final layer

        model.fc = nn.Sequential(
            nn.Dropout(0.4),             # Drop neurons randomly to reduce overfitting
            nn.Linear(in_features, 512), # Reduce feature size and learn task-specific representation
            nn.ReLU(inplace=True),       # Non-linearity to model complex patterns
            nn.Dropout(0.3),             # Extra regularization
            nn.Linear(512, num_classes)  # Final layer outputs class scores
        )
        # Original ResNet output layer (1000 ImageNet classes) is replaced

        input_size = 224
        # ResNet50 expects 224x224 input

    # -----------------------------------------------------
    # INCEPTIONV3
    # -----------------------------------------------------
    elif model_name == "inceptionv3":
        weights = models.Inception_V3_Weights.DEFAULT if pretrained else None
        model = models.inception_v3(weights=weights, aux_logits=True)
        # aux_logits=True enables auxiliary classifier, useful during training

        in_features = model.fc.in_features

        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        if model.AuxLogits is not None:
            aux_in_features = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(aux_in_features, num_classes)
            # Auxiliary classifier must also match target class count

        input_size = 299
        # InceptionV3 expects 299x299 input

    # -----------------------------------------------------
    # EFFICIENTNETB0
    # -----------------------------------------------------
    elif model_name == "efficientnetb0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)

        in_features = model.classifier[1].in_features
        # EfficientNet stores classifier in classifier[1]

        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        input_size = 224

    # -----------------------------------------------------
    # DENSENET121
    # -----------------------------------------------------
    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)

        in_features = model.classifier.in_features

        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        input_size = 224

    # -----------------------------------------------------
    # VGG16
    # -----------------------------------------------------
    elif model_name == "vgg16":
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        model = models.vgg16(weights=weights)
        # VGG16 uses deep convolution blocks followed by large classifier layers

        in_features = model.classifier[6].in_features
        # Final original VGG16 classifier layer input size

        model.classifier[6] = nn.Linear(in_features, num_classes)
        # Replace only the last classification layer
        # Original output = 1000 ImageNet classes
        # New output = num_classes tumor classes

        input_size = 224

    # -----------------------------------------------------
    # VGG19
    # -----------------------------------------------------
    elif model_name == "vgg19":
        weights = models.VGG19_Weights.DEFAULT if pretrained else None
        model = models.vgg19(weights=weights)

        in_features = model.classifier[6].in_features

        model.classifier[6] = nn.Linear(in_features, num_classes)
        # Same idea as VGG16 but deeper backbone

        input_size = 224

    else:
        raise ValueError(
            "Model must be one of: resnet50, inceptionv3, efficientnetb0, densenet121, vgg16, vgg19"
        )

    return model, input_size


# =========================================================
# FREEZE BACKBONE
# =========================================================

def freeze_backbone(model: nn.Module, model_name: str) -> None:
    """
    Phase 1:
    Freeze pretrained layers so only classifier head learns.
    This protects pretrained knowledge at the beginning.
    """
    for param in model.parameters():
        param.requires_grad = False
        # Do not update any parameters initially

    if model_name == "resnet50":
        for param in model.fc.parameters():
            param.requires_grad = True
        # Train only ResNet classifier head

    elif model_name == "inceptionv3":
        for param in model.fc.parameters():
            param.requires_grad = True

        if model.AuxLogits is not None:
            for param in model.AuxLogits.parameters():
                param.requires_grad = True
        # Inception auxiliary head must also remain trainable

    elif model_name == "efficientnetb0":
        for param in model.classifier.parameters():
            param.requires_grad = True

    elif model_name == "densenet121":
        for param in model.classifier.parameters():
            param.requires_grad = True

    elif model_name in ["vgg16", "vgg19"]:
        for param in model.classifier.parameters():
            param.requires_grad = True
        # VGG classifier block remains trainable


# =========================================================
# UNFREEZE TOP LAYERS
# =========================================================

def unfreeze_top_layers(model: nn.Module, model_name: str, trainable_children: int = 2) -> None:
    """
    Phase 2:
    Unfreeze top layers of the backbone so model can adapt
    higher-level pretrained features to MRI domain.
    """
    for param in model.parameters():
        param.requires_grad = False
        # Reset everything to frozen first

    children = list(model.children())
    # Top-level modules of the network

    for child in children[-trainable_children:]:
        for param in child.parameters():
            param.requires_grad = True
    # Unfreeze only last few modules
    # Lower layers learn generic edges/textures
    # Upper layers learn more task-specific features

    if model_name == "resnet50":
        for param in model.fc.parameters():
            param.requires_grad = True

    elif model_name == "inceptionv3":
        for param in model.fc.parameters():
            param.requires_grad = True

        if model.AuxLogits is not None:
            for param in model.AuxLogits.parameters():
                param.requires_grad = True

    elif model_name == "efficientnetb0":
        for param in model.classifier.parameters():
            param.requires_grad = True

    elif model_name == "densenet121":
        for param in model.classifier.parameters():
            param.requires_grad = True

    elif model_name in ["vgg16", "vgg19"]:
        for param in model.classifier.parameters():
            param.requires_grad = True


# =========================================================
# IMAGE TRANSFORMS
# =========================================================

def get_transforms(input_size: int):
    """
    Returns:
    - training transform with augmentation
    - evaluation transform without random augmentation
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    # Mean RGB values of ImageNet dataset

    imagenet_std = [0.229, 0.224, 0.225]
    # Standard deviation RGB values of ImageNet dataset

    # These values are used because pretrained models were trained
    # using this exact normalization.
    # Normalize formula:
    # normalized_pixel = (pixel - mean) / std

    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        # Resize all images to match model input size

        transforms.RandomHorizontalFlip(),
        # Data augmentation: improves robustness to left-right variation

        transforms.RandomRotation(15),
        # Data augmentation: simulates slight MRI orientation changes

        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        # Data augmentation: simulates intensity variation between scans

        transforms.ToTensor(),
        # Convert PIL image to tensor and scale pixel values to [0,1]

        transforms.Normalize(imagenet_mean, imagenet_std),
        # Normalize to match pretrained backbone input distribution
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        # Same resize for validation and test

        transforms.ToTensor(),

        transforms.Normalize(imagenet_mean, imagenet_std),
        # No random augmentation during validation/test
    ])

    return train_transform, eval_transform


# =========================================================
# DATALOADERS
# =========================================================

def build_dataloaders(train_dir: str, test_dir: str, input_size: int, batch_size: int, val_split: float):
    """
    Build train, validation, and test dataloaders.
    Uses:
    - stratified split
    - weighted random sampling for class imbalance
    """
    train_transform, eval_transform = get_transforms(input_size)

    base_dataset = datasets.ImageFolder(train_dir)
    # Read dataset structure once without transform to get labels

    class_names = base_dataset.classes
    # Stores ordered class names

    targets = np.array(base_dataset.targets)
    # Integer labels for each image

    indices = np.arange(len(base_dataset))
    # Numeric indices for all training images

    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_split,
        random_state=RANDOM_SEED,
        stratify=targets
    )
    # Stratified split preserves class proportion in train/val sets

    train_dataset_full = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset_full = datasets.ImageFolder(train_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    train_targets = targets[train_indices]
    # Labels only for training subset

    class_counts = np.bincount(train_targets)
    # Count how many samples each class has

    class_weights = 1.0 / class_counts
    # Smaller classes get larger weights

    sample_weights = [class_weights[t] for t in train_targets]
    # Each sample weight depends on its class frequency

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
    # Oversamples minority classes to reduce imbalance bias

    pin_memory = torch.cuda.is_available()
    # Can speed up data transfer when using CUDA

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory
    )
    # Uses sampler instead of shuffle because sampler already controls sampling

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader, class_names


# =========================================================
# SAVE CONFIG
# =========================================================

def save_config(output_dir: str, model_name: str, class_names, input_size: int) -> str:
    """
    Save metadata needed for reproducibility and later testing.
    """
    config = {
        "model_name": model_name,
        "class_names": class_names,
        "num_classes": len(class_names),
        "input_size": input_size,
        "train_dir": TRAIN_DIR,
        "test_dir": TEST_DIR,
        "batch_size": BATCH_SIZE,
        "epochs_phase1": EPOCHS_PHASE1,
        "epochs_phase2": EPOCHS_PHASE2,
        "saved_at": datetime.now().isoformat()
    }

    config_path = os.path.join(output_dir, f"{model_name}_config.json")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return config_path


# =========================================================
# TRAIN ONE EPOCH
# =========================================================

def train_one_epoch(model, loader, criterion, optimizer, device, model_name):
    """
    Train model for one full epoch.
    Returns:
    - average loss
    - accuracy
    - multi-class AUC
    """
    model.train()
    # Training mode:
    # - dropout active
    # - batchnorm updates running statistics

    running_loss = 0.0
    running_correct = 0
    total = 0
    all_labels = []
    all_probs = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        # Clear old gradients because PyTorch accumulates them by default

        if model_name == "inceptionv3":
            outputs, aux_outputs = model(inputs)
            # Inception returns main output + auxiliary output

            loss1 = criterion(outputs, labels)
            # Main classifier loss

            loss2 = criterion(aux_outputs, labels)
            # Auxiliary classifier loss

            loss = loss1 + 0.4 * loss2
            # Auxiliary loss is weighted to help training stability

            logits = outputs
            # Use main output for prediction
        else:
            logits = model(inputs)
            # Standard forward pass for other models

            loss = criterion(logits, labels)

        loss.backward()
        # Backpropagation computes gradients

        optimizer.step()
        # Update trainable weights

        probs = torch.softmax(logits, dim=1)
        # Convert raw logits to class probabilities

        preds = torch.argmax(probs, dim=1)
        # Pick highest probability class

        running_loss += loss.item() * inputs.size(0)
        # Multiply by batch size so final average is correct

        running_correct += torch.sum(preds == labels).item()
        total += labels.size(0)

        all_labels.extend(labels.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total

    try:
        y_true = np.array(all_labels)
        y_prob = np.array(all_probs)
        y_true_bin = np.eye(y_prob.shape[1])[y_true]
        epoch_auc = roc_auc_score(y_true_bin, y_prob, multi_class="ovr")
        # One-vs-Rest AUC measures class separability
    except Exception:
        epoch_auc = 0.0

    return epoch_loss, epoch_acc, epoch_auc


# =========================================================
# VALIDATE ONE EPOCH
# =========================================================

@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    """
    Run one validation epoch.
    No gradients are computed, so validation is faster and uses less memory.
    """
    model.eval()
    # Evaluation mode:
    # - dropout disabled
    # - batchnorm uses stored statistics

    running_loss = 0.0
    running_correct = 0
    total = 0
    all_labels = []
    all_probs = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        if isinstance(outputs, tuple):
            outputs = outputs[0]
            # Handle models like Inception safely

        loss = criterion(outputs, labels)

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        running_loss += loss.item() * inputs.size(0)
        running_correct += torch.sum(preds == labels).item()
        total += labels.size(0)

        all_labels.extend(labels.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total

    try:
        y_true = np.array(all_labels)
        y_prob = np.array(all_probs)
        y_true_bin = np.eye(y_prob.shape[1])[y_true]
        epoch_auc = roc_auc_score(y_true_bin, y_prob, multi_class="ovr")
    except Exception:
        epoch_auc = 0.0

    return epoch_loss, epoch_acc, epoch_auc


# =========================================================
# FIT ONE PHASE
# =========================================================

def fit_phase(model, train_loader, val_loader, device, model_name, output_dir, epochs, learning_rate, phase_name):
    """
    Train one phase (phase1 or phase2).
    Saves best model based on validation AUC.
    Uses:
    - Adam optimizer
    - ReduceLROnPlateau scheduler
    - early stopping
    """
    criterion = nn.CrossEntropyLoss()
    # Standard loss for multi-class single-label classification

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    # Adam is used because:
    # - adaptive learning rate
    # - fast convergence
    # - works well for deep learning
    # Only parameters with requires_grad=True are updated

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    # If validation loss does not improve, learning rate is reduced by half

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_auc": [],
        "val_auc": []
    }

    best_weights = copy.deepcopy(model.state_dict())
    # Keep copy of best model weights

    best_val_auc = -1.0
    # Track best validation AUC

    early_stop_patience = 6
    # Stop if no improvement happens for 6 epochs

    no_improve = 0

    for epoch in range(epochs):
        start_time = time.time()

        train_loss, train_acc, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, model_name
        )

        val_loss, val_acc, val_auc = validate_one_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)
        # Scheduler monitors validation loss, not AUC

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_auc"].append(train_auc)
        history["val_auc"].append(val_auc)

        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
            f"Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | "
            f"LR: {current_lr:.8f} | Time: {epoch_time:.1f}s"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, os.path.join(output_dir, f"best_{phase_name}.pth"))
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= early_stop_patience:
            print(f"Early stopping triggered in {phase_name}")
            break
            # Prevents overfitting and unnecessary computation

    model.load_state_dict(best_weights)
    # Restore best weights before returning

    return model, history


# =========================================================
# MERGE HISTORIES
# =========================================================

def merge_histories(history1, history2):
    """
    Merge phase 1 and phase 2 histories so one continuous training
    plot can be drawn.
    """
    return {
        "train_loss": history1["train_loss"] + history2["train_loss"],
        "val_loss": history1["val_loss"] + history2["val_loss"],
        "train_acc": history1["train_acc"] + history2["train_acc"],
        "val_acc": history1["val_acc"] + history2["val_acc"],
        "train_auc": history1["train_auc"] + history2["train_auc"],
        "val_auc": history1["val_auc"] + history2["val_auc"],
    }


# =========================================================
# PLOT TRAINING HISTORY
# =========================================================

def plot_history(history, save_path):
    """
    Save one figure containing:
    - loss curve
    - accuracy curve
    - AUC curve
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    # Loss helps detect underfitting or overfitting

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["train_acc"], label="Train")
    plt.plot(epochs, history["val_acc"], label="Val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history["train_auc"], label="Train")
    plt.plot(epochs, history["val_auc"], label="Val")
    plt.title("AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.grid(True, alpha=0.3)
    plt.legend()
    # AUC is especially useful in medical classification tasks

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# EVALUATE MODEL ON TEST SET
# =========================================================

@torch.no_grad()
def evaluate_model(model, loader, class_names, device, save_dir):
    """
    Run final evaluation on held-out test set.
    Saves:
    - classification report
    - raw confusion matrix
    - normalized confusion matrix
    - JSON metrics
    """
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    print("\nClassification Report:\n")
    print(report)

    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()

    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    # Row-wise normalization gives per-class prediction proportions

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix_normalized.png"), dpi=300, bbox_inches="tight")
    plt.close()

    try:
        y_true_bin = np.eye(len(class_names))[y_true]
        auc_score = roc_auc_score(y_true_bin, y_prob, multi_class="ovr")
    except Exception:
        auc_score = 0.0

    metrics = {
        "accuracy": float((y_true == y_pred).mean()),
        "auc_ovr": float(auc_score)
    }

    with open(os.path.join(save_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nTest Metrics:")
    print(metrics)


# =========================================================
# TRAIN ONE MODEL END-TO-END
# =========================================================

def train_and_save_one_model(model_name: str, device: torch.device):
    """
    Full workflow for one model:
    1. Create output folder
    2. Build dataloaders
    3. Build model
    4. Train phase 1
    5. Train phase 2
    6. Save final model
    7. Save training plots
    8. Evaluate on test set
    """
    print("\n" + "=" * 80)
    print(f"TRAINING MODEL: {model_name.upper()}")
    print("=" * 80)

    output_dir = os.path.join("outputs_pytorch_NewDataset", model_name)
    os.makedirs(output_dir, exist_ok=True)

    temp_model, input_size = get_model(model_name, num_classes=2, pretrained=True)
    # Temporary model just to read required input size
    del temp_model

    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        TRAIN_DIR, TEST_DIR, input_size, BATCH_SIZE, VAL_SPLIT
    )

    num_classes = len(class_names)
    # Number of tumor categories derived from dataset folders

    print(f"Classes: {class_names}")
    print(f"Number of classes: {num_classes}")
    print(f"Input size: {input_size}")

    model, input_size = get_model(model_name, num_classes=num_classes, pretrained=True)
    model = model.to(device)

    save_config(output_dir, model_name, class_names, input_size)

    print("\nPHASE 1: Train classifier head")
    freeze_backbone(model, model_name)

    model, history1 = fit_phase(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        model_name=model_name,
        output_dir=output_dir,
        epochs=EPOCHS_PHASE1,
        learning_rate=LR_PHASE1,
        phase_name="phase1"
    )

    print("\nPHASE 2: Fine tune top layers")
    unfreeze_top_layers(model, model_name, trainable_children=2)

    model, history2 = fit_phase(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        model_name=model_name,
        output_dir=output_dir,
        epochs=EPOCHS_PHASE2,
        learning_rate=LR_PHASE2,
        phase_name="phase2"
    )

    final_model_path = os.path.join(output_dir, f"{model_name}_final.pth")
    torch.save(model.state_dict(), final_model_path)

    full_history = merge_histories(history1, history2)
    plot_history(full_history, os.path.join(output_dir, f"{model_name}_training_history.png"))

    print("\nTEST EVALUATION")
    evaluate_model(model, test_loader, class_names, device, output_dir)

    print(f"\nSaved final model: {final_model_path}")
    print(f"Saved outputs in: {output_dir}")


# =========================================================
# MAIN FUNCTION
# =========================================================

def main():
    """
    Main entry point:
    - set seed
    - select device
    - train all models
    """
    set_seed(RANDOM_SEED)

    device = get_device()

    print(f"Using device: {device}")
    print(f"Train folder: {TRAIN_DIR}")
    print(f"Test folder: {TEST_DIR}")
    print(f"Models to train: {MODEL_LIST}")

    for model_name in MODEL_LIST:
        train_and_save_one_model(model_name, device)

    print("\nALL MODELS TRAINED AND SAVED SEPARATELY.")


# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()