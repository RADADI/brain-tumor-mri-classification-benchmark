"""
Brain Tumor Classification - Testing Script

This script evaluates all trained deep learning models on the final test set.

Models supported:
- ResNet50
- InceptionV3
- EfficientNetB0
- DenseNet121
- VGG16
- VGG19

Main features:
1. Rebuilds the exact same model architecture used during training
2. Loads saved model weights from disk
3. Applies deterministic preprocessing using ImageNet normalization
4. Supports Test-Time Augmentation (TTA) for more robust inference
5. Computes multiple evaluation metrics:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - AUC
6. Saves:
   - classification report
   - confusion matrix
   - normalized confusion matrix
   - combined results for all models

Goal:
To test all trained models under a consistent and reproducible evaluation pipeline.
"""

# =========================================================
# IMPORT REQUIRED LIBRARIES
# =========================================================

import os
# Used for:
# - joining folder and file paths safely
# - checking whether saved model/config files exist
# - creating output directories for reports and figures

import json
# Used for:
# - reading saved config files created during training
# - saving final combined results into JSON format

import numpy as np
# Used for:
# - converting predictions and labels into arrays
# - one-hot encoding labels for AUC computation
# - argmax to select highest-probability class

import matplotlib
matplotlib.use("Agg")
# Use non-interactive backend so plots can be saved directly to files
# without needing a display window

import matplotlib.pyplot as plt
# Used to create confusion matrix plots

import seaborn as sns
# Used to create cleaner and more readable heatmaps

from PIL import Image
# Used for optional single-image prediction

import torch
# Main PyTorch library for tensor operations and deep learning inference

import torch.nn as nn
# Used to rebuild classifier heads for pretrained models

from torch.utils.data import DataLoader
# Used to load testing data in batches efficiently

from torchvision import datasets, transforms, models
# datasets.ImageFolder -> load dataset from class-based folder structure
# transforms -> preprocessing pipeline
# models -> rebuild pretrained CNN architectures

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
# Metrics used for model evaluation:
# - classification_report = class-wise precision/recall/F1/support
# - confusion_matrix = count of correct/incorrect predictions per class
# - accuracy_score = overall correctness
# - precision_score = quality of positive predictions
# - recall_score = how many true positives were found
# - f1_score = balance between precision and recall
# - roc_auc_score = class separability ability


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
# Match exactly with training script model list for fair comparison

TEST_DIR = "DatasetNew/Testing"
# Folder containing final unseen testing images

BASE_OUTPUT_DIR = "BrainTumor_Project/outputs_pytorch_NewDataset"
# Folder containing trained model folders and saved weights/config files

COMBINED_RESULTS_FILE = os.path.join(BASE_OUTPUT_DIR, "all_models_test_summary.txt")
# Combined text file where summary of all models will be saved

COMBINED_RESULTS_JSON = os.path.join(BASE_OUTPUT_DIR, "all_models_test_summary.json")
# Combined JSON file where all metrics will be saved in structured format

BATCH_SIZE = 16
# Number of test images processed together in one batch
# Same batch size as training is fine, but can be reduced if memory is limited

NUM_WORKERS = 2
# Number of worker processes used to load data from disk in parallel

RUN_SINGLE_IMAGE = False
# If True, the script also predicts one individual image and saves that result

SINGLE_IMAGE_PATH = "DatasetNew/Testing/meningioma_tumor/image(7).jpg"
# Example path for optional single-image prediction
# Updated to match your new DatasetNew folder

USE_TTA = True
# Enables Test-Time Augmentation (TTA)
# If True, each test image is evaluated in multiple augmented forms and predictions are averaged

TTA_TRANSFORMS = ["original", "hflip", "vflip"]
# TTA modes used during testing:
# - original = original test image
# - hflip = horizontally flipped image
# - vflip = vertically flipped image
# Final prediction = average of probability outputs across all TTA versions


# =========================================================
# DEVICE SELECTION
# =========================================================

def get_device() -> torch.device:
    """
    Select the best available hardware automatically.

    Priority:
    1. Apple MPS (Apple Silicon GPU acceleration)
    2. CUDA (NVIDIA GPU acceleration)
    3. CPU (fallback if no GPU is available)
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
        # Best option for Apple Silicon Macs

    if torch.cuda.is_available():
        return torch.device("cuda")
        # Best option for NVIDIA GPU systems

    return torch.device("cpu")
    # Safe fallback if no GPU is available


# =========================================================
# REBUILD MODEL ARCHITECTURE
# =========================================================

def get_model(model_name: str, num_classes: int):
    """
    Rebuild the exact same architecture that was used during training.

    Why this is required:
    load_state_dict() only works if the model structure exactly matches
    the structure used when the weights were saved.
    """
    model_name = model_name.lower()
    # Convert model name to lowercase so matching is case-insensitive

    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        # Rebuild ResNet50 architecture only
        # weights=None because we will load trained weights from disk

        in_features = model.fc.in_features
        # Number of input features entering original ResNet classifier

        model.fc = nn.Sequential(
            nn.Dropout(0.4),              # Same dropout as training
            nn.Linear(in_features, 512),  # Same hidden layer as training
            nn.ReLU(inplace=True),        # Same activation as training
            nn.Dropout(0.3),              # Same second dropout as training
            nn.Linear(512, num_classes)   # Final output layer for tumor classes
        )

        input_size = 224
        # ResNet50 expects 224x224 input

    elif model_name == "inceptionv3":
        model = models.inception_v3(weights=None, aux_logits=True)
        # Rebuild InceptionV3 with auxiliary classifier
        # Must match training exactly

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
            # Auxiliary classifier output must also match training

        input_size = 299
        # InceptionV3 expects 299x299 input

    elif model_name == "efficientnetb0":
        model = models.efficientnet_b0(weights=None)
        # Rebuild EfficientNetB0 architecture only

        in_features = model.classifier[1].in_features
        # EfficientNet stores final linear layer inside classifier[1]

        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        input_size = 224

    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
        # Rebuild DenseNet121 architecture only

        in_features = model.classifier.in_features

        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        input_size = 224

    elif model_name == "vgg16":
        model = models.vgg16(weights=None)
        # Rebuild VGG16 architecture only
        # Must match training script exactly

        in_features = model.classifier[6].in_features
        # Final original VGG16 classifier takes 4096 features

        model.classifier[6] = nn.Linear(in_features, num_classes)
        # Replace final classifier layer only
        # Same as training script

        input_size = 224

    elif model_name == "vgg19":
        model = models.vgg19(weights=None)
        # Rebuild VGG19 architecture only

        in_features = model.classifier[6].in_features

        model.classifier[6] = nn.Linear(in_features, num_classes)
        # Replace final classifier layer only
        # Same as training script

        input_size = 224

    else:
        raise ValueError(
            "Model must be one of: resnet50, inceptionv3, efficientnetb0, densenet121, vgg16, vgg19"
        )

    return model, input_size
    # Return:
    # - rebuilt model
    # - correct input image size


# =========================================================
# EVALUATION TRANSFORM
# =========================================================

def get_eval_transform(input_size: int):
    """
    Build preprocessing transform for testing/inference.

    Only deterministic transforms are used here because test results
    should be stable and reproducible.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    # Channel-wise ImageNet mean for RGB channels

    imagenet_std = [0.229, 0.224, 0.225]
    # Channel-wise ImageNet standard deviation for RGB channels

    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        # Resize all images to match the model input size

        transforms.ToTensor(),
        # Convert image to tensor and scale to [0,1]

        transforms.Normalize(imagenet_mean, imagenet_std),
        # Normalize to match pretrained model expectations
    ])


# =========================================================
# TEST-TIME AUGMENTATION HELPER
# =========================================================

def apply_tta(batch: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Apply a safe test-time augmentation to a batch.

    Batch shape: [B, C, H, W]
    """
    if mode == "original":
        return batch
        # Keep input unchanged

    elif mode == "hflip":
        return torch.flip(batch, dims=[3])
        # Flip along width dimension

    elif mode == "vflip":
        return torch.flip(batch, dims=[2])
        # Flip along height dimension

    else:
        raise ValueError(f"Unsupported TTA mode: {mode}")


# =========================================================
# EVALUATE WHOLE TEST SET
# =========================================================

@torch.no_grad()
def evaluate_test_set(model, loader, class_names, device, use_tta=False, tta_transforms=None):
    """
    Run inference on the full test set and compute final metrics.

    If TTA is enabled:
    - generate multiple versions of each test batch
    - run inference on each version
    - average the predicted probabilities
    """
    model.eval()
    # Evaluation mode disables dropout randomness and stabilizes inference

    if tta_transforms is None:
        tta_transforms = ["original"]

    all_labels = []
    all_preds = []
    all_probs = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # ---------------------------------------------------
        # Standard inference (no TTA)
        # ---------------------------------------------------
        if not use_tta:
            outputs = model(inputs)

            if isinstance(outputs, tuple):
                outputs = outputs[0]
                # Needed for models like Inception

            probs = torch.softmax(outputs, dim=1)

        # ---------------------------------------------------
        # TTA inference
        # ---------------------------------------------------
        else:
            probs_list = []

            for tta_mode in tta_transforms:
                aug_inputs = apply_tta(inputs, tta_mode)
                # Apply one TTA mode to this batch

                outputs = model(aug_inputs)

                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                aug_probs = torch.softmax(outputs, dim=1)
                probs_list.append(aug_probs)
                # Save probability output from this augmentation

            probs = torch.mean(torch.stack(probs_list), dim=0)
            # Final probability = average of all TTA outputs

        preds = torch.argmax(probs, dim=1)
        # Predicted class index

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        # Overall proportion of correct predictions

        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        # Weighted precision handles class imbalance

        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        # Weighted recall measures how many true positives are found

        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        # Weighted F1 balances precision and recall
    }

    try:
        y_true_bin = np.eye(len(class_names))[y_true]
        # One-hot encoding for AUC

        metrics["auc"] = roc_auc_score(y_true_bin, y_prob, multi_class="ovr")
        # One-vs-Rest AUC for multi-class classification
    except Exception:
        metrics["auc"] = 0.0

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    # Detailed class-wise report

    cm = confusion_matrix(y_true, y_pred)
    # Raw confusion matrix

    return metrics, report, cm


# =========================================================
# SAVE CONFUSION MATRICES
# =========================================================

def save_confusion_matrices(cm, class_names, output_dir, model_name):
    """
    Save raw and normalized confusion matrices.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()
    # Save raw confusion matrix

    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    # Normalize each row by total true samples in that class

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f"{model_name} - Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix_normalized.png"), dpi=300, bbox_inches="tight")
    plt.close()
    # Save normalized confusion matrix


# =========================================================
# SINGLE IMAGE PREDICTION
# =========================================================

@torch.no_grad()
def predict_single_image(model, image_path, class_names, input_size, device, output_dir, model_name):
    """
    Predict one image and save:
    - text summary
    - image visualization with label/confidence
    """
    transform = get_eval_transform(input_size)

    image = Image.open(image_path).convert("RGB")
    # Force image into RGB format

    image_tensor = transform(image).unsqueeze(0).to(device)
    # Add batch dimension [1, C, H, W]

    outputs = model(image_tensor)

    if isinstance(outputs, tuple):
        outputs = outputs[0]

    probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    result_lines = [
        f"Single Image Prediction - {model_name}",
        f"Image: {image_path}",
        f"Predicted Class: {pred_class}",
        f"Confidence: {confidence * 100:.2f}%",
        "All Probabilities:"
    ]

    for i, cls in enumerate(class_names):
        result_lines.append(f"  {cls}: {probs[i]:.4f}")

    result_text = "\n".join(result_lines)

    with open(os.path.join(output_dir, f"{model_name}_single_prediction.txt"), "w") as f:
        f.write(result_text)

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"{model_name}: {pred_class} ({confidence * 100:.2f}%)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_single_prediction.png"), dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# BUILD TEST LOADER
# =========================================================

def build_test_loader(test_dir, input_size):
    """
    Build testing dataset and dataloader.
    """
    transform = get_eval_transform(input_size)

    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    # Folder-based test dataset

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        # No need to shuffle test data

        num_workers=NUM_WORKERS,
        # Parallel data loading

        pin_memory=torch.cuda.is_available()
        # Faster transfer on CUDA systems
    )

    return test_dataset, test_loader


# =========================================================
# TEST ONE MODEL
# =========================================================

def test_one_model(model_name, device):
    """
    Full evaluation pipeline for one trained model.
    """
    model_dir = os.path.join(BASE_OUTPUT_DIR, model_name)
    model_path = os.path.join(model_dir, f"{model_name}_final.pth")
    config_path = os.path.join(model_dir, f"{model_name}_config.json")
    test_output_dir = os.path.join(model_dir, "dynamic_test_results")

    os.makedirs(test_output_dir, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"Skipping {model_name} - model file not found: {model_path}")
        return None

    if not os.path.exists(config_path):
        print(f"Skipping {model_name} - config file not found: {config_path}")
        return None

    with open(config_path, "r") as f:
        config = json.load(f)
    # Load metadata saved during training

    class_names = config["class_names"]
    num_classes = config["num_classes"]

    model, input_size = get_model(model_name, num_classes)
    # Rebuild exact model structure

    state_dict = torch.load(model_path, map_location=device)
    # Load trained weights onto current device

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    test_dataset, test_loader = build_test_loader(TEST_DIR, input_size)

    metrics, report, cm = evaluate_test_set(
        model,
        test_loader,
        class_names,
        device,
        use_tta=USE_TTA,
        tta_transforms=TTA_TRANSFORMS
    )

    with open(os.path.join(test_output_dir, f"{model_name}_classification_report.txt"), "w") as f:
        f.write(report)

    save_confusion_matrices(cm, class_names, test_output_dir, model_name)

    print("\n" + "=" * 80)
    print(f"MODEL: {model_name.upper()}")
    print("=" * 80)
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Accuracy : {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall   : {metrics['recall']:.2%}")
    print(f"F1-Score : {metrics['f1_score']:.2%}")
    print(f"AUC      : {metrics['auc']:.2%}")
    print(f"TTA Used  : {USE_TTA}")
    if USE_TTA:
        print(f"TTA Modes : {TTA_TRANSFORMS}")

    summary_text = (
        "\n" + "=" * 80 + "\n"
        + f"MODEL: {model_name.upper()}\n"
        + "=" * 80 + "\n\n"
        + "=" * 80 + "\n"
        + "TEST RESULTS\n"
        + "=" * 80 + "\n"
        + f"Accuracy : {metrics['accuracy']:.2%}\n"
        + f"Precision: {metrics['precision']:.2%}\n"
        + f"Recall   : {metrics['recall']:.2%}\n"
        + f"F1-Score : {metrics['f1_score']:.2%}\n"
        + f"AUC      : {metrics['auc']:.2%}\n"
        + f"TTA Used  : {USE_TTA}\n"
        + (f"TTA Modes : {TTA_TRANSFORMS}\n" if USE_TTA else "")
    )

    with open(os.path.join(test_output_dir, f"{model_name}_test_summary.txt"), "w") as f:
        f.write(summary_text)
        f.write("\n\nClassification Report:\n")
        f.write(report)

    if RUN_SINGLE_IMAGE:
        predict_single_image(
            model=model,
            image_path=SINGLE_IMAGE_PATH,
            class_names=class_names,
            input_size=input_size,
            device=device,
            output_dir=test_output_dir,
            model_name=model_name
        )

    result_dict = {
        "model_name": model_name,
        "metrics": metrics,
        "report_path": os.path.join(test_output_dir, f"{model_name}_classification_report.txt"),
        "summary_path": os.path.join(test_output_dir, f"{model_name}_test_summary.txt"),
        "tta_used": USE_TTA,
        "tta_modes": TTA_TRANSFORMS if USE_TTA else []
    }

    return result_dict, summary_text, report


# =========================================================
# MAIN FUNCTION
# =========================================================

def main():
    """
    Evaluate all trained models and save combined summaries.
    """
    device = get_device()

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Testing folder: {TEST_DIR}")
    print(f"Models to test: {MODEL_LIST}")
    print(f"TTA Enabled   : {USE_TTA}")
    if USE_TTA:
        print(f"TTA Modes     : {TTA_TRANSFORMS}")

    all_results = []
    combined_text_parts = []

    for model_name in MODEL_LIST:
        output = test_one_model(model_name, device)

        if output is None:
            continue

        result_dict, summary_text, report = output

        all_results.append(result_dict)
        combined_text_parts.append(summary_text)
        combined_text_parts.append("\nClassification Report:\n")
        combined_text_parts.append(report)
        combined_text_parts.append("\n\n")

    combined_text = "\n".join(combined_text_parts)

    with open(COMBINED_RESULTS_FILE, "w") as f:
        f.write(combined_text)

    with open(COMBINED_RESULTS_JSON, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print("ALL MODEL TESTING COMPLETED")
    print("=" * 80)
    print(f"Combined summary saved to: {COMBINED_RESULTS_FILE}")
    print(f"Combined JSON saved to   : {COMBINED_RESULTS_JSON}")


# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()