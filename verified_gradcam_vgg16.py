#!/usr/bin/env python3
"""Verified qualitative + Grad-CAM analysis for the VGG16 brain-tumor model.

Designed for the publication experiment produced by brain_tumor_publication_train_resume.py.
The script performs validation checks before creating any paper figure:

1) strict checkpoint loading (no missing/unexpected keys),
2) exact VGG16 architecture / target-layer check,
3) re-runs all held-out test predictions using the original evaluation transform,
4) compares class labels/predictions with the saved seed-123 test_predictions.csv,
5) computes Grad-CAM without backward hooks by splitting VGG16 at its final conv layer,
6) checks that the split-forward logits exactly reproduce the normal model forward,
7) reports a simple CAM masking faithfulness diagnostic for each selected case,
8) saves separate publication-oriented figures for correct and error cases.

Important interpretation note:
Grad-CAM is an attribution map, not a tumor segmentation mask. A map that attends to
background, skull, text, or other non-lesion regions can be a *real model failure/shortcut*,
not necessarily a Grad-CAM implementation error. Do not claim anatomical localization
without independent localization ground truth.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PRETTY = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No tumor",
    "no_tumor": "No tumor",
    "pituitary": "Pituitary",
}


def build_model(num_classes: int = 4) -> nn.Module:
    model = models.vgg16(weights=None)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model


def load_checkpoint_strict(model: nn.Module, path: Path) -> None:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise RuntimeError("Checkpoint does not contain a recognizable state_dict.")
    obj = {k[7:] if k.startswith("module.") else k: v for k, v in obj.items()}
    # For publication verification we do NOT allow silent mismatch.
    model.load_state_dict(obj, strict=True)


def choose_prediction_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def eval_transform():
    # This intentionally matches build_transforms(...).eval_transform from
    # brain_tumor_publication_train_resume.py for VGG16 (input size 224).
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def raw_transform():
    return transforms.Resize((224, 224))


def predict_dataset(model: nn.Module, ds, device: torch.device, batch_size: int):
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    true_all, pred_all, prob_all = [], [], []
    model = model.to(device).eval()
    with torch.inference_mode():
        for xb, yb in loader:
            logits = model(xb.to(device))
            probs = torch.softmax(logits, dim=1).cpu()
            true_all.append(yb.cpu())
            pred_all.append(probs.argmax(1))
            prob_all.append(probs)
    return (
        torch.cat(true_all).numpy(),
        torch.cat(pred_all).numpy(),
        torch.cat(prob_all).numpy(),
    )


def verify_against_reference(ds, y_true, y_pred, y_prob, ref_csv: Path) -> Dict[str, float]:
    ref = pd.read_csv(ref_csv)
    if len(ref) != len(ds):
        raise RuntimeError(f"Reference CSV has {len(ref)} rows, but ImageFolder has {len(ds)} images.")

    required = {"true_index", "pred_index"} | {f"prob_{c}" for c in ds.classes}
    missing = required - set(ref.columns)
    if missing:
        raise RuntimeError(f"Reference CSV is missing columns: {sorted(missing)}")

    ref_true = ref["true_index"].to_numpy(dtype=int)
    ref_pred = ref["pred_index"].to_numpy(dtype=int)
    ref_prob = ref[[f"prob_{c}" for c in ds.classes]].to_numpy(dtype=float)

    true_match = float(np.mean(y_true == ref_true))
    pred_match = float(np.mean(y_pred == ref_pred))
    max_prob_diff = float(np.max(np.abs(y_prob - ref_prob)))
    mean_prob_diff = float(np.mean(np.abs(y_prob - ref_prob)))

    if true_match != 1.0:
        bad = np.flatnonzero(y_true != ref_true)[:10]
        raise RuntimeError(
            "Dataset ordering/labels do not reproduce the original test run. "
            f"First mismatched row indices: {bad.tolist()}"
        )
    if pred_match != 1.0:
        bad = np.flatnonzero(y_pred != ref_pred)[:10]
        details = []
        for i in bad:
            details.append(
                f"row {i}: file={Path(ds.samples[i][0]).name}, now={y_pred[i]}, ref={ref_pred[i]}"
            )
        raise RuntimeError(
            "Checkpoint/model evaluation does NOT reproduce the saved test predictions.\n" + "\n".join(details)
        )

    return {
        "true_label_match_fraction": true_match,
        "predicted_class_match_fraction": pred_match,
        "max_probability_abs_difference": max_prob_diff,
        "mean_probability_abs_difference": mean_prob_diff,
    }


def pretty(name: str) -> str:
    return PRETTY.get(name, name.replace("_", " ").title())


def median_case(indices: List[int], confidences: np.ndarray) -> int | None:
    if not indices:
        return None
    vals = confidences[indices]
    med = float(np.median(vals))
    return min(indices, key=lambda i: abs(float(confidences[i]) - med))


def select_cases(ds, y_true, y_pred, y_prob, excludes: set[str]):
    conf = y_prob.max(axis=1)
    available = [i for i, (p, _) in enumerate(ds.samples) if Path(p).name not in excludes]
    selected_correct: List[int] = []
    selected_errors: List[int] = []

    # Correct: one median-confidence example per class. Selection is independent of CAM appearance.
    for c in range(len(ds.classes)):
        cand = [i for i in available if y_true[i] == c and y_pred[i] == c]
        pick = median_case(cand, conf)
        if pick is not None:
            selected_correct.append(pick)

    name_to_idx = {n.replace("_", "").replace(" ", "").lower(): i for i, n in enumerate(ds.classes)}
    def ci(name):
        return name_to_idx.get(name.replace("_", "").replace(" ", "").lower())

    priorities = [(ci("glioma"), ci("meningioma")),
                  (ci("glioma"), ci("notumor")),
                  (ci("meningioma"), ci("glioma"))]

    used = set(selected_correct)
    for t, p in priorities:
        if t is None or p is None:
            continue
        cand = [i for i in available if i not in used and y_true[i] == t and y_pred[i] == p]
        pick = median_case(cand, conf)
        if pick is not None:
            selected_errors.append(pick); used.add(pick)

    # Fill to four errors from the largest remaining confusion groups.
    groups: Dict[Tuple[int, int], List[int]] = {}
    for i in available:
        if i in used or y_true[i] == y_pred[i]:
            continue
        groups.setdefault((int(y_true[i]), int(y_pred[i])), []).append(i)
    for _, inds in sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True):
        pick = median_case(inds, conf)
        if pick is not None:
            selected_errors.append(pick); used.add(pick)
        if len(selected_errors) >= 4:
            break

    return selected_correct[:4], selected_errors[:4]


def vgg16_gradcam_direct(model: nn.Module, x_cpu: torch.Tensor, target_idx: int):
    """Grad-CAM at VGG16's final Conv2d (features[28]), without hooks.

    We split the network exactly at the final convolution. The continuation receives
    a clone so VGG16's in-place ReLU cannot overwrite the saved pre-ReLU activation.
    """
    model = model.cpu().eval()
    model.zero_grad(set_to_none=True)

    if not isinstance(model.features[28], nn.Conv2d):
        raise RuntimeError(f"Expected model.features[28] to be Conv2d, got {type(model.features[28]).__name__}")
    if not isinstance(model.features[29], nn.ReLU):
        raise RuntimeError(f"Expected model.features[29] to be ReLU, got {type(model.features[29]).__name__}")

    # Exact ordinary forward for verification.
    with torch.no_grad():
        normal_logits = model(x_cpu).detach()

    # Split forward at last convolutional feature map.
    activation = model.features[:29](x_cpu)
    activation.retain_grad()
    z = model.features[29:](activation.clone())
    z = model.avgpool(z)
    z = torch.flatten(z, 1)
    logits = model.classifier(z)

    max_logit_diff = float(torch.max(torch.abs(logits.detach() - normal_logits)).item())
    if max_logit_diff > 1e-6:
        raise RuntimeError(f"Split VGG16 forward does not match normal forward (max logit diff={max_logit_diff:.3g}).")

    score = logits[0, target_idx]
    score.backward()
    grad = activation.grad
    if grad is None:
        raise RuntimeError("No gradient was retained for the final convolution activation.")

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activation.detach()).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x_cpu.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    if not torch.isfinite(cam).all():
        raise RuntimeError("Grad-CAM contains non-finite values.")
    raw_max = float(cam.max().item())
    if raw_max <= 1e-12:
        raise RuntimeError("Grad-CAM is effectively all-zero for this sample/target.")
    cam = cam - cam.min()
    cam = cam / cam.max().clamp_min(1e-12)
    return cam.detach().numpy(), logits.detach(), max_logit_diff


def cam_masking_diagnostic(model: nn.Module, x_cpu: torch.Tensor, cam: np.ndarray, target_idx: int, frac: float = 0.15):
    """Simple faithfulness diagnostic: mask highest-CAM pixels with normalized ImageNet mean (=0)."""
    model = model.cpu().eval()
    with torch.no_grad():
        p_before = float(torch.softmax(model(x_cpu), dim=1)[0, target_idx].item())
        threshold = float(np.quantile(cam, 1.0 - frac))
        mask = torch.from_numpy((cam >= threshold).astype(np.bool_)).unsqueeze(0).unsqueeze(0)
        x_masked = x_cpu.clone()
        x_masked = torch.where(mask.expand_as(x_masked), torch.zeros_like(x_masked), x_masked)
        p_after = float(torch.softmax(model(x_masked), dim=1)[0, target_idx].item())
    return p_before, p_after, p_before - p_after


def save_correct_figure(ds, indices, model, tfm, output_path: Path, diag_rows: list):
    fig, axes = plt.subplots(len(indices), 2, figsize=(7.4, 3.1 * len(indices)))
    if len(indices) == 1:
        axes = np.array([axes])
    rt = raw_transform()
    for row, idx in enumerate(indices):
        path, true_idx = ds.samples[idx]
        img = Image.open(path).convert("RGB")
        x = tfm(img).unsqueeze(0).cpu()
        with torch.no_grad():
            probs = torch.softmax(model.cpu()(x), dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())
        cam, _, split_diff = vgg16_gradcam_direct(model, x, pred_idx)
        p0, p1, drop = cam_masking_diagnostic(model, x, cam, pred_idx)
        disp = np.asarray(rt(img)).astype(np.float32) / 255.0

        axes[row, 0].imshow(disp); axes[row, 0].axis("off")
        axes[row, 0].set_title(
            f"True: {pretty(ds.classes[true_idx])} | Pred: {pretty(ds.classes[pred_idx])}\nConfidence: {float(probs[pred_idx])*100:.1f}%",
            fontsize=9,
        )
        axes[row, 1].imshow(disp)
        axes[row, 1].imshow(cam, cmap="turbo", alpha=0.45, vmin=0, vmax=1)
        axes[row, 1].axis("off"); axes[row, 1].set_title("Predicted-class Grad-CAM", fontsize=9)
        diag_rows.append({
            "file": Path(path).name, "case": "correct", "true": ds.classes[true_idx],
            "pred": ds.classes[pred_idx], "confidence": float(probs[pred_idx]),
            "cam_target": ds.classes[pred_idx], "mask_top_fraction": 0.15,
            "target_prob_before_mask": p0, "target_prob_after_mask": p1,
            "probability_drop": drop, "split_forward_max_logit_diff": split_diff,
        })
    fig.suptitle("VGG16 representative correct predictions and Grad-CAM", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_error_figure(ds, indices, model, tfm, output_path: Path, diag_rows: list):
    # For errors, show both the predicted-class explanation and the true-class explanation.
    fig, axes = plt.subplots(len(indices), 3, figsize=(10.7, 3.1 * len(indices)))
    if len(indices) == 1:
        axes = np.array([axes])
    rt = raw_transform()
    for row, idx in enumerate(indices):
        path, true_idx = ds.samples[idx]
        img = Image.open(path).convert("RGB")
        x = tfm(img).unsqueeze(0).cpu()
        with torch.no_grad():
            probs = torch.softmax(model.cpu()(x), dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())

        cam_pred, _, split_diff1 = vgg16_gradcam_direct(model, x, pred_idx)
        cam_true, _, split_diff2 = vgg16_gradcam_direct(model, x, int(true_idx))
        p0, p1, drop = cam_masking_diagnostic(model, x, cam_pred, pred_idx)
        disp = np.asarray(rt(img)).astype(np.float32) / 255.0

        axes[row, 0].imshow(disp); axes[row, 0].axis("off")
        axes[row, 0].set_title(
            f"True: {pretty(ds.classes[true_idx])} | Pred: {pretty(ds.classes[pred_idx])}\nPred. confidence: {float(probs[pred_idx])*100:.1f}%",
            fontsize=9,
        )
        axes[row, 1].imshow(disp); axes[row, 1].imshow(cam_pred, cmap="turbo", alpha=0.45, vmin=0, vmax=1)
        axes[row, 1].axis("off"); axes[row, 1].set_title(f"CAM for predicted {pretty(ds.classes[pred_idx])}", fontsize=9)
        axes[row, 2].imshow(disp); axes[row, 2].imshow(cam_true, cmap="turbo", alpha=0.45, vmin=0, vmax=1)
        axes[row, 2].axis("off"); axes[row, 2].set_title(f"CAM for true {pretty(ds.classes[true_idx])}", fontsize=9)

        diag_rows.append({
            "file": Path(path).name, "case": "error", "true": ds.classes[true_idx],
            "pred": ds.classes[pred_idx], "confidence": float(probs[pred_idx]),
            "cam_target": ds.classes[pred_idx], "mask_top_fraction": 0.15,
            "target_prob_before_mask": p0, "target_prob_after_mask": p1,
            "probability_drop": drop,
            "split_forward_max_logit_diff": max(split_diff1, split_diff2),
        })
    fig.suptitle("VGG16 misclassifications: predicted-class vs. true-class Grad-CAM", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--reference-predictions", required=True,
                    help="Original seed-123 test_predictions.csv saved by the publication training script.")
    ap.add_argument("--output-dir", default="figures/verified_gradcam")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--prediction-device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--exclude", nargs="*", default=["Te-no_1.jpg"],
                    help="Filenames to exclude for objective image-quality/artifact reasons.")
    args = ap.parse_args()

    test_dir = Path(args.test_dir)
    checkpoint = Path(args.checkpoint)
    ref_csv = Path(args.reference_predictions)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    for p in (test_dir, checkpoint, ref_csv):
        if not p.exists():
            raise FileNotFoundError(p)

    tfm = eval_transform()
    ds = datasets.ImageFolder(str(test_dir), transform=tfm)
    if ds.classes != ["glioma", "meningioma", "notumor", "pituitary"]:
        raise RuntimeError(f"Unexpected class order: {ds.classes}")
    if len(ds) != 1600:
        raise RuntimeError(f"Expected 1600 held-out test images; found {len(ds)}")

    model = build_model(4)
    load_checkpoint_strict(model, checkpoint)
    print("[PASS] Checkpoint loaded with strict=True; no missing/unexpected parameters.")
    print("[PASS] Target layer verified as VGG16 features[28] (final Conv2d).")

    pred_device = choose_prediction_device(args.prediction_device)
    print("Prediction verification device:", pred_device)
    y_true, y_pred, y_prob = predict_dataset(model, ds, pred_device, args.batch_size)
    stats = verify_against_reference(ds, y_true, y_pred, y_prob, ref_csv)
    print(f"[PASS] Dataset labels reproduce saved run: {stats['true_label_match_fraction']*100:.1f}%")
    print(f"[PASS] Predicted classes reproduce saved run: {stats['predicted_class_match_fraction']*100:.1f}%")
    print(f"[INFO] Max |probability difference| vs saved CSV: {stats['max_probability_abs_difference']:.8f}")
    print(f"[INFO] Mean |probability difference| vs saved CSV: {stats['mean_probability_abs_difference']:.8f}")

    # Use CPU for attribution to avoid backend-specific hook/autograd issues; only 8 cases are processed.
    model = model.cpu().eval()
    correct_idx, error_idx = select_cases(ds, y_true, y_pred, y_prob, set(args.exclude))

    print("\nSelected correct cases (selection independent of CAM):")
    for i in correct_idx:
        print(f"  {Path(ds.samples[i][0]).name}: {ds.classes[y_true[i]]} -> {ds.classes[y_pred[i]]}, p={y_prob[i, y_pred[i]]:.4f}")
    print("Selected error cases (selection independent of CAM):")
    for i in error_idx:
        print(f"  {Path(ds.samples[i][0]).name}: {ds.classes[y_true[i]]} -> {ds.classes[y_pred[i]]}, p={y_prob[i, y_pred[i]]:.4f}")

    diagnostics = []
    save_correct_figure(ds, correct_idx, model, tfm, outdir / "vgg16_verified_gradcam_correct.png", diagnostics)
    save_error_figure(ds, error_idx, model, tfm, outdir / "vgg16_verified_gradcam_errors.png", diagnostics)
    pd.DataFrame(diagnostics).to_csv(outdir / "gradcam_diagnostics.csv", index=False)

    # Save verification summary.
    summary_lines = [
        "VGG16 Grad-CAM verification summary",
        "===================================",
        f"Test images: {len(ds)}",
        f"Class order: {ds.classes}",
        f"Checkpoint: {checkpoint}",
        f"Reference predictions: {ref_csv}",
        "Checkpoint strict load: PASS",
        "Target layer: model.features[28] (last Conv2d)",
        f"Saved true-label agreement: {stats['true_label_match_fraction']:.6f}",
        f"Saved predicted-class agreement: {stats['predicted_class_match_fraction']:.6f}",
        f"Max probability absolute difference: {stats['max_probability_abs_difference']:.10f}",
        f"Mean probability absolute difference: {stats['mean_probability_abs_difference']:.10f}",
        "Grad-CAM implementation: direct split-forward, no backward hooks",
        "Grad-CAM attribution device: CPU",
        "Interpretation: attribution map only; not a tumor segmentation/localization ground truth.",
    ]
    (outdir / "verification_summary.txt").write_text("\n".join(summary_lines) + "\n")

    print("\n[PASS] Direct split-forward Grad-CAM completed without backward hooks.")
    print("Saved:")
    print(" ", outdir / "vgg16_verified_gradcam_correct.png")
    print(" ", outdir / "vgg16_verified_gradcam_errors.png")
    print(" ", outdir / "gradcam_diagnostics.csv")
    print(" ", outdir / "verification_summary.txt")
    print("\nIMPORTANT: A visually off-target CAM can be a genuine model shortcut/failure. It is not automatically a code error.")


if __name__ == "__main__":
    main()
