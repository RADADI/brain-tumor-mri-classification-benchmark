#!/usr/bin/env python3
"""Generate representative VGG16 qualitative + Grad-CAM examples.

Selection is intentionally not based on the most impressive examples:
- one correct example per class is chosen near the median confidence among
  correct predictions for that class;
- error examples prioritize glioma confusions and are chosen near the median
  confidence for each available error type.

Example:
python3 scripts/qualitative_gradcam.py \
  --test-dir DatasetNew/Testing \
  --checkpoint /path/to/vgg16_seed123_best.pth \
  --output figures/vgg16_qualitative_gradcam.png
"""

import argparse
from pathlib import Path
import math
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from PIL import Image
import matplotlib.pyplot as plt


def build_model(num_classes: int = 4):
    model = models.vgg16(weights=None)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    # Grad-CAM needs gradients through the final convolutional feature map.
    # torchvision VGG uses in-place ReLUs by default; those can conflict with
    # autograd hooks (especially on recent PyTorch/MPS). Make them non-inplace.
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False

    return model


def load_checkpoint(model, path, device):
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise RuntimeError("Checkpoint does not contain a recognizable state_dict.")
    # Remove common DataParallel prefix.
    obj = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in obj.items()}
    missing, unexpected = model.load_state_dict(obj, strict=False)
    if missing or unexpected:
        print("WARNING: checkpoint loaded with non-strict matching.")
        if missing:
            print("  Missing keys:", missing[:20])
        if unexpected:
            print("  Unexpected keys:", unexpected[:20])
    return model


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        # A tensor-level gradient hook avoids the view + in-place ReLU conflict
        # that can occur with register_full_backward_hook on VGG/MPS.
        self.fwd = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach().clone()
        if out.requires_grad:
            out.register_hook(self._save_tensor_gradient)

    def _save_tensor_gradient(self, grad):
        self.gradients = grad.detach().clone()

    def __call__(self, x, class_idx):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = logits[:, class_idx].sum()
        score.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam -= cam.min()
        denom = cam.max().clamp_min(1e-8)
        cam /= denom
        return cam.detach().cpu().numpy(), logits.detach()

    def close(self):
        self.fwd.remove()


def median_representative(rows):
    """Pick row with confidence closest to median confidence."""
    if not rows:
        return None
    med = float(np.median([r["confidence"] for r in rows]))
    return min(rows, key=lambda r: abs(r["confidence"] - med))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", default="figures/vgg16_qualitative_gradcam.png")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    test_dir = Path(args.test_dir)
    ckpt = Path(args.checkpoint)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not test_dir.exists():
        raise FileNotFoundError(test_dir)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    device = choose_device()
    print("Device:", device)

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    raw_tfm = transforms.Compose([transforms.Resize((224, 224))])
    ds = datasets.ImageFolder(test_dir, transform=tfm)
    print("Class order:", ds.classes)
    if len(ds.classes) != 4:
        raise RuntimeError(f"Expected four classes, found {ds.classes}")

    model = build_model(len(ds.classes))
    model = load_checkpoint(model, ckpt, device).to(device).eval()

    # Predict all images first.
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    rows = []
    offset = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu()
            conf, pred = probs.max(dim=1)
            for j in range(len(yb)):
                idx = offset + j
                rows.append({
                    "index": idx,
                    "path": ds.samples[idx][0],
                    "true": int(yb[j]),
                    "pred": int(pred[j]),
                    "confidence": float(conf[j]),
                })
            offset += len(yb)

    selected = []
    # One representative correct example per class.
    for c in range(len(ds.classes)):
        cand = [r for r in rows if r["true"] == c and r["pred"] == c]
        r = median_representative(cand)
        if r:
            selected.append(("Correct", r))

    # Prioritize clinically relevant observed error pattern: glioma -> meningioma,
    # then glioma -> no tumor, then meningioma -> glioma, then any remaining error.
    name_to_idx = {n.lower().replace(" ", "").replace("_", ""): i for i, n in enumerate(ds.classes)}
    def idx_for(*names):
        for n in names:
            key = n.lower().replace(" ", "").replace("_", "")
            if key in name_to_idx:
                return name_to_idx[key]
        return None

    glioma = idx_for("glioma")
    mening = idx_for("meningioma")
    notumor = idx_for("notumor", "no tumor", "no_tumor")
    priorities = []
    if glioma is not None and mening is not None:
        priorities.append((glioma, mening))
    if glioma is not None and notumor is not None:
        priorities.append((glioma, notumor))
    if mening is not None and glioma is not None:
        priorities.append((mening, glioma))

    used = {r["index"] for _, r in selected}
    for t, p in priorities:
        cand = [r for r in rows if r["true"] == t and r["pred"] == p and r["index"] not in used]
        r = median_representative(cand)
        if r:
            selected.append(("Error", r)); used.add(r["index"])
        if len(selected) >= 7:
            break

    # Fill to eight total with representative errors from remaining error groups.
    if len(selected) < 8:
        groups = {}
        for r in rows:
            if r["true"] != r["pred"] and r["index"] not in used:
                groups.setdefault((r["true"], r["pred"]), []).append(r)
        # Prefer larger error groups, avoiding cherry-picking by selecting group median confidence.
        for key, group in sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True):
            r = median_representative(group)
            if r:
                selected.append(("Error", r)); used.add(r["index"])
            if len(selected) >= 8:
                break

    selected = selected[:8]
    print("Selected cases:")
    for kind, r in selected:
        print(kind, Path(r["path"]).name, ds.classes[r["true"]], "->", ds.classes[r["pred"]], f"{r['confidence']:.3f}")

    # Last convolution layer in VGG16 features.
    conv_layers = [m for m in model.features.modules() if isinstance(m, nn.Conv2d)]
    cam_engine = GradCAM(model, conv_layers[-1])

    # 4 rows x 4 columns = two cases per row; each case has original + Grad-CAM.
    fig, axes = plt.subplots(4, 4, figsize=(13, 14))
    axes = axes.ravel()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for k, (kind, r) in enumerate(selected):
        img = Image.open(r["path"]).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)
        cam, logits = cam_engine(x, r["pred"])
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        disp = np.asarray(raw_tfm(img)).astype(np.float32) / 255.0
        a0 = axes[2*k]
        a1 = axes[2*k+1]
        a0.imshow(disp)
        a0.axis("off")
        a0.set_title(f"{kind}: true={ds.classes[r['true']]}\npred={ds.classes[r['pred']]} ({probs[r['pred']]*100:.1f}%)", fontsize=9)
        a1.imshow(disp)
        a1.imshow(cam, alpha=0.45, cmap="jet", vmin=0, vmax=1)
        a1.axis("off")
        a1.set_title("Grad-CAM", fontsize=9)

    for ax in axes[2*len(selected):]:
        ax.axis("off")

    fig.suptitle("Representative VGG16 qualitative predictions and Grad-CAM explanations", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=300, bbox_inches="tight")
    cam_engine.close()
    print("Saved:", out)


if __name__ == "__main__":
    main()
