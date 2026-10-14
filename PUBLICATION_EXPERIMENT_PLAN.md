# Brain Tumor Publication Experiment Plan

## Goal
Create new, reproducible experimental evidence for a peer-reviewed manuscript. The old class report and poster are background only; publication results should come from fresh controlled runs.

## Step 1 — Dataset integrity
Run the SHA-256 duplicate check built into `brain_tumor_publication_train.py`.

If the script reports exact duplicate images across `Training` and `Testing`, review/remove leakage before using any results in the manuscript.

## Step 2 — Fair six-model benchmark
Run:

```bash
python brain_tumor_publication_train.py --mode benchmark --seeds 42
```

This compares:
- ResNet50
- InceptionV3
- EfficientNetB0
- DenseNet121
- VGG16
- VGG19

All models use the same baseline training logic. This produces the fair architecture benchmark.

## Step 3 — DenseNet121 ablation
Run:

```bash
python brain_tumor_publication_train.py --mode ablation --seeds 42
```

Ablation sequence:
- D0 baseline
- D1 + CLAHE/median filtering
- D2 + weighted sampling
- D3 + label smoothing 0.1
- D4 + AdamW and cosine annealing
- D5 + TTA (original + horizontal flip only; vertical flip is avoided because it is anatomically difficult to justify)

This establishes which additions actually improve performance.

## Step 4 — Diagnose InceptionV3
If InceptionV3 remains near chance-level performance, inspect:
- training/validation curves
- class mapping
- learning rate
- early stopping
- auxiliary-head behavior
- confusion matrix

Do not interpret a near-25% four-class result as normal architecture performance without investigation.

## Step 5 — Final 3-seed manuscript runs
After the one-seed runs are verified, run:

```bash
python brain_tumor_publication_train.py --mode all --seeds 42 123 2026
```

Use the generated `aggregate_mean_std.csv` and `manuscript_results_table.csv` for the paper.

## Core manuscript metrics
Report at minimum:
- Accuracy
- Macro precision
- Macro recall
- Macro F1
- Weighted F1
- Macro One-vs-Rest AUC
- Per-class precision/recall/F1
- Confusion matrix
- ROC curves
- Mean ± standard deviation over 3 seeds
- Parameter count
- Inference time

## Proposed paper framing
Working title:

**Enhanced Transfer Learning for Four-Class Brain Tumor MRI Classification: A Multi-Architecture Benchmark and Ablation Study**

Main contribution should be reproducible benchmarking + ablation + class-level analysis, not a claim that 91.94% is state of the art.

## After experiments
Once the clean result files are generated, prepare:
1. Results table
2. Ablation table
3. Confusion matrices
4. ROC figure
5. Training curves
6. Grad-CAM analysis for the final best model
7. Full manuscript
