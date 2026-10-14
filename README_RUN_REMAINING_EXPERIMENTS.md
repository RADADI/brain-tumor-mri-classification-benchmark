# IPAS Remaining Experiments — Run Order

These scripts finish the items requested by Dr. Alam **without inventing results**. Run them from:

`/Users/hemishradadiya/Downloads/Final Project Brain Tumor`

Keep `brain_tumor_publication_train_resume.py`, `DatasetNew/`, and `publication_outputs/` in that folder.

## 0. Install the small extra dependencies

```bash
python3 -m pip install fvcore ImageHash
```

## 1. Feature visualization (VGG16 seed 123)

Copy `feature_tsne_verified.py` into the project folder, then run:

```bash
python3 feature_tsne_verified.py \
  --test-dir DatasetNew/Testing \
  --checkpoint publication_outputs/benchmark_vgg16/seed_123/final_model.pth \
  --reference-predictions publication_outputs/benchmark_vgg16/seed_123/test_predictions.csv \
  --output-dir publication_outputs_ipas/feature_visualization
```

Expected files:
- `publication_outputs_ipas/feature_visualization/vgg16_feature_tsne.png`
- `vgg16_tsne_points.csv`
- `vgg16_feature_tsne_summary.json`

The script must report 100% prediction agreement with the saved seed-123 run before the plot is used in the paper.

## 2. Modern baselines: ConvNeXt-Tiny and Swin-Tiny

Copy `train_modern_baselines.py` into the project folder and run:

```bash
python3 train_modern_baselines.py \
  --train-dir DatasetNew/Training \
  --test-dir DatasetNew/Testing \
  --seeds 42 123 2026 \
  --num-workers 0 \
  --resume \
  --skip-integrity-check
```

This uses the same benchmark protocol as the original six models: ImageNet initialization, 20% stratified validation split per seed, batch size 16, phase 1 up to 15 epochs at 1e-3, phase 2 up to 10 epochs at 1e-5, Adam, cross entropy, ReduceLROnPlateau, early stopping, and validation macro OVR AUC for model selection.

Outputs go to `publication_outputs_modern/`.

If training is interrupted, rerun the same command with `--resume`.

## 3. Efficiency profile for all eight models

Only run this after Step 2 finishes all six new runs (2 models × 3 seeds).

```bash
python3 profile_efficiency.py \
  --old-root publication_outputs \
  --new-root publication_outputs_modern \
  --seed 123 \
  --output-dir publication_outputs_ipas/efficiency
```

This produces one consistent, in-memory, batch-1 timing protocol on the same Mac and reports:
- parameters
- FP32 parameter memory
- approximate forward GFLOPs
- mean/SD/median/p95 inference latency

It intentionally avoids disk I/O so the previous VGG16 storage-timeout outlier cannot contaminate the timing result.

## 4. External validation — BDNeuro-MRI

Use the independent four-class **BDNeuro-MRI: A Bangladeshi Clinical Brain Tumor MRI Dataset for Four-Class Deep Learning Classification**, Mendeley Data V7, DOI `10.17632/zwr4ntf94j.7`.

Official dataset page:
`https://data.mendeley.com/datasets/zwr4ntf94j/7`

Download and extract it. The official dataset has 5,941 images from Epic & CSCR Hospital, Bangladesh and a fixed test split of 889 images. We evaluate **only the external test split** and do not fine-tune on it.

Then copy `external_validate_bdneuro.py` into the project folder and run, replacing the path with the extracted folder:

```bash
python3 external_validate_bdneuro.py \
  --external-root "/PATH/TO/Brain_Tumor_MRI_Dataset_Final" \
  --internal-train DatasetNew/Training \
  --internal-test DatasetNew/Testing \
  --checkpoint-root publication_outputs/benchmark_vgg16 \
  --seeds 42 123 2026 \
  --output-dir publication_outputs_ipas/external_validation
```

The script performs SHA-256 cross-dataset overlap checking and pHash near-duplicate screening before reporting external metrics. If it finds overlap candidates, do not use the external-validation result in the paper until those candidates are reviewed.

## 5. Verify all requested experiments are complete

```bash
python3 combine_ipas_results.py
```

It will stop if any of the eight models lacks three-seed results, the efficiency table is incomplete, or external/feature results are missing.

If it prints:

`All requested experiment outputs are complete.`

then create one ZIP for review:

```bash
zip -r IPAS_completed_experiments.zip \
  publication_outputs_modern \
  publication_outputs_ipas
```

Upload `IPAS_completed_experiments.zip` to ChatGPT. The manuscript should only be updated to the final submission-ready version after these real outputs are checked.
