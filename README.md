# Eight-Architecture Four-Class Brain Tumor MRI Benchmark

Research repository for the manuscript **Reproducible Deep Learning for Four-Class Brain Tumor MRI Classification**.

**Scientific status (October 2026): PROVISIONAL.** The original model benchmarks used image-level partitions subsequently found to have duplicate-image leakage. The original reported accuracy, F1, AUC, class-level comparisons, and ablation results **must not** be presented as leakage-free results. Corrected group-aware partitions and full reruns are pending.

## Model configurations

ImageNet-pretrained ResNet50, InceptionV3, EfficientNetB0, DenseNet121, VGG16, VGG19, ConvNeXt-Tiny, and Swin-Tiny. Common high-level evaluation protocol does not isolate backbone architecture alone: model-specific classifier heads and input sizes differ (InceptionV3 299×299; others 224×224).

## Data and provenance

Original training/testing directory: `DatasetNew/Training/<class>` and `DatasetNew/Testing/<class>` where `<class>` is `glioma`, `meningioma`, `notumor`, or `pituitary`. The locally evaluated copy contains 7,200 files (5,600 supplied training and 1,600 supplied testing, balanced by class), consistent in counts with Kaggle Brain Tumor MRI Dataset version 2: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset . **Exact download-version provenance is not independently verified.** Check source licenses and image redistribution rights before sharing raw MRI files. `DatasetNew_Clean_v2` represents exact-hash deduplication **only**, not a certified leakage-free dataset.

## Integrity audit and limitation

- The 7,200 original files had **187 redundant exact copies**, leaving **7,013 unique SHA-256 hashes**.
- Original saved image-level train/validation splits for seeds 42, 123, and 2026 contained **49, 51, and 56** cross-boundary exact duplicate-hash groups respectively.
- Original source Training vs Testing folder comparison found **zero exact SHA-256 cross-partition hashes**; this does not rule out perceptual or patient-level overlap.
- After exact deduplication and resplitting, pHash Hamming distance <= 5 screening identified **2,835 cross-split candidate pairs** (909 train–validation, 1,501 train–test, 425 validation–test), including **848 pHash-identical pairs**. Similarity is a screening signal, not verified patient identity.
- **Do not train final claimed leakage-free results on these unreviewed splits.** Adjudicate candidate matches and construct group-aware splits before retraining. Patient-level independence cannot be verified without patient IDs.

## External dataset screening

- BDNeuro-MRI (Mendeley Data v7, DOI `10.17632/zwr4ntf94j.7`): 889 screened test images, 472 exact SHA-256 match records, 605 external images with nearest pHash distance <= 5.
- Multi-Class Brain Tumor MRI: 1,137 screened images, 0 SHA-256 matches, 724 images with nearest pHash distance <= 5.

These datasets were **not** used to claim independent external performance.

## Main code entry points

- `brain_tumor_publication_train_resume.py`: original six-CNN benchmark and DenseNet121 ablation; `--help` prints runtime options.
- `train_modern_baselines.py`: ConvNeXt-Tiny and Swin-Tiny benchmark; `--help` prints runtime options.
- `prepare_clean_brain_dataset.py`: exact-hash deduplication and pHash candidate screening; **not** a validated group-aware split.
- `profile_efficiency.py`: model size and inference latency; uses Apple MPS synchronization where relevant and reports FP32 parameters in MiB.
- `verified_gradcam_vgg16.py` and `feature_tsne_verified.py`: retrospective qualitative interpretation; verification of checkpoint-consistent predictions is not attribution-map validation.
- `external_validate_bdneuro.py` and `external_validate_multiclass_mri.py`: external similarity/overlap screening.
- `combine_ipas_results.py`: combine pre-existing metrics.

**Reproduction caveat:** Check the exact dependency versions and available checkpoint/config/manifest paths before running. A complete environment lockfile and independently tested end-to-end reproduction are **not yet verified**. Older script names and paths may not line up with all directories after code was copied; do not treat these run commands as a successful reproduction record.

## Original benchmark rerun commands — NOT a corrected experiment

```bash
python brain_tumor_publication_train_resume.py --help
python train_modern_baselines.py --help
python prepare_clean_brain_dataset.py --help
```

These commands only inspect CLI options. Running the original training scripts with their default dataset paths repeats the original image-level splitting methodology and **does not resolve the leakage**.

## Outstanding requirements before a final manuscript claim

1. Establish and document dataset provenance, version, license, and integrity.
2. Review near-duplicate clusters, resolve any cross-label conflicts, form leakage-controlled groups, and verify group separation including held-out testing; document patient-independence limitations.
3. Save new class-count summaries, SHA-256 and pHash audits, stable group split manifests, all configuration JSONs, and environment lock/version outputs.
4. Retrain all eight architectures under the corrected splits across all reported seeds; rerun ablation if retaining its results.
5. Regenerate all benchmark tables, class metrics, confusion matrices, ROC plots, Grad-CAM examples and discussion where affected.
6. Independently test a clean setup from README and record reproducibility results.
7. Confirm venue review anonymity requirements before placing an identifying GitHub URL into the submission.

## Research/clinical use

Research only; this is not a clinically validated diagnostic device. Grad-CAM is qualitative and not validated against expert tumor annotations.
