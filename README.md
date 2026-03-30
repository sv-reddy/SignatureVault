# SignatureVault

**Offline Handwritten Signature Verification System**

SignatureVault is an end-to-end deep-learning pipeline for verifying handwritten signatures. It unifies multiple public datasets, extracts a rich 4-channel biometric feature representation, trains a Siamese-Transformer model (TAV-Net), verifies questioned signatures against a personal vault of genuine references, and produces explainability dashboards via Grad-CAM backbone attention heatmaps.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [End-to-End Workflow Summary](#2-end-to-end-workflow-summary)
3. [Architecture](#3-architecture)
4. [Repository Structure](#4-repository-structure)
5. [Datasets](#5-datasets)
6. [Pipeline — Step by Step](#6-pipeline--step-by-step)
   - [Step 1 — Unify Datasets](#step-1--unify-datasets-unify_datasetpy)
   - [Step 2 — Extract Features](#step-2--extract-features-extract_featurespy)
   - [Step 3 — Train TAV-Net](#step-3--train-tav-net-train_tavnetpy)
   - [Step 4 — Evaluate](#step-4--evaluate-evaluate_tavnetpy)
   - [Step 5 — Verify](#step-5--verify-verify_vaultpy)
    - [Step 6 — Forensic Explainability Report](#step-6--forensic-explainability-report-grad_campy)
   - [Auxiliary: Feature Visualization](#auxiliary-tool--feature-visualization-visualize_featurespy)
   - [Auxiliary: Cleaning Validation](#auxiliary-tool--cleaning-validation-check_image_cleaningpy)
7. [Model — TAV-Net](#7-model--tav-net)
8. [Scoring System](#8-scoring-system)
9. [Installation](#9-installation)
10. [Recent Optimizations & Improvements](#recent-optimizations--improvements)
11. [Quick Start](#11-quick-start)
12. [Configuration Reference](#12-configuration-reference)
13. [Output Files](#13-output-files)
14. [Dependencies](#14-dependencies)

---

## 1. Project Overview

Signature verification is a critical part of document authentication in banking, legal proceedings, and identity management. SignatureVault addresses the **offline** variant — where only a scanned image of the signature is available, with no pen-pressure or timing data.

**Key design decisions:**
- **Multi-dataset training** (CEDAR, BHSig260 Hindi, BHSig260 Bengali, GPDS, ICDAR2011, Independent) with 750+ writers to maximise generalisation.
- **4-channel feature tensor** (Shape, Pseudo-Pressure, Stroke Angle, Skeleton) encodes complementary biometric cues that are otherwise discarded by single-channel grayscale pipelines.
- **TAV-Net** combines a CNN backbone (ResNet-50), channel-and-spatial attention (CBAM), and a Transformer encoder to produce writer-discriminative 512-d L2-normalised embeddings.
- **Sub-Center ArcFace loss** with spherical margin ensures that intra-class variation (multiple signing styles per writer) is modelled without collapsing clusters.
- **Z-score adaptive thresholding** in the verifier means that a writer whose genuine signatures vary more gets a proportionally wider acceptance envelope — no fixed cosine cut-off is applied globally.
- **Grad-CAM explainability** shows which spatial regions of the signature the backbone attends to, broken down per biometric channel.

---

## 2. End-to-End Workflow Summary

This is the full project flow from raw datasets to final analysis outputs.

1. **Datasets used:** CEDAR, BHSig260 (Hindi), BHSig260 (Bengali), GPDS, ICDAR2011, and Independent.
2. **Unification (`unify_dataset.py`):** all raw images are copied into a single flat directory `DATA/unification_Data/` with canonical names `<DATASET>_<UID>_<G|F>_<NNN>.<ext>`, and metadata is written to `DATA/unification_Data/manifest.json`.
3. **Image cleaning (`extract_features.py`, phase 1):** illumination normalization, adaptive thresholding, ROI cropping with padding, letterbox resize, and stronger form-line removal (morphology + Hough long-line suppression).
4. **Feature extraction (`extract_features.py`, phase 2):** each cleaned signature is converted into a 4-channel tensor (Shape, Pseudo-Pressure, Stroke Angle, Skeleton) and saved as `.npy` in `DATA/process_data/`.
5. **Training (`train_tavnet.py`):** TAV-Net is trained on writer-disjoint 70/10/20 splits using Sub-Center ArcFace + APN bundles, producing `checkpoints/best_tavnet.pt` and `checkpoints/manifest.json`.
6. **Evaluation (`evaluate_tavnet.py`):** randomized vault protocol (5-8 genuine refs per trial) computes EER, AUC, Accuracy, Precision/Recall/F1, script-wise and questioned-type breakdowns, then saves `results/evaluate/testing.json`.
7. **Verification (`verify_vault.py`):** deployment-style verdicts with adaptive Z-score thresholding and vault scoring.
8. **Forensic Explainability (`grad_cam.py`):** Grad-CAM backbone attention heatmaps, per-channel attribution maps, contrastive analysis, and detailed metrics visualizations.
9. **Auxiliary tools:** `visualize_features.py` (1×5 feature quality panel) and `check_image_cleaning.py` (preprocessing validation).

---

## 3. Architecture

```
Input image (any size, any color)
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  extract_features.py  —  Phase 1: Preprocessing           │
│  BGR → LAB → L-channel → illumination normalisation       │
│  → Adaptive Gaussian Threshold → ROI crop → letterbox     │
│  Output: (384, 384) uint8 binary mask  (0=ink, 255=bg)    │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  extract_features.py  —  Phase 2: 4-Channel Extraction    │
│  Ch-0  Shape          Inverted binary ink mask            │
│  Ch-1  Pseudo-Pressure  GPU Gaussian blur σ=1.0 of Ch-0  │
│  Ch-2  Stroke Angle     Sobel arctan2 → [0, 255]          │
│  Ch-3  Skeleton         Zhang-Suen / morphological thin   │
│  Output: (4, 384, 384) uint8 .npy tensor                  │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  TAV-Net  (train_tavnet.py)                  │
│                                                           │
│  ResNet-50 (conv1: 4-ch, ImageNet weights)                │
│    layer1, layer2, layer3                                 │
│    layer4 (2048 feature maps, 12×12)                      │
│        │                                                  │
│        └──► CBAM attention                                │
│               Channel Attention (avg+max pool → MLP)      │
│               Spatial Attention (7×7 conv)                │
│        │                                                  │
│        ▼                                                  │
│  Flatten spatial: (B, 2048, 12, 12) → 144 tokens         │
│  + learnable cls token + learnable positional embedding   │
│        │                                                  │
│        ▼                                                  │
│  TransformerEncoder (1 layer, 8 heads, FFN=2048, Pre-LN)  │
│  Use cls token output → (B, 2048)                         │
│        │                                                  │
│        ▼                                                  │
│  Linear(2048→512) → BatchNorm1d(512) → L2 normalise       │
│  Output: 512-d unit-sphere embedding                      │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  verify_vault.py  —  4-Component Scoring                  │
│                                                           │
│  centroid_sim   (0.5)  cosine to mean vault embedding     │
│  subcenter_sim  (0.5)  cosine to nearest K-means center   │
│                                                           │
│  combined = weighted sum of the four components           │
│  z = (combined − vault_mean) / max(vault_std, 0.01)       │
│  GENUINE  if  z ≥ z_threshold   else  FORGERY             │
└───────────────────────────────────────────────────────────┘
```

---

## 4. Repository Structure

```
SignatureVault/
├── backend/
│   ├── unify_dataset.py            # Dataset unification (Step 1)
│   ├── extract_features.py         # 4-channel feature extraction (Step 2)
│   ├── train_tavnet.py # TAV-Net training (Step 3)
│   ├── evaluate_tavnet.py          # Rigorous evaluation (Step 4)
│   ├── verify_vault.py             # Signature verification (Step 5)
│   ├── visualize_features.py       # Feature quality visualiser (Step 6)
│   ├── tune_weights.py             # Grid search for optimal scoring weights
│   ├── check_image_cleaning.py     # Cleaning-phase visual checker
│   ├── grad_cam.py                 # Grad-CAM forensic report generator (Step 6)
│   ├── requirements.txt
│   ├── checkpoints/
│   │   ├── best_tavnet.pt         # Saved TAV-Net checkpoint
│   │   └── manifest.json           # Writer train/val/test split record
│   ├── sample_check/
│   │   ├── vault/                  # Genuine reference signatures
│   │   └── questioned/             # Questioned signatures to verify
│   └── results/
│       ├── cleaned_image/          # Cleaning-check visual outputs
│       ├── visualize_features/     # Feature panel PNG outputs
│       └── grad_cam/               # Forensic explainability report PNGs
└── DATA/
    ├── BHSig260/
    │   ├── Bengali/                # 100 Bengali writers
    │   └── Hindi/                  # 160 Hindi writers
    ├── CEDAR/                      # 55 Latin-script writers
    │   ├── full_org/
    │   └── full_forg/
    ├── GPDS/                       # 150 writers (subset)
    │   ├── genuine/
    │   └── forge/
    ├── unification_Data/           # Flat unified image directory (Step 1 output)
    └── process_data/               # .npy feature tensors (Step 2 output)
```

---

## 5. Datasets

| Dataset | Script | Language | Writers | UIDs | Genuine/writer | Forgeries/writer |
|---------|--------|----------|---------|------|----------------|-----------------|
| CEDAR | CEDAR | Latin | 55 | 101–155 | 24 | 24 |
| BHSig260 Hindi | BHSig260 | Hindi (Devanagari) | 160 | 201–360 | 24 | 24 |
| BHSig260 Bengali | BHSig260 | Bengali | 100 | 401–500 | 24 | 30 |
| GPDS | GPDS | Latin | 150 | 601–750 | 16 | Variable |
| ICDAR2011 | ICDAR | Latin | 64 | 801–869 | 12 | Variable |
| Independent | IND | Latin | 223 | 1001–1223 | 10 | 10 |

**Filename convention after unification:**
```
<DATASET>_<UID>_G_<NNN>.<ext>    genuine signature number NNN for writer UID
<DATASET>_<UID>_F_<NNN>.<ext>    forgery signature number NNN for writer UID
```
Examples: `BS_401_G_003.png`, `GPDS_601_F_012.png`, `ICDAR_849_G_011.png`, `IND_1001_F_004.jpg`.

**Writer split (writer-disjoint):**

| Split | Writers | Ratio |
|-------|---------|-------|
| Train | ~526 | 70% |
| Val | ~75 | 10% |
| Test | ~151 | 20% |

The exact split is recorded in `checkpoints/manifest.json` at training time, ensuring evaluation is always performed on writers the model has never seen.

---

## 6. Pipeline — Step by Step

### Step 1 — Unify Datasets (`unify_dataset.py`)

**Purpose:** Scan all six raw dataset directories, copy every genuine and forgery image into a single flat directory (`DATA/unification_Data/`), and rename files to the canonical `<DATASET>_<UID>_<G|F>_<NNN>.<ext>` convention. Generates `DATA/unification_Data/manifest.json` and `DATA/unification_Data/unification.log`.

**How to run:**
```bash
python unify_dataset.py
```
No arguments required. Paths are resolved relative to the script's parent directory.

**What it does internally:**
1. Iterates over CEDAR (`full_org/` → genuine, `full_forg/` → forgery), assigning UIDs 101–155.
2. Iterates over BHSig260 Hindi (UIDs 201–360) and Bengali (UIDs 401–500).
3. Iterates over GPDS (UIDs 601–750).
4. Iterates over ICDAR2011 flat folders `DATA/ICDAR2011/{id}/` and `DATA/ICDAR2011/{id}_forg/` (UIDs 801–869).
5. Iterates over Independent folders `DATA/Independent/{id}/` and `DATA/Independent/{id}_forg/` (UIDs 1001–1223).
6. Copies all discovered images to `unification_Data/` with canonical prefixed names and writes manifest/log.

---

### Step 2 — Extract Features (`extract_features.py`)

**Purpose:** Convert every unified image in `DATA/unification_Data/` into a `(4, 384, 384)` uint8 NumPy tensor, applying GPU-accelerated preprocessing, and save each as a `.npy` file in `DATA/process_data/`. Existing `.npy` files are skipped automatically (incremental mode).

**How to run:**
```bash
python extract_features.py
```

**Two-phase pipeline per image:**

**Phase 1 — Robust Preprocessing (Illumination Normalisation):**
1. Convert BGR → LAB, extract L channel (luminance only; eliminates ink-colour variation).
2. Divide-normalisation: `L / GaussBlur(L, σ=max_dim/4) × 128` — cancels slow illumination gradients and shadows.
3. Adaptive Gaussian Threshold (blockSize=41, C=10) → binary ink mask (0=ink, 255=background).
4. **Bounding-box ROI crop** with 15 px padding to remove excess whitespace and focus the model on stroke pixels.
5. Letterbox resize to 384×384, preserving aspect ratio (white padding).

**Recent extract_features updates (quality improvements):**
- **Form-line removal:** horizontal/vertical template lines are suppressed before channel extraction using:
  - wide morphological opening kernels (`max(31, dim//3)`),
  - HoughLinesP-based long-line detection,
  - mask dilation and subtraction from the ink mask.
- **Effect:** cleaner Shape/Skeleton channels and fewer box-line artifacts leaking into pseudo-pressure and angle channels.

**Phase 2 — 4-Channel Extraction (GPU-accelerated):**

| Channel | Name | Description | Values |
|---------|------|-------------|--------|
| 0 | Shape | Inverted binary ink mask | 0=bg, 255=ink |
| 1 | Pseudo-Pressure | GPU Gaussian blur σ=1.0 on Ch-0 | Stroke density |
| 2 | Stroke Angle | Sobel arctan2(Gy,Gx) → [0,255] | Orientation map |
| 3 | Skeleton | Zhang-Suen / morphological 1-px paths | Stroke topology |

**GPU backend priority:** cv2.cuda → CuPy → PyTorch CUDA → SciPy CPU. Detected once per spawned process and cached.

**Parallelism:** Uses `ProcessPoolExecutor` with up to 3 workers when GPU is active (to preserve VRAM), or all CPU cores on CPU-only mode.

---

### Step 3 — Train TAV-Net (`train_tavnet.py`)

**Purpose:** Train the Transformer-Attention-Vision Network (TAV-Net) on the processed feature tensors. Produces `checkpoints/best_tavnet.pt` (saved whenever validation EER improves) and `checkpoints/manifest.json` (writer train/val/test split).

**How to run:**
```bash
# Default (50 epochs, batch=8, accum=4 → effective batch=32 APN-bundles = 256 tensors)
python train_tavnet.py

# Resume from checkpoint
python train_tavnet.py --resume checkpoints/best_tavnet.pt

# Smaller physical batch with equivalent effective batch
python train_tavnet.py --batch-size 4 --accum-steps 8

# Enable hard-negative mining from epoch 5
python train_tavnet.py --hard-mining
```

**Training details:**

- **Loss Function:** Sub-Center ArcFace (angular margin m=0.55 radians ≈ 31.5°, logit scale s=64.0, K=7 sub-centers per class). Applied exclusively to genuine training samples. Sub-centers naturally model multiple distinct signing styles per writer without requiring explicit style labels.
  - **Mathematical form:** 
    ```
    L_ArcFace = -log( exp(s·cos(θ + m)) / (exp(s·cos(θ + m)) + Σ_{j≠i} exp(s·cos(θ_j))) )
    ```
    where θ is the angle between the embedding and the nearest sub-center of its class.
  - **Triplet Loss (complementary):** Batch-hard softplus triplet loss on hard-negatives to enforce relative embeddings distances.
    - **Warmup:** `triplet_weight = min(1.0, (epoch-1)/10.0)` scales from 0→1 over first 10 epochs; prevents loss-component conflict.
    - **Margin decay:** Triplet margin decays from 0.30 (epoch 1) → 0.05 (epoch 50) via formula `margin(t) = 0.30 - 0.25 × (t-1)/49` to harmonise with ArcFace's spherical boundaries late in training.
  - **Hard-negative mining (optional HNM):** If enabled, skilled forgeries sampled from per-writer top-10 hardest pool, activating after epoch 5.

- **APN-Bundle curriculum (8 tensors per writer per step):**
  - **Epochs 1–10:**
    - Slot 0: Anchor (genuine, writer W)
    - Slots 1–3: Positives ×3 (genuine, writer W)
    - Slots 4–5: Skilled negatives ×2 (writer W forgeries, or top-10 hardest if HNM)
    - Slots 6–7: Random negatives ×2 (genuine, writer W' ≠ W)
  - **Epoch 11+:**
    - Slot 0: Anchor (genuine, writer W)
    - Slots 1–3: Positives ×3 (genuine, writer W)
    - Slots 4–7: Skilled negatives ×4 (writer W forgeries; top-10 hardest if HNM active)
  - **Effect:** Provides curriculum learning from general writing patterns (epochs 1–10) to fine-grained forgery discrimination (epochs 11+).

- **Data augmentation (training split only):** Applied per-sample in `APNBundleDataset(..., augment=True)` for train loader; validation loader uses `augment=False`. Augmentations operate on float tensors of shape (4, 384, 384) ∈ [0,1].

**Augmentation pipeline (exact order in code):**

| Stage | Transform | Probability | Parameters / Notes |
|-------|-----------|-------------|-------------------|
| 1 | Elastic deformation | 0.50 | `_elastic_deform(alpha=50.0, sigma=5.0)` dense displacement field for natural handwriting variation |
| 2 | Stroke morphology | 0.30 | Random dilate/erode on Channels 0 (Shape) and 3 (Skeleton) to simulate pen-thickness variation |
| 3 | Random rotation | 0.70 | `RandomRotation(degrees=30, interpolation=BILINEAR)` for document skew up to ±30° |
| 4 | Random affine | 1.00 ← always | `RandomAffine(degrees=0, scale=(0.8,1.2), shear=10, interpolation=BILINEAR)` for tilt/slant/zoom variation |
| 5 | Random perspective | 0.30 | `RandomPerspective(distortion_scale=0.3, p=1.0)` for 3D viewing angle variation |
| 6 | Channel dropout | 0.15 | Randomly zero one of 4 channels to encourage cross-channel robustness without style information |
| 7 | Random erasing | 0.50 | `RandomErasing(scale=(0.02,0.10), ratio=(0.3,3.3))` to simulate stroke fragmentation and ink dropout |
| 8 | Additive noise | 0.30 | Gaussian noise `std=0.05`;  models scan-acquisition noise; output clamped to [0,1] |

**Implementation notes:**
- Augmentations are applied on float tensors of shape (4, 384, 384) after loading .npy and scaling by 1/255.
- If torchvision transform classes are unavailable, geometric augmentations are skipped gracefully (except Noise which has CPU fallback).
- This stochastic policy improves robustness to writing-style drift, document distortions, and scan noise without changing writer identity labels.

- **Optimiser:** AdamW (lr=1e-4, wd=5e-4 weight decay, betas=(0.9, 0.999), eps=1e-8).

- **LR Scheduling:** ReduceLROnPlateau (mode='min' on val_eer, factor=0.5, patience=3, min_lr=1e-6) monitors validation EER and reduces learning rate by 50% whenever improvement stalls for 3 consecutive epochs.

- **Gradient accumulation:** 4 steps (effective batch = 8 writers × 32 tensors/writer = 256 samples, allowing large effective batch on modest VRAM).

- **Checkpoint saving:** Model saved to `checkpoints/best_tavnet.pt` when `is_best = (val_eer < best_eer) and (tar_at_08 ≥ 0.80)`, ensuring only production-ready models with both high accuracy AND high genuine acceptance are retained.

- **Validation metrics:** 
  - **val_loss:** batch-hard softplus triplet loss on validation split
  - **val_EER:** Equal Error Rate (threshold where FAR = FRR)
  - **Forg_EER:** EER computed only on skilled forgery impostors vs. genuine
  - **Rand_EER:** EER on random-writer impostors vs. genuine
  - **TAR@0.80:** True Accept Rate (%) at cosine similarity ≥ 0.80

- **Writer split:** 70/10/20 writer-disjoint train/val/test; split recorded in `checkpoints/manifest.json` to ensure held-out evaluation on writers never seen during training.

---

### Step 4 — Evaluate (`evaluate_tavnet.py`)

**Purpose:** Rigorous evaluation of a trained checkpoint on held-out test writers using a vault simulation protocol. Reports EER, ROC-AUC, Accuracy, Precision/Recall/F1 overall and script-wise, plus questioned-type diagnostics. Also writes a JSON summary to `results/evaluate/testing.json`.

**How to run:**
```bash
python evaluate_tavnet.py --checkpoint checkpoints/best_tavnet.pt

# Custom checkpoint
python evaluate_tavnet.py --checkpoint checkpoints/my_model.pt

# More signing-style sub-centers per writer vault
python evaluate_tavnet.py --n-subcenters 5

# More randomized vault/questioned trials per writer
python evaluate_tavnet.py --vault-trials 5
```

**Evaluation protocol:**

1. Load test writer split from `checkpoints/manifest.json` (writers held-out from training).
2. Preload all test signatures and compute embeddings with TAV-Net in batches (progress bar shown).
3. **Per-writer random sampling** (3 trials per writer by default):
   - Sample random vault of 5–8 genuine signatures from the test writer.
   - For each vault, score three questioned types:
     - **Genuine (same writer):** Other genuine signature from same writer.
     - **Skilled forgery:** Forgery signature by a skilled forger targeting this writer.
     - **Random impostor:** Genuine signature from a different writer (negative control).
4. **Scoring per trial:**
   - Compute vault Leave-One-Out (LOO) statistics: vault_mean, vault_std via held-out scoring.
   - Compute combined score for each questioned signature using 2-component weighting (centroid 0.50, subcenter 0.50).
   - Compute z-score: `z = (combined - vault_mean) / max(vault_std, 0.01)`.
5. **Threshold search:** Tune cosine similarity threshold via ROC curve to find EER (Equal Error Rate).
6. **Metrics reported:**
   - **Overall:** EER, AUC (ROC), Accuracy, Precision, Recall, F1-score
   - **Script-wise breakdown:** Latin, Hindi, Bengali (if multiple scripts present)
   - **Questioned-type summary:** separate metrics for genuine/forgery/random-impostor questions
   - **Special cases:** Hardest writers (highest FRR at thresholds), outlier analysis
7. Save structured summary to `backend/results/evaluate/testing.json` (detailed per-writer/per-trial metrics, aggregated statistics, threshold values).

---

### Step 5 — Verify (`verify_vault.py`)

**Purpose:** Given a folder of genuine reference signatures (the "vault") and one or more questioned signatures, produce a GENUINE / FORGERY verdict with a confidence Z-score for each questioned sample. Results are saved as JSON.

**How to run:**
```bash
# Single questioned image
python verify_vault.py --vault path/to/vault/ --questioned path/to/sig.png

# Entire folder of questioned images
python verify_vault.py --vault path/to/vault/ --questioned path/to/questioned_folder/

# Tune strictness (more negative = more lenient)
python verify_vault.py --vault vault/ --questioned q.png --z-threshold -1.5

# Adjust number of signing-style sub-centers
python verify_vault.py --vault vault/ --questioned q.png --n-subcenters 5

# Save to specific JSON file
python verify_vault.py --vault vault/ --questioned q.png --output results/run1.json
```

**Scoring pipeline:**

1. **Embed all vault signatures** (genuine references) and the questioned signature using TAV-Net (with automatic mixed precision on CUDA).
2. **Vault centroid** — mean L2-normalised embedding of all vault samples.
3. **Sub-centers (signing styles)** — spherical K-means (default k=3, max_k adaptively selected via silhouette score) clusters the vault to capture natural variation in a writer's signature style without requiring explicit style labels.
4. **Leave-One-Out (LOO) calibration** — each vault sample is scored against the centroid/sub-centers computed from remaining vault samples, establishing `vault_mean` and `vault_std` — a writer-adaptive baseline for intra-class consistency.
   - **Formula:** For each i, leave vault[i] out, compute combined score against remaining vault embeddings, accumulate LOO scores.
   - **Robust statistics:** If fewer than 2 LOO scores available, use fallback values (vault_mean=0.70, vault_std=0.05).
5. **2-Component score for questioned signature:**

| Component | Weight | Description |
|-----------|--------|-------------|
| centroid_sim | 0.50 | Cosine similarity to vault centroid (global writer identity) |
| subcenter_sim | 0.50 | Cosine similarity to nearest signing-style sub-center (style-specific score) |

6. **Z-score adaptive threshold:**

```
combined = 0.50 · centroid_sim + 0.50 · subcenter_sim

acceptance_threshold = vault_mean - vault_std

verdict = GENUINE  if  combined ≥ acceptance_threshold  else  FORGERY

z_score = (combined - vault_mean) / max(vault_std, 0.01)  [diagnostic metric]
```

**Adaptive thresholding rationale:** A stylistically consistent writer achieves high vault_mean and low vault_std, tightening the acceptance envelope. A writer with naturally variable signatures has wide tolerance automatically. This writer-centric approach avoids global fixed thresholds.

**Output JSON structure:**
```json
{
  "checkpoint": "checkpoints/best_tavnet.pt",
  "vault_size": integer,
  "vault_mean": float,
  "vault_std": float,
  "acceptance_threshold": float,
  "results": [
    {
      "questioned_file": "string",
      "centroid_sim": float,
      "subcenter_sim": float,
      "combined_score": float,
      "z_score": float,
      "verdict": "GENUINE" | "FORGERY",
      "individual_vault_similarities": { "vault_001.png":  +0.854, ... }
    }
  ]
}
```

---

### Step 6 — Forensic Explainability Report (`grad_cam.py`)

**Purpose:** Generate forensic-grade XAI visualization reports for TAV-Net signature verification with Grad-CAM backbone attention heatmaps, per-channel attribution maps, contrastive difference maps, attention rollout, and comprehensive metrics tables with mathematical symbols and formulas.

**How to run:**

```bash
# Sample-dir mode (requires vault/ and questioned/ subfolders with images or .npy tensors)
python grad_cam.py --sample-dir sample_check

# Writer-ID mode (scans DATA/process_data for .npy tensors)
python grad_cam.py --writer-id 401

# Explicit questioned sample
python grad_cam.py --writer-id 401 --questioned path/to/sig.npy

# Custom checkpoint and output directory
python grad_cam.py --writer-id 401 \
    --checkpoint checkpoints/best_tavnet.pt \
    --out-dir results/grad_cam/

# Force CPU, disable attention rollout, adjust heatmap opacity
python grad_cam.py --sample-dir sample_check --cpu --no-rollout --alpha 0.70

# Set Z-score threshold and number of sub-centers
python grad_cam.py --writer-id 401 --z-threshold -1.5 --n-subcenters 5
```

**Evidence report pipeline:**

1. **Load questioned signature** and vault (genuine reference) embeddings.
2. **Compute vault statistics:**
   - Centroid: mean L2-normalised embedding of vault samples.
   - Sub-centers: spherical K-means (default K=3) for signing-style clusters.
   - Leave-One-Out (LOO) calibration: per-vault-sample scoring against remaining vault samples produces `vault_mean` and `vault_std`.
3. **Grad-CAM computation (questioned signature):**
   - Compute embedding and cosine similarity to vault centroid (differentiable).
   - Backpropagate through TAV-Net to `layer4` activations.
   - Weight feature maps by gradient means, ReLU, bilinear-upsample to 384×384.
   - Normalise to [0, 1].
4. **Best-matched genuine selection:** Find the vault sample with highest cosine similarity to questioned, run Grad-CAM on that genuine signature.
5. **Contrastive Δ-Map:** `vault_heatmap - questioned_heatmap` mapped to [0, 1] via RdBu_r colormap (Red = vault feature, Blue = questioned anomaly).
6. **Per-channel attribution maps (4 channels):** For each channel (Shape, Pressure, Angle, Skeleton):
   - Zero all other channels in tensor.
   - Compute gradient × input saliency: `∂(cosine_similarity)/∂input_c × |input_c|`.
   - Normalise to [0, 1].
7. **Per-channel similarity analysis:** Compute cosine similarity between questioned and reference channel attribution maps for 4 signature channels.
8. **Report layout (Faculty-Presentation Style):** GridSpec 3 rows × 4 columns structure:
   - **Row 0 (Raw Signatures):** Questioned signature vs. Best-matched authentic signature.
   - **Row 1 (Grad-CAM overlays):** Questioned heatmap vs. Authentic heatmap (Jet colormap).
   - **Row 2 (Analysis panels):**
     - **Channel Similarity Table:** Cosine similarity scores between the questioned and authentic signature broken down by Shape, Pressure, Angle, and Skeleton.
     - **Contrastive Map:** RdBu_r colormap (Red = authentic feature, Blue = questioned anomaly).
     - **Global Contrast Index Map:** YlOrRd colormap showing absolute difference magnitude.
     - **Verdict Box:** Mathematical breakdown of centroid similarity, subcenter similarity, vault thresholds, and the final Z-Score verdict.

9. **Verdict declaration:** GENUINE if Z ≥ z_threshold, else FORGERY (colour-coded symbol and text on report).


**Key computed metrics in report:**

- **Centroid similarity:** cosine(questioned_embedding, mean_vault_embedding)
- **Subcenter similarity:** cosine(questioned_embedding, nearest_kmeans_center)
- **Combined score:** (0.5 × centroid_sim) + (0.5 × subcenter_sim)
- **Vault LOO mean:** average combined score over vault samples when each is scored against remaining samples
- **Vault LOO std:** standard deviation of LOO combined scores
- **Z-score:** (combined − vault_mean) / max(vault_std, 0.01) — standardised decision metric
- **Per-channel similarity:** cosine similarity between questioned and reference channel attribution maps (Shape, Pressure, Angle, Skeleton)

**Output file:**
- `results/grad_cam/evidence_report_<writer_id>_<questioned_stem>_<timestamp>.png` at 200 DPI (publication quality).

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--sample-dir` | — | Root directory with vault/ and questioned/ subfolders |
| `--writer-id` | — | Writer UID (scans DATA/process_data/) |
| `--questioned` | auto | Path to questioned signature image or .npy file |
| `--checkpoint` | checkpoints/best_tavnet.pt | TAV-Net checkpoint |
| `--out-dir` | results/grad_cam/ | Output directory for PNG reports |
| `--alpha` | 0.55 | Heatmap overlay opacity (0=transparent, 1=opaque) |
| `--no-rollout` | False | Skip Transformer Attention Rollout computation |
| `--max-vault` | 8 | Maximum genuine samples for vault Grad-CAM averaging |
| `--cpu` | False | Force CPU inference |
| `--z-threshold` | -1.0 | Z-score acceptance threshold |
| `--n-subcenters` | 3 | Number of K-means signing-style sub-centers per writer |

---

### Auxiliary Tool — Feature Visualization (`visualize_features.py`)

**Purpose:** Quality-verification tool that renders a 1×5 figure panel for one input file (`.npy` tensor or image path), letting you inspect `cleaned_image` and all 4 feature channels in one view.

**How to run:**
```bash
# Direct image path
python visualize_features.py --file "sample_check/questioned/12.jpeg"

# Direct .npy path
python visualize_features.py --file "../DATA/process_data/BS_401_G_001.npy"
```

**Panel layout:**

| Panel | Content | Colormap | Description |
|-------|---------|----------|-------------|
| Ax-0 | Cleaned image | gray | Binary ink mask after Phase-1 preprocessing |
| Ax-1 | Ch-0 Shape | gray | Inverted binary ink mask (ink=255, bg=0) |
| Ax-2 | Ch-1 Pseudo-Pressure | inferno | Gaussian blur of Shape → stroke density |
| Ax-3 | Ch-2 Stroke Angle | hsv | Sobel arctan2 orientation map [0, 255] |
| Ax-4 | Ch-3 Skeleton | gray | Zhang-Suen morphological 1-px thinning |

Each channel subplot carries static min/max/mean statistics overlay and a colourbar. Console output prints: file name, tensor shape, source path, and saved output location.

Output is saved to `backend/results/visualize_features/vf_<stem>_<timestamp>.png` at 200 DPI.

---

### Auxiliary Tool — Cleaning Validation (`check_image_cleaning.py`)

**Purpose:** Quick visual validation of Phase-1 preprocessing quality for one image — side-by-side comparison of original vs. cleaned output.

**How to run:**
```bash
python check_image_cleaning.py --file-name "sample_check/questioned/12.jpeg"
```

**Output:**
- A 1×2 comparative report panel: `Original Image` and `Cleaned Output (Phase-1 Binary Mask)`.
- White figure background with black borders around both image panels.
- Saved output: `backend/results/cleaned_image/clean_<stem>_<timestamp>.png` at 200 DPI.

**Terminal output:**
- Input image path and shape
- Output binary mask shape
- Saved PNG report pathname

---

## 7. Model — TAV-Net

```
TAVNet(embed_dim=512)
├── backbone: ResNet-50
│     conv1: Conv2d(4, 64, 7×7, stride=2)  ← 4-channel input
│     layer1, layer2, layer3  (frozen-optionally)
│     layer4: 2048-channel 7×7 feature maps
│
├── cbam: CBAM
│     ChannelAttention: AvgPool + MaxPool → MLP(2048→128→2048) → Sigmoid
│     SpatialAttention: concat(avg,max) → Conv2d(2,1,7×7) → Sigmoid
│
├── cls_token: LearnableEmbedding(1, 2048)
├── pos_enc: LearnableEmbedding(50, 2048)
│
├── transformer: TransformerEncoder
│     1 layer | 8 heads | dim_feedforward=2048 | Pre-LN | dropout=0.1
│
└── head:
      Linear(2048, 512) → BatchNorm1d(512) → L2 normalise
```

**Checkpoint format (`best_tavnet.pt`):**
```python
{
    "epoch":       int,          # epoch at which best EER was achieved
    "model_state": OrderedDict,  # TAVNet.state_dict()
    "best_eer":    float,        # best validation EER seen so far
    "embed_dim":   int,          # embedding dimension (default 512)
}
```

---

## 8. Scoring System

The verification scoring system is designed around two principles:

1. **Multi-cue fusion:** No single cosine similarity metric is sufficient. The 4-component weighted score captures global writer identity (centroid), style clusters (subcenter), robust central tendency (median), and a soft veto against extreme dissimilarity (min).

2. **Writer-adaptive thresholding:** The Z-score normalises the combined score by each writer's own internal consistency (`vault_mean`, `vault_std` from Leave-One-Out). This automatically compensates for writers with naturally variable signatures (wider tolerance) and writers with very consistent signatures (tighter tolerance).

**ArcFace Sub-Centers (training):** Each writer is represented by K=7 sub-centers in embedding space. The loss pulls genuine embeddings towards the nearest sub-center rather than a single class prototype, naturally handling writers who switch between multiple dominant signing styles.

---

## 9. Installation

### Requirements

- Python 3.11+
- CUDA 12.x (optional but recommended for GPU-accelerated feature extraction and training)
- 6 GB+ VRAM for training with default batch size

### Setup

```bash
# Clone / download the project
cd "e:\Major Project\SignatureVault"

# Create and activate virtual environment
python -m venv backend/.venv
backend\.venv\Scripts\activate        # Windows
# source backend/.venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r backend/requirements.txt
```

### Key dependencies

| Package | Purpose |
|---------|---------|
| torch | TAV-Net training and inference |
| torchvision | ResNet-50 backbone |
| opencv-python | Image I/O and GPU-accelerated preprocessing |
| cupy-cuda12x | GPU numerical primitives (optional) |
| numpy | Array operations |
| scikit-image | Zhang-Suen skeletonization |
| scikit-learn | K-means, ROC/EER metrics |
| scipy | CPU fallback Gaussian/Sobel |
| matplotlib | Visualisation and heatmap dashboards |
| pillow | Robust image loading |
| tqdm | Progress bars |

---

## Recent Optimizations & Improvements

This section documents the system-level improvements made to maximise accuracy and stability:

### Training Improvements

**1. Augmentation Tuning**
- **Gentler geometric transforms:** Rotation 10° (p=0.5), scale 0.9-1.1, shear 5° to preserve biometric integrity
- **Reduced occlusion:** Random erasing p=0.2, elastic deformation p=0.2
- **Removed channel dropout:** All 4 channels (Shape, Pressure, Angle, Skeleton) are now preserved during training to maintain complementary biometric signals
- **Effect:** Stabilised gradient flow and prevented embedding space collapse in late training

**2. Triplet Loss Warmup**
- **Explicit warmup schedule:** `triplet_weight = min(1.0, (epoch-1)/10.0)` scales triplet loss from 0→1 over 10 epochs
- **Motivation:** Prevents loss component conflicts; ArcFace trains first on consistent class centroids, then triplet loss bootstraps cluster refinement
- **Effect:** Smoother convergence and better intra-class heterogeneity modelling

**3. Adaptive Learning Rate Scheduling**
- **ReduceLROnPlateau scheduler:** Replaces fixed CosineAnnealingLR; reduces LR by 50% when validation EER plateaus for 3 consecutive epochs
- **Motivation:** Learning rate adapts to actual performance, not a predefined schedule
- **Configuration:** mode='min' on val_eer, factor=0.5, patience=3

**4. Strict Checkpoint Saving Logic**
- **Dual condition:** Model saved only when `is_best = (val_eer < best_eer) and (tar_at_08 ≥ 0.80)`
- **Fallback:** First model in training is always saved to ensure initial weights are preserved
- **Motivation:** Prevents pathological late-epoch configurations with improved EER but collapsed True Accept Rate (TAR)
- **Effect:** Only production-ready models (high accuracy AND high genuine acceptance) are retained

### Evaluation Improvements

**5. Test-Time Augmentation (TTA) at Dataset Level**
- **3-view stacking:** `EvaluateDataset` returns stacked tensors (3, 4, 384, 384) with original, +5°, −5° rotations
- **Efficient batch processing:** Single batch forward pass through model with reshape trick: (Batch, 3, 4, 384, 384) → (Batch×3, 4, 384, 384) → embed → reshape & average
- **Effect:** ~2-3% EER improvement vs. single-view inference; reduces embedding variance across writing-style variation

**6. Weight Optimization Grid Search**
- **4-component scoring:** Automated tool (`tune_weights.py`) evaluates 1200+ weight combinations for the scoring formula:
  - `score = w_centroid·c + w_subcenter·s + w_median·m + w_min·min`
  - Constraint: weights sum to 1.0
- **Output:** Top 5 configurations ranked by (EER ascending, Accuracy descending) with copy-paste-ready Python dictionaries
- **Mathematics:** Uses sklearn `roc_curve` with correct semantics for similarity scores (higher = more genuine)

### Advanced Optimizations (Phase 2)

**7. Resolution Bump: 384×384 Tensors (Micro-Detail Unlock)** 
- **Concept:** The original pipeline at 224×224 resolution heavily compressed long signatures, causing loss of microscopic hesitation marks, micro-tremors, blunt pen-lifts, and stroke velocity variation—skilled forgeries appeared perfect globally. The current system upgrades to **384×384 tensors** throughout the pipeline (feature extraction, training, evaluation, and inference) to capture these micro-level biometric details that expose forgeries at the finest granule.
- **Implementation:**
  - **Current system:** 384×384 tensors → 12×12 ResNet layer4 tokens (144 total)
  - Positional embeddings, Transformer input, and all downstream processing adjusted to handle 144 tokens
  - All pipeline scripts (`extract_features.py`, `train_tavnet.py`, `evaluate_tavnet.py`) process 384×384 tensors by default
- **Architecture compatibility:** GeM pooling and Transformer naturally handle variable token counts; architecture is fully backward-compatible
- **Expected improvement:** 2-5% EER reduction by capturing fine-grained forgery artifacts; particularly effective for distinguishing skilled forgeries

**8. Triplet Margin Decay (Loss Harmonization)**
- **Concept:** ArcFace ($m=0.55$) and Triplet Loss ($margin=0.30$) initialize together. Early in training, this is beneficial. Late in training, they fight each other: triplet loss constantly pushes the hardest forgery away, potentially warping the spherical boundaries ArcFace is building.
- **Implementation:**
  - Triplet margin dynamically decays from 0.30 (epoch 1) → 0.05 (epoch 50)
  - Formula: `margin(t) = 0.30 - 0.25 × (t-1)/49`
  - Triplet loss still scales by warmup weight (0→1 over 10 epochs)
- **Training dynamics:**
  - Epochs 1–10: Triplet loss gradually activates; ArcFace dominates; margin = 0.30–0.28
  - Epochs 10–30: Both losses fully active; margin decays to 0.13; hard negatives separate classes
  - Epochs 30–50: Triplet margin → 0.05; ArcFace tightens spherical boundary; triplet loss provides fine stability
- **Expected improvement:** 1-3% EER reduction by preventing loss-component conflicts and allowing ArcFace to round out clusters without distortion

**9. Dynamic Sub-Center Vaults (Writer-Adaptive Clustering)**
- **Concept:** Current vault protocol forces K-means to find fixed $K=3$ or $K=7$ sub-centers for every writer. However, human writers vary: Writer A may have 1 consistent signature style, while Writer B has 4 distinct styles. Forcing K-means to find 7 clusters for a 1-style writer creates artificial noise and inflates vault_std, making the decision boundary slack.
- **Implementation:**
  - New function `_dynamic_kmeans_subcenters()` compares Silhouette Scores for K ∈ {1, 2, 3}
  - Automatically selects K that maximizes cluster cohesion and separation for each writer's vault
  - Uses sklearn's `silhouette_score` metric (cosine distance)
  - Applied to both main vault and Leave-One-Out (LOO) validation
- **Execution:**
  - Per-writer clustering is computed once per trial (minimal overhead)
  - K ranges from 1 (perfectly consistent writer) to 3 (highly variable writer)
  - Silhouette Score naturally balances within-cluster compactness and between-cluster separation
- **Expected improvement:** 1-2% EER reduction + more robust Z-score thresholds by matching K to actual signature variability per writer

### Integration

All optimizations are integrated into the default pipeline:
- **Feature extraction:** `python extract_features.py` now generates (4, 384, 384) tensors with preserved microscopic biometric detail
- **Training:** `python train_tavnet.py` automatically applies triplet margin decay (no configuration needed); processes 384×384 tensors with 144 Transformer tokens
- **Evaluation:** `python evaluate_tavnet.py` applies dynamic K-means selection via Silhouette Score; TTA processes enlarged tensors for reduced embedding variance
- **Optimization:** `python tune_weights.py` works with enhanced representations to find optimal component weights on high-resolution feature space

---

## 11. Quick Start

```bash
cd backend

# Step 1 — Unify all dataset images into a flat directory
python unify_dataset.py

# Step 2 — Extract 4-channel feature tensors (GPU auto-detected)
python extract_features.py

# Step 3 — Train TAV-Net (50 epochs, saves best checkpoint automatically)
python train_tavnet.py

# Step 4 — Evaluate on held-out test writers
python evaluate_tavnet.py

# Step 5 — Verify a questioned signature against a vault
python verify_vault.py --vault sample_check/vault --questioned sample_check/questioned

# Optional — Visually inspect a feature tensor (quality check)
python visualize_features.py --file "sample_check/questioned/12.jpeg"

# Step 6 — Generate forensic explainability report (Grad-CAM + metrics)
python grad_cam.py --sample-dir sample_check
```

---

## 12. Configuration Reference

### `train_tavnet.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 8 | Physical batch size (APN-bundles) |
| `--accum-steps` | 4 | Gradient accumulation steps |
| `--lr` | 1e-4 | AdamW learning rate |
| `--wd` | 5e-4 | AdamW weight decay |
| `--embed-dim` | 512 | Embedding dimension |
| `--arcface-m` | 0.55 | ArcFace angular margin (radians) |
| `--arcface-s` | 64.0 | ArcFace logit scale factor |
| `--arcface-k` | 7 | Sub-centers per writer class |
| `--num-workers` | 2 | DataLoader worker processes |
| `--hard-mining` | False | Enable top-10 hard-negative mining |
| `--hard-mining-start` | 5 | Epoch when HNM activates |
| `--resume` | None | Path to checkpoint to resume from |
| `--processed-dir` | auto | Override path to process_data/ |

### `evaluate_tavnet.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | checkpoints/best_tavnet.pt | TAV-Net checkpoint path |
| `--n-subcenters` | 3 | K-means sub-center count per vault |
| `--vault-trials` | 3 | Randomized vault/questioned trials per test writer |
| `--seed` | 42 | Random seed for vault sampling |

### `verify_vault.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--vault` | required | Folder of genuine reference signatures |
| `--questioned` | required | Path to questioned image file or folder |
| `--checkpoint` | checkpoints/best_tavnet.pt | TAV-Net checkpoint |
| `--z-threshold` | -1.0 | Z-score acceptance threshold (more negative = lenient) |
| `--n-subcenters` | 3 | K-means signing-style sub-center count |
| `--output` | auto | Custom JSON output path |

### `grad_cam.py` (Forensic Report Generator)

| Argument | Default | Description |
|----------|---------|-------------|
| `--sample-dir` | — | Root directory with vault/ and questioned/ subfolders |
| `--writer-id` | — | Writer UID (scans DATA/process_data/) |
| `--questioned` | auto | Path to questioned signature (image or .npy) |
| `--checkpoint` | checkpoints/best_tavnet.pt | TAV-Net checkpoint |
| `--out-dir` | results/grad_cam/ | Output PNG directory |
| `--alpha` | 0.55 | Heatmap overlay opacity (0–1) |
| `--no-rollout` | False | Skip Transformer Attention Rollout |
| `--max-vault` | 8 | Max genuine samples for vault Grad-CAM averaging |
| `--report-style` | 4panel | Output layout style: simple, forensic (faculty comparison), or 4panel |
| `--cpu` | False | Force CPU inference |
| `--z-threshold` | -1.0 | Z-score acceptance threshold for verdict |
| `--n-subcenters` | 3 | K-means sub-center count for vault clustering |

### `visualize_features.py` (Feature Quality Inspector)

| Argument | Default | Description |
|----------|---------|-------------|
| `--file` | required | Path to image (.png/.jpg) or tensor (.npy) |

### `check_image_cleaning.py` (Preprocessing Validator)

| Argument | Default | Description |
|----------|---------|-------------|
| `--file-name` | required | Path to image to validate preprocessing |

---

## 13. Output Files

| File | Produced by | Description |
|------|-------------|-------------|
| `DATA/unification_Data/<DATASET>_<UID>_G/F_<NNN>.<ext>` | `unify_dataset.py` | Unified flat image directory with canonical dataset prefixes |
| `DATA/unification_Data/manifest.json` | `unify_dataset.py` | Metadata: per-file dataset, script, status, writer UID |
| `DATA/unification_Data/unification.log` | `unify_dataset.py` | Detailed copy log with copy statistics and warnings |
| `DATA/process_data/<DATASET>_<UID>_G/F_<NNN>.npy` | `extract_features.py` | (4, 384, 384) uint8 feature tensors: Shape, Pressure, Angle, Skeleton |
| `DATA/process_data/feature_extraction.log` | `extract_features.py` | Extraction run log with backend GPU detection and parallelism stats |
| `checkpoints/best_tavnet.pt` | `train_tavnet.py` | TAV-Net checkpoint (best validation EER + TAR≥0.80) |
| `checkpoints/manifest.json` | `train_tavnet.py` | Writer split metadata: train/val/test writer UIDs, model config |
| `results/evaluate/testing.json` | `evaluate_tavnet.py` | Comprehensive evaluation metrics: EER, AUC, Accuracy, Precision/Recall/F1, script/category breakdowns |
| `results/verification/results_<timestamp>.json` | `verify_vault.py` | Verification verdicts and component scores per questioned signature |
| `results/cleaned_image/clean_<stem>_<ts>.png` | `check_image_cleaning.py` | 1×2 comparison panel: Original image vs. Phase-1 binary mask output |
| `results/visualize_features/vf_<stem>_<ts>.png` | `visualize_features.py` | 1×5 feature quality panel: cleaned image + 4 channels with statistics |
| `results/grad_cam/evidence_report_<writer>_<stem>_<ts>.png` | `grad_cam.py` | Forensic report (22×18 in @ 200 DPI): 2 image rows + 4-panel analysis with Grad-CAM, attribution maps, contrast diff, metrics table |

---

## 14. Dependencies

Full pinned dependency list in `backend/requirements.txt`. Core ML stack:

```
torch
torchvision
torchaudio
numpy
opencv-python
scikit-learn
scikit-image
scipy
matplotlib
pillow
cupy-cuda12x    # optional GPU numeric backend
tqdm
```

---

*SignatureVault — Offline Signature Verification via TAV-Net (ResNet-50 + CBAM + Transformer)*
