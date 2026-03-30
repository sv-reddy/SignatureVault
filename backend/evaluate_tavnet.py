"""
evaluate_tavnet.py - TAVNet Rigorous Evaluation Protocol
=========================================================

HOW IT WORKS
------------
1. Loads the writer-disjoint test split from checkpoints/manifest.json.
2. Embeds all test signatures using TAVNet (ResNet-50 + CBAM + Transformer).
3. For each test writer and each trial:
      - randomly selects a vault of 5-8 genuine signatures of the same writer,
      - chooses questioned signatures from:
            (a) same writer genuine,
            (b) same writer forgery,
            (c) different writer genuine,
            - applies the same 2-component scoring as verify_vault.py.
    Components:
         centroid_sim  (0.50) -- cosine similarity to mean genuine embedding
         subcenter_sim (0.50) -- cosine similarity to nearest K-means sub-center
     combined = weighted sum of the two components.
4. A per-trial Z-score is computed from Leave-One-Out vault statistics:
     z = (combined - vault_mean) / max(vault_std, 0.01)
    The combined score is used as the ranking signal for EER/AUC.
5. Reports Equal Error Rate (EER), ROC-AUC, Precision/Recall/F1 overall
   and broken down by script (Latin, Hindi, Bengali), plus outlier analysis.

HOW TO RUN
----------
Basic evaluation (uses default checkpoint):
    python evaluate_tavnet.py

Custom checkpoint:
    python evaluate_tavnet.py --checkpoint checkpoints/my_model.pt

Change number of signing-style sub-centers per writer vault:
    python evaluate_tavnet.py --n-subcenters 5

Change number of random vault/questioned trials per writer:
    python evaluate_tavnet.py --vault-trials 5

ARGUMENTS
---------
--checkpoint     Path to TAVNet .pt checkpoint.
                 Default: checkpoints/best_tavnet.pt
--n-subcenters   Number of K-means signing-style sub-centers per writer vault.
                 Default: 7
--vault-trials   Number of randomized vault/questioned trials per writer.
                 Default: 3

SCORING WEIGHTS
---------------
centroid_sim  : 0.50
subcenter_sim : 0.50
"""

import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF, InterpolationMode
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, silhouette_score, confusion_matrix

try:
    from tqdm.auto import tqdm as _tqdm

    def _pbar(it, **kw):
        return _tqdm(it, **kw)

except ImportError:
    def _pbar(it, **kw):
        return it

from train_tavnet import TAVNet, _scan_dir

_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_ROOT  = _SCRIPT_DIR.parent / "DATA"
_PROC_      = _DATA_ROOT / "process_data"

CKPT_DIR       = _SCRIPT_DIR / "checkpoints"
BEST_CKPT_PATH = CKPT_DIR / "best_tavnet.pt"
MANIFEST_PATH  = CKPT_DIR / "manifest.json"
RESULT_JSON_PATH = _SCRIPT_DIR / "results" / "evaluate" / "testing.json"

DEFAULT_N_SUBCENTERS = 7
DEFAULT_VAULT_TRIALS = 3
MIN_VAULT_SIZE       = 5
MAX_VAULT_SIZE       = 8

SCORING_WEIGHTS: Dict[str, float] = {
    "centroid":  0.50,
    "subcenter": 0.50
}

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")


def _extract_model_state_dict(checkpoint: dict) -> dict:
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "model_state" in checkpoint:
        return checkpoint["model_state"]
    return checkpoint


def _load_tavnet_checkpoint(model: TAVNet, checkpoint: dict) -> None:
    state_dict = _extract_model_state_dict(checkpoint)
    model_state = model.state_dict()

    patched = dict(state_dict)
    changed = False
    has_cls_model = ("cls_token" in model_state)
    has_cls_ckpt = ("cls_token" in patched)

    if has_cls_model and not has_cls_ckpt:
        patched["cls_token"] = model_state["cls_token"].clone()
        changed = True
    elif has_cls_ckpt and not has_cls_model:
        patched.pop("cls_token", None)
        changed = True

    if "pos_embed" in patched and "pos_embed" in model_state:
        src = patched["pos_embed"]
        tgt = model_state["pos_embed"]
        if src.shape != tgt.shape and src.ndim == 3 and tgt.ndim == 3 and src.shape[1:] == tgt.shape[1:]:
            src = src.to(device=tgt.device, dtype=tgt.dtype)
            if src.shape[0] + 1 == tgt.shape[0]:
                patched["pos_embed"] = torch.cat([tgt[:1].clone(), src], dim=0)
                changed = True
            elif src.shape[0] == tgt.shape[0] + 1:
                patched["pos_embed"] = src[1:].clone()
                changed = True

    if changed:
        missing, unexpected = model.load_state_dict(patched, strict=False)
        if missing:
            logging.warning("Missing keys after compatibility load: %s", missing)
        if unexpected:
            logging.warning("Unexpected keys after compatibility load: %s", unexpected)
        return

    try:
        model.load_state_dict(patched)
        return
    except RuntimeError as exc:
        logging.warning("Strict checkpoint load failed: %s", exc)
        raise


class EvaluateDataset(Dataset):
    """
    High-resolution evaluation dataset with Test-Time Augmentation (TTA).
    Loads (4, 384, 384) feature tensors and applies 3-view TTA: original, +5°, -5° rotations.
    Returns stacked (3, 4, 384, 384) tensors for efficient batch processing.
    """
    def __init__(self, paths, uids, types):
        self.paths = paths
        self.uids = uids
        self.types = types

    def __len__(self):
        return len(self.paths)

    def _load(self, path: str) -> torch.Tensor:
        arr_u8 = np.load(path).astype(np.uint8)
        arr = arr_u8.astype(np.float32) / 255.0
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        # Load original tensor (4, 384, 384)
        t_orig = self._load(self.paths[idx])
        
        # Create TTA views: original, +5°, -5°
        t_p5 = TF.rotate(t_orig, angle=5.0, interpolation=InterpolationMode.BILINEAR, fill=1.0)
        t_m5 = TF.rotate(t_orig, angle=-5.0, interpolation=InterpolationMode.BILINEAR, fill=1.0)
        
        # Stack to (3, 4, 384, 384): [original, +5°, -5°]
        t_tta = torch.stack([t_orig, t_p5, t_m5], dim=0)
        
        return t_tta, self.uids[idx], self.types[idx]


def _dynamic_kmeans_subcenters(
    embeddings: List[torch.Tensor],
    max_k: int = 3,
    n_iter: int = 30,
    seed: int = 0,
) -> torch.Tensor:
    """
    Dynamic K-means clustering for signing-style sub-centers.
    Automatically selects optimal K based on Silhouette Score with smart fallback.
    
    Args:
        embeddings: List of embedding tensors (each of shape [512] or similar)
        max_k: Maximum number of clusters to evaluate (default 3)
        n_iter: Number of K-means iterations (default 30)
        seed: Random seed for reproducibility
    
    Returns:
        Normalized cluster centers as torch.Tensor of shape [K, embedding_dim]
        where K is either 1 or the optimal K found via Silhouette Score.
    
    Edge Cases:
        - If len(embeddings) <= 2: Returns mean embedding (K=1)
        - If best silhouette score < 0.10: Returns mean embedding (K=1)
        - Otherwise: Returns K-Means centers from highest-scoring K
    """
    n = len(embeddings)
    
    # Edge case: too few embeddings for clustering
    if n <= 2:
        stacked = torch.stack(embeddings).float()
        return F.normalize(stacked.mean(0, keepdim=True), p=2, dim=1)
    
    # Convert to numpy for K-means computation
    stacked = torch.stack(embeddings).float().cpu().numpy()
    
    best_score = -1.0
    best_centers = None
    
    # Search K from 2 to min(max_k, len(embeddings) - 1)
    k_values = range(2, min(max_k, n - 1) + 1)
    
    for k in k_values:
        # Initialize centers randomly from embeddings
        rng = np.random.default_rng(seed + k)  # Vary seed per K
        idxs = rng.choice(n, k, replace=False)
        centers = stacked[idxs].copy()
        
        # Normalize centers to unit L2 norm
        centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
        
        # K-means iterations
        assigns_prev = None
        for iteration in range(n_iter):
            # Compute cosine similarities and get cluster assignments
            sims = stacked @ centers.T
            assigns = np.argmax(sims, axis=1)
            
            # Check for convergence
            if assigns_prev is not None and np.array_equal(assigns, assigns_prev):
                break
            assigns_prev = assigns
            
            # Update cluster centers
            for i in range(k):
                mask = (assigns == i)
                if mask.any():
                    center_sum = stacked[mask].mean(axis=0)
                    center_norm = np.linalg.norm(center_sum)
                    if center_norm > 1e-8:
                        centers[i] = center_sum / center_norm
        
        # Compute Silhouette Score using cosine metric
        try:
            score = silhouette_score(stacked, assigns, metric='cosine')
            
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
        except Exception as e:
            # Silhouette computation failed; skip this K
            logging.debug(f"Silhouette score failed for K={k}: {e}")
            continue
    
    # Fallback logic: if best score below threshold, use K=1 (mean-only)
    if best_score < 0.10:
        logging.debug(
            f"Best silhouette score ({best_score:.4f}) < 0.10 threshold. "
            "Falling back to K=1 (mean centroid only)."
        )
        stacked_torch = torch.from_numpy(stacked).float()
        return F.normalize(stacked_torch.mean(0, keepdim=True), p=2, dim=1)
    
    # Return the best K centers
    return torch.from_numpy(best_centers).float()





def _compute_combined(
    q_emb:      torch.Tensor,
    centroid:   torch.Tensor,
    subcenters: torch.Tensor,
    ref_embs:   List[torch.Tensor],
    weights:    Dict[str, float],
) -> Tuple[float, float, float]:
    centroid_sim  = torch.dot(q_emb, centroid).item()
    subcenter_sim = float((q_emb @ subcenters.T).max().item())
    combined = (
        weights["centroid"]    * centroid_sim
        + weights["subcenter"] * subcenter_sim
    )
    return combined, centroid_sim, subcenter_sim


def compute_metrics(y_true, y_scores):
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    if len(np.unique(y_true)) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_thresh = float(thresholds[eer_idx])
    auc = float(roc_auc_score(y_true, y_scores))
    y_pred = (y_scores >= eer_thresh).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return eer, eer_thresh, auc, p, r, f


def compute_dataset_report(y_true, y_scores):
    y_true = np.array(y_true, dtype=np.int32)
    y_scores = np.array(y_scores, dtype=np.float32)

    if len(y_true) == 0:
        return {
            "samples": 0,
            "eer": 0.0,
            "threshold": 0.0,
            "auc": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "far": 0.0,
            "frr": 0.0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }

    eer, eer_thresh, auc, p, _, f1 = compute_metrics(y_true, y_scores)
    y_pred = (y_scores >= eer_thresh).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    neg_den = (fp + tn)
    pos_den = (tp + fn)
    far = float(fp / neg_den) if neg_den > 0 else 0.0
    frr = float(fn / pos_den) if pos_den > 0 else 0.0
    acc = float((y_pred == y_true).mean())

    return {
        "samples": int(len(y_true)),
        "eer": float(eer),
        "threshold": float(eer_thresh),
        "auc": float(auc),
        "accuracy": float(acc),
        "precision": float(p),
        "f1": float(f1),
        "far": float(far),
        "frr": float(frr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def get_script(uid: int) -> str:
    if 200 < uid <= 400:
        return "Hindi"
    elif 400 < uid <= 600:
        return "Bengali"
    elif uid <= 200 or 600 < uid <= 900 or uid >= 1000:
        return "Latin"
    else:
        return "Unknown"


def main():
    parser = argparse.ArgumentParser(description="TAVNet Evaluation Protocol")
    parser.add_argument("--checkpoint",   type=str, default=str(BEST_CKPT_PATH),
                        help="Path to TAVNet checkpoint")
    parser.add_argument("--n-subcenters", type=int, default=DEFAULT_N_SUBCENTERS,
                        help="Maximum style clusters to evaluate (dynamic K in [1, 7])")
    parser.add_argument("--vault-trials", type=int, default=DEFAULT_VAULT_TRIALS,
                        help="Number of randomized vault/questioned trials per writer")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for vault/questioned sampling")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    n_subcenters    = max(1, min(7, int(args.n_subcenters)))
    vault_trials = max(1, int(args.vault_trials))
    rng = random.Random(args.seed)

    logging.info("=" * 60)
    logging.info(" TAV-Net Evaluation Protocol (Siamese-Transformer)")
    logging.info("=" * 60)
    logging.info("Step 1/5: loading split + configuration")

    if not MANIFEST_PATH.exists():
        logging.error(f"Manifest not found at {MANIFEST_PATH}")
        return

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    test_ids = set(manifest["split"]["test"])
    logging.info(f"Loaded {len(test_ids)} test writers from manifest (writer-disjoint zero-overlap).")

    proc_dir = _PROC_
    if not proc_dir.exists():
        logging.error(f"Data dir {proc_dir} not found. Run extract_features.py first.")
        return

    sample_map = _scan_dir(proc_dir)

    test_paths = []
    test_uids  = []
    test_types = []

    for uid in sorted(test_ids):
        if uid not in sample_map:
            continue
        for p in sample_map[uid].get("G", []):
            test_paths.append(p)
            test_uids.append(uid)
            test_types.append(0)
        for p in sample_map[uid].get("F", []):
            test_paths.append(p)
            test_uids.append(uid)
            test_types.append(1)

    dataset = EvaluateDataset(test_paths, test_uids, test_types)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    logging.info(
        "Test signature tensors: %d (DataLoader batches: %d, batch_size=%d)",
        len(dataset),
        len(loader),
        loader.batch_size if loader.batch_size is not None else 1,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = TAVNet()

    if not checkpoint_path.exists():
        logging.error(f"Checkpoint not found at {checkpoint_path}")
        return

    logging.info(f"Loading checkpoint {checkpoint_path.name}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    _load_tavnet_checkpoint(model, checkpoint)

    model.to(device)
    model.eval()

    logging.info("Step 2/5: extracting embeddings")
    all_embs  = []
    all_uids  = []
    all_types = []

    use_amp = (device.type == "cuda")

    with torch.no_grad():
        for tensors, uids, types in _pbar(
            loader,
            total=len(loader),
            desc="Embedding test batches",
            unit="batch",
        ):
            tensors = tensors.to(device, non_blocking=True)  # (Batch, 3, 4, 384, 384)
            batch_size = tensors.shape[0]
            
            # Reshape to (Batch * 3, 4, 384, 384) for efficient model inference
            tensors_reshaped = tensors.view(batch_size * 3, 4, 384, 384)
            
            with torch.amp.autocast(device.type, enabled=use_amp):
                emb_tta = model(tensors_reshaped)  # (Batch * 3, 512)
            
            # Reshape back to (Batch, 3, 512)
            emb_tta = emb_tta.view(batch_size, 3, 512)
            
            # Average across TTA views and re-normalize to unit sphere
            emb = emb_tta.mean(dim=1)  # (Batch, 512)
            emb = F.normalize(emb, p=2, dim=1)
            
            all_embs.append(emb.cpu().float())
            all_uids.append(uids)
            all_types.append(types)

    embs_cat  = torch.cat(all_embs)
    uids_cat  = torch.cat(all_uids).numpy()
    types_cat = torch.cat(all_types).numpy()

    logging.info("Step 3/5: grouping embeddings by writer/type")
    writer_bank: Dict[int, Dict[str, List[torch.Tensor]]] = defaultdict(lambda: {"G": [], "F": []})
    for i in range(len(uids_cat)):
        uid = int(uids_cat[i])
        key = "G" if int(types_cat[i]) == 0 else "F"
        writer_bank[uid][key].append(F.normalize(embs_cat[i].unsqueeze(0), p=2, dim=1).squeeze(0))

    populated_writers = sorted(uid for uid in test_ids if uid in writer_bank)
    eligible_writers = [
        uid for uid in populated_writers
        if len(writer_bank[uid]["G"]) >= (MIN_VAULT_SIZE + 1) and len(writer_bank[uid]["F"]) >= 1
    ]
    if not eligible_writers:
        logging.error(
            "No eligible test writers found. Need >=%d genuine and >=1 forgery per writer.",
            MIN_VAULT_SIZE + 1,
        )
        return

    logging.info(
        "Eligible writers for vault protocol: %d/%d",
        len(eligible_writers), len(populated_writers),
    )

    logging.info("Step 4/5: running randomized vault/questioned protocol")
    overall_y_true: List[int] = []
    overall_y_score: List[float] = []
    script_y_true: Dict[str, List[int]] = defaultdict(list)
    script_y_score: Dict[str, List[float]] = defaultdict(list)
    category_y_true: Dict[str, List[int]] = defaultdict(list)
    category_y_score: Dict[str, List[float]] = defaultdict(list)

    writer_avg_pos: Dict[int, float] = defaultdict(float)
    writer_pos_count: Dict[int, int] = defaultdict(int)

    trial_rows = 0

    for uid in eligible_writers:
        g_list = writer_bank[uid]["G"]
        f_list = writer_bank[uid]["F"]

        other_uids = [ou for ou in populated_writers if ou != uid and len(writer_bank[ou]["G"]) > 0]
        if not other_uids:
            continue

        for _ in range(vault_trials):
            max_vault = min(MAX_VAULT_SIZE, len(g_list) - 1)
            if max_vault < MIN_VAULT_SIZE:
                continue

            vault_size = rng.randint(MIN_VAULT_SIZE, max_vault)
            vault_indices = rng.sample(range(len(g_list)), vault_size)
            vault_embs = [g_list[i] for i in vault_indices]
            vault_idx_set = set(vault_indices)
            remaining_g = [g_list[i] for i in range(len(g_list)) if i not in vault_idx_set]
            if not remaining_g:
                continue

            vault_centroid = F.normalize(torch.stack(vault_embs).mean(0, keepdim=True), p=2, dim=1).squeeze(0)
            # Use dynamic K-means with Silhouette Score to select optimal cluster count
            vault_sub = _dynamic_kmeans_subcenters(vault_embs, max_k=n_subcenters)

            loo_scores: List[float] = []
            for i in range(len(vault_embs)):
                others = [vault_embs[j] for j in range(len(vault_embs)) if j != i]
                if not others:
                    continue
                loo_centroid = F.normalize(torch.stack(others).mean(0, keepdim=True), p=2, dim=1).squeeze(0)
                # Use dynamic K-means for leave-one-out validation as well
                loo_sub = _dynamic_kmeans_subcenters(others, max_k=n_subcenters)
                sc, *_ = _compute_combined(vault_embs[i], loo_centroid, loo_sub, others, SCORING_WEIGHTS)
                loo_scores.append(sc)

            vault_mean = float(np.mean(loo_scores)) if loo_scores else 0.70
            vault_std = float(np.std(loo_scores)) if len(loo_scores) > 1 else 0.05

            q_same_g = rng.choice(remaining_g)
            q_same_f = rng.choice(f_list)
            q_other_uid = rng.choice(other_uids)
            q_other_g = rng.choice(writer_bank[q_other_uid]["G"])

            questioned = [
                ("same_user_genuine", 1, q_same_g),
                ("same_user_forgery", 0, q_same_f),
                ("different_user_genuine", 0, q_other_g),
            ]

            for category, label, q_emb in questioned:
                combined, *_ = _compute_combined(q_emb, vault_centroid, vault_sub, vault_embs, SCORING_WEIGHTS)
                _ = (combined - vault_mean) / max(vault_std, 0.01)

                overall_y_true.append(label)
                overall_y_score.append(combined)

                script = get_script(uid)
                script_y_true[script].append(label)
                script_y_score[script].append(combined)

                category_y_true[category].append(label)
                category_y_score[category].append(combined)

                if label == 1:
                    writer_avg_pos[uid] += combined
                    writer_pos_count[uid] += 1
                trial_rows += 1

    if not overall_y_true:
        logging.error("No evaluation samples were generated by the vault protocol.")
        return

    writer_avg_pos_final = {
        uid: (writer_avg_pos[uid] / max(1, writer_pos_count[uid]))
        for uid in writer_avg_pos
    }

    overall_report = compute_dataset_report(overall_y_true, overall_y_score)

    w = SCORING_WEIGHTS
    formula = f"{w['centroid']:.2f}*centroid + {w['subcenter']:.2f}*subcenter"

    print("\n")
    print(f"Scoring Schema : {formula}")
    print(f"Sub-Centers    : Dynamic K in [1-{n_subcenters}] (Silhouette-selected)")
    print(f"Vault Protocol : {MIN_VAULT_SIZE}-{MAX_VAULT_SIZE} random genuine refs per trial")
    print(f"Trials/Writer  : {vault_trials}")
    print(f"Total Samples  : {trial_rows}")
    print("\n")
    print("| Dataset/Script | Writers | Samples | Accuracy | Precision | F1-Score | FAR | FRR | Threshold |")
    print("| :------------- | :------ | :------ | :------- | :-------- | :------- | :-- | :-- | :-------- |")

    script_reports: Dict[str, Dict[str, float]] = {}
    for scr in ["Overall", "Latin", "Hindi", "Bengali"]:
        if scr == "Overall":
            report = overall_report
            w_cnt = len(eligible_writers)
        else:
            if scr not in script_y_true:
                continue
            report = compute_dataset_report(script_y_true[scr], script_y_score[scr])
            w_cnt = sum(1 for u in eligible_writers if get_script(u) == scr)

        script_reports[scr] = {
            "writers": int(w_cnt),
            **report,
        }

        print(
            f"| {scr:14s} | {w_cnt:7d} | {report['samples']:7d} | "
            f"{report['accuracy']*100:7.2f}% | {report['precision']*100:8.2f}% | "
            f"{report['f1']*100:7.2f}% | {report['far']*100:5.2f}% | {report['frr']*100:5.2f}% | "
            f"{report['threshold']:9.4f} |"
        )

    print("\n--- Overall Confusion Matrix (Threshold = Overall EER threshold) ---")
    overall_cm = script_reports["Overall"]
    print("                 Pred 0   Pred 1")
    print(f"True 0 (negative) {overall_cm['tn']:7d} {overall_cm['fp']:8d}")
    print(f"True 1 (positive) {overall_cm['fn']:7d} {overall_cm['tp']:8d}")

    print("\n| Questioned Type       | Samples | Positive% | Mean Score |")
    print("| :------------------- | :------ | :-------- | :--------- |")
    for cat in ["same_user_genuine", "same_user_forgery", "different_user_genuine"]:
        if cat not in category_y_true:
            continue
        labels = np.array(category_y_true[cat], dtype=np.float32)
        scores = np.array(category_y_score[cat], dtype=np.float32)
        pos_pct = float(labels.mean() * 100.0) if len(labels) > 0 else 0.0
        mean_score = float(scores.mean()) if len(scores) > 0 else 0.0
        print(
            f"| {cat:20s} | {len(category_y_true[cat]):7d} | {pos_pct:8.2f}% | {mean_score:10.4f} |"
        )

    print("\n--- Additional Metrics (Overall) ---")
    print(f"Accuracy @ EER   : {overall_report['accuracy']*100:.2f}%")
    print(f"Precision @ EER  : {overall_report['precision']*100:.2f}%")
    print(f"F1-Score @ EER   : {overall_report['f1']*100:.2f}%")
    print(f"FAR @ EER        : {overall_report['far']*100:.2f}%")
    print(f"FRR @ EER        : {overall_report['frr']*100:.2f}%")

    script_metrics: Dict[str, Dict[str, float]] = {
        scr: {
            "writers": int(rep["writers"]),
            "samples": int(rep["samples"]),
            "eer": float(rep["eer"]),
            "auc": float(rep["auc"]),
            "accuracy": float(rep["accuracy"]),
            "precision": float(rep["precision"]),
            "f1": float(rep["f1"]),
            "far_at_eer": float(rep["far"]),
            "frr_at_eer": float(rep["frr"]),
            "threshold": float(rep["threshold"]),
            "confusion_matrix": {
                "tn": int(rep["tn"]),
                "fp": int(rep["fp"]),
                "fn": int(rep["fn"]),
                "tp": int(rep["tp"]),
            },
        }
        for scr, rep in script_reports.items()
    }

    category_metrics: Dict[str, Dict[str, float]] = {}
    for cat in ["same_user_genuine", "same_user_forgery", "different_user_genuine"]:
        if cat not in category_y_true:
            continue
        labels = np.array(category_y_true[cat], dtype=np.float32)
        scores = np.array(category_y_score[cat], dtype=np.float32)
        category_metrics[cat] = {
            "samples": int(len(labels)),
            "positive_percent": float(labels.mean() * 100.0) if len(labels) > 0 else 0.0,
            "mean_score": float(scores.mean()) if len(scores) > 0 else 0.0,
        }

    result_payload = {
        "checkpoint": str(checkpoint_path),
        "n_subcenters": int(n_subcenters),
        "vault_trials": int(vault_trials),
        "vault_protocol": {
            "min_vault_size": int(MIN_VAULT_SIZE),
            "max_vault_size": int(MAX_VAULT_SIZE),
        },
        "total_samples": int(trial_rows),
        "eligible_writers": int(len(eligible_writers)),
        "overall": {
            "samples": int(overall_report["samples"]),
            "eer": float(overall_report["eer"]),
            "auc": float(overall_report["auc"]),
            "threshold": float(overall_report["threshold"]),
            "accuracy": float(overall_report["accuracy"]),
            "precision": float(overall_report["precision"]),
            "f1": float(overall_report["f1"]),
            "far_at_eer": float(overall_report["far"]),
            "frr_at_eer": float(overall_report["frr"]),
            "confusion_matrix": {
                "tn": int(overall_report["tn"]),
                "fp": int(overall_report["fp"]),
                "fn": int(overall_report["fn"]),
                "tp": int(overall_report["tp"]),
            },
        },
        "scripts": script_metrics,
        "questioned_types": category_metrics
    }

    RESULT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)
    logging.info("Saved evaluation JSON to %s", RESULT_JSON_PATH)

    logging.info("Step 5/5: reporting complete")
    logging.info("Evaluation complete.")


if __name__ == "__main__":
    main()

