"""
verify_vault.py - SignVault Offline Signature Verification
==========================================================

HOW IT WORKS
------------
1. TAV-Net (ResNet-50 + CBAM + Transformer) produces a 512-d L2-normalised
   embedding for every signature image.
2. All genuine reference signatures in the vault folder are embedded.
   Their mean embedding forms the vault centroid. Spherical K-means
   (default k=3) groups them into signing-style sub-centers.
3. Leave-One-Out (LOO) evaluation over the vault produces vault_mean and
   vault_std -- a calibrated measure of how consistent the writer is.
   For well-calibrated Z-scores, provide 5-8 genuine references (matching
   the evaluation protocol used during training).
4. The questioned signature is embedded and scored on two components:
         centroid_sim  (0.50) -- cosine similarity to vault centroid
         subcenter_sim (0.50) -- similarity to nearest signing-style cluster
     combined = weighted sum of the two components.
5. A Z-score is computed relative to the vault's own internal stats:
     z = (combined - vault_mean) / max(vault_std, 0.01)
    Verdict: GENUINE if combined >= (vault_mean - vault_std), else FORGERY.
   A tight vault produces a stricter threshold automatically; a writer whose
   natural signatures vary more gets proportionally wider tolerance.

HOW TO RUN
----------
Verify a single questioned signature:
    python verify_vault.py --vault path/to/vault_dir --questioned path/to/sig.png

Verify an entire folder of questioned signatures:
    python verify_vault.py --vault path/to/vault_dir --questioned path/to/folder/

Specify a custom model checkpoint:
    python verify_vault.py --vault vault/ --questioned q.png --checkpoint checkpoints/best_tavnet.pt

ARGUMENTS
---------
--vault          (required) Folder of genuine reference signature images.
--questioned     (required) Questioned image file or folder of images.
--checkpoint     Path to TAV-Net .pt checkpoint.
                 Default: checkpoints/best_tavnet.pt

SUPPORTED FILE FORMATS
----------------------
Images : .png  .jpg  .jpeg  .tif  .tiff  .bmp
Arrays : .npy  (pre-extracted 4-channel feature arrays, uint8, shape (4,224,224))
"""

import os
import json
import logging
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from sklearn.metrics import silhouette_score

from train_tavnet import TAVNet
from extract_features import preprocess, extract_channels, _read_image

DEFAULT_CHECKPOINT   = "checkpoints/best_tavnet.pt"
IMAGE_EXTS           = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
DEFAULT_N_SUBCENTERS = 3

DEFAULT_WEIGHTS: Dict[str, float] = {
    "centroid":  0.50,
    "subcenter": 0.50,
}

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)


def _dynamic_kmeans_subcenters(
    embeddings: List[torch.Tensor],
    max_k: int = 3,
    n_iter: int = 30,
    seed: int = 0,
) -> torch.Tensor:
    device = embeddings[0].device
    dtype = embeddings[0].dtype
    n = len(embeddings)
    if n <= 2:
        stacked = torch.stack(embeddings).float()
        return F.normalize(stacked.mean(0, keepdim=True), p=2, dim=1).to(device=device, dtype=dtype)

    stacked = torch.stack(embeddings).float().cpu().numpy()
    best_score = -1.0
    best_centers = None

    for k in range(2, min(max_k, n - 1) + 1):
        rng = np.random.default_rng(seed + k)
        idxs = rng.choice(n, k, replace=False)
        centers = stacked[idxs].copy()
        centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)

        assigns_prev = None
        for _ in range(n_iter):
            sims = stacked @ centers.T
            assigns = np.argmax(sims, axis=1)
            if assigns_prev is not None and np.array_equal(assigns, assigns_prev):
                break
            assigns_prev = assigns

            for i in range(k):
                mask = (assigns == i)
                if mask.any():
                    center_sum = stacked[mask].mean(axis=0)
                    center_norm = np.linalg.norm(center_sum)
                    if center_norm > 1e-8:
                        centers[i] = center_sum / center_norm

        try:
            score = silhouette_score(stacked, assigns, metric="cosine")
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
        except Exception:
            continue

    if best_score < 0.10:
        stacked_torch = torch.from_numpy(stacked).float()
        return F.normalize(stacked_torch.mean(0, keepdim=True), p=2, dim=1).to(device=device, dtype=dtype)

    return torch.from_numpy(best_centers).float().to(device=device, dtype=dtype)


def _vault_acceptance_threshold(vault_mean: float, vault_std: float) -> float:
    """Derive acceptance threshold directly from vault consistency stats."""
    return float(vault_mean - vault_std)


def _compute_combined(
    q_emb:      torch.Tensor,
    centroid:   torch.Tensor,
    subcenters: torch.Tensor,
    named_embs: List[Tuple[str, torch.Tensor]],
    weights:    Dict[str, float],
) -> Tuple[float, float, float]:
    centroid_sim  = torch.dot(q_emb, centroid).item()
    subcenter_sim = float((q_emb @ subcenters.T).max().item())
    combined = (
        weights["centroid"]    * centroid_sim
        + weights["subcenter"] * subcenter_sim
    )
    return combined, centroid_sim, subcenter_sim


class SignatureVerifier:
    def __init__(self, checkpoint_path: str, device: torch.device):
        self.device = device
        self.model  = self._load_model(checkpoint_path)
        self.model.eval()
        logger.info(f"Model loaded and moved to {device}")

    def _load_model(self, checkpoint_path: str) -> TAVNet:
        model = TAVNet()
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("model_state", checkpoint))

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
            state_dict = patched
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning("Missing keys after compatibility load: %s", missing)
            if unexpected:
                logger.warning("Unexpected keys after compatibility load: %s", unexpected)
        else:
            model.load_state_dict(state_dict)

        model.to(self.device)
        return model

    def _load_and_preprocess(self, path: Path) -> torch.Tensor:
        if path.suffix.lower() == ".npy":
            arr = np.load(str(path)).astype(np.float32) / 255.0
        elif path.suffix.lower() in IMAGE_EXTS:
            img  = _read_image(path)
            mask = preprocess(img)
            arr  = extract_channels(mask).astype(np.float32) / 255.0
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        return torch.from_numpy(arr).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def get_embedding(self, path: Path) -> torch.Tensor:
        tensor  = self._load_and_preprocess(path)
        use_amp = (self.device.type == "cuda")
        with torch.amp.autocast(self.device.type, enabled=use_amp):
            embedding = self.model(tensor)
        return embedding.squeeze(0)

    def compute_vault_centroid(
        self,
        vault_dir:    Path,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[str, torch.Tensor]], float, float]:
        vault_files = sorted([
            f for f in vault_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTS or f.suffix.lower() == ".npy"
        ])
        if not vault_files:
            raise ValueError(f"No valid signatures found in vault: {vault_dir}")

        logger.info(f"Processing {len(vault_files)} signatures from vault...")
        embeddings, named_embeddings = [], []
        for f in vault_files:
            emb = self.get_embedding(f)
            embeddings.append(emb)
            named_embeddings.append((f.name, emb))

        all_embs   = torch.stack(embeddings)
        centroid   = F.normalize(all_embs.mean(0, keepdim=True), p=2, dim=1).squeeze(0)
        style_max_k = DEFAULT_N_SUBCENTERS
        subcenters = _dynamic_kmeans_subcenters(embeddings, max_k=style_max_k)
        logger.info(f"Vault: {len(vault_files)} signatures -> {subcenters.shape[0]} sub-centers")

        loo_scores: List[float] = []
        for i, (_, q_emb) in enumerate(named_embeddings):
            others = [(n, e) for j, (n, e) in enumerate(named_embeddings) if j != i]
            if not others:
                continue
            other_stack  = torch.stack([e for _, e in others])
            loo_centroid = F.normalize(other_stack.mean(0, keepdim=True), p=2, dim=1).squeeze(0)
            loo_sub      = _dynamic_kmeans_subcenters([e for _, e in others], max_k=style_max_k)
            score, *_    = _compute_combined(q_emb, loo_centroid, loo_sub, others, DEFAULT_WEIGHTS)
            loo_scores.append(score)

        vault_mean = float(np.mean(loo_scores))       if loo_scores else 0.70
        vault_std  = float(np.std(loo_scores))        if len(loo_scores) > 1 else 0.05
        logger.info(
            f"Vault stats -- mean: {vault_mean:.4f}  std: {vault_std:.4f}  "
            f"({len(loo_scores)} LOO scores)"
        )
        return centroid, subcenters, named_embeddings, vault_mean, vault_std

    def verify(
        self,
        questioned_path: Path,
        vault_centroid:  torch.Tensor,
        subcenters:      torch.Tensor,
        named_embs:      List[Tuple[str, torch.Tensor]],
        vault_mean:      float,
        vault_std:       float,
    ) -> Dict[str, Any]:
        logger.info(f"Verifying questioned signature: {questioned_path.name}")
        q_emb = self.get_embedding(questioned_path)

        combined, centroid_sim, subcenter_sim = _compute_combined(
            q_emb, vault_centroid, subcenters, named_embs, DEFAULT_WEIGHTS
        )

        acceptance_threshold = _vault_acceptance_threshold(vault_mean, vault_std)
        z_score = (combined - vault_mean) / max(vault_std, 0.01)
        verdict = "GENUINE" if combined >= acceptance_threshold else "FORGERY"

        indiv_sims = [(name, round(torch.dot(q_emb, emb).item(), 4)) for name, emb in named_embs]

        return {
            "questioned_file":  questioned_path.name,
            "centroid_sim":     round(centroid_sim,  4),
            "subcenter_sim":    round(subcenter_sim, 4),
            "combined_score":   round(combined,      4),
            "similarity_score": round(combined,      4),
            "vault_mean":       round(vault_mean,    4),
            "vault_std":        round(vault_std,     4),
            "acceptance_threshold": round(acceptance_threshold, 4),
            "z_score":          round(z_score,       4),
            "verdict":          verdict,
            "q_embedding":      q_emb,
            "pairwise_sims":    indiv_sims,
        }


def print_report(results: Dict[str, Any], vault_size: int):
    w = DEFAULT_WEIGHTS
    formula = f"{w['centroid']:.2f}*centroid + {w['subcenter']:.2f}*subcenter"
    print("\n" + "="*62)
    print("            SIGNVAULT VERIFICATION REPORT")
    print("="*62)
    print(f" Questioned File : {results['questioned_file']}")
    print(f" Vault Size      : {vault_size} reference signatures")
    print(f" Vault Consist.  : mean={results['vault_mean']:.4f}  std={results['vault_std']:.4f}")
    print(f" Threshold       : {results['acceptance_threshold']:.4f}  (from vault stats)")
    print("-" * 62)
    print(f" Centroid Sim    : {results['centroid_sim']:.4f}  (global vault centroid)")
    print(f" Sub-Center Sim  : {results['subcenter_sim']:.4f}  (nearest style cluster)")
    print(f" COMBINED SCORE  : {results['combined_score']:.4f}  ({formula})")
    print(f" Z-SCORE         : {results['z_score']:.4f}")
    status_color = "\033[92m" if results["verdict"] == "GENUINE" else "\033[91m"
    reset_color  = "\033[0m"
    print(f" VERDICT         : {status_color}{results['verdict']}{reset_color}")
    print("-" * 62)
    print(" Pairwise Similarity Breakdown:")
    for name, score in results["pairwise_sims"]:
        print(f"  +- {name:20s} : {score:.4f}")
    print("="*62 + "\n")


def main():
    parser = argparse.ArgumentParser(description="SignVault Verification Script")
    parser.add_argument("--vault",        type=str, required=True,
                        help="Path to folder containing genuine reference signatures")
    parser.add_argument("--questioned",   type=str, required=True,
                        help="Path to questioned signature image or folder")
    parser.add_argument("--checkpoint",   type=str, default=DEFAULT_CHECKPOINT,
                        help="Path to model checkpoint")
    args = parser.parse_args()

    vault_path      = Path(args.vault).resolve()
    q_path          = Path(args.questioned).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()

    if not vault_path.is_dir():
        logger.error(f"Vault path is not a directory: {vault_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(f"Hardware Acceleration: {torch.cuda.get_device_name(0)} detected.")

    try:
        verifier = SignatureVerifier(str(checkpoint_path), device)

        centroid, subcenters, named_embs, vault_mean, vault_std = (
            verifier.compute_vault_centroid(vault_path)
        )
        n_vault = len(named_embs)
        if n_vault < 2:
            logger.warning("Vault has only 1 signature -- Z-score cannot be calibrated. "
                           "Provide at least 5-8 genuine references for reliable verification.")
        elif n_vault < 5:
            logger.warning(
                "Vault has only %d genuine references. "
                "5-8 references are recommended for well-calibrated Z-scores "
                "(matching the evaluation protocol).", n_vault
            )

        q_files = []
        if q_path.is_dir():
            q_files = sorted([f for f in q_path.iterdir()
                               if f.suffix.lower() in IMAGE_EXTS or f.suffix.lower() == ".npy"])
        elif q_path.is_file():
            q_files = [q_path]

        if not q_files:
            logger.error(f"No valid questioned signatures found at {q_path}")
            return

        all_results = []
        for qf in q_files:
            res = verifier.verify(
                qf, centroid, subcenters, named_embs,
                vault_mean, vault_std,
            )
            res["style_clusters"] = int(subcenters.shape[0])

            res_report = {k: v for k, v in res.items() if k not in ("q_embedding", "pairwise_sims")}
            res_report["pairwise_sims"] = res["pairwise_sims"]
            print_report(res_report, len(named_embs))

            res.pop("q_embedding", None)
            res["pairwise_scores"] = [
                {"ref_file": n, "score": s} for n, s in res.pop("pairwise_sims")
            ]
            all_results.append(res)

        timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M")
        results_dir = Path("results") / "verification"
        results_dir.mkdir(parents=True, exist_ok=True)
        save_path = results_dir / f"results_{timestamp}.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4)
        logger.info(f"Results saved to {save_path}")

    except Exception:
        logger.error("Verification failed!")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
