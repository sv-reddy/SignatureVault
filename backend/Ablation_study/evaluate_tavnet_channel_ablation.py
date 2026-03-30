"""
evaluate_tavnet_ablation.py - Unified Verification for Ablation Variants
=======================================================================

Runs the verification protocol for one or all ablation variants:
  - baseline: [0]
  - step1:    [0, 1]
  - step2:    [0, 1, 2]
  - full:     [0, 1, 2, 3]

This script does not modify existing scripts.
It evaluates variant checkpoints trained by train_tavnet_ablation.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

_SCRIPT_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _SCRIPT_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from train_tavnet import _scan_dir
from train_tavnet_channel_ablation import AblationTAVNet, VARIANTS

try:
    from tqdm.auto import tqdm as _tqdm

    def _pbar(it, **kw):
        return _tqdm(it, **kw)

except ImportError:
    def _pbar(it, **kw):
        return it


_DATA_ROOT = _BACKEND_DIR.parent / "DATA"
_DEFAULT_PROC_DIR = _DATA_ROOT / "process_data"
_DEFAULT_CKPT_ROOT = _BACKEND_DIR / "checkpoints" / "ablation"
_DEFAULT_MANIFEST = _DEFAULT_CKPT_ROOT / "manifest.json"
_DEFAULT_OUTPUT_JSON = _BACKEND_DIR / "results" / "evaluate" / "ablation_summary.json"

MIN_VAULT_SIZE = 5
MAX_VAULT_SIZE = 8
DEFAULT_N_SUBCENTERS = 7
DEFAULT_VAULT_TRIALS = 3

SCORING_WEIGHTS: Dict[str, float] = {
    "centroid": 0.50,
    "subcenter": 0.50,
}

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")


class EvaluateAblationDataset(Dataset):
    def __init__(self, paths: List[str], uids: List[int], types: List[int], channel_indices: List[int]) -> None:
        self.paths = paths
        self.uids = uids
        self.types = types
        self.channel_indices = list(channel_indices)

    def __len__(self) -> int:
        return len(self.paths)

    def _load(self, path: str) -> torch.Tensor:
        arr_u8 = np.load(path).astype(np.uint8)
        arr = arr_u8[self.channel_indices, :, :].astype(np.float32) / 255.0
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int):
        t_orig = self._load(self.paths[idx])
        t_p5 = TF.rotate(t_orig, angle=5.0, interpolation=InterpolationMode.BILINEAR, fill=1.0)
        t_m5 = TF.rotate(t_orig, angle=-5.0, interpolation=InterpolationMode.BILINEAR, fill=1.0)
        t_tta = torch.stack([t_orig, t_p5, t_m5], dim=0)
        return t_tta, self.uids[idx], self.types[idx]


def _extract_model_state_dict(checkpoint: dict) -> dict:
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "model_state" in checkpoint:
        return checkpoint["model_state"]
    return checkpoint


def _load_ablation_checkpoint(model: AblationTAVNet, checkpoint: dict) -> None:
    state_dict = _extract_model_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logging.warning("Missing keys during checkpoint load: %s", missing)
    if unexpected:
        logging.warning("Unexpected keys during checkpoint load: %s", unexpected)


def _dynamic_kmeans_subcenters(
    embeddings: List[torch.Tensor],
    max_k: int = 3,
    n_iter: int = 30,
    seed: int = 0,
) -> torch.Tensor:
    from sklearn.metrics import silhouette_score

    n = len(embeddings)
    if n <= 2:
        stacked = torch.stack(embeddings).float()
        return F.normalize(stacked.mean(0, keepdim=True), p=2, dim=1)

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
                mask = assigns == i
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
        return F.normalize(stacked_torch.mean(0, keepdim=True), p=2, dim=1)

    return torch.from_numpy(best_centers).float()


def _compute_combined(
    q_emb: torch.Tensor,
    centroid: torch.Tensor,
    subcenters: torch.Tensor,
    weights: Dict[str, float],
) -> Tuple[float, float, float]:
    centroid_sim = torch.dot(q_emb, centroid).item()
    subcenter_sim = float((q_emb @ subcenters.T).max().item())
    combined = (
        weights["centroid"] * centroid_sim
        + weights["subcenter"] * subcenter_sim
    )
    return combined, centroid_sim, subcenter_sim


def _compute_metrics(y_true: List[int], y_scores: List[float]) -> Tuple[float, float, float, float, float, float]:
    y_true_np = np.array(y_true)
    y_scores_np = np.array(y_scores)

    if len(np.unique(y_true_np)) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    fpr, tpr, thresholds = roc_curve(y_true_np, y_scores_np)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_thresh = float(thresholds[eer_idx])
    auc = float(roc_auc_score(y_true_np, y_scores_np))

    y_pred = (y_scores_np >= eer_thresh).astype(int)
    p, _r, f1, _ = precision_recall_fscore_support(y_true_np, y_pred, average="binary", zero_division=0)
    return eer, eer_thresh, auc, p, 0.0, f1


def compute_dataset_report(y_true: List[int], y_scores: List[float]) -> dict:
    y_true_np = np.array(y_true, dtype=np.int32)
    y_scores_np = np.array(y_scores, dtype=np.float32)

    if len(y_true_np) == 0:
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

    eer, eer_thresh, auc, precision, _recall, f1 = _compute_metrics(y_true, y_scores)
    y_pred = (y_scores_np >= eer_thresh).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(y_true_np, y_pred, labels=[0, 1]).ravel()
    neg_den = fp + tn
    pos_den = tp + fn
    far = float(fp / neg_den) if neg_den > 0 else 0.0
    frr = float(fn / pos_den) if pos_den > 0 else 0.0
    acc = float((y_pred == y_true_np).mean())

    return {
        "samples": int(len(y_true_np)),
        "eer": float(eer),
        "threshold": float(eer_thresh),
        "auc": float(auc),
        "accuracy": float(acc),
        "precision": float(precision),
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
    if 400 < uid <= 600:
        return "Bengali"
    if uid <= 200 or 600 < uid <= 900 or uid >= 1000:
        return "Latin"
    return "Unknown"


def _evaluate_variant(
    *,
    variant_name: str,
    channel_indices: List[int],
    checkpoint_path: Path,
    sample_map: dict,
    test_ids: set[int],
    n_subcenters: int,
    vault_trials: int,
    seed: int,
    embed_dim: int,
    batch_size: int,
    num_workers: int,
) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for {variant_name}: {checkpoint_path}")

    test_paths: List[str] = []
    test_uids: List[int] = []
    test_types: List[int] = []

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

    dataset = EvaluateAblationDataset(test_paths, test_uids, test_types, channel_indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AblationTAVNet(in_channels=len(channel_indices), embed_dim=embed_dim)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    _load_ablation_checkpoint(model, checkpoint)

    model.to(device)
    model.eval()

    all_embs: List[torch.Tensor] = []
    all_uids: List[torch.Tensor] = []
    all_types: List[torch.Tensor] = []

    use_amp = device.type == "cuda"

    with torch.no_grad():
        for tensors, uids, types in _pbar(
            loader,
            total=len(loader),
            desc=f"Embed {variant_name}",
            unit="batch",
        ):
            tensors = tensors.to(device, non_blocking=True)
            bsz = tensors.shape[0]
            ch = len(channel_indices)

            tensors_reshaped = tensors.view(bsz * 3, ch, 384, 384)
            with torch.amp.autocast(device.type, enabled=use_amp):
                emb_tta = model(tensors_reshaped)

            emb_tta = emb_tta.view(bsz, 3, 512)
            emb = emb_tta.mean(dim=1)
            emb = F.normalize(emb, p=2, dim=1)

            all_embs.append(emb.cpu().float())
            all_uids.append(uids)
            all_types.append(types)

    embs_cat = torch.cat(all_embs)
    uids_cat = torch.cat(all_uids).numpy()
    types_cat = torch.cat(all_types).numpy()

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
        raise RuntimeError(f"No eligible writers for variant {variant_name}")

    rng = random.Random(seed)

    overall_y_true: List[int] = []
    overall_y_score: List[float] = []
    script_y_true: Dict[str, List[int]] = defaultdict(list)
    script_y_score: Dict[str, List[float]] = defaultdict(list)

    total_samples = 0

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
            vault_sub = _dynamic_kmeans_subcenters(vault_embs, max_k=n_subcenters)

            loo_scores: List[float] = []
            for i in range(len(vault_embs)):
                others = [vault_embs[j] for j in range(len(vault_embs)) if j != i]
                if not others:
                    continue
                loo_centroid = F.normalize(torch.stack(others).mean(0, keepdim=True), p=2, dim=1).squeeze(0)
                loo_sub = _dynamic_kmeans_subcenters(others, max_k=n_subcenters)
                sc, *_ = _compute_combined(vault_embs[i], loo_centroid, loo_sub, SCORING_WEIGHTS)
                loo_scores.append(sc)

            vault_mean = float(np.mean(loo_scores)) if loo_scores else 0.70
            vault_std = float(np.std(loo_scores)) if len(loo_scores) > 1 else 0.05

            q_same_g = rng.choice(remaining_g)
            q_same_f = rng.choice(f_list)
            q_other_uid = rng.choice(other_uids)
            q_other_g = rng.choice(writer_bank[q_other_uid]["G"])

            questioned = [
                (1, q_same_g),
                (0, q_same_f),
                (0, q_other_g),
            ]

            for label, q_emb in questioned:
                combined, *_ = _compute_combined(q_emb, vault_centroid, vault_sub, SCORING_WEIGHTS)
                _ = (combined - vault_mean) / max(vault_std, 0.01)

                overall_y_true.append(label)
                overall_y_score.append(combined)

                scr = get_script(uid)
                script_y_true[scr].append(label)
                script_y_score[scr].append(combined)
                total_samples += 1

    overall_report = compute_dataset_report(overall_y_true, overall_y_score)

    scripts_payload = {}
    for scr in ["Latin", "Hindi", "Bengali"]:
        if scr not in script_y_true:
            continue
        rep = compute_dataset_report(script_y_true[scr], script_y_score[scr])
        scripts_payload[scr] = rep

    return {
        "variant": variant_name,
        "channels": channel_indices,
        "checkpoint": str(checkpoint_path),
        "eligible_writers": int(len(eligible_writers)),
        "total_samples": int(total_samples),
        "overall": overall_report,
        "scripts": scripts_payload,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify ablation variants with vault protocol.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=["baseline", "step1", "step2", "full", "all"],
        help="Variant to evaluate.",
    )
    p.add_argument(
        "--checkpoint-root",
        type=str,
        default=str(_DEFAULT_CKPT_ROOT),
        help="Root directory containing variant checkpoint folders.",
    )
    p.add_argument(
        "--manifest",
        type=str,
        default=str(_DEFAULT_MANIFEST),
        help="Manifest JSON path containing split with test writer IDs.",
    )
    p.add_argument(
        "--processed-dir",
        type=str,
        default=str(_DEFAULT_PROC_DIR),
        help="Processed .npy tensors directory.",
    )
    p.add_argument("--embed-dim", type=int, default=512, help="Embedding dimensionality.")
    p.add_argument("--n-subcenters", type=int, default=DEFAULT_N_SUBCENTERS, help="Dynamic K upper bound.")
    p.add_argument("--vault-trials", type=int, default=DEFAULT_VAULT_TRIALS, help="Trials per writer.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--batch-size", type=int, default=8, help="Embedding batch size.")
    p.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    p.add_argument(
        "--output-json",
        type=str,
        default=str(_DEFAULT_OUTPUT_JSON),
        help="Path to write ablation evaluation summary.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    checkpoint_root = Path(args.checkpoint_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    processed_dir = Path(args.processed_dir).resolve()
    output_json = Path(args.output_json).resolve()

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed directory not found: {processed_dir}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    test_ids = set(manifest["split"]["test"])
    sample_map = _scan_dir(processed_dir)

    if args.variant == "all":
        run_list = ["baseline", "step1", "step2", "full"]
    else:
        run_list = [args.variant]

    n_subcenters = max(1, min(7, int(args.n_subcenters)))
    vault_trials = max(1, int(args.vault_trials))

    summary = {
        "manifest": str(manifest_path),
        "processed_dir": str(processed_dir),
        "n_subcenters": n_subcenters,
        "vault_trials": vault_trials,
        "seed": int(args.seed),
        "variants": {},
        "skipped": {},
    }

    row_fmt = (
        "| {variant:<9} | {channels:<12} | {writers:>7} | {samples:>7} | "
        "{acc:>8} | {prec:>9} | {f1:>7} | {eer:>7} | {status:<7} |"
    )
    header = row_fmt.format(
        variant="Variant",
        channels="Channels",
        writers="Writers",
        samples="Samples",
        acc="Accuracy",
        prec="Precision",
        f1="F1",
        eer="EER",
        status="Status",
    )
    sep = "=" * len(header)

    print("\nAblation Verification")
    print(sep)
    print(header)
    print(sep)

    for variant_name in run_list:
        channels = VARIANTS[variant_name]["channels"]
        ckpt_path = checkpoint_root / variant_name / f"best_tavnet_{variant_name}.pt"

        logging.info("Evaluating variant=%s channels=%s checkpoint=%s", variant_name, channels, ckpt_path)

        if not ckpt_path.exists():
            msg = f"Checkpoint not found: {ckpt_path}"
            if args.variant == "all":
                logging.warning("Skipping %s: %s", variant_name, msg)
                summary["skipped"][variant_name] = msg
                print(row_fmt.format(
                    variant=variant_name,
                    channels=str(channels),
                    writers="-",
                    samples="-",
                    acc="N/A",
                    prec="N/A",
                    f1="N/A",
                    eer="N/A",
                    status="SKIPPED",
                ))
                continue
            raise FileNotFoundError(f"Checkpoint not found for {variant_name}: {ckpt_path}")

        payload = _evaluate_variant(
            variant_name=variant_name,
            channel_indices=channels,
            checkpoint_path=ckpt_path,
            sample_map=sample_map,
            test_ids=test_ids,
            n_subcenters=n_subcenters,
            vault_trials=vault_trials,
            seed=int(args.seed),
            embed_dim=int(args.embed_dim),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
        )

        summary["variants"][variant_name] = payload
        ov = payload["overall"]
        print(row_fmt.format(
            variant=variant_name,
            channels=str(channels),
            writers=f"{payload['eligible_writers']}",
            samples=f"{payload['total_samples']}",
            acc=f"{ov['accuracy']*100:.2f}%",
            prec=f"{ov['precision']*100:.2f}%",
            f1=f"{ov['f1']*100:.2f}%",
            eer=f"{ov['eer']*100:.2f}%",
            status="OK",
        ))

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(sep)
    print(
        f"Evaluated: {len(summary['variants'])}  "
        f"Skipped: {len(summary['skipped'])}"
    )
    print(f"Saved ablation summary: {output_json}")


if __name__ == "__main__":
    main()
