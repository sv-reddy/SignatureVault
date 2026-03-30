"""
evaluate_tavnet_arch_ablation.py - Architecture Ablation + Dataset Breakdown + FP Cases
=========================================================================================

Evaluates architecture ablation checkpoints and writes:
1) Architecture ablation table payload (ResNet vs ResNet+CBAM vs ResNet+Transformer)
2) Dataset-wise metrics table with 6 rows (CEDAR, BHSig260-H, BHSig260-B, GPDS, ICDAR2011, Independent)
3) Qualitative false-positive examples (top-N highest-score FP cases)

"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
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
from train_tavnet_arch_ablation import ArchitectureAblationTAVNet, VARIANTS

try:
    from tqdm.auto import tqdm as _tqdm

    def _pbar(it, **kw):
        return _tqdm(it, **kw)
except ImportError:
    def _pbar(it, **kw):
        return it


_DATA_ROOT = _BACKEND_DIR.parent / "DATA"
_DEFAULT_PROC_DIR = _DATA_ROOT / "process_data"
_DEFAULT_CKPT_ROOT = _BACKEND_DIR / "checkpoints" / "arch_ablation"
_DEFAULT_MANIFEST = _DEFAULT_CKPT_ROOT / "manifest.json"
_DEFAULT_OUTPUT_JSON = _BACKEND_DIR / "results" / "evaluate" / "arch_ablation_report.json"

MIN_VAULT_SIZE = 5
MAX_VAULT_SIZE = 8
DEFAULT_N_SUBCENTERS = 7
DEFAULT_VAULT_TRIALS = 3

SCORING_WEIGHTS: Dict[str, float] = {
    "centroid": 0.50,
    "subcenter": 0.50,
}

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")


@dataclass
class EmbSample:
    embedding: torch.Tensor
    path: str
    uid: int
    dataset: str
    sample_type: str  # "G" or "F"


def get_dataset_name(uid: int) -> str:
    if 101 <= uid <= 155:
        return "CEDAR"
    if 201 <= uid <= 360:
        return "BHSig260-H"
    if 401 <= uid <= 500:
        return "BHSig260-B"
    if 601 <= uid <= 750:
        return "GPDS"
    if 801 <= uid <= 869:
        return "ICDAR2011"
    if 1001 <= uid <= 1223:
        return "Independent"
    return "Unknown"


def _extract_model_state_dict(checkpoint: dict) -> dict:
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "model_state" in checkpoint:
        return checkpoint["model_state"]
    return checkpoint


def _load_checkpoint(model: ArchitectureAblationTAVNet, checkpoint: dict) -> None:
    state_dict = _extract_model_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logging.warning("Missing keys during checkpoint load: %s", missing)
    if unexpected:
        logging.warning("Unexpected keys during checkpoint load: %s", unexpected)


class EvaluateDataset(Dataset):
    def __init__(self, paths: List[str], uids: List[int], types: List[int]) -> None:
        self.paths = paths
        self.uids = uids
        self.types = types

    def __len__(self) -> int:
        return len(self.paths)

    def _load(self, path: str) -> torch.Tensor:
        arr_u8 = np.load(path).astype(np.uint8)
        arr = arr_u8.astype(np.float32) / 255.0
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int):
        t_orig = self._load(self.paths[idx])
        t_p5 = TF.rotate(t_orig, angle=5.0, interpolation=InterpolationMode.BILINEAR, fill=1.0)
        t_m5 = TF.rotate(t_orig, angle=-5.0, interpolation=InterpolationMode.BILINEAR, fill=1.0)
        t_tta = torch.stack([t_orig, t_p5, t_m5], dim=0)
        return t_tta, self.uids[idx], self.types[idx], self.paths[idx]


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

    if best_score < 0.10 or best_centers is None:
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
    combined = weights["centroid"] * centroid_sim + weights["subcenter"] * subcenter_sim
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


def _infer_question_type(claimed_uid: int, sample: EmbSample) -> str:
    if sample.sample_type == "F" and sample.uid == claimed_uid:
        return "same_writer_forgery"
    if sample.sample_type == "G" and sample.uid == claimed_uid:
        return "same_writer_genuine"
    return "different_writer_genuine"


def _heuristic_fp_reason(case: dict) -> str:
    qtype = case["question_type"]
    c_sim = case["centroid_sim"]
    s_sim = case["subcenter_sim"]

    if qtype == "different_writer_genuine":
        return "Cross-writer style overlap in global structure"
    if s_sim > c_sim + 0.03:
        return "Forgery aligned with a local sub-style more than global centroid"
    return "High global similarity despite impostor evidence"


def _evaluate_variant(
    *,
    variant_name: str,
    checkpoint_path: Path,
    sample_map: dict,
    test_ids: set[int],
    n_subcenters: int,
    vault_trials: int,
    seed: int,
    embed_dim: int,
    batch_size: int,
    num_workers: int,
    fp_top_k: int,
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

    dataset = EvaluateDataset(test_paths, test_uids, test_types)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = VARIANTS[variant_name]
    model = ArchitectureAblationTAVNet(
        use_cbam=cfg["use_cbam"],
        use_transformer=cfg["use_transformer"],
        embed_dim=embed_dim,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    _load_checkpoint(model, checkpoint)

    model.to(device)
    model.eval()

    all_embs: List[torch.Tensor] = []
    all_uids: List[int] = []
    all_types: List[int] = []
    all_paths: List[str] = []

    use_amp = device.type == "cuda"

    with torch.no_grad():
        for tensors, uids, types, paths in _pbar(
            loader,
            total=len(loader),
            desc=f"Embed {variant_name}",
            unit="batch",
        ):
            tensors = tensors.to(device, non_blocking=True)
            bsz = tensors.shape[0]

            tensors_reshaped = tensors.view(bsz * 3, 4, 384, 384)
            with torch.amp.autocast(device.type, enabled=use_amp):
                emb_tta = model(tensors_reshaped)

            emb_tta = emb_tta.view(bsz, 3, embed_dim)
            emb = emb_tta.mean(dim=1)
            emb = F.normalize(emb, p=2, dim=1)

            all_embs.append(emb.cpu().float())
            all_uids.extend([int(v) for v in uids.tolist()])
            all_types.extend([int(v) for v in types.tolist()])
            all_paths.extend(list(paths))

    embs_cat = torch.cat(all_embs)

    writer_bank: Dict[int, Dict[str, List[EmbSample]]] = defaultdict(lambda: {"G": [], "F": []})
    for idx in range(len(all_uids)):
        uid = int(all_uids[idx])
        key = "G" if int(all_types[idx]) == 0 else "F"
        sample = EmbSample(
            embedding=F.normalize(embs_cat[idx].unsqueeze(0), p=2, dim=1).squeeze(0),
            path=all_paths[idx],
            uid=uid,
            dataset=get_dataset_name(uid),
            sample_type=key,
        )
        writer_bank[uid][key].append(sample)

    populated_writers = sorted(uid for uid in test_ids if uid in writer_bank)
    eligible_writers = [
        uid for uid in populated_writers
        if len(writer_bank[uid]["G"]) >= (MIN_VAULT_SIZE + 1) and len(writer_bank[uid]["F"]) >= 1
    ]
    if not eligible_writers:
        raise RuntimeError(f"No eligible writers for variant {variant_name}")

    rng = random.Random(seed)

    records: List[dict] = []
    overall_y_true: List[int] = []
    overall_y_score: List[float] = []
    script_y_true: Dict[str, List[int]] = defaultdict(list)
    script_y_score: Dict[str, List[float]] = defaultdict(list)
    dataset_y_true: Dict[str, List[int]] = defaultdict(list)
    dataset_y_score: Dict[str, List[float]] = defaultdict(list)

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
            vault_samples = [g_list[i] for i in vault_indices]
            vault_idx_set = set(vault_indices)
            remaining_g = [g_list[i] for i in range(len(g_list)) if i not in vault_idx_set]
            if not remaining_g:
                continue

            vault_embs = [s.embedding for s in vault_samples]
            vault_centroid = F.normalize(torch.stack(vault_embs).mean(0, keepdim=True), p=2, dim=1).squeeze(0)
            vault_sub = _dynamic_kmeans_subcenters(vault_embs, max_k=n_subcenters)

            loo_scores: List[float] = []
            for i in range(len(vault_embs)):
                others = [vault_embs[j] for j in range(len(vault_embs)) if j != i]
                if not others:
                    continue
                loo_centroid = F.normalize(torch.stack(others).mean(0, keepdim=True), p=2, dim=1).squeeze(0)
                loo_sub = _dynamic_kmeans_subcenters(others, max_k=n_subcenters)
                sc, _, _ = _compute_combined(vault_embs[i], loo_centroid, loo_sub, SCORING_WEIGHTS)
                loo_scores.append(sc)

            vault_mean = float(np.mean(loo_scores)) if loo_scores else 0.70
            vault_std = float(np.std(loo_scores)) if len(loo_scores) > 1 else 0.05
            adaptive_threshold = float(vault_mean - vault_std)

            q_same_g = rng.choice(remaining_g)
            q_same_f = rng.choice(f_list)
            q_other_uid = rng.choice(other_uids)
            q_other_g = rng.choice(writer_bank[q_other_uid]["G"])

            questioned = [
                (1, q_same_g),
                (0, q_same_f),
                (0, q_other_g),
            ]

            for gt, q_sample in questioned:
                combined, centroid_sim, subcenter_sim = _compute_combined(
                    q_sample.embedding,
                    vault_centroid,
                    vault_sub,
                    SCORING_WEIGHTS,
                )
                z_score = (combined - vault_mean) / max(vault_std, 0.01)
                pred_adaptive = 1 if combined >= adaptive_threshold else 0

                rec = {
                    "claimed_uid": int(uid),
                    "claimed_dataset": get_dataset_name(uid),
                    "true_uid": int(q_sample.uid),
                    "true_dataset": q_sample.dataset,
                    "question_type": _infer_question_type(uid, q_sample),
                    "ground_truth": int(gt),
                    "score": float(combined),
                    "centroid_sim": float(centroid_sim),
                    "subcenter_sim": float(subcenter_sim),
                    "vault_mean": float(vault_mean),
                    "vault_std": float(vault_std),
                    "adaptive_threshold": float(adaptive_threshold),
                    "pred_adaptive": int(pred_adaptive),
                    "z_score": float(z_score),
                    "query_path": q_sample.path,
                    "vault_paths": [s.path for s in vault_samples],
                }
                records.append(rec)

                overall_y_true.append(gt)
                overall_y_score.append(combined)

                script_key = "Latin"
                if 201 <= uid <= 360:
                    script_key = "Hindi"
                elif 401 <= uid <= 500:
                    script_key = "Bengali"

                script_y_true[script_key].append(gt)
                script_y_score[script_key].append(combined)

                ds_key = get_dataset_name(uid)
                dataset_y_true[ds_key].append(gt)
                dataset_y_score[ds_key].append(combined)

    overall_report = compute_dataset_report(overall_y_true, overall_y_score)
    eer_threshold = float(overall_report["threshold"])

    y_true_np = np.array(overall_y_true, dtype=np.int32)
    y_score_np = np.array(overall_y_score, dtype=np.float32)
    y_pred_np = (y_score_np >= eer_threshold).astype(np.int32)

    for i, rec in enumerate(records):
        rec["pred_eer"] = int(y_pred_np[i])

    fp_cases = [r for r in records if r["ground_truth"] == 0 and r["pred_eer"] == 1]
    fp_cases_sorted = sorted(fp_cases, key=lambda r: r["score"], reverse=True)
    fp_examples = fp_cases_sorted[: max(2, fp_top_k)]
    for case in fp_examples:
        case["likely_failure_reason"] = _heuristic_fp_reason(case)

    scripts_payload = {}
    for scr in ["Latin", "Hindi", "Bengali"]:
        if scr in script_y_true:
            scripts_payload[scr] = compute_dataset_report(script_y_true[scr], script_y_score[scr])

    datasets_order = ["CEDAR", "BHSig260-H", "BHSig260-B", "GPDS", "ICDAR2011", "Independent"]
    datasets_payload = {}
    for ds in datasets_order:
        if ds in dataset_y_true:
            datasets_payload[ds] = compute_dataset_report(dataset_y_true[ds], dataset_y_score[ds])
        else:
            datasets_payload[ds] = compute_dataset_report([], [])

    return {
        "variant": variant_name,
        "label": VARIANTS[variant_name]["label"],
        "checkpoint": str(checkpoint_path),
        "eligible_writers": int(len(eligible_writers)),
        "total_samples": int(len(records)),
        "overall": overall_report,
        "scripts": scripts_payload,
        "datasets": datasets_payload,
        "false_positive_examples": fp_examples,
    }


def _render_latex_tables(summary: dict, out_dir: Path) -> None:
    # Architecture ablation table
    arch_lines = [
        "\\begin{table}[htbp]",
        "\\caption{Architectural Ablation Results}",
        "\\label{tab:arch_ablation}",
        "\\begin{center}",
        "\\begin{tabular}{|l|c|c|c|c|}",
        "\\hline",
        "\\textbf{Variant} & \\textbf{Accuracy (\\%)} & \\textbf{Precision (\\%)} & \\textbf{F1-score (\\%)} & \\textbf{EER (\\%)} \\\\",
        "\\hline",
    ]

    for key in ["resnet", "resnet_cbam", "resnet_transformer"]:
        if key not in summary["variants"]:
            continue
        p = summary["variants"][key]
        ov = p["overall"]
        arch_lines.append(
            f"{p['label']} & {ov['accuracy']*100:.2f} & {ov['precision']*100:.2f} & {ov['f1']*100:.2f} & {ov['eer']*100:.2f} \\\\" 
        )
    arch_lines.extend(["\\hline", "\\end{tabular}", "\\end{center}", "\\end{table}"])

    # Dataset table from best variant (lowest EER)
    best_variant = None
    best_eer = float("inf")
    for key, payload in summary["variants"].items():
        eer = payload["overall"]["eer"]
        if eer < best_eer:
            best_eer = eer
            best_variant = key

    ds_lines = [
        "\\begin{table}[htbp]",
        "\\caption{Dataset-wise Results (Best Architecture Variant)}",
        "\\label{tab:dataset_breakdown}",
        "\\begin{center}",
        "\\begin{tabular}{|l|c|c|c|c|}",
        "\\hline",
        "\\textbf{Dataset} & \\textbf{Accuracy (\\%)} & \\textbf{Precision (\\%)} & \\textbf{F1-score (\\%)} & \\textbf{EER (\\%)} \\\\",
        "\\hline",
    ]

    if best_variant is not None:
        ds_payload = summary["variants"][best_variant]["datasets"]
        for ds in ["CEDAR", "BHSig260-H", "BHSig260-B", "GPDS", "ICDAR2011", "Independent"]:
            m = ds_payload[ds]
            ds_lines.append(
                f"{ds} & {m['accuracy']*100:.2f} & {m['precision']*100:.2f} & {m['f1']*100:.2f} & {m['eer']*100:.2f} \\\\" 
            )

    ds_lines.extend(["\\hline", "\\end{tabular}", "\\end{center}", "\\end{table}"])

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "arch_ablation_table.tex").write_text("\n".join(arch_lines), encoding="utf-8")
    (out_dir / "dataset_breakdown_table.tex").write_text("\n".join(ds_lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate architecture ablation variants with vault protocol.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--variant", type=str, default="all", choices=["resnet", "resnet_cbam", "resnet_transformer", "all"], help="Variant to evaluate.")
    p.add_argument("--checkpoint-root", type=str, default=str(_DEFAULT_CKPT_ROOT), help="Root directory containing variant checkpoints.")
    p.add_argument("--manifest", type=str, default=str(_DEFAULT_MANIFEST), help="Split manifest JSON path.")
    p.add_argument("--processed-dir", type=str, default=str(_DEFAULT_PROC_DIR), help="Processed .npy tensors directory.")
    p.add_argument("--embed-dim", type=int, default=512, help="Embedding dimensionality.")
    p.add_argument("--n-subcenters", type=int, default=DEFAULT_N_SUBCENTERS, help="Dynamic K upper bound.")
    p.add_argument("--vault-trials", type=int, default=DEFAULT_VAULT_TRIALS, help="Trials per writer.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--batch-size", type=int, default=8, help="Embedding batch size.")
    p.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    p.add_argument("--fp-top-k", type=int, default=3, help="Number of top false-positive examples to keep.")
    p.add_argument("--output-json", type=str, default=str(_DEFAULT_OUTPUT_JSON), help="Output JSON report path.")
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
        run_list = ["resnet", "resnet_cbam", "resnet_transformer"]
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

    for variant_name in run_list:
        ckpt_path = checkpoint_root / variant_name / f"best_tavnet_{variant_name}.pt"
        logging.info("Evaluating variant=%s checkpoint=%s", variant_name, ckpt_path)

        if not ckpt_path.exists():
            msg = f"Checkpoint not found: {ckpt_path}"
            if args.variant == "all":
                logging.warning("Skipping %s: %s", variant_name, msg)
                summary["skipped"][variant_name] = msg
                continue
            raise FileNotFoundError(msg)

        payload = _evaluate_variant(
            variant_name=variant_name,
            checkpoint_path=ckpt_path,
            sample_map=sample_map,
            test_ids=test_ids,
            n_subcenters=n_subcenters,
            vault_trials=vault_trials,
            seed=int(args.seed),
            embed_dim=int(args.embed_dim),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            fp_top_k=int(args.fp_top_k),
        )
        summary["variants"][variant_name] = payload

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _render_latex_tables(summary, output_json.parent)

    print("\nArchitecture ablation evaluation complete")
    print(f"Saved JSON report: {output_json}")
    print(f"Saved LaTeX table snippets: {output_json.parent / 'arch_ablation_table.tex'}")
    print(f"Saved LaTeX table snippets: {output_json.parent / 'dataset_breakdown_table.tex'}")


if __name__ == "__main__":
    main()
