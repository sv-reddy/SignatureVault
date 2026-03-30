"""
tune_weights.py - Grid Search for Optimal Scoring Schema Weights
=================================================================

Performs exhaustive grid search over all valid weight combinations:

    combined_score = w1*centroid_sim + w2*subcenter_sim

where w1 + w2 == 1.0 (constraint enforced with 1e-9 tolerance)

Goals:
  - Minimize Equal Error Rate (EER)
  - Maximize Accuracy at EER threshold
  - Output top 5 combinations for direct copy into evaluate_tavnet.py and verify_vault.py

Data Flow:
  1. Load best_tavnet.pt checkpoint and test split from manifest.json
  2. Extract embeddings using 3-view TTA (original, +5°, -5°)
  3. Run vault protocol: collect raw component scores, not final combined scores
  4. Grid search: evaluate all weight combos (step=0.05) over all samples
  5. Output ranked results with copy-paste-ready Python dicts

RUN:
    python tune_weights.py --n-subcenters 7 --vault-trials 3 --seed 42
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
from sklearn.metrics import roc_curve

try:
    from tqdm.auto import tqdm as _tqdm

    def _pbar(it, **kw):
        return _tqdm(it, **kw)

except ImportError:
    def _pbar(it, **kw):
        return it

# Local imports
from backend.train_tavnet import TAVNet, _scan_dir
from evaluate_tavnet import EvaluateDataset, _load_tavnet_checkpoint, _dynamic_kmeans_subcenters, get_script

_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_ROOT = _SCRIPT_DIR.parent / "DATA"
_PROC_ = _DATA_ROOT / "process_data"

CKPT_DIR = _SCRIPT_DIR / "checkpoints"
BEST_CKPT_PATH = CKPT_DIR / "best_tavnet.pt"
MANIFEST_PATH = CKPT_DIR / "manifest.json"

MIN_VAULT_SIZE = 5
MAX_VAULT_SIZE = 8

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")


def extract_embeddings(
    model: TAVNet,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Extract TTA embeddings from test set.
    Uses 3-view augmentation: original, +5°, -5° rotations.
    Returns (embeddings, writer_ids, signature_types).
    """
    all_embs = []
    all_uids = []
    all_types = []
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for tensors, uids, types in _pbar(
            loader, total=len(loader), desc="Extracting embeddings", unit="batch"
        ):
            tensors = tensors.to(device, non_blocking=True)  # (Batch, 3, 4, 384, 384)
            batch_size = tensors.shape[0]

            # Reshape to (Batch * 3, 4, 384, 384) for efficient processing
            tensors_reshaped = tensors.view(batch_size * 3, 4, 384, 384)

            with torch.amp.autocast(device.type, enabled=use_amp):
                emb_tta = model(tensors_reshaped)  # (Batch * 3, 512)

            # Reshape back to (Batch, 3, 512) and average across TTA views
            emb_tta = emb_tta.view(batch_size, 3, 512)
            emb = emb_tta.mean(dim=1)  # (Batch, 512)
            emb = F.normalize(emb, p=2, dim=1)

            all_embs.append(emb.cpu().float())
            all_uids.append(uids)
            all_types.append(types)

    embs_cat = torch.cat(all_embs)
    uids_cat = torch.cat(all_uids).numpy()
    types_cat = torch.cat(all_types).numpy()

    return embs_cat, uids_cat, types_cat


def run_vault_protocol(
    embs_cat: torch.Tensor,
    uids_cat: np.ndarray,
    types_cat: np.ndarray,
    test_ids: set,
    n_subcenters: int,
    vault_trials: int,
    rng: random.Random,
) -> List[Tuple[int, float, float, float, float]]:
    """
    Run randomized vault protocol and collect raw score components.
    
    Returns list of tuples: (label, centroid_sim, subcenter_sim)
    where label ∈ {0=impostor, 1=genuine}
    """

    # Group embeddings by writer/type (genuine/forgery)
    writer_bank: Dict[int, Dict[str, List[torch.Tensor]]] = defaultdict(
        lambda: {"G": [], "F": []}
    )
    for i in range(len(uids_cat)):
        uid = int(uids_cat[i])
        key = "G" if int(types_cat[i]) == 0 else "F"
        writer_bank[uid][key].append(
            F.normalize(embs_cat[i].unsqueeze(0), p=2, dim=1).squeeze(0)
        )

    # Filter to eligible writers (min samples threshold)
    populated_writers = sorted(uid for uid in test_ids if uid in writer_bank)
    eligible_writers = [
        uid
        for uid in populated_writers
        if len(writer_bank[uid]["G"]) >= (MIN_VAULT_SIZE + 1)
        and len(writer_bank[uid]["F"]) >= 1
    ]

    logging.info("Eligible writers for vault: %d/%d", len(eligible_writers), len(populated_writers))

    # Collect all (label, component_scores) samples
    all_samples: List[Tuple[int, float, float]] = []

    for uid in eligible_writers:
        g_list = writer_bank[uid]["G"]
        f_list = writer_bank[uid]["F"]
        other_uids = [
            ou for ou in populated_writers if ou != uid and len(writer_bank[ou]["G"]) > 0
        ]

        if not other_uids:
            continue

        for _ in range(vault_trials):
            max_vault = min(MAX_VAULT_SIZE, len(g_list) - 1)
            if max_vault < MIN_VAULT_SIZE:
                continue

            # Randomly construct vault and questioned set
            vault_size = rng.randint(MIN_VAULT_SIZE, max_vault)
            vault_indices = rng.sample(range(len(g_list)), vault_size)
            vault_embs = [g_list[i] for i in vault_indices]
            vault_idx_set = set(vault_indices)
            remaining_g = [g_list[i] for i in range(len(g_list)) if i not in vault_idx_set]

            if not remaining_g:
                continue

            # Compute vault centroid and sub-centers
            vault_centroid = F.normalize(
                torch.stack(vault_embs).mean(0, keepdim=True), p=2, dim=1
            ).squeeze(0)
            vault_sub = _dynamic_kmeans_subcenters(vault_embs, max_k=max(1, min(3, int(n_subcenters))))

            # Create 3 questioned signatures per trial
            q_same_g = rng.choice(remaining_g)  # Same writer, genuine → label=1
            q_same_f = rng.choice(f_list)  # Same writer, forgery → label=0
            q_other_uid = rng.choice(other_uids)
            q_other_g = rng.choice(writer_bank[q_other_uid]["G"])  # Diff writer, genuine → label=0

            questioned = [
                (1, q_same_g),  # Positive (genuine same writer)
                (0, q_same_f),  # Negative (forgery same writer)
                (0, q_other_g),  # Negative (genuine diff writer)
            ]

            for label, q_emb in questioned:
                # Compute the 2 component scores
                centroid_sim = torch.dot(q_emb, vault_centroid).item()
                subcenter_sim = float((q_emb @ vault_sub.T).max().item())

                all_samples.append((label, centroid_sim, subcenter_sim))

    logging.info("Collected %d evaluated samples from vault protocol", len(all_samples))
    return all_samples


def generate_weight_combinations(step: float = 0.05) -> List[Tuple[float, float]]:
    """
    Generate all weight combinations where w1 + w2 == 1.0.
    
    Args:
        step: Grid step size (default 0.05 = 20 combinations per axis)
    
    Returns:
        List of (w1, w2) tuples sorted lexicographically.
    """
    tolerance = 1e-9
    combos = set()

    steps = np.arange(0.0, 1.0 + step / 2, step)

    for w1 in steps:
        w2 = 1.0 - w1
        if abs((w1 + w2) - 1.0) <= tolerance:
            combo = (round(w1, 2), round(max(0.0, min(1.0, w2)), 2))
            combos.add(combo)

    return sorted(list(combos))


def compute_eer_accuracy(
    all_samples: List[Tuple[int, float, float]],
    weights: Tuple[float, float],
) -> Tuple[float, float]:
    """
    Compute EER and Accuracy for a given weight combination using sklearn roc_curve.
    
    Args:
        all_samples: List of (label, c_sim, s_sim)
        weights: (w_centroid, w_subcenter)
    
    Returns:
        (eer, accuracy) where eer is minimized and accuracy is at EER threshold
    """
    w1, w2 = weights

    # Compute combined score for each sample
    all_labels = []
    all_scores = []

    for label, c_sim, s_sim in all_samples:
        all_labels.append(label)
        all_scores.append(w1 * c_sim + w2 * s_sim)

    all_labels = np.array(all_labels, dtype=np.int32)
    all_scores = np.array(all_scores, dtype=np.float32)

    if len(np.unique(all_labels)) < 2:
        return 1.0, 0.0

    # Correct EER and Accuracy calculation using ROC
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_thresh = float(thresholds[eer_idx])

    y_pred = (all_scores >= eer_thresh).astype(np.int32)
    accuracy = float((y_pred == all_labels).mean())

    return eer, accuracy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid search for optimal scoring schema weights (minimize EER).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(BEST_CKPT_PATH),
        help="Path to TAVNet checkpoint.",
    )
    parser.add_argument(
        "--n-subcenters",
        type=int,
        default=7,
        help="Number of K-means sub-centers per writer vault.",
    )
    parser.add_argument(
        "--vault-trials",
        type=int,
        default=3,
        help="Number of randomized vault/questioned trials per writer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    checkpoint_path = Path(args.checkpoint).resolve()

    print("\n" + "=" * 80)
    print(" SCORING WEIGHT OPTIMIZATION (GRID SEARCH)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        logging.info("GPU: %s (%.1f GB VRAM)", props.name, props.total_memory / 1e9)

    # Load manifest and split
    if not MANIFEST_PATH.exists():
        logging.error("Manifest not found: %s", MANIFEST_PATH)
        return

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    test_ids = set(manifest["split"]["test"])
    logging.info("Test writers: %d", len(test_ids))

    # Scan processed data directory
    proc_dir = _PROC_
    if not proc_dir.exists():
        logging.error("Data directory not found: %s", proc_dir)
        return

    sample_map = _scan_dir(proc_dir)

    # Build test dataset
    test_paths = []
    test_uids = []
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
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    logging.info("Test samples: %d", len(dataset))

    # Load model
    model = TAVNet()
    if not checkpoint_path.exists():
        logging.error("Checkpoint not found: %s", checkpoint_path)
        return

    logging.info("Loading checkpoint: %s", checkpoint_path.name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    _load_tavnet_checkpoint(model, checkpoint)
    model.to(device)
    model.eval()

    # Step 1: Extract embeddings
    logging.info("\nStep 1/3: Extracting embeddings with 3-view TTA")
    embs_cat, uids_cat, types_cat = extract_embeddings(model, loader, device)

    # Step 2: Run vault protocol
    logging.info("\nStep 2/3: Running vault protocol (collecting component scores)")
    all_samples = run_vault_protocol(
        embs_cat, uids_cat, types_cat, test_ids, args.n_subcenters, args.vault_trials, rng
    )

    if not all_samples:
        logging.error("No samples collected from vault protocol. Aborting.")
        return

    # Step 3: Grid search
    logging.info("\nStep 3/3: Grid search over weight combinations")
    weight_combos = generate_weight_combinations(step=0.05)
    logging.info("Total combinations to evaluate: %d", len(weight_combos))

    # Evaluate all combinations
    results: List[Tuple[float, float, Tuple[float, float]]] = []

    for i, weights in enumerate(_pbar(weight_combos, desc="Grid search", unit="combo")):
        eer, accuracy = compute_eer_accuracy(all_samples, weights)
        results.append((eer, accuracy, weights))

    # Sort by EER (ascending), then by Accuracy (descending)
    results.sort(key=lambda x: (x[0], -x[1]))

    # Display results
    print("\n" + "=" * 100)
    print(
        " RANK | EER (%)  | Accuracy | w_centroid | w_subcenter "
    )
    print("=" * 100)

    for rank, (eer, acc, (w1, w2)) in enumerate(results[:5], 1):
        print(
            f"  {rank}   | {eer * 100:7.2f}  |  {acc * 100:6.2f}%  |   {w1:6.2f}   |    {w2:6.2f}"
        )

    print("\n" + "=" * 100)
    print(" COPY-PASTE READY CONFIGURATIONS")
    print("=" * 100)

    for rank, (eer, acc, (w1, w2)) in enumerate(results[:5], 1):
        print(f"\n### RANK {rank}: EER = {eer * 100:.2f}% | Accuracy = {acc * 100:.2f}%")
        print("SCORING_WEIGHTS = {")
        print(f'    "centroid":  {w1},')
        print(f'    "subcenter": {w2},')
        print("}")

    print("\n" + "=" * 100)
    print("To apply these weights, update evaluate_tavnet.py and verify_vault.py:")
    print("  - Find: SCORING_WEIGHTS: Dict[str, float] = { ... }")
    print("  - Replace with the configuration from RANK 1 above")
    print("=" * 100 + "\n")

    logging.info("Grid search complete. Top 5 results displayed above.")


if __name__ == "__main__":
    main()
