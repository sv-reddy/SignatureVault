"""
generate_heatmap.py
===================
Generate Grad-CAM and 4-channel tensor visualizations for SignatureVault.

This script is designed for API usage and prints one JSON payload to stdout.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from extract_features import _read_image, preprocess, extract_channels
from grad_cam import load_model, generate_evidence_report

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".npy"}
CHANNEL_NAMES = ["Shape", "Pseudo-Pressure", "Stroke Angle", "Skeleton"]
CHANNEL_CMAPS = ["gray", "inferno", "hsv", "gray"]


def _scan_images(path: Path) -> List[Path]:
    return sorted([p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def _load_channels(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        data = np.load(str(path))
        if data.ndim != 3 or data.shape[0] != 4:
            raise ValueError(f"Expected (4,H,W) tensor in {path}, got {data.shape!r}")
        return data.astype(np.uint8)

    image = _read_image(path)
    mask = preprocess(image)
    return extract_channels(mask).astype(np.uint8)


def _save_channel_artifacts(channels: np.ndarray, out_dir: Path, stem: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    channel_paths = []
    for i, (name, cmap) in enumerate(zip(CHANNEL_NAMES, CHANNEL_CMAPS)):
        channel = channels[i]
        channel_path = out_dir / f"{stem}_ch{i}_{name.lower().replace(' ', '_')}.png"
        plt.figure(figsize=(4, 4))
        plt.imshow(channel, cmap=cmap)
        plt.axis("off")
        plt.title(name)
        plt.tight_layout()
        plt.savefig(channel_path, dpi=140)
        plt.close()
        channel_paths.append(channel_path)

    grid_path = out_dir / f"{stem}_4channel_grid.png"
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(channels[i], cmap=CHANNEL_CMAPS[i])
        ax.set_title(CHANNEL_NAMES[i], fontsize=11, fontweight="bold")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(grid_path, dpi=160)
    plt.close(fig)

    return {
        "grid": str(grid_path),
        "channels": [str(p) for p in channel_paths],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM and 4-channel visualizations")
    parser.add_argument("--sample-dir", required=True, type=str, help="Folder containing vault/ and questioned/")
    parser.add_argument("--questioned", type=str, default=None, help="Optional explicit questioned file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_tavnet.pt", help="Model checkpoint")
    parser.add_argument("--out-dir", type=str, default="results/grad_cam", help="Output directory for Grad-CAM")
    parser.add_argument("--feature-out-dir", type=str, default="results/visualize_features", help="Output directory for tensor channels")
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    vault_dir = sample_dir / "vault"
    questioned_dir = sample_dir / "questioned"

    if not vault_dir.is_dir() or not questioned_dir.is_dir():
        raise FileNotFoundError("sample-dir must contain vault/ and questioned/ folders")

    genuine_paths = _scan_images(vault_dir)
    if not genuine_paths:
        raise ValueError(f"No valid files found in {vault_dir}")

    if args.questioned:
        questioned_path = Path(args.questioned)
        if not questioned_path.exists():
            raise FileNotFoundError(f"Questioned file not found: {questioned_path}")
    else:
        questioned_files = _scan_images(questioned_dir)
        if not questioned_files:
            raise ValueError(f"No valid files found in {questioned_dir}")
        questioned_path = questioned_files[0]

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{questioned_path.stem}_{run_stamp}"

    channels = _load_channels(questioned_path)
    feature_artifacts = _save_channel_artifacts(channels, Path(args.feature_out_dir), stem)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(checkpoint, device)
    gradcam_path = generate_evidence_report(
        model=model,
        writer_id=0,
        genuine_paths=genuine_paths,
        questioned_path=questioned_path,
        device=device,
        out_dir=Path(args.out_dir),
        alpha=0.55,
        enable_rollout=True,
        max_vault=min(8, len(genuine_paths)),
        questioned_stem=questioned_path.stem,
    )

    payload = {
        "questioned_file": questioned_path.name,
        "grad_cam_report": str(gradcam_path),
        "feature_visualization": feature_artifacts,
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        raise
