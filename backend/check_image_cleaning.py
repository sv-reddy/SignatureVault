"""
check_image_cleaning.py
=======================
Quick quality check for image cleaning used in extract_features.py.

What it does:
- Loads one image.
- Reuses extract_features._read_image() and extract_features.preprocess().
- Computes simple cleaning metrics (ink ratio, connected components, tiny-noise count).
- Saves a comparison figure.

Usage:
    python check_image_cleaning.py --file-name image_name.png

If --file-name is not an absolute/relative path to an existing file,
it is searched under DATA/unification_Data/.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from extract_features import _read_image, preprocess

_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_ROOT = _SCRIPT_DIR.parent / "DATA"
_UNIFIED_DIR = _DATA_ROOT / "unification_Data"
_RESULTS_DIR = _SCRIPT_DIR / "results" / "cleaned_image"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check cleaning quality for one image using extract_features preprocess()."
    )
    parser.add_argument(
        "--file-name",
        required=True,
        help="Image file path or filename (searched in DATA/unification_Data).",
    )
    return parser.parse_args()


def _resolve_image_path(file_name: str) -> Path:
    candidate = Path(file_name)

    if candidate.exists():
        return candidate.resolve()

    fallback = _UNIFIED_DIR / file_name
    if fallback.exists():
        return fallback.resolve()

    raise SystemExit(
        f"[ERROR] Image not found: {file_name}\n"
        f"Checked:\n  {candidate}\n  {fallback}"
    )


def _compute_metrics(binary_mask: np.ndarray) -> dict[str, float | int]:
    # preprocess() returns binary with background=255 and ink=0.
    ink = (255 - binary_mask).astype(np.uint8)
    ink_bool = ink > 0

    total_px = ink_bool.size
    ink_px = int(np.count_nonzero(ink_bool))
    ink_ratio = (ink_px / total_px) if total_px else 0.0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)

    if num_labels <= 1:
        components = 0
        tiny_noise_components = 0
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
        components = int(len(areas))
        tiny_noise_components = int(np.count_nonzero(areas < 10))

    return {
        "ink_pixels": ink_px,
        "total_pixels": int(total_px),
        "ink_ratio": float(ink_ratio),
        "components": int(components),
        "tiny_noise_components": int(tiny_noise_components),
    }


def _save_visual_report(
    img_bgr: np.ndarray,
    binary_mask: np.ndarray,
    image_path: Path,
) -> Path:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    original_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    cleaned_rgb = np.full((*binary_mask.shape, 3), 255, dtype=np.uint8)
    cleaned_rgb[binary_mask == 0] = np.array([0, 0, 0], dtype=np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    axes[0].imshow(original_rgb)
    axes[0].set_title("Original")
    axes[0].text(
        0.5,
        -0.08,
        "Step 1: Input photo",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color="#3b342a",
    )
    h0, w0 = original_rgb.shape[:2]
    axes[0].add_patch(
        Rectangle((-0.5, -0.5), w0, h0, fill=False, edgecolor="black", linewidth=2)
    )
    axes[0].axis("off")

    axes[1].imshow(cleaned_rgb)
    axes[1].set_title("Cleaned Output (Phase-1)")
    axes[1].text(
        0.5,
        -0.08,
        "Step 2: Normalize light -> threshold -> crop/resize",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color="#3b342a",
    )
    h1, w1 = cleaned_rgb.shape[:2]
    axes[1].add_patch(
        Rectangle((-0.5, -0.5), w1, h1, fill=False, edgecolor="black", linewidth=2)
    )
    axes[1].axis("off")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = _RESULTS_DIR / f"clean_{image_path.stem}_{timestamp}.png"
    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)

    return out_path


def main() -> None:
    args = _parse_args()
    image_path = _resolve_image_path(args.file_name)

    img_bgr = _read_image(image_path)
    cleaned_binary = preprocess(img_bgr)

    report_path = _save_visual_report(
        img_bgr,
        cleaned_binary,
        image_path,
    )

    print(f"Input image            : {image_path}")
    print(f"Output image size      : {cleaned_binary.shape[1]}x{cleaned_binary.shape[0]} px")
    print(f"Saved visual report    : {report_path}")


if __name__ == "__main__":
    main()
