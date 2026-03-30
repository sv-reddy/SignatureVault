"""
visualize_features.py  –  4-Channel Feature Quality Visualiser
===============================================================

HOW IT WORKS
------------
Renders a 1×5 figure panel for one 4-channel tensor (from .npy or direct image):
    [ cleaned_image │ Ch-0 Shape │ Ch-1 Pressure │ Ch-2 Angle │ Ch-3 Skeleton ]
Each channel subplot displays uint8 data mapped to [0, 1] with a per-channel
min / max / mean statistics overlay and a colourbar so normalisation can be
verified at a glance.  The source image is located automatically in
DATA/unification_Data/ by matching the .npy file stem.

Channels
  Ch-0  Shape          – Inverted binary ink mask (ink = 255).   Colormap: gray.
  Ch-1  Pseudo-Pressure – Gaussian-blur density approximation.   Colormap: inferno.
  Ch-2  Stroke Angle   – Sobel-gradient orientation [0–255].     Colormap: hsv.
  Ch-3  Skeleton       – Zhang-Suen / morphological 1-px paths.  Colormap: gray.

Output is saved to backend/results/visualize_features/vf_<stem>_<timestamp>.png.

Supported file-name prefixes (produced by unify_dataset.py)
  BS_<UID>_<G|F>_<seq>    BHSig-Bengali / CEDAR / GPDS  (UID   1–400)
  BNG_<UID>_<G|F>_<seq>   BHSig-Bengali                 (UID 401–500)
  HND_<UID>_<G|F>_<seq>   BHSig-Hindi                   (UID 501–600)
  GPDS_<UID>_<G|F>_<seq>  GPDS (extended)                (UID 601–800)
  ICDAR_<UID>_<G|F>_<seq> ICDAR-2011  (flat {001..069}/{001..069}_forg) (Latin)
  IND_<UID>_<G|F>_<seq>   Independent dataset           (UID 1001–1223)

HOW TO RUN
----------
Visualise directly from an input file path:
    python visualize_features.py --file "image_path.jpeg"

Requires:
  DATA/process_data/*.npy        (produced by extract_features.py)
  DATA/unification_Data/*        (source images, optional but recommended)
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

_SCRIPT_DIR   = Path(__file__).resolve().parent
_DATA_ROOT    = _SCRIPT_DIR.parent / "DATA"
UNIFIED_DIR   = _DATA_ROOT / "unification_Data"

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

_CHANNEL_INFO = [
    {
        "index": 0,
        "title": "Ch-0  Shape\n(Binary Ink Mask)",
        "cmap": "gray",
        "note": "Check: clean edges, no noise speckles",
    },
    {
        "index": 1,
        "title": "Ch-1  Pseudo-Pressure\n(Stroke Density)",
        "cmap": "inferno",
        "note": "Check: bright peaks at stroke junctions",
    },
    {
        "index": 2,
        "title": "Ch-2  Stroke Angle\n(Sobel Orientation)",
        "cmap": "hsv",
        "note": "Check: smooth hue transitions along strokes",
    },
    {
        "index": 3,
        "title": "Ch-3  Skeleton\n(Zhang-Suen Thinning)",
        "cmap": "gray",
        "note": "Check: 1-px paths, no hairy artifacts",
    },
]


def _find_source_image(stem: str) -> Path | None:
    for ext in _IMAGE_EXTS:
        candidate = UNIFIED_DIR / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def _load_original_rgb(path: Path) -> np.ndarray | None:
    try:
        from PIL import Image
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with Image.open(path) as img:
                return np.array(img.convert("RGB"), dtype=np.uint8)
    except Exception:
        pass
    try:
        import cv2
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is not None:
            return bgr[:, :, ::-1]
    except Exception:
        pass
    return None


def _overlay_stats(ax: plt.Axes, data: np.ndarray, color: str = "white") -> None:
    txt = f"min  {data.min():.3f}\nmax  {data.max():.3f}\nmean {data.mean():.3f}"
    ax.text(
        0.02, 0.98, txt,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=7.5,
        color=color,
        fontfamily="monospace",
        bbox={"facecolor": "black", "alpha": 0.50, "pad": 3, "boxstyle": "round,pad=0.3"},
    )


def _render_visualization(
    raw: np.ndarray,
    stem: str,
    file_label: str,
    source_hint: str,
    cleaned_image_rgb: np.ndarray,
) -> None:
    if raw.ndim != 3 or raw.shape[0] != 4:
        sys.exit(
            f"[ERROR] Unexpected tensor shape {raw.shape!r} in {file_label}.\n"
            "Expected (4, H, W) uint8 from extract_features.py."
        )

    ch0_shape    = raw[0].astype(np.float32) / 255.0
    ch1_pressure = raw[1].astype(np.float32) / 255.0
    ch2_angle    = raw[2].astype(np.float32) / 255.0
    ch3_skeleton = raw[3].astype(np.float32) / 255.0
    tensor_f     = np.stack([ch0_shape, ch1_pressure, ch2_angle, ch3_skeleton])

    print(f"File   : {file_label}")
    print(f"Shape  : {raw.shape}")
    print(f"Source : {source_hint}")


    fig = plt.figure(figsize=(22, 5), layout="constrained")
    fig.suptitle(
        f"SignVault Feature Quality Check  —  {stem}",
        fontsize=13, fontweight="bold",
    )
    gs   = gridspec.GridSpec(1, 5, figure=fig, wspace=0.04)
    axes = [fig.add_subplot(gs[0, c]) for c in range(5)]


    ax0 = axes[0]
    ax0.imshow(cleaned_image_rgb, aspect="equal")
    ax0.set_title("Ax-0  cleaned_image", fontsize=9, pad=4)
    gray = cleaned_image_rgb.mean(axis=2) / 255.0
    _overlay_stats(ax0, gray, color="yellow")


    channel_data = [ch0_shape, ch1_pressure, ch2_angle, ch3_skeleton]

    for plot_col, (info, ch_f) in enumerate(zip(_CHANNEL_INFO, channel_data), start=1):
        ax = axes[plot_col]

        im = ax.imshow(
            ch_f,
            cmap=info["cmap"],
            vmin=0.0, vmax=1.0,
            interpolation="nearest",
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, format="%.2f")

        ax.set_title(f"Ax-{plot_col}  {info['title']}", fontsize=9, pad=4)

        _overlay_stats(ax, ch_f)

        ax.text(
            0.5, -0.04, info["note"],
            transform=ax.transAxes,
            ha="center", va="top", fontsize=6.5,
            color="#555555", style="italic",
        )

    for ax in axes:
        ax.axis("off")

    results_dir = _SCRIPT_DIR / "results" / "visualize_features"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_path = results_dir / f"vf_{stem}_{timestamp}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved : {out_path}")

    plt.show()


def visualize(npy_path: Path) -> None:
    raw = np.load(str(npy_path))
    stem = npy_path.stem
    src_path = _find_source_image(stem)
    cleaned_binary = 255 - raw[0]
    cleaned_image_rgb = np.repeat(cleaned_binary[:, :, None], 3, axis=2)

    source_hint = (
        str(src_path.relative_to(_DATA_ROOT))
        if src_path
        else f"not found in unification_Data/  (stem={stem!r})"
    )

    _render_visualization(
        raw=raw,
        stem=stem,
        file_label=npy_path.name,
        source_hint=source_hint,
        cleaned_image_rgb=cleaned_image_rgb,
    )


def visualize_image(image_path: Path) -> None:
    try:
        from extract_features import _read_image, extract_channels, preprocess
    except Exception as exc:
        sys.exit(f"[ERROR] Unable to import extract_features helpers: {exc}")

    img_bgr = _read_image(image_path)
    mask = preprocess(img_bgr)
    raw = extract_channels(mask)
    cleaned_image_rgb = np.repeat(mask[:, :, None], 3, axis=2)

    _render_visualization(
        raw=raw,
        stem=image_path.stem,
        file_label=image_path.name,
        source_hint=str(image_path),
        cleaned_image_rgb=cleaned_image_rgb,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualise a 4-channel tensor from an input file path (.npy or image)."
        )
    )
    parser.add_argument(
        "--file", metavar="FILENAME",
        required=True,
        help="Input file path (.npy or image file).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    candidate = Path(args.file)

    if not candidate.exists():
        sys.exit(f"[ERROR] File not found: {candidate}")

    if candidate.suffix.lower() == ".npy":
        visualize(candidate.resolve())
        return

    if candidate.suffix.lower() in _IMAGE_EXTS:
        visualize_image(candidate.resolve())
        return

    sys.exit(
        f"[ERROR] Unsupported input extension: {candidate.suffix}\n"
        f"Use one of {', '.join(_IMAGE_EXTS)} or .npy"
    )


if __name__ == "__main__":
    main()