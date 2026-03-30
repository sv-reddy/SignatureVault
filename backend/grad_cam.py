"""
grad_cam.py — Forensic-Grade XAI Visualizations for TAV-Net
==================================================================

Generates evidence-quality explainability reports for the TAV-Net
Siamese-Transformer signature verification model.

Visualization pipeline
1. Grad-CAM at model.layer4          — spatial decision map (7×7 ? 224×224)
2. Per-channel Attribution Maps      — ?sim/?input_c × input_c  for each of
                                       the 4 signature channels (Shape, Pressure,
                                       Angle, Skeleton)
3. Contrastive ?-Map                 — vault-centroid heatmap minus questioned
                                       heatmap; red = vault feature, blue = spurious
4. Transformer Attention Rollout     — propagated multi-head attention collapsed
                                       onto the 7×7 token spatial grid

Evidence report layout (22 × 18 in @ 200 dpi ˜ 4400 × 3600 px)
  [Banner — writer ID, similarity score, overall verdict]
  Row 0 : Questioned (raw) | Questioned + Grad-CAM | Vault + Grad-CAM | ?-Map
  Row 1 : Ch-0 Shape       | Ch-1 Pressure         | Ch-2 Angle       | Ch-3 Skeleton
  Row 2 : Attention Rollout (left 2 cols)  |  Salience Score Table (right 2 cols)

Supported dataset UID ranges (from unify_dataset.py manifest)
  UID   1 –  400  : BHSig-Bengali / CEDAR / GPDS          (Latin / Bengali)
  UID 401 –  500  : BHSig-Bengali                         (Bengali)
  UID 501 –  600  : BHSig-Hindi                           (Hindi)
  UID 601 –  800  : GPDS (extended)                       (Latin)
  UID 801 –  869  : ICDAR-2011  (flat {id}/{id}_forg)    (Latin)
  UID 1001 – 1223 : Independent dataset                   (Latin)

Usage
  # Compare first forgery for writer 401 against their vault centroid
  python grad_cam.py --writer-id 401

  # Explicit questioned sample path
  python grad_cam.py --writer-id 401 --questioned path/to/sig.npy

  # Specify checkpoint and output directory
  python grad_cam.py --writer-id 401 \\
      --checkpoint checkpoints/best_tavnet.pt \\
      --out-dir results/

Hardware
  Gradient computation runs on the RTX 3050 via CUDA.  CPU fallback is
  supported but slower.  All tensors are streamed one-at-a-time to keep
  VRAM well under 6 GB.
"""

from __future__ import annotations

import argparse
import logging
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # headless rendering — must precede pyplot import

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

_SCRIPT_DIR  = Path(__file__).resolve().parent
_DATA_ROOT   = _SCRIPT_DIR.parent / "DATA"
_PROC_       = _DATA_ROOT / "process_data"        # 4-ch uint8  (primary)
CKPT_DIR     = _SCRIPT_DIR / "checkpoints"
DEFAULT_CKPT = CKPT_DIR / "best_tavnet.pt"
DEFAULT_OUT  = _SCRIPT_DIR / "results" / "grad_cam"

_CHANNEL_NAMES  = ["Shape",   "Pressure", "Angle",  "Skeleton"]
_CHANNEL_CMAPS  = ["gray",    "inferno",  "hsv",    "gray"]   # raw display colormaps
_CHANNEL_COLORS = ["#4C9BE8", "#E87B4C",  "#8FB054", "#9B8FE8"]  # accent per channel

# Supported raw image extensions (case-insensitive) for --sample-dir mode
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

# Import model from training script
# The training script is guarded with  if __name__ == "__main__":  so only
# class/function definitions are executed on import.

try:
    from backend.train_tavnet import (   # noqa: E402
        TAVNet,
        CBAM,
        ChannelAttention,
        SpatialAttention,
    )
except ImportError as _err:
    sys.exit(
        "[ERROR] Cannot import model from train_tavnet.py.\n"
        "        Run this script from the backend/ directory:\n"
        f"            python grad_cam.py --writer-id <ID>\n"
        f"        Import error: {_err}"
    )

# Lazy import of the feature-extraction pipeline (only needed when processing
# raw images rather than pre-built .npy tensors).
try:
    from extract_features import preprocess, extract_channels, _read_image
    _HAS_EXTRACTOR = True
except ImportError:
    _HAS_EXTRACTOR = False

# Scoring helpers — imported from verify_vault for parity with the verifier.
# Inline fallbacks are provided so grad_cam.py can run standalone.
try:
    from verify_vault import (
        _dynamic_kmeans_subcenters,
        _compute_combined,
        DEFAULT_WEIGHTS,
        DEFAULT_Z_THRESHOLD,
        DEFAULT_N_SUBCENTERS,
    )
except ImportError:
    DEFAULT_Z_THRESHOLD:  float = -1.0
    DEFAULT_N_SUBCENTERS: int   = 3
    DEFAULT_WEIGHTS: dict[str, float] = {
        "centroid":  0.50,
        "subcenter": 0.50,
    }

    def _kmeans_subcenters(embeddings, k, n_iter=30, seed=0):
        """Backward compatibility wrapper; delegates to _dynamic_kmeans_subcenters."""
        return _dynamic_kmeans_subcenters(embeddings, max_k=k, n_iter=n_iter, seed=seed)

    def _dynamic_kmeans_subcenters(embeddings, max_k=3, n_iter=30, seed=0):
        """Dynamic K-means with silhouette-based k selection."""
        from sklearn.metrics import silhouette_score
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

    def _compute_combined(q_emb, centroid, subcenters, named_embs, weights):
        centroid_sim  = torch.dot(q_emb, centroid).item()
        subcenter_sim = float((q_emb @ subcenters.T).max().item())
        combined = (
            weights["centroid"]    * centroid_sim
            + weights["subcenter"] * subcenter_sim
        )
        return combined, centroid_sim, subcenter_sim

log = logging.getLogger("grad_cam")


# -----------------------------------------------------------------------------
# Grad-CAM
# -----------------------------------------------------------------------------

class GradCAM:
    """
    Grad-CAM for TAV-Net, targeting model.layer4.

    Registers forward + backward hooks on the target layer.  After a
    forward+backward pass the 7×7 weighted activation map is accessible
    via .heatmap().

    Usage
    -----
        gcam = GradCAM(model, model.layer4)
        emb  = gcam.forward(tensor)           # returns embedding (1, D)
        sim  = (emb[0] * centroid).sum()
        model.zero_grad(); sim.backward()
        hmap = gcam.heatmap(out_size=224)      # (224, 224) float32 [0, 1]
        gcam.remove()
    """

    def __init__(self, model: TAVNet, target_layer: nn.Module) -> None:
        self.model = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None

        def _fwd(module, inp, out):
            self._activations = out.detach()

        def _bwd(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self._fwd_h = target_layer.register_forward_hook(_fwd)
        self._bwd_h = target_layer.register_full_backward_hook(_bwd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass. Must NOT be inside torch.no_grad()."""
        return self.model(x)

    def heatmap(self, out_size: int = 384) -> np.ndarray:
        """
        Compute the Grad-CAM heatmap.

        Weights each activation channel by the global-average-pooled gradient
        (Selvaraju et al., eq. 1), applies ReLU, upsamples to out_size, and
        min-max normalises to [0, 1].

        Returns: (out_size, out_size) float32 array.
        """
        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "Call forward() followed by backward() before calling heatmap()."
            )
        # (B, C) — importance weights for each activation channel
        weights = self._gradients.mean(dim=(2, 3))          # GAP over (H, W)
        # Weighted sum of feature maps ? (B, H, W)
        cam = (weights[:, :, None, None] * self._activations).sum(dim=1)
        cam = F.relu(cam)                                   # discard negatives
        # Bilinear upsample ? (B, 1, 12, 12) ? (B, out_size, out_size)
        if cam.shape[-1] != out_size:
            cam = F.interpolate(
                cam.unsqueeze(1),
                size=(out_size, out_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        cam = cam[0].cpu().numpy()                          # first (only) image
        vmin, vmax = cam.min(), cam.max()
        if vmax - vmin > 1e-8:
            cam = (cam - vmin) / (vmax - vmin)
        return cam.astype(np.float32)

    def remove(self) -> None:
        self._fwd_h.remove()
        self._bwd_h.remove()


# -----------------------------------------------------------------------------
# Transformer Attention Rollout
# -----------------------------------------------------------------------------

class AttentionRollout:
    """
    Attention Rollout for TAV-Net's nn.TransformerEncoder (Abnar & Zuidema, 2020).

    PyTorch's TransformerEncoderLayer calls self_attn with need_weights=False
    by default, which discards the weight tensor.  This class monkey-patches
    _sa_block on each TransformerEncoderLayer to force need_weights=True so
    that attention maps are captured and rolled out.

    Usage
    -----
        ar = AttentionRollout(model)
        with torch.no_grad():
            model(tensor)
        grid = ar.rollout()     # (7, 7) float32 [0, 1]
        ar.remove()
    """

    def __init__(self, model: TAVNet) -> None:
        self.model = model
        self._weights: list[torch.Tensor] = []
        self._originals: list = []

        weight_store = self._weights   # closure reference

        for layer in model.transformer.layers:
            # Preserve original _sa_block (a bound method of the layer instance)
            self._originals.append(layer._sa_block)

            # Build a patched _sa_block that passes need_weights=True
            # and stores the returned attention weights.
            def _patched_sa_block(
                self_layer,
                x,
                attn_mask,
                key_padding_mask,
                is_causal: bool = False,
                _store=weight_store,
            ):
                # Call self_attn directly with need_weights=True
                x_out, w = self_layer.self_attn(
                    x, x, x,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=True,
                    average_attn_weights=True,   # (B, S, S) — averaged over heads
                    is_causal=is_causal,
                )
                if w is not None:
                    _store.append(w.detach())
                return self_layer.dropout1(x_out)

            # Bind as a method of the layer instance so `self_layer` resolves
            layer._sa_block = types.MethodType(_patched_sa_block, layer)

    def rollout(self) -> np.ndarray:
        """
        Compute Attention Rollout across all captured layers.

        For each layer: A_aug = 0.5 * A_mean + 0.5 * I  (residual addition),
        then rows are re-normalised.  Rollout = product of all A_aug matrices.

        The resulting (144, 144) matrix is averaged over "query" tokens to produce
        a (144,) aggregate attention vector, then reshaped to (12, 12).

        Returns: (12, 12) float32 array in [0, 1].
        """
        if not self._weights:
            raise RuntimeError(
                "No attention weights captured.  Ensure a forward pass was run "
                "after constructing AttentionRollout."
            )
        S   = self._weights[0].shape[-1]            # sequence length (144)
        eye = torch.eye(S, device=self._weights[0].device)
        rollout = eye.unsqueeze(0)                  # (1, S, S)

        for a in self._weights:
            # a: (B, S, S) — already head-averaged by average_attn_weights=True
            a_aug  = 0.5 * a + 0.5 * eye.unsqueeze(0)
            # Re-normalise each row so it sums to 1
            a_aug  = a_aug / a_aug.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            rollout = torch.bmm(a_aug, rollout)

        # Average over query tokens ? (B, S) aggregate token importance
        token_imp = rollout[0].mean(dim=0)          # (144,)
        vmin, vmax = token_imp.min(), token_imp.max()
        if vmax - vmin > 1e-8:
            token_imp = (token_imp - vmin) / (vmax - vmin)

        grid = token_imp.reshape(12, 12).cpu().numpy().astype(np.float32)
        self._weights.clear()
        return grid

    def remove(self) -> None:
        """Restore original _sa_block methods."""
        for layer, orig in zip(self.model.transformer.layers, self._originals):
            layer._sa_block = orig
        self._weights.clear()


# Data loading
def _load_npy(path: str | Path) -> torch.Tensor:
    """
    Load a pre-built .npy signature tensor.
    Handles (uint8, 4-ch) 
    Returns: (4, 384, 384) float32 in [0, 1].
    """
    arr = np.load(str(path))
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
        if arr.shape[0] == 3:
            pad = np.zeros((1, arr.shape[1], arr.shape[2]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
    return torch.from_numpy(arr).float()   # (4, H, W) float32


def _load_image_as_tensor(path: Path) -> torch.Tensor:
    """
    Process a raw signature image (PNG / JPG / TIFF / BMP / …) through the
    full 4-channel feature-extraction pipeline from extract_features.py.

    Pipeline:
        BGR image  ?  preprocess()  ?  (384, 384) uint8 binary mask
                   ?  extract_channels()  ?  (4, 384, 384) uint8
                   ?  float32 / 255  ?  torch.Tensor (4, 384, 384)
    """
    if not _HAS_EXTRACTOR:
        raise RuntimeError(
            "extract_features.py could not be imported. "
            "Make sure it is in the same directory as grad_cam.py."
        )
    img_bgr = _read_image(path)                # BGR uint8 ndarray
    binary  = preprocess(img_bgr)              # (384, 384) uint8 binary mask
    arr     = extract_channels(binary)         # (4, 384, 384) uint8
    t       = torch.from_numpy(arr.astype(np.float32) / 255.0)
    return t                                   # (4, 384, 384) float32 [0, 1]


def _load_file(path: Path) -> torch.Tensor:
    """
    Dispatch loader based on file extension.
    .npy  ? _load_npy()       (fast, pre-processed tensor)
    image ? _load_image_as_tensor()   (inline feature extraction)
    """
    if path.suffix.lower() == ".npy":
        return _load_npy(path)
    if path.suffix.lower() in _IMAGE_EXTS:
        return _load_image_as_tensor(path)
    raise ValueError(f"Unsupported file type: {path.suffix}  ({path.name})")


def _scan_folder(folder: Path) -> list[Path]:
    """
    Return all supported files (raw images + .npy tensors) inside a folder,
    sorted alphabetically.  Does not recurse into sub-directories.
    """
    files: list[Path] = []
    for ext in list(_IMAGE_EXTS) + [".npy"]:
        files.extend(folder.glob(f"*{ext}"))
        files.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(set(files))


def _scan_writer(writer_id: int, proc_dir: Path) -> dict[str, list[Path]]:
    """Return {"G": [...], "F": [...]} of .npy paths for one writer UID."""
    found: dict[str, list[Path]] = {"G": [], "F": []}
    for f in sorted(proc_dir.glob("*.npy")):
        parts = f.stem.split("_")
        if len(parts) < 4:
            continue
        status = parts[-2].upper()
        if status not in ("G", "F"):
            continue
        try:
            uid = int(parts[-3])
        except ValueError:
            continue
        if uid == writer_id:
            found[status].append(f)
    return found


def _resolve_proc_dir(override: Optional[str]) -> Path:
    """Return the processed-data directory, preferring, falling back to v1."""
    if override:
        p = Path(override)
        if not p.is_dir():
            log.error("--proc-dir does not exist: %s", p)
            sys.exit(1)
        return p
    if _PROC_.is_dir() and any(_PROC_.glob("*.npy")):
        return _PROC_
    log.error(
        "No processed .npy directory found. "
        "Run extract_features.py first, or pass --proc-dir."
    )
    sys.exit(1)

# Model loading
def load_model(ckpt_path: Path, device: torch.device) -> tuple[TAVNet, dict]:
    """
    Load TAVNet from checkpoint.
    Returns (model in eval mode, raw checkpoint dict).
    """
    if not ckpt_path.exists():
        log.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)
    ckpt  = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    dim   = ckpt.get("embed_dim", 512)
    model = TAVNet(embed_dim=dim).to(device)
    state_dict = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))

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
            log.warning("Missing keys after compatibility load: %s", missing)
        if unexpected:
            log.warning("Unexpected keys after compatibility load: %s", unexpected)
    else:
        model.load_state_dict(state_dict)
    model.eval()
    return model, ckpt


# Embedding utilities
@torch.no_grad()
def compute_centroid(
    model:   TAVNet,
    paths:   list[Path],
    device:  torch.device,
    batch:   int = 16,
) -> torch.Tensor:
    """
    Mean L2-normalised embedding (vault centroid) for a set of genuine paths.
    Accepts both .npy tensors and raw image files.
    Returns: (D,) float32 on CPU.
    """
    chunks: list[torch.Tensor] = []
    for i in range(0, len(paths), batch):
        tensors = torch.stack([_load_file(p) for p in paths[i : i + batch]]).to(device)
        embs    = model(tensors)          # already L2-normalised
        chunks.append(embs.cpu().float())
    stacked  = torch.cat(chunks, dim=0)  # (N, D)
    centroid = stacked.mean(dim=0)       # (D,)
    return F.normalize(centroid, p=2, dim=0)


# -----------------------------------------------------------------------------
# Per-channel attribution: Gradient × Input
# -----------------------------------------------------------------------------

def channel_attribution_maps(
    model:      TAVNet,
    tensor:     torch.Tensor,    # (4, 224, 224) float32 on device — will grad
    target_emb: torch.Tensor,    # (D,) centroid or comparison embedding on CPU
) -> list[np.ndarray]:
    """
    Compute per-channel Gradient × Input saliency maps.

    For each channel c:
        saliency_c = |?(cosine_similarity) / ?input_c| ? |input_c|

    All 4 channels are differentiated in a single backward pass.

    Returns: list of 4 × (384, 384) float32 arrays in [0, 1].
    """
    model.eval()
    x = tensor.unsqueeze(0).detach().requires_grad_(True)   # (1, 4, 384, 384)
    emb  = model(x)                                         # (1, D)
    sim  = (emb[0] * target_emb.to(x.device)).sum()
    model.zero_grad()
    sim.backward()

    grad = x.grad[0].abs()     # (4, 384, 384)  |?sim/?input|
    inp  = tensor.detach().abs()               # (4, 384, 384)

    maps: list[np.ndarray] = []
    for c in range(4):
        smap = (grad[c] * inp[c]).cpu().numpy()
        vmin, vmax = smap.min(), smap.max()
        if vmax - vmin > 1e-8:
            smap = (smap - vmin) / (vmax - vmin)
        maps.append(smap.astype(np.float32))
    return maps


# Grad-CAM: single image
def run_gradcam(
    model:      TAVNet,
    tensor:     torch.Tensor,    # (4, 384, 384) float32 on device (no batch dim)
    target_emb: torch.Tensor,    # (D,) on CPU
) -> np.ndarray:
    """
    Run one forward+backward Grad-CAM pass; returns (384, 384) heatmap in [0,1].
    The cosine similarity with target_emb is used as the scalar score.
    """
    gcam = GradCAM(model, model.layer4)
    emb  = gcam.forward(tensor.unsqueeze(0))
    sim  = (emb[0] * target_emb.to(tensor.device)).sum()
    model.zero_grad()
    sim.backward()
    hmap = gcam.heatmap(out_size=384)
    gcam.remove()
    return hmap


# -----------------------------------------------------------------------------
# Salience scores
# -----------------------------------------------------------------------------

def salience_scores(
    channel_maps:       list[np.ndarray],
    cosine_similarity:  float,
) -> list[dict]:
    """
    Compute a numeric per-channel 'Salience Score'.

        raw_score  = mean activation in the top-20% of pixels (Grad×Input)
        conf_score = raw_score × max(cosine_similarity, 0)

    Activation strength labels (based on raw_score percentile across channels):
        raw = p67 of channel raws ? HIGH
        raw = p33 of channel raws ? MEDIUM
        raw <  p33 of channel raws ? LOW

    These labels describe *which channels the model focused on*, NOT whether
    the signature is genuine or forged.  The overall GENUINE/FORGERY verdict
    is determined solely by cosine similarity vs. threshold.

    Returns list[dict] with keys: channel, raw, conf, activation.
    """
    results: list[dict] = []
    raws: list[float] = []
    for name, cmap in zip(_CHANNEL_NAMES, channel_maps):
        flat  = cmap.flatten()
        p80   = np.percentile(flat, 80)
        top20 = flat[flat >= p80]
        raw   = float(top20.mean()) if len(top20) > 0 else 0.0
        conf  = raw * max(cosine_similarity, 0.0)
        raws.append(raw)
        results.append({"channel": name, "raw": raw, "conf": conf, "activation": ""})

    # Rank activation strength relative to the other channels
    p33 = float(np.percentile(raws, 33))
    p67 = float(np.percentile(raws, 67))
    for entry in results:
        entry["activation"] = (
            "HIGH"   if entry["raw"] >= p67 else
            "MEDIUM" if entry["raw"] >= p33 else
            "LOW"
        )
    return results


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------

def _upsample_np(arr: np.ndarray, size: int = 384) -> np.ndarray:
    """Bilinear upsample a 2-D numpy array to (size, size)."""
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.squeeze().numpy()


def _overlay(
    base:    np.ndarray,   # (H, W) float32 [0, 1] — grayscale background
    heatmap: np.ndarray,   # (H, W) float32 [0, 1] — saliency
    cmap:    str  = "jet",
    alpha:   float = 0.55,
) -> np.ndarray:
    """
    Blend a grayscale base image with a false-colour heatmap.
    Returns (H, W, 3) uint8 RGB suitable for imshow().
    Alpha is applied uniformly; adjust for stronger/softer overlay.
    """
    base_rgb = np.stack([base, base, base], axis=-1)           # (H, W, 3) [0,1]
    heat_rgb = plt.get_cmap(cmap)(heatmap)[..., :3]            # (H, W, 3) [0,1]
    blended  = (1.0 - alpha) * base_rgb + alpha * heat_rgb
    return (blended.clip(0.0, 1.0) * 255).astype(np.uint8)


def _ax_style(ax: plt.Axes, title: str, title_color: str = "#C8C8D0") -> None:
    """Apply dark forensic styling: dark background, monospace title, no ticks."""
    ax.set_facecolor("#0A0A0F")
    ax.set_title(title, color=title_color, fontsize=8.5,
                 fontfamily="monospace", pad=4, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#303038")


def _add_colorbar(
    fig:   plt.Figure,
    ax:    plt.Axes,
    cmap:  str,
    label: str,
    vmin:  float = 0.0,
    vmax:  float = 1.0,
) -> None:
    """Attach a thin, styled colorbar to an axes."""
    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.03, orientation="vertical")
    cb.set_label(label, color="#A0A0A8", fontsize=6.5, fontfamily="monospace")
    cb.ax.yaxis.set_tick_params(colors="#A0A0A8", labelsize=6)
    cb.outline.set_edgecolor("#404048")


# -----------------------------------------------------------------------------
# Salience table renderer
# -----------------------------------------------------------------------------

def _draw_salience_table(
    ax:              plt.Axes,
    scores:          list[dict],
    combined_score:  float,
    z_score:         float,
    z_threshold:     float,
    overall_verdict: str,
) -> None:
    """Render the per-channel salience score table inside the given axes."""
    # Channel activation strength colours (not forgery verdicts)
    ACT_COLORS = {
        "HIGH":   "#2ECC71",
        "MEDIUM": "#F39C12",
        "LOW":    "#778899",
    }
    OV_COLORS = {
        "GENUINE": "#2ECC71",
        "FORGERY": "#E74C3C",
    }

    ax.set_facecolor("#0D0D15")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#303038")

    def _t(x, y, txt, **kw):
        ax.text(x, y, txt, transform=ax.transAxes,
                fontfamily="monospace", **kw)

    # Title
    _t(0.5, 0.97, "CHANNEL SALIENCE SCORES",
       ha="center", va="top", fontsize=10, fontweight="bold", color="#EAEAEA")
    _t(0.5, 0.87, f"Combined: {combined_score:.4f}   Z-score: {z_score:+.4f}  (thresh {z_threshold:+.1f})",
       ha="center", va="top", fontsize=8.5, color="#A0A0A8")

    # Column headers
    col_x = [0.04, 0.44, 0.74]
    for cx, hdr in zip(col_x, ["Channel", "Raw Score", "Conf Score"]):
        _t(cx, 0.76, hdr, va="top", fontsize=8, fontweight="bold",
           color="#C0C0CC", ha="left" if cx == col_x[0] else "center")

    ax.axhline(0.725, color="#333340", linewidth=0.8)

    # Data rows
    row_y   = 0.65
    row_gap = 0.135
    for i, (sc, cc) in enumerate(zip(scores, _CHANNEL_COLORS)):
        y  = row_y - i * row_gap
        _t(col_x[0], y, sc["channel"],    va="center", fontsize=9,
           color=cc, fontweight="bold")
        _t(col_x[1], y, f"{sc['raw']:.4f}",  va="center", fontsize=9,
           color="#CACACC", ha="center")
        _t(col_x[2], y, f"{sc['conf']:.4f}", va="center", fontsize=9,
           color="#CACACC", ha="center")
        # Confidence bar (width ? conf_score, max 0.75 of axes width)
        bar_w = max(sc["conf"] * 0.75, 0.003)
        ax.add_patch(plt.Rectangle(
            (col_x[0], y - row_gap * 0.27), bar_w, row_gap * 0.16,
            transform=ax.transAxes, color=cc, alpha=0.35, zorder=2,
        ))

    # Overall verdict box
    ov_c = OV_COLORS.get(overall_verdict, "#EAEAEA")
    _t(0.5, 0.04, f"OVERALL VERDICT:  {overall_verdict}   (Z = {z_score:+.3f}  /  thresh {z_threshold:+.1f})",
       ha="center", va="bottom", fontsize=11.5, fontweight="bold", color=ov_c,
       bbox=dict(boxstyle="round,pad=0.45", fc="#12121A", ec=ov_c,
                 lw=1.4, alpha=0.92))


# Evidence report (main pipeline)
def generate_evidence_report(
    model:           TAVNet,
    writer_id:       int,
    genuine_paths:   list[Path],
    questioned_path: Path,
    device:          torch.device,
    out_dir:         Path,
    alpha:           float = 0.55,
    enable_rollout:  bool  = True,
    max_vault:       int   = 8,
    questioned_stem: str   = "",
    n_subcenters:    int   = DEFAULT_N_SUBCENTERS,
    z_threshold:     float = DEFAULT_Z_THRESHOLD,
) -> Path:
    """
    Run the full XAI pipeline and save a high-resolution PNG evidence report.

    Steps
    -----
    1. Load questioned tensor.
    2. Compute vault centroid, sub-centers, and LOO calibration stats.
    3. Compute questioned embedding, 4-component score, and Z-score verdict.
    4. Grad-CAM (questioned)       — differentiating cosine similarity.
    5. Grad-CAM (vault mean)       — average over up to max_vault genuine samples.
    6. Contrastive ?-Map           — vault - questioned heatmap.
    7. Per-channel attribution     — Grad × Input for each of 4 channels.
    8. Attention Rollout           — Transformer token-level context map.
    9. Salience scores             — numeric confidence per channel.
    10. Compose + save figure.

    Parameters
    ----------
    max_vault : cap on number of vault samples used for Grad-CAM averaging
                (keeps runtime predictable on the RTX 3050).

    Returns
    -------
    Path to the saved PNG file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    stem      = f"{writer_id}_{questioned_stem}" if questioned_stem else str(writer_id)
    out_path  = out_dir / f"evidence_report_{stem}_{timestamp}.png"

    # -- 1. Load questioned tensor ---------------------------------------------
    log.debug("Loading questioned signature: %s", questioned_path.name)
    q_tensor = _load_file(questioned_path)         # (4, 384, 384) CPU float32
    q_base   = q_tensor[0].numpy()                 # Shape channel — grayscale base

    # -- 2. Vault embeddings, centroid, sub-centers, and LOO calibration --------
    log.debug("Computing vault embeddings from %d genuine samples …", len(genuine_paths))
    _vault_embs: list[torch.Tensor] = []
    with torch.no_grad():
        for _vp in genuine_paths:
            _vt = _load_file(_vp).unsqueeze(0).to(device)
            _vault_embs.append(model(_vt)[0].cpu().float())
    _vault_stack = torch.stack(_vault_embs)                          # (N, D)
    centroid     = F.normalize(_vault_stack.mean(0, keepdim=True), p=2, dim=1).squeeze(0)
    subcenters   = _dynamic_kmeans_subcenters(_vault_embs, max_k=n_subcenters)   # (K, D)
    _named_embs  = [(p.name, e) for p, e in zip(genuine_paths, _vault_embs)]
    log.debug("  Vault centroid + %d sub-centers computed.", subcenters.shape[0])

    _loo_scores: list[float] = []
    for _i, (_, _qe) in enumerate(_named_embs):
        _others = [(n, e) for _j, (n, e) in enumerate(_named_embs) if _j != _i]
        if not _others:
            continue
        _os   = torch.stack([e for _, e in _others])
        _oc   = F.normalize(_os.mean(0, keepdim=True), p=2, dim=1).squeeze(0)
        _osub = _dynamic_kmeans_subcenters([e for _, e in _others], max_k=min(n_subcenters, len(_others)))
        _sc, *_ = _compute_combined(_qe, _oc, _osub, _others, DEFAULT_WEIGHTS)
        _loo_scores.append(_sc)
    vault_mean = float(np.mean(_loo_scores))    if _loo_scores          else 0.70
    vault_std  = float(np.std(_loo_scores))     if len(_loo_scores) > 1 else 0.05
    log.debug("  Vault LOO — mean: %.4f  std: %.4f  (%d scores)",
             vault_mean, vault_std, len(_loo_scores))

    # -- 3. Questioned embedding, 4-component score, and Z-score verdict --------
    with torch.no_grad():
        q_emb = model(q_tensor.unsqueeze(0).to(device))[0].cpu().float()
    combined, centroid_sim, subcenter_sim = _compute_combined(
        q_emb, centroid, subcenters, _named_embs, DEFAULT_WEIGHTS
    )
    cos_sim = centroid_sim   # centroid similarity — used as the visual reference throughout
    z_score = (combined - vault_mean) / max(vault_std, 0.01)
    log.debug("  Centroid: %.4f  Sub-center: %.4f",
             centroid_sim, subcenter_sim)
    log.debug("  Combined: %.4f  Z-score: %.4f  Z-threshold: %.2f",
             combined, z_score, z_threshold)

    overall_verdict = "GENUINE" if z_score >= z_threshold else "FORGERY"
    log.info("  Verdict: %s  (Z: %+.2f)", overall_verdict, z_score)

    # -- Find best-matched genuine signature (highest similarity to questioned) --
    best_idx = 0
    best_sim = -2.0
    for idx, (_, emb) in enumerate(_named_embs):
        sim = torch.dot(q_emb, emb).item()
        if sim > best_sim:
            best_sim = sim
            best_idx = idx
    best_genuine_path = genuine_paths[best_idx]
    best_genuine_tensor = _load_file(best_genuine_path).to(device)

    # -- 4. Grad-CAM: questioned signature -------------------------------------
    log.debug("Computing Grad-CAM for questioned signature …")
    hmap_q = run_gradcam(model, q_tensor.to(device), centroid)   # (384, 384)

    # -- 5. Grad-CAM: best-matched genuine -------------------------------------
    logger_msg = f"Computing Grad-CAM for best-matched genuine: {best_genuine_path.name}"
    log.debug(logger_msg)
    hmap_best = run_gradcam(model, best_genuine_tensor, centroid)  # (384, 384)

    # -- 5.5. Contrastive ?-Map (vault - questioned) ---------------------------
    diff_map  = hmap_best - hmap_q                  # ? [-1, 1]
    diff_norm = (diff_map + 1.0) / 2.0             # ? [0, 1] for RdBu_r cmap

    # -- 7. Per-channel attribution maps ---------------------------------------
    log.debug("Computing per-channel Gradient × Input attribution maps …")
    q_dev    = q_tensor.to(device)
    ch_maps  = channel_attribution_maps(model, q_dev, centroid)  # 4 × (384,384)

    # -- 8. Attention Rollout --------------------------------------------------
    # Note: rollout_grid not used in simplified report, skipping

    # -- 9. Salience scores ----------------------------------------------------
    scores = salience_scores(ch_maps, cos_sim)

    # -- 10. Compose figure with 3-row layout -------------------------------------
    log.debug("Composing 3-row comparison report …")
    _BG  = "#FFFFFF"  # white background for professional presentation
    
    # Create 3x2 grid: top 2 rows are images (2 cols each), bottom row is analysis (4 panels)
    fig = plt.figure(figsize=(18, 16), facecolor=_BG)
    fig.patch.set_facecolor(_BG)
    
    gs = gridspec.GridSpec(
        3, 4,
        figure=fig,
        top=0.92,
        bottom=0.05,
        left=0.05,
        right=0.98,
        hspace=0.30,
        wspace=0.24,
    )

    # -- Title at Top ------------------------------------------------------------
    V_COLOR = {"GENUINE": "#2ECC71", "FORGERY": "#E74C3C"}
    title_color = V_COLOR.get(overall_verdict, "#000000")
    
    fig.text(
        0.5, 0.97,
        f"Signature Verification Report",
        ha="center", va="top", fontsize=22, fontweight="bold", color="#000000",
    )

    # Get numpy version of best-matched genuine signature grayscale
    best_genuine_base = best_genuine_tensor[0].cpu().numpy()  # shape channel grayscale

    # -- Row 0, Cols 0-1: Questioned Raw Grayscale --------------------------------
    ax_q_raw = fig.add_subplot(gs[0, :2])
    ax_q_raw.imshow(q_base, cmap="gray", vmin=0, vmax=1)
    ax_q_raw.set_title("QUESTIONED SIGNATURE", fontsize=14, fontweight="bold", pad=10)
    ax_q_raw.set_xticks([])
    ax_q_raw.set_yticks([])
    for spine in ax_q_raw.spines.values():
        spine.set_edgecolor("#000000")
        spine.set_linewidth(2)

    # -- Row 0, Cols 2-3: Authentic Raw Grayscale -------------------------------
    ax_g_raw = fig.add_subplot(gs[0, 2:])
    ax_g_raw.imshow(best_genuine_base, cmap="gray", vmin=0, vmax=1)
    ax_g_raw.set_title(f"AUTHENTIC SIGNATURE (Reference)", 
                       fontsize=14, fontweight="bold", pad=10)
    ax_g_raw.set_xticks([])
    ax_g_raw.set_yticks([])
    for spine in ax_g_raw.spines.values():
        spine.set_edgecolor("#000000")
        spine.set_linewidth(2)

    # -- Row 1, Cols 0-1: Questioned Grad-CAM -----------------------------------
    ax_q_cam = fig.add_subplot(gs[1, :2])
    overlay_q = _overlay(q_base, hmap_q, cmap="jet", alpha=0.60)
    im_q = ax_q_cam.imshow(overlay_q)
    ax_q_cam.set_title("GRAD-CAM MAP (QUESTIONED)", fontsize=14, fontweight="bold", pad=10)
    ax_q_cam.set_xticks([])
    ax_q_cam.set_yticks([])
    for spine in ax_q_cam.spines.values():
        spine.set_edgecolor("#000000")
        spine.set_linewidth(2)
    # Add colorbar for heatmap intensity
    cbar_q = plt.colorbar(im_q, ax=ax_q_cam, orientation="horizontal", pad=0.08, fraction=0.046)
    cbar_q.set_label("Attention Intensity", fontsize=10, fontweight="bold")

    # -- Row 1, Cols 2-3: Authentic Grad-CAM ------------------------------------
    ax_g_cam = fig.add_subplot(gs[1, 2:])
    overlay_g = _overlay(best_genuine_base, hmap_best, cmap="jet", alpha=0.60)
    im_g = ax_g_cam.imshow(overlay_g)
    ax_g_cam.set_title("GRAD-CAM MAP (AUTHENTIC)", fontsize=14, fontweight="bold", pad=10)
    ax_g_cam.set_xticks([])
    ax_g_cam.set_yticks([])
    for spine in ax_g_cam.spines.values():
        spine.set_edgecolor("#000000")
        spine.set_linewidth(2)
    # Add colorbar for heatmap intensity
    cbar_g = plt.colorbar(im_g, ax=ax_g_cam, orientation="horizontal", pad=0.08, fraction=0.046)
    cbar_g.set_label("Attention Intensity", fontsize=12, fontweight="bold")

    # -- Row 2: Analysis Panels --------------------------------------------------
    
    # Compute per-channel cosine similarity (questioned vs vault/reference)
    ch_sims = []
    best_gen_dev = best_genuine_tensor.to(device).requires_grad_(True)
    best_ch_maps = channel_attribution_maps(model, best_gen_dev, centroid)  # precompute all channels
    
    for c in range(4):
        ch_q = ch_maps[c]  # questioned channel map
        ch_best = best_ch_maps[c]  # authentic channel map
        q_flat = ch_q.reshape(-1)
        b_flat = ch_best.reshape(-1)
        denom = (np.linalg.norm(q_flat) * np.linalg.norm(b_flat)) + 1e-8
        sim = float(np.dot(q_flat, b_flat) / denom)
        ch_sims.append(sim)

    # -- Panel 0: Channel Similarity Analysis Table -----------------------------
    ax_table = fig.add_subplot(gs[2, 0])
    ax_table.axis("off")
    
    table_data = [
        ["Channel", "Similarity"],
    ]
    for i, name in enumerate(_CHANNEL_NAMES):
        table_data.append([
            name,
            f"{ch_sims[i]:.3f}",
        ])
    
    table = ax_table.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        bbox=[0.05, 0.1, 0.9, 0.85],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.4)
    
    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor("#2C3E50")
        table[(0, i)].set_text_props(weight="bold", color="white", fontsize=12)
    
    # Style data rows with channel colors
    for i, cc in enumerate(_CHANNEL_COLORS, 1):
        table[(i, 0)].set_facecolor(cc)
        table[(i, 0)].set_text_props(color="white", weight="bold", fontsize=11)
        table[(i, 1)].set_facecolor("#ECEFF1")
        table[(i, 1)].set_text_props(fontsize=11)
    
    # Style combined row (last row)
    combined_row = len(table_data) - 1
    table[(combined_row, 0)].set_facecolor("#34495E")
    table[(combined_row, 0)].set_text_props(weight="bold", color="white", fontsize=10)
    table[(combined_row, 1)].set_facecolor("#D5DBDB")
    table[(combined_row, 1)].set_text_props(fontsize=11, weight="bold")
    
    ax_table.text(0.5, 1.05, "Channel-wise Similarity Analysis", 
                  ha="center", va="bottom", transform=ax_table.transAxes,
                  fontsize=12, fontweight="bold", color="#1A3A3A")

    # -- Panel 1: Contrastive Difference Map (left) ------------------------------
    ax_contrast_l = fig.add_subplot(gs[2, 1])
    contrast_overlay = _overlay(q_base, diff_norm, cmap="RdBu_r", alpha=0.80)
    im_contrast_l = ax_contrast_l.imshow(contrast_overlay)
    ax_contrast_l.set_title("Contrastive Map\n(Red=Authentic | Blue=Questioned)", 
                            fontsize=12, fontweight="bold", pad=8, color="#000000")
    ax_contrast_l.set_xticks([])
    ax_contrast_l.set_yticks([])
    for spine in ax_contrast_l.spines.values():
        spine.set_edgecolor("#333333")
        spine.set_linewidth(1.5)
    # Add colorbar
    cbar_contrast_l = plt.colorbar(im_contrast_l, ax=ax_contrast_l, orientation="vertical", pad=0.03, fraction=0.046)
    cbar_contrast_l.set_label("Difference", fontsize=9, fontweight="bold")

    # -- Panel 2: Global Contrast Index Map (right) ------------------------------
    ax_contrast_r = fig.add_subplot(gs[2, 2])
    # Create a global contrast index (absolute difference overlay)
    abs_diff = np.abs(diff_map)
    abs_diff_norm = (abs_diff - abs_diff.min()) / (abs_diff.max() - abs_diff.min() + 1e-8)
    contrast_idx_overlay = _overlay(q_base, abs_diff_norm, cmap="YlOrRd", alpha=0.75)
    im_contrast_r = ax_contrast_r.imshow(contrast_idx_overlay)
    ax_contrast_r.set_title("Global Contrast\nIndex Map",
                            fontsize=12, fontweight="bold", pad=8, color="#000000")
    ax_contrast_r.set_xticks([])
    ax_contrast_r.set_yticks([])
    for spine in ax_contrast_r.spines.values():
        spine.set_edgecolor("#333333")
        spine.set_linewidth(1.5)
    # Add colorbar
    cbar_contrast_r = plt.colorbar(im_contrast_r, ax=ax_contrast_r, orientation="vertical", pad=0.03, fraction=0.046)
    cbar_contrast_r.set_label("Magnitude", fontsize=9, fontweight="bold")

    # -- Panel 3: Verdict Declaration --------------------------------------------
    ax_verdict = fig.add_subplot(gs[2, 3])
    ax_verdict.axis("off")
    ax_verdict.set_xlim(0, 10)
    ax_verdict.set_ylim(0, 10)
    
    # Verdict box background
    ax_verdict.add_patch(plt.Rectangle(
        (0.1, 0.8), 9.8, 8.5,
        facecolor="#FAFAFA",
        edgecolor=title_color,
        linewidth=3,
    ))
    
    symbol = "?" if overall_verdict.lower() == "genuine" else "?"
    ax_verdict.text(
    5, 7.8, f"{symbol} {overall_verdict}",
    fontsize=18, fontweight="bold",
    color=title_color,
    ha="center", va="center",
    fontfamily="sans-serif"
    )

    
    # Z-score analysis block with compact lines so text stays inside the box
    ax_verdict.text(
        5, 6.55, f"centroid_sim: cosine(q, mean_vault) = {centroid_sim:+.3f}",
        fontsize=7.8,
        fontweight="bold",
        color="#333333",
        ha="center", va="center",
        fontfamily="monospace",
        clip_on=True,
    )
    ax_verdict.text(
        5, 5.95, f"subcenter_sim: cosine(q, nearest_kmeans) = {subcenter_sim:+.3f}",
        fontsize=7.8,
        fontweight="bold",
        color="#333333",
        ha="center", va="center",
        fontfamily="monospace",
        clip_on=True,
    )
    ax_verdict.text(
        5, 5.35, f"combined = avg(centroid_sim,subcenter_sim) = {combined:+.3f}",
        fontsize=7.6,
        fontweight="bold",
        color="#1F2937",
        ha="center", va="center",
        fontfamily="monospace",
        clip_on=True,
    )
    ax_verdict.text(
        5, 4.80, f"vault_mean = {vault_mean:+.3f} vault_std = {vault_std:+.3f}",
        fontsize=7.8,
        fontweight="bold",
        color="#1F2937",
        ha="center", va="center",
        fontfamily="monospace",
        clip_on=True,
    )
    ax_verdict.text(
        5, 4.30, "z = (combined - vault_mean) / max(vault_std, 0.01)",
        fontsize=7.6,
        fontweight="bold",
        color="#222222",
        ha="center", va="center",
        fontfamily="monospace",
        clip_on=True,
    )
    ax_verdict.text(
        5, 3.85, f"z = ({combined:+.3f} - {vault_mean:+.3f}) / {max(vault_std, 0.01):.3f} = {z_score:+.3f}",
        fontsize=7.5,
        fontweight="bold",
        color="#222222",
        ha="center", va="center",
        fontfamily="monospace",
        clip_on=True,
    )
    ax_verdict.text(
        5, 3.25, f"threshold = {z_threshold:+.1f}",
        fontsize=8.2,
        fontweight="bold",
        color="#333333",
        ha="center", va="center",
        fontfamily="monospace",
        clip_on=True,
    )
    ax_verdict.text(
        5, 2.60, f"Z-score = {z_score:+.3f}",
        fontsize=11.0,
        fontweight="bold",
        color="#111111",
        ha="center", va="center",
        fontfamily="monospace",
        clip_on=True,
    )

    # -- Save ------------------------------------------------------------------
    fig.savefig(
        str(out_path),
        dpi=200,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    log.info("Evidence report saved ? %s", out_path)
    return out_path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "grad_cam.py — Forensic XAI report for TAV-Net"
            "(Grad-CAM + per-channel attribution + Attention Rollout).\n"
            "\n"
            "Two modes:\n"
            "  1. --sample-dir DIR  :  DIR must contain vault/ and questioned/\n"
            "                         subfolders with images (PNG/JPG/TIFF/BMP)\n"
            "                         or .npy tensors in any mix.\n"
            "  2. --writer-id N     :  scans DATA/process_data for the writer."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # -- Mode 1: sample-dir (images / npy in vault + questioned folders) -------
    p.add_argument(
        "--sample-dir", type=str, default=None, metavar="DIR",
        help="Root directory that contains 'vault/' and 'questioned/' subfolders. "
             "Files may be raw images (PNG/JPG/JPEG/BMP/TIFF/WEBP) or .npy tensors. "
             "When provided, --writer-id is optional (defaults to 0).",
    )

    # -- Mode 2: writer-id (scans processed .npy directory) -------------------
    p.add_argument(
        "--writer-id", type=int, default=None,
        help="Writer UID whose vault (genuine samples) form the centroid. "
             "Valid ranges: 1-400 (BHSig/CEDAR/GPDS), 401-500 (Bengali), "
             "501-600 (Hindi), 601-800 (GPDS ext), 801-869 (ICDAR-2011), "
             "1001-1223 (Independent). "
             "Required when --sample-dir is not given.",
    )
    p.add_argument(
        "--questioned", type=str, default=None, metavar="PATH",
        help="Path to the questioned signature (image or .npy). "
             "If omitted in writer-id mode, uses the first forgery found.",
    )
    p.add_argument(
        "--proc-dir", type=str, default=None, metavar="DIR",
        help="Processed .npy directory override "
             "(default: DATA/process_data, fallback: DATA/Processed_data). "
             "Only used in writer-id mode.",
    )

    # -- Shared options --------------------------------------------------------
    p.add_argument(
        "--checkpoint", type=str, default=str(DEFAULT_CKPT), metavar="PATH",
        help="Path to the TAV-Net checkpoint (.pt).",
    )
    p.add_argument(
        "--out-dir", type=str, default=str(DEFAULT_OUT), metavar="DIR",
        help="Directory to save the evidence report PNG.",
    )
    p.add_argument(
        "--alpha", type=float, default=0.55,
        help="Heatmap overlay transparency (0=invisible, 1=fully opaque).",
    )
    p.add_argument(
        "--no-rollout", action="store_true",
        help="Skip Transformer Attention Rollout (faster, fewer pages).",
    )
    p.add_argument(
        "--max-vault", type=int, default=8,
        help="Max genuine samples to average for the vault Grad-CAM.",
    )
    p.add_argument(
        "--cpu", action="store_true",
        help="Force CPU inference even if CUDA is available.",
    )
    p.add_argument(
        "--z-threshold", type=float, default=DEFAULT_Z_THRESHOLD,
        help=f"Z-score acceptance threshold for GENUINE/FORGERY verdict "
             f"(default: {DEFAULT_Z_THRESHOLD}, matches verify_vault.py).",
    )
    p.add_argument(
        "--n-subcenters", type=int, default=DEFAULT_N_SUBCENTERS,
        help=f"Number of K-means signing-style sub-centers extracted from vault "
             f"(default: {DEFAULT_N_SUBCENTERS}, matches verify_vault.py).",
    )

    args = p.parse_args()
    if args.sample_dir is None and args.writer_id is None:
        p.error("one of --sample-dir or --writer-id is required.")
    return args


# Entry point
def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level  = logging.INFO,
        format = "%(levelname)-8s %(message)s",
        handlers = [logging.StreamHandler()],
    )

    # -- Device ----------------------------------------------------------------
    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        log.info("Running on CPU.")
    else:
        device = torch.device("cuda")

    # -------------------------------------------------------------------------
    # Mode 1 — --sample-dir: vault/ + questioned/ subfolders with raw files
    # -------------------------------------------------------------------------
    if args.sample_dir:
        sample_root = Path(args.sample_dir)
        vault_dir   = sample_root / "vault"
        quest_dir   = sample_root / "questioned"

        for d, label in [(vault_dir, "vault"), (quest_dir, "questioned")]:
            if not d.is_dir():
                log.error(
                    "Expected sub-directory not found: %s\n"
                    "  --sample-dir must contain 'vault/' and 'questioned/' folders.",
                    d,
                )
                sys.exit(1)

        genuine_paths   = _scan_folder(vault_dir)
        questioned_list = _scan_folder(quest_dir)

        if not genuine_paths:
            log.error("No files found in vault directory: %s", vault_dir)
            sys.exit(1)
        if not questioned_list:
            log.error("No files found in questioned directory: %s", quest_dir)
            sys.exit(1)

        # All questioned files are processed; one report per file
        writer_id       = args.writer_id if args.writer_id is not None else 0
        vault_names = ",".join([f.name for f in genuine_paths])
        quest_names = ",".join([f.name for f in questioned_list])
        log.info(f"Vault ({len(genuine_paths)}): {vault_names}")
        log.info(f"Questioned ({len(questioned_list)}): {quest_names}")

        # Load model once
        model, ckpt = load_model(Path(args.checkpoint), device)

        # Generate one report per questioned file
        for i, questioned_path in enumerate(questioned_list, 1):
            log.info(f"Processing [{i}/{len(questioned_list)}]: {questioned_path.name}")
            out_path = generate_evidence_report(
                model            = model,
                writer_id        = writer_id,
                genuine_paths    = genuine_paths,
                questioned_path  = questioned_path,
                device           = device,
                out_dir          = Path(args.out_dir),
                alpha            = args.alpha,
                enable_rollout   = not args.no_rollout,
                max_vault        = args.max_vault,
                questioned_stem  = questioned_path.stem,
                n_subcenters     = args.n_subcenters,
                z_threshold      = args.z_threshold,
            )
        return   # ? end of sample-dir mode

    # -------------------------------------------------------------------------
    # Mode 2 — --writer-id: scan processed .npy directory
    # -------------------------------------------------------------------------
    writer_id = args.writer_id  # guaranteed non-None by arg validation above

    # -- Resolve processed-data directory -------------------------------------
    proc_dir = _resolve_proc_dir(args.proc_dir)
    log.info("Processed .npy directory: %s", proc_dir)

    # -- Scan for writer samples -----------------------------------------------
    paths = _scan_writer(writer_id, proc_dir)
    if not paths["G"]:
        log.error(
            "No genuine (.npy) samples found for writer %d in %s.",
            writer_id, proc_dir,
        )
        sys.exit(1)
    log.info(
        "Writer %d — %d genuine, %d forgery samples found.",
        writer_id, len(paths["G"]), len(paths["F"]),
    )

    # -- Resolve questioned path -----------------------------------------------
    if args.questioned:
        questioned_path = Path(args.questioned)
        if not questioned_path.exists():
            log.error("--questioned path does not exist: %s", questioned_path)
            sys.exit(1)
    elif paths["F"]:
        questioned_path = paths["F"][0]
        log.info(
            "No --questioned provided; using first forgery: %s",
            questioned_path.name,
        )
    else:
        # No forgery available ? use last genuine as self-test
        questioned_path = paths["G"][-1]
        log.warning(
            "No forgery samples found for writer %d. "
            "Using last genuine sample as questioned (expect high similarity).",
            writer_id,
        )

    # -- Load model ------------------------------------------------------------
    model, _ = load_model(Path(args.checkpoint), device)

    # -- Generate report -------------------------------------------------------
    out_path = generate_evidence_report(
        model           = model,
        writer_id       = writer_id,
        genuine_paths   = paths["G"],
        questioned_path = questioned_path,
        device          = device,
        out_dir         = Path(args.out_dir),
        alpha           = args.alpha,
        enable_rollout  = not args.no_rollout,
        max_vault       = args.max_vault,
        n_subcenters    = args.n_subcenters,
        z_threshold     = args.z_threshold,
    )


if __name__ == "__main__":
    main()