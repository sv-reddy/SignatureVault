"""
train_tavnet.py - TAVNet Training Script
======================================================

HOW IT WORKS
------------
Architecture: ResNet-50 (ImageNet weights; conv1 patched to 4 input channels)
  → layer1, layer2, layer3
  → layer4 → CBAM (channel + spatial attention on 2048-dim feature maps)
        → Spatial flatten: (B, 2048, 12, 12) → (144, B, 2048) token sequence
        → learnable positional embedding (144 tokens for 384×384 input)
  → TransformerEncoder (1 layer | 8 heads | FFN 2048 | Pre-LN)
        → GeM pooling over 12x12 tokens → (B, 2048)
  → Linear(2048 → 512) → BatchNorm1d(512) → L2 normalise

Input : (4, 384, 384) uint8 tensor scaled to float32 [0, 1]
        Channels: Shape | Pseudo-Pressure | Stroke Angle | Skeleton

Loss  : Sub-Center ArcFace (margin m=0.45, scale s=64, K=3 sub-centers per class)
        Applied on genuine training samples only.

Validation metrics : val_loss (batch-hard softplus triplet, no label mapping)
                     val_EER  (Equal Error Rate at FAR = FRR threshold)
                     Forg_EER / Rand_EER (split EERs by impostor type)
                     TAR@.8  (True Accept Rate at cosine similarity ≥ 0.80)

APN-Bundle (8 slots per writer):
  slot 0     Anchor         genuine, writer W
  slots 1-3  Positives ×3  genuine, writer W
  slots 4-5  Skilled Neg   forgery of writer W (top-10 HNM pool)
  slots 6-7  Random Neg    genuine, writer W' ≠ W

Optimiser    : AdamW  lr=1e-4  wd=5e-4
Scheduler    : ReduceLROnPlateau (factor=0.5, patience=3, metric=val_eer)
Warmup       : Triplet loss scales from 0 → 1 over 10 epochs
Margin Decay : Triplet loss margin decays from 0.30 → 0.05 over 50 epochs
Augmentation : Gentler transforms (rotation 10°, scale 0.9-1.1, erase p=0.2, elastic p=0.2)
Accumulation : 4 gradient-accumulation steps (effective batch = 256 tensors)
Writer split : 70 / 10 / 20 writer-disjoint
Output       : checkpoints/best_tavnet.pt  |  checkpoints/manifest.json

HOW TO RUN
----------
Default training (50 epochs, batch=8, accum=4):
    python train_tavnet.py

Resume from a saved checkpoint:
    python train_tavnet.py --resume checkpoints/best_tavnet.pt

Smaller physical batch with more accumulation (same effective batch):
    python train_tavnet.py --batch-size 8 --accum-steps 4

Enable hard-negative mining (activates after epoch 5 by default):
    python train_tavnet.py --hard-mining

Use more ArcFace sub-centers for high-variance writers:
    python train_tavnet.py --arcface-k 5

Override the processed data directory:
    python train_tavnet.py --processed-dir /path/to/process_data

ARGUMENTS
---------
--epochs            Total training epochs. Default: 50
--batch-size        Writers per physical batch (8 tensors each). Default: 8
--accum-steps       Gradient accumulation steps. Default: 4
--lr                Peak learning rate. Default: 1e-4
--wd                AdamW weight decay. Default: 5e-4
--embed-dim         Embedding dimensionality. Default: 512
--arcface-m         ArcFace angular margin (radians). Default: 0.55
--arcface-s         ArcFace logit scale. Default: 64.0
--arcface-k         Sub-centers per class. Default: 7
--num-workers       DataLoader worker processes (0 = main process). Default: 2
--seed              Global random seed. Default: 42
--resume            Resume from checkpoint (.pt path).
--processed-dir     Override path to the processed .npy directory.
--hard-mining       Enable hard-negative mining (top-10 hardest forgeries).
--hard-mining-start Epoch after which HNM activates. Default: 5
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

import torchvision.models as tvm
from torchvision.models import ResNet50_Weights

from scipy.ndimage import gaussian_filter as _gauss_filter
from scipy.ndimage import map_coordinates as _map_coords

try:
    from tqdm import tqdm as _tqdm

    def _pbar(it, **kw):
        return _tqdm(it, **kw)

except ImportError:
    def _pbar(it, total=None, desc=None, leave=True, unit="it", **kw):
        return it

try:
    from torchvision.transforms import (
        RandomRotation,
        RandomAffine,
        RandomPerspective,
        InterpolationMode,
        RandomErasing,
    )

    _ROT_AUG = RandomRotation(
        degrees=10,
        interpolation=InterpolationMode.BILINEAR
    )
    _AFFINE_AUG = RandomAffine(
        degrees=0,
        scale=(0.9, 1.1),
        shear=5,
        interpolation=InterpolationMode.BILINEAR,
    )
    _PERSP_AUG = RandomPerspective(
        distortion_scale=0.3,
        p=1.0,
    )
    _ERASE_AUG = RandomErasing(
        p=0.2,
        scale=(0.01, 0.05),
        ratio=(0.3, 3.3),
        value=0,
        inplace=False,
    )
    _NOISE_AUG = None
except ImportError:
    _ROT_AUG = _AFFINE_AUG = _PERSP_AUG = _ERASE_AUG = _NOISE_AUG = None

_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_ROOT  = _SCRIPT_DIR.parent / "DATA"
_PROC_      = _DATA_ROOT / "process_data"

CKPT_DIR         = _SCRIPT_DIR / "checkpoints"
BEST_CKPT_PATH   = CKPT_DIR / "best_tavnet.pt"
MANIFEST_PATH    = CKPT_DIR / "manifest.json"

_BUNDLE_SIZE   = 8
_MIN_GENUINE   = 4
_MIN_FORGERY   = 2
_ELASTIC_ALPHA = 50.0
_ELASTIC_SIGMA = 5.0

_BUNDLE_TYPES = torch.tensor([0, 0, 0, 0, 1, 1, 0, 0], dtype=torch.long)

_BIG = 1e9

_HN_WARMUP_EPOCHS  = 5
_HN_SIM_THRESHOLD  = 0.70
_HN_TOP_K          = 10
_HIGH_VAR_DIST_P75 = 0.75

_TRAIN_RATIO = 0.70
_VAL_RATIO   = 0.10
_TEST_RATIO  = 0.20

_CURRICULUM_PHASE2_EPOCH = 11



class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size,
                                 padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg      = x.mean(dim=1, keepdim=True)
        mx, _    = x.max(dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(
        self,
        channels:  int = 2048,
        reduction: int = 16,
        spatial_k: int = 7,
    ) -> None:
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x



class SubCenterArcFaceLoss(nn.Module):
    def __init__(
        self,
        in_features: int,
        n_classes:   int,
        K:           int   = 3,
        s:           float = 64.0,
        m:           float = 0.50,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.K         = K
        self.s         = s
        self.m         = m
        self.weight = nn.Parameter(
            torch.FloatTensor(n_classes * K, in_features)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels:     torch.Tensor,
    ) -> torch.Tensor:
        W = F.normalize(self.weight, p=2, dim=1)
        cos_all = embeddings @ W.T
        cos_theta = cos_all.view(-1, self.n_classes, self.K).max(dim=2).values
        cos_theta_clamped = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta              = torch.acos(cos_theta_clamped)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        theta_with_margin = theta + one_hot * self.m
        logits = torch.cos(theta_with_margin) * self.s
        return F.cross_entropy(logits, labels)


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.avg_pool2d(x, kernel_size=(x.size(-2), x.size(-1)))
        return x.pow(1.0 / self.p)


class TAVNet(nn.Module):
    _SEQ_LEN = 12 * 12  # 384x384 input: layer4 outputs 12x12 tokens (instead of 7x7 for 224x224)
    _N_TOKENS = _SEQ_LEN

    def __init__(self, embed_dim: int = 512) -> None:
        super().__init__()

        base = tvm.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        old_w     = base.conv1.weight.data
        new_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv1.weight[:, :3, :, :] = old_w
            new_conv1.weight[:, 3:, :, :] = old_w.mean(dim=1, keepdim=True)
        base.conv1 = new_conv1

        self.stem   = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        # Freeze early backbone stages to reduce overfitting.
        for module in (self.stem, self.layer1, self.layer2):
            for param in module.parameters():
                param.requires_grad = False

        self.cbam = CBAM(channels=2048, reduction=16, spatial_k=7)

        self.pos_embed = nn.Parameter(torch.zeros(self._N_TOKENS, 1, 2048))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model         = 2048,
            nhead           = 8,
            dim_feedforward = 2048,
            dropout         = 0.1,
            activation      = "gelu",
            batch_first     = False,
            norm_first      = True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=1, enable_nested_tensor=False
        )
        self.gem = GeM(p=3.0, eps=1e-6)

        self.head = nn.Sequential(
            nn.Linear(2048, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cbam(x)
        B = x.shape[0]
        x = x.flatten(2)
        x = x.permute(2, 0, 1)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = x.permute(1, 2, 0).contiguous().view(B, 2048, 12, 12)
        x = self.gem(x).flatten(1)
        x = self.head(x)
        return F.normalize(x, p=2, dim=1)


def _load_model_state_compat(model: TAVNet, state_dict: dict) -> None:
    model_state = model.state_dict()
    patched = dict(state_dict)

    has_cls_model = ("cls_token" in model_state)
    has_cls_ckpt = ("cls_token" in patched)

    if has_cls_model and not has_cls_ckpt:
        patched["cls_token"] = model_state["cls_token"].clone()
    elif has_cls_ckpt and not has_cls_model:
        patched.pop("cls_token", None)

    if "pos_embed" in patched and "pos_embed" in model_state:
        src = patched["pos_embed"]
        tgt = model_state["pos_embed"]
        if src.shape != tgt.shape and src.ndim == 3 and tgt.ndim == 3 and src.shape[1:] == tgt.shape[1:]:
            if src.shape[0] + 1 == tgt.shape[0]:
                src = src.to(device=tgt.device, dtype=tgt.dtype)
                patched["pos_embed"] = torch.cat([tgt[:1].clone(), src], dim=0)
            elif src.shape[0] == tgt.shape[0] + 1:
                src = src.to(device=tgt.device, dtype=tgt.dtype)
                patched["pos_embed"] = src[1:].clone()

    missing, unexpected = model.load_state_dict(patched, strict=False)
    if missing:
        logging.getLogger("tavnet").warning("Missing keys after model compatibility load: %s", missing)
    if unexpected:
        logging.getLogger("tavnet").warning("Unexpected keys after model compatibility load: %s", unexpected)


class RandomStrokeMorphology:
    """
    Randomly dilate or erode stroke maps to simulate pen-thickness variation.
    Applies only to channels 0 (Shape) and 3 (Skeleton).
    """

    def __init__(self, p: float = 0.3) -> None:
        self.p = p

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return t
        out = t.clone()
        for c in (0, 3):
            ch = out[c:c + 1].unsqueeze(0)
            if random.random() < 0.5:
                morphed = F.max_pool2d(ch, kernel_size=3, stride=1, padding=1)
            else:
                morphed = -F.max_pool2d(-ch, kernel_size=3, stride=1, padding=1)
            out[c] = morphed.squeeze(0).squeeze(0)
        return out


_MORPH_AUG = RandomStrokeMorphology(p=0.3)


def _elastic_deform(
    t:     torch.Tensor,
    alpha: float = _ELASTIC_ALPHA,
    sigma: float = _ELASTIC_SIGMA,
    rng:   np.random.Generator | None = None,
) -> torch.Tensor:
    if rng is None:
        rng = np.random.default_rng()
    C, H, W = t.shape
    arr      = t.numpy()
    dx = _gauss_filter(rng.standard_normal((H, W)).astype(np.float32) * alpha, sigma)
    dy = _gauss_filter(rng.standard_normal((H, W)).astype(np.float32) * alpha, sigma)
    xs, ys  = np.meshgrid(np.arange(W, dtype=np.float32),
                          np.arange(H, dtype=np.float32))
    idx_y   = np.clip(ys + dy, 0, H - 1)
    idx_x   = np.clip(xs + dx, 0, W - 1)
    out     = np.stack(
        [_map_coords(arr[c], [idx_y, idx_x], order=1, mode="nearest") for c in range(C)],
        axis=0,
    )
    return torch.from_numpy(out.astype(np.float32))


def _augment(t: torch.Tensor) -> torch.Tensor:
    if random.random() < 0.2:
        t = _elastic_deform(t)
    t = _MORPH_AUG(t)
    if _ROT_AUG is not None and random.random() < 0.5:
        t = _ROT_AUG(t)
    if _AFFINE_AUG is not None:
        t = _AFFINE_AUG(t)
    if _PERSP_AUG is not None and random.random() < 0.3:
        t = _PERSP_AUG(t)
    if _ERASE_AUG is not None:
        t = _ERASE_AUG(t)
    if random.random() < 0.3:
        if _NOISE_AUG is not None:
            t = _NOISE_AUG(t)
        else:
            t = (t + torch.randn_like(t) * 0.05).clamp_(0.0, 1.0)
    return t


class APNBundleDataset(Dataset):
    def __init__(
        self,
        writer_ids: list[int],
        sample_map: dict[int, dict[str, list[str]]],
        augment:    bool = False,
    ) -> None:
        self.sample_map = sample_map
        self.augment    = augment
        self.writer_ids = [
            w for w in writer_ids
            if (
                len(sample_map.get(w, {}).get("G", [])) >= _MIN_GENUINE
                and len(sample_map.get(w, {}).get("F", [])) >= _MIN_FORGERY
            )
        ]
        if not self.writer_ids:
            raise RuntimeError(
                f"No writer has ≥{_MIN_GENUINE} genuine and ≥{_MIN_FORGERY} forgery "
                "samples in the given split. Cannot construct APN-Bundles."
            )
        self.forgery_scores: dict[int, dict[str, float]] = {}
        self.current_epoch = 1

    def __len__(self) -> int:
        return len(self.writer_ids)

    def update_forgery_scores(
        self,
        scores: dict[int, dict[str, float]],
    ) -> None:
        self.forgery_scores = scores

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def _load(self, path: str) -> torch.Tensor:
        arr = np.load(path).astype(np.float32) / 255.0
        t = torch.from_numpy(arr)
        if self.augment:
            t = _augment(t)
        return t

    def __getitem__(self, idx: int):
        w       = self.writer_ids[idx]
        genuine = self.sample_map[w]["G"]
        forgery = self.sample_map[w]["F"]

        chosen_g = random.sample(genuine, 4)

        phase2 = (self.current_epoch >= _CURRICULUM_PHASE2_EPOCH)
        n_skilled = 4 if phase2 else 2
        n_random = 0 if phase2 else 2

        writer_scores = self.forgery_scores.get(w)
        if writer_scores:
            sorted_f = sorted(writer_scores.items(), key=lambda kv: -kv[1])
            pool = [p for p, _ in sorted_f[:_HN_TOP_K]]
        else:
            pool = []
        if not pool:
            pool = list(forgery)

        if len(pool) >= n_skilled:
            chosen_f = random.sample(pool, n_skilled)
        else:
            chosen_f = random.choices(pool, k=n_skilled)

        neg_writers: list[int] = []
        neg_paths: list[str] = []
        if n_random > 0:
            others = [u for u in self.writer_ids if u != w]
            if others:
                neg_writers = random.choices(others, k=n_random)
                neg_paths = [random.choice(self.sample_map[nw]["G"]) for nw in neg_writers]
            else:
                # Safety fallback if only one writer exists in split.
                neg_paths = random.choices(forgery, k=n_random)
                neg_writers = [w] * n_random

        all_paths = chosen_g + chosen_f + neg_paths
        bundle    = torch.stack([self._load(p) for p in all_paths])

        # Slot layout:
        #   0-3: genuine (anchor + positives)
        #   4-?: skilled negatives (forgeries of same writer)
        #   ...: random negatives (genuine of different writers) in phase 1
        w_ids_list = [w, w, w, w] + [w] * n_skilled + neg_writers
        types_list = [0, 0, 0, 0] + [1] * n_skilled + [0] * n_random

        w_ids = torch.tensor(w_ids_list, dtype=torch.long)
        types = torch.tensor(types_list, dtype=torch.long)
        return bundle, w_ids, types


def _collate_bundles(batch):
    bundles, w_ids_list, types_list = zip(*batch)
    return (
        torch.cat(bundles),
        torch.cat(w_ids_list),
        torch.cat(types_list),
    )


def _worker_init_fn(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)



@torch.no_grad()
def compute_hard_forgery_scores(
    model:      "TAVNet",
    sample_map: dict[int, dict[str, list[str]]],
    writer_ids: list[int],
    device:     torch.device,
    batch_size: int = 32,
) -> dict[int, dict[str, float]]:
    use_amp = (device.type == "cuda")

    def _load_batch(paths: list[str]) -> torch.Tensor:
        arrays = [np.load(p).astype(np.float32) / 255.0 for p in paths]
        return torch.from_numpy(np.stack(arrays)).to(device)

    def _embed_paths(paths: list[str]) -> torch.Tensor:
        chunks = []
        for i in range(0, len(paths), batch_size):
            batch = _load_batch(paths[i : i + batch_size])
            with torch.amp.autocast("cuda", enabled=use_amp):
                emb = model(batch)
            chunks.append(emb.cpu().float())
        return torch.cat(chunks, dim=0)

    model.eval()
    all_scores: dict[int, dict[str, float]] = {}

    for wid in writer_ids:
        genuine_paths = sample_map.get(wid, {}).get("G", [])
        forgery_paths = sample_map.get(wid, {}).get("F", [])
        if not genuine_paths or not forgery_paths:
            continue
        g_embs   = _embed_paths(genuine_paths)
        centroid = F.normalize(g_embs.mean(0), p=2, dim=0)
        f_embs   = _embed_paths(forgery_paths)
        cos_sims = (f_embs @ centroid).clamp(min=0.0)
        wscores: dict[str, float] = {
            fp: float(s) for fp, s in zip(forgery_paths, cos_sims.tolist())
        }
        all_scores[wid] = wscores

    return all_scores

def _batch_pair_stats(
    emb:   torch.Tensor,
    w_ids: torch.Tensor,
    types: torch.Tensor,
) -> tuple[float, float]:
    N          = emb.shape[0]
    device     = emb.device
    dists      = torch.cdist(emb, emb, p=2)
    is_genuine = (types == 0)
    same_w     = w_ids.unsqueeze(0) == w_ids.unsqueeze(1)
    eye        = torch.eye(N, dtype=torch.bool, device=device)
    pos_mask = same_w & is_genuine.unsqueeze(0) & is_genuine.unsqueeze(1) & ~eye
    neg_mask = (
        (~same_w & is_genuine.unsqueeze(0) & is_genuine.unsqueeze(1))
        | ((types == 1).unsqueeze(0) & is_genuine.unsqueeze(1))
        | (is_genuine.unsqueeze(0) & (types == 1).unsqueeze(1))
    )
    ap = dists[pos_mask].mean().item() if pos_mask.any() else 0.0
    an = dists[neg_mask].mean().item() if neg_mask.any() else 0.0
    return ap, an


def _eer_from_distances(
    genuine_d:    np.ndarray,
    impostor_d:   np.ndarray,
    n_thresholds: int = 200,
) -> float:
    if len(genuine_d) == 0 or len(impostor_d) == 0:
        return 1.0
    all_d      = np.concatenate([genuine_d, impostor_d])
    thresholds = np.linspace(float(all_d.min()), float(all_d.max()), n_thresholds)
    min_diff   = float("inf")
    best_eer   = 1.0
    for thresh in thresholds:
        frr  = float((genuine_d   > thresh).mean())
        far  = float((impostor_d <= thresh).mean())
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            best_eer = (far + frr) / 2.0
    return float(best_eer)


def _compute_eer(
    emb:          torch.Tensor,
    w_ids:        torch.Tensor,
    types:        torch.Tensor,
    n_thresholds: int = 200,
) -> float:
    N          = emb.shape[0]
    dists      = torch.cdist(emb.float(), emb.float(), p=2)
    is_genuine = (types == 0)
    same_w     = w_ids.unsqueeze(0) == w_ids.unsqueeze(1)
    eye        = torch.eye(N, dtype=torch.bool)
    genuine_mask  = same_w & is_genuine.unsqueeze(0) & is_genuine.unsqueeze(1) & ~eye
    impostor_mask = (
        (~same_w & is_genuine.unsqueeze(0) & is_genuine.unsqueeze(1))
        | ((types == 1).unsqueeze(0) & is_genuine.unsqueeze(1))
        | (is_genuine.unsqueeze(0) & (types == 1).unsqueeze(1))
    )
    return _eer_from_distances(
        dists[genuine_mask].numpy(),
        dists[impostor_mask].numpy(),
        n_thresholds,
    )


def _compute_eer_split(
    emb:          torch.Tensor,
    w_ids:        torch.Tensor,
    types:        torch.Tensor,
    n_thresholds: int = 200,
) -> tuple[float, float]:
    N          = emb.shape[0]
    dists      = torch.cdist(emb.float(), emb.float(), p=2)
    is_genuine = (types == 0)
    same_w     = w_ids.unsqueeze(0) == w_ids.unsqueeze(1)
    eye        = torch.eye(N, dtype=torch.bool)
    genuine_mask = same_w & is_genuine.unsqueeze(0) & is_genuine.unsqueeze(1) & ~eye
    forgery_mask = (
        ((types == 1).unsqueeze(0) & is_genuine.unsqueeze(1))
        | (is_genuine.unsqueeze(0) & (types == 1).unsqueeze(1))
    )
    random_mask = ~same_w & is_genuine.unsqueeze(0) & is_genuine.unsqueeze(1) & ~eye
    genuine_d = dists[genuine_mask].numpy()
    return (
        _eer_from_distances(genuine_d, dists[forgery_mask].numpy(), n_thresholds),
        _eer_from_distances(genuine_d, dists[random_mask].numpy(),  n_thresholds),
    )


def _compute_similarity_eer(
    emb:          torch.Tensor,
    w_ids:        torch.Tensor,
    types:        torch.Tensor,
    n_thresholds: int   = 200,
    target_sim:   float = 0.80,
) -> tuple[float, float, float]:
    emb   = emb.float()
    N     = emb.shape[0]
    sims       = (emb @ emb.T).clamp(-1.0, 1.0)
    is_genuine = (types == 0)
    same_w     = w_ids.unsqueeze(0) == w_ids.unsqueeze(1)
    eye        = torch.eye(N, dtype=torch.bool)
    genuine_mask  = same_w & is_genuine.unsqueeze(0) & is_genuine.unsqueeze(1) & ~eye
    impostor_mask = (
        (~same_w & is_genuine.unsqueeze(0) & is_genuine.unsqueeze(1))
        | ((types == 1).unsqueeze(0) & is_genuine.unsqueeze(1))
        | (is_genuine.unsqueeze(0) & (types == 1).unsqueeze(1))
    )
    if not genuine_mask.any() or not impostor_mask.any():
        return 1.0, 0.0, 1.0
    genuine_sim  = sims[genuine_mask].numpy()
    impostor_sim = sims[impostor_mask].numpy()
    tar_at_target = float((genuine_sim  >= target_sim).mean())
    far_at_target = float((impostor_sim >= target_sim).mean())
    all_sim    = np.concatenate([genuine_sim, impostor_sim])
    thresholds = np.linspace(float(all_sim.min()), float(all_sim.max()), n_thresholds)
    min_diff   = float("inf")
    best_eer   = 1.0
    for thresh in thresholds:
        frr  = float((genuine_sim  < thresh).mean())
        far  = float((impostor_sim >= thresh).mean())
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            best_eer = (far + frr) / 2.0
    return float(best_eer), tar_at_target, far_at_target


def _per_writer_pos_dist(
    emb:   torch.Tensor,
    w_ids: torch.Tensor,
    types: torch.Tensor,
) -> dict[int, float]:
    is_genuine = (types == 0)
    dists      = torch.cdist(emb.float(), emb.float(), p=2)
    eye        = torch.eye(len(emb), dtype=torch.bool)
    writer_dists: dict[int, float] = {}
    for wid in w_ids[is_genuine].unique().tolist():
        wid       = int(wid)
        wid_mask  = (w_ids == wid)
        pos_mask  = (
            wid_mask.unsqueeze(0) & wid_mask.unsqueeze(1)
            & is_genuine.unsqueeze(0) & is_genuine.unsqueeze(1)
            & ~eye
        )
        if pos_mask.any():
            writer_dists[wid] = dists[pos_mask].mean().item()
    return writer_dists


def _soft_triplet_val_loss(
    emb:    torch.Tensor,
    w_ids:  torch.Tensor,
    types:  torch.Tensor,
    margin: float = 0.3,
) -> float:
    N          = emb.shape[0]
    dists      = torch.cdist(emb.float(), emb.float(), p=2)
    is_genuine = (types == 0)
    same_w     = w_ids.unsqueeze(0) == w_ids.unsqueeze(1)
    eye        = torch.eye(N, dtype=torch.bool)
    pos_mask  = same_w & is_genuine.unsqueeze(0) & is_genuine.unsqueeze(1) & ~eye
    neg_mask  = (
        (~same_w & is_genuine.unsqueeze(0) & is_genuine.unsqueeze(1))
        | ((types == 1).unsqueeze(0) & is_genuine.unsqueeze(1))
    )
    pos_d     = dists.masked_fill(~pos_mask, -_BIG)
    neg_d     = dists.masked_fill(~neg_mask,  _BIG)
    hard_pos  = pos_d.max(dim=1).values
    hard_neg  = neg_d.min(dim=1).values
    valid = is_genuine & (hard_pos > -_BIG / 2) & (hard_neg < _BIG / 2)
    if not valid.any():
        return 0.0
    return F.softplus(hard_pos[valid] - hard_neg[valid] + margin).mean().item()



def _run_train_epoch(
    model:              TAVNet,
    arcface:            SubCenterArcFaceLoss,
    loader:             DataLoader,
    device:             torch.device,
    optimizer:          AdamW,
    scaler:             torch.amp.GradScaler,
    wid_to_idx_lut:     torch.Tensor,
    current_epoch:      int,
    accumulation_steps: int = 4,
) -> tuple[float, float, float]:
    model.train()
    arcface.train()
    optimizer.zero_grad(set_to_none=True)
    
    # Triplet loss warmup: scale from 0 to 1 over 10 epochs
    triplet_weight = min(1.0, (current_epoch - 1) / 10.0)
    
    # Triplet margin decay: linearly decay from 0.30 (epoch 1) to 0.05 (epoch 50),
    # then keep it at 0.05 for later epochs.
    margin_start, margin_end, total_epochs = 0.30, 0.05, 50
    decay_progress = min(1.0, max(0.0, (current_epoch - 1) / max(1, total_epochs - 1)))
    triplet_margin = margin_start + (margin_end - margin_start) * decay_progress
    triplet_criterion = nn.TripletMarginLoss(margin=triplet_margin)

    sum_loss = sum_ap = sum_an = 0.0
    n_steps  = 0
    use_amp  = (device.type == "cuda")

    for step, (tensors, w_ids, types) in enumerate(
        _pbar(loader, desc="  trn", total=len(loader), leave=False, unit="batch")
    ):
        tensors   = tensors.to(device, non_blocking=True)
        w_ids_dev = w_ids.to(device,   non_blocking=True)
        types_dev = types.to(device,   non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            emb = model(tensors)

            is_genuine = (types_dev == 0)
            gen_emb    = emb[is_genuine]
            gen_wids   = w_ids_dev[is_genuine]

            gen_labels = wid_to_idx_lut[gen_wids]

            arcface_loss = arcface(gen_emb, gen_labels)

            # Batch-hard triplet over APN bundles:
            # slot 0 = anchor, slots 1-3 = positives, slots 4-7 = negatives.
            emb_bundles = emb.view(-1, _BUNDLE_SIZE, emb.shape[1])
            types_bundles = types_dev.view(-1, _BUNDLE_SIZE)

            anchors = emb_bundles[:, 0, :]
            pos_candidates = emb_bundles[:, 1:4, :]
            neg_candidates = emb_bundles[:, 4:, :]

            # Hard positive: farthest positive from anchor.
            pos_dists = torch.norm(anchors.unsqueeze(1) - pos_candidates, p=2, dim=2)
            hard_pos_idx = pos_dists.argmax(dim=1)
            row_idx = torch.arange(anchors.shape[0], device=anchors.device)
            hard_positives = pos_candidates[row_idx, hard_pos_idx]

            # Hard negative: closest forgery among negative slots.
            neg_dists = torch.norm(anchors.unsqueeze(1) - neg_candidates, p=2, dim=2)
            neg_types = types_bundles[:, 4:]
            forgery_mask = (neg_types == 1)

            if forgery_mask.any():
                neg_dists_forg = neg_dists.masked_fill(~forgery_mask, float("inf"))
                no_forgery_row = ~forgery_mask.any(dim=1)
                hard_neg_idx_forg = neg_dists_forg.argmin(dim=1)
                hard_neg_idx_any = neg_dists.argmin(dim=1)
                hard_neg_idx = torch.where(no_forgery_row, hard_neg_idx_any, hard_neg_idx_forg)
            else:
                hard_neg_idx = neg_dists.argmin(dim=1)

            hard_negatives = neg_candidates[row_idx, hard_neg_idx]

            triplet_loss = triplet_criterion(anchors, hard_positives, hard_negatives)

            total_loss = arcface_loss + (triplet_weight * triplet_loss)
            loss = total_loss / accumulation_steps

        scaler.scale(loss).backward()

        is_last    = (step + 1) == len(loader)
        should_opt = ((step + 1) % accumulation_steps == 0) or is_last
        if should_opt:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(arcface.parameters()),
                max_norm=1.0,
            )
            scale_before = scaler.get_scale()
            scaler.step(optimizer)  
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            ap, an = _batch_pair_stats(emb.detach(), w_ids_dev, types_dev)

        sum_loss += loss.item() * accumulation_steps
        sum_ap   += ap
        sum_an   += an
        n_steps  += 1

    n = max(n_steps, 1)
    return sum_loss / n, sum_ap / n, sum_an / n



def _run_val_epoch(
    model:  "TAVNet",
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float, float, float, float, float, dict[int, float]]:
    model.eval()
    use_amp = (device.type == "cuda")

    all_embs:  list[torch.Tensor] = []
    all_wids:  list[torch.Tensor] = []
    all_types: list[torch.Tensor] = []

    with torch.no_grad():
        for tensors, w_ids, types in _pbar(
            loader, desc="  val", total=len(loader), leave=False, unit="batch"
        ):
            tensors = tensors.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                emb = model(tensors)
            all_embs.append(emb.cpu().float())
            all_wids.append(w_ids.cpu())
            all_types.append(types.cpu())

    embs_cat  = torch.cat(all_embs)
    wids_cat  = torch.cat(all_wids)
    types_cat = torch.cat(all_types)

    val_loss             = _soft_triplet_val_loss(embs_cat, wids_cat, types_cat)
    val_eer              = _compute_eer(embs_cat, wids_cat, types_cat)
    forg_eer, rand_eer   = _compute_eer_split(embs_cat, wids_cat, types_cat)
    _sim_eer, tar_at_08, _far_at_08 = _compute_similarity_eer(
        embs_cat, wids_cat, types_cat, target_sim=0.80
    )

    is_genuine = (types_cat == 0)
    same_w     = wids_cat.unsqueeze(0) == wids_cat.unsqueeze(1)
    eye        = torch.eye(len(embs_cat), dtype=torch.bool)
    dists      = torch.cdist(embs_cat, embs_cat, p=2)

    pos_mask   = same_w & is_genuine.unsqueeze(0) & is_genuine.unsqueeze(1) & ~eye
    neg_mask   = (
        (~same_w & is_genuine.unsqueeze(0) & is_genuine.unsqueeze(1))
        | ((types_cat == 1).unsqueeze(0) & is_genuine.unsqueeze(1))
        | (is_genuine.unsqueeze(0) & (types_cat == 1).unsqueeze(1))
    )
    avg_pos = dists[pos_mask].mean().item() if pos_mask.any() else 0.0
    avg_neg = dists[neg_mask].mean().item() if neg_mask.any() else 0.0

    writer_pos_dists = _per_writer_pos_dist(embs_cat, wids_cat, types_cat)

    return val_loss, val_eer, forg_eer, rand_eer, avg_pos, avg_neg, tar_at_08, writer_pos_dists



def _scan_dir(
    processed_dir: Path,
) -> dict[int, dict[str, list[str]]]:
    sample_map: dict[int, dict[str, list[str]]] = {}
    skipped = 0
    for f in sorted(processed_dir.glob("*.npy")):
        parts = f.stem.split("_")
        if len(parts) < 4:
            skipped += 1
            continue
        status = parts[-2].upper()
        if status not in ("G", "F"):
            skipped += 1
            continue
        try:
            uid = int(parts[-3])
        except ValueError:
            skipped += 1
            continue
        sample_map.setdefault(uid, {"G": [], "F": []})[status].append(str(f))

    if skipped:
        logging.getLogger("tavnet").warning(
            "%d .npy files had unexpected names and were skipped.", skipped
        )
    return sample_map


def _make_split(
    sample_map: dict[int, dict[str, list[str]]],
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    eligible = sorted(
        uid for uid, v in sample_map.items()
        if len(v["G"]) >= _MIN_GENUINE and len(v["F"]) >= _MIN_FORGERY
    )
    arr = np.array(eligible, dtype=np.int64)
    np.random.default_rng(seed).shuffle(arr)
    n = len(arr)

    n_train = int(round(n * _TRAIN_RATIO))
    n_val   = int(round(n * _VAL_RATIO))
    n_test  = n - n_train - n_val

    if n >= 3:
        n_train = max(1, n_train)
        n_val   = max(1, n_val)
        n_test  = max(1, n_test)

    while n_train + n_val + n_test > n:
        if n_train >= n_test and n_train > 1:
            n_train -= 1
        elif n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            break
    while n_train + n_val + n_test < n:
        n_train += 1

    return (
        arr[:n_train].tolist(),
        arr[n_train : n_train + n_val].tolist(),
        arr[n_train + n_val :].tolist(),
    )


def _build_manifest(
    train_ids:  list[int],
    val_ids:    list[int],
    test_ids:   list[int],
    sample_map: dict[int, dict[str, list[str]]],
) -> dict:
    def _cnt(ids, key):
        return sum(len(sample_map.get(w, {}).get(key, [])) for w in ids)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split_ratios": {
            "train": _TRAIN_RATIO,
            "val":   _VAL_RATIO,
            "test":  _TEST_RATIO,
        },
        "split": {
            "train": sorted(train_ids),
            "val":   sorted(val_ids),
            "test":  sorted(test_ids),
        },
        "stats": {
            "n_writers": {
                "train": len(train_ids),
                "val":   len(val_ids),
                "test":  len(test_ids),
                "total": len(train_ids) + len(val_ids) + len(test_ids),
            },
            "n_genuine": {
                "train": _cnt(train_ids, "G"),
                "val":   _cnt(val_ids,   "G"),
                "test":  _cnt(test_ids,  "G"),
            },
            "n_forgery": {
                "train": _cnt(train_ids, "F"),
                "val":   _cnt(val_ids,   "F"),
                "test":  _cnt(test_ids,  "F"),
            },
        },
    }


def _manifest_has_target_split(manifest: dict) -> bool:
    ratios = manifest.get("split_ratios")
    if not isinstance(ratios, dict):
        return False
    try:
        return (
            abs(float(ratios.get("train", -1)) - _TRAIN_RATIO) < 1e-9
            and abs(float(ratios.get("val", -1)) - _VAL_RATIO) < 1e-9
            and abs(float(ratios.get("test", -1)) - _TEST_RATIO) < 1e-9
        )
    except (TypeError, ValueError):
        return False


_HDR = (
    f"{'Epoch':>5}  {'trn_loss':>8}  {'val_loss':>8}  "
    f"{'val_EER':>7}  {'Forg_EER':>8}  {'Rand_EER':>8}  "
    f"{'Pos_Dist':>8}  {'Neg_Dist':>8}  {'TAR@.8':>7}  {'LR':>10}  Best"
)
_SEP = "─" * (len(_HDR) + 2)


def _print_header() -> None:
    print(f"\n{_SEP}")
    print(_HDR)
    print(_SEP)


def _print_hint(text: str) -> None:
    print(f"          ↳ {text}")


def _print_row(
    epoch:     int,
    trn_loss:  float,
    val_loss:  float,
    val_eer:   float,
    forg_eer:  float,
    rand_eer:  float,
    pos_dist:  float,
    neg_dist:  float,
    tar_at_08: float,
    lr:        float,
    best:      bool,
) -> None:
    mark = "  ✅" if best else "    "

    def _fmt(v: float) -> str:
        return f"{v:.2%}" if not math.isnan(v) else "  N/A "

    print(
        f"{epoch:>5}  {trn_loss:>8.4f}  {val_loss:>8.4f}  "
        f"{_fmt(val_eer):>7}  {_fmt(forg_eer):>8}  {_fmt(rand_eer):>8}  "
        f"{pos_dist:>8.4f}  {neg_dist:>8.4f}  {_fmt(tar_at_08):>7}  {lr:>10.2e}{mark}"
    )



def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train TAV-Net: ResNet-50 + CBAM + Transformer + Sub-Center ArcFace"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--epochs",       type=int,   default=50,
                   help="Total training epochs.")
    p.add_argument("--batch-size",   type=int,   default=8,
                   help="Writers per physical batch (8 tensors each → 64 tensors/batch).")
    p.add_argument("--accum-steps",  type=int,   default=4,
                   help="Gradient accumulation steps. Reduces VRAM without shrinking "
                        "effective batch size.")
    p.add_argument("--lr",           type=float, default=1e-4,
                   help="Peak learning rate (linear warmup → CosineAnnealingLR).")
    p.add_argument("--wd",           type=float, default=5e-4,
                   help="AdamW weight decay (5e-4 works well with compact sub-center heads).")
    p.add_argument("--embed-dim",    type=int,   default=512,
                   help="Embedding dimensionality.")
    p.add_argument("--arcface-m",    type=float, default=0.55,
                   help="ArcFace angular margin (radians).")
    p.add_argument("--arcface-s",    type=float, default=64.0,
                   help="ArcFace logit scale.")
    p.add_argument("--arcface-k",    type=int,   default=7,
                   help="Number of ArcFace sub-centers per class.")
    p.add_argument("--num-workers",  type=int,   default=2,
                   help="DataLoader worker processes  (0 = main process).")
    p.add_argument("--seed",         type=int,   default=42,
                   help="Global random seed.")
    p.add_argument("--resume",       type=str,   default=None, metavar="PATH",
                   help="Resume from a saved checkpoint (.pt).")
    p.add_argument("--processed-dir", type=str,  default=None, metavar="DIR",
                   help="Override path to the processed .npy directory.")
    p.add_argument("--hard-mining",  action="store_true",
                   help="Enable hard-negative mining (N=10): after "
                        "--hard-mining-start epochs, the top-%d hardest forgeries "
                        "per writer are used as the sampling pool (difficulty = "
                        "cosine similarity to genuine centroid). Requires "
                        "persistent_workers=False and one extra inference pass "
                        "per epoch." % _HN_TOP_K)
    p.add_argument("--hard-mining-start", type=int, default=_HN_WARMUP_EPOCHS,
                   help="Epoch (1-indexed) after which hard-negative mining "
                        "activates.  Model should be somewhat trained first.")
    return p.parse_args()



def main() -> None:
    args = _parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(message)s",
        handlers=[logging.StreamHandler()],
    )
    log = logging.getLogger("tavnet")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device : %s", device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        props = torch.cuda.get_device_properties(0)
        log.info(
            "GPU    : %s  (%.1f GB VRAM)",
            props.name, props.total_memory / 1e9,
        )

    if args.processed_dir:
        processed_dir = Path(args.processed_dir)
    elif _PROC_.exists() and any(_PROC_.glob("*.npy")):
        processed_dir = _PROC_
        log.info("Using tensors (4-ch uint8): %s", processed_dir)
    else:
        log.error(
            "No processed .npy directory found.  "
            "Run extract_features.py first."
        )
        sys.exit(1)

    log.info("Scanning %s …", processed_dir)
    sample_map = _scan_dir(processed_dir)
    if not sample_map:
        log.error("No .npy files found in %s", processed_dir)
        sys.exit(1)

    eligible = [
        uid for uid, v in sample_map.items()
        if len(v["G"]) >= _MIN_GENUINE and len(v["F"]) >= _MIN_FORGERY
    ]
    log.info(
        "Writers: %d total  |  %d eligible (≥%d genuine, ≥%d forgery)",
        len(sample_map), len(eligible), _MIN_GENUINE, _MIN_FORGERY,
    )
    if len(eligible) < 6:
        log.error(
            "Need ≥6 eligible writers for a 70/10/20 split. "
            "Found only %d.", len(eligible)
        )
        sys.exit(1)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    if MANIFEST_PATH.exists() and args.resume is None:
        log.info("Found existing split manifest: %s", MANIFEST_PATH)
        with MANIFEST_PATH.open("r", encoding="utf-8") as fh:
            manifest  = json.load(fh)
        if _manifest_has_target_split(manifest):
            log.info("Reusing existing writer split (70/10/20).")
            train_ids = manifest["split"]["train"]
            val_ids   = manifest["split"]["val"]
            test_ids  = manifest["split"]["test"]
        else:
            log.warning("Existing manifest is not 70/10/20; regenerating split.")
            train_ids, val_ids, test_ids = _make_split(sample_map, seed=args.seed)
            manifest = _build_manifest(train_ids, val_ids, test_ids, sample_map)
            with MANIFEST_PATH.open("w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2)
            log.info("Writer split saved → %s", MANIFEST_PATH)
    else:
        train_ids, val_ids, test_ids = _make_split(sample_map, seed=args.seed)
        manifest = _build_manifest(train_ids, val_ids, test_ids, sample_map)
        with MANIFEST_PATH.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        log.info("Writer split saved → %s", MANIFEST_PATH)

    log.info(
        "Split  train=%d  val=%d  test=%d  writers",
        len(train_ids), len(val_ids), len(test_ids),
    )
    if manifest.get("stats"):
        s = manifest["stats"]
        log.info(
            "       train  %d genuine  %d forgery",
            s["n_genuine"]["train"], s["n_forgery"]["train"],
        )
        log.info(
            "       val    %d genuine  %d forgery",
            s["n_genuine"]["val"], s["n_forgery"]["val"],
        )

    train_writers   = sorted(set(train_ids))
    writer_to_idx   = {w: i for i, w in enumerate(train_writers)}
    n_train_writers = len(train_writers)
    log.info("ArcFace: %d training writers (classes)", n_train_writers)

    max_wid         = max(writer_to_idx.keys())
    wid_to_idx_lut  = torch.full((max_wid + 1,), -1, dtype=torch.long, device=device)
    for wid, idx in writer_to_idx.items():
        wid_to_idx_lut[wid] = idx

    train_ds = APNBundleDataset(train_ids, sample_map, augment=True)
    val_ds   = APNBundleDataset(val_ids,   sample_map, augment=False)
    log.info(
        "Datasets  train=%d writers  val=%d writers  (after bundle-eligibility filter)",
        len(train_ds), len(val_ds),
    )

    _loader_kw = dict(
        collate_fn         = _collate_bundles,
        worker_init_fn     = _worker_init_fn,
        pin_memory         = (device.type == "cuda"),
        # Must be False so per-epoch dataset state (curriculum epoch) is visible to workers.
        persistent_workers = False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        drop_last   = True,
        **_loader_kw,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        drop_last   = False,
        **_loader_kw,
    )

    log.info(
        "Building TAV-Net  (ResNet-50 + CBAM@layer4 + Transformer + embed=%d) …",
        args.embed_dim,
    )
    model = TAVNet(embed_dim=args.embed_dim).to(device)

    arcface = SubCenterArcFaceLoss(
        in_features = args.embed_dim,
        n_classes   = n_train_writers,
        K           = args.arcface_k,
        s           = args.arcface_s,
        m           = args.arcface_m,
    ).to(device)

    all_params = list(model.parameters()) + list(arcface.parameters())
    optimizer  = AdamW(all_params, lr=args.lr, weight_decay=args.wd)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    start_epoch = 1
    best_eer    = float("inf")

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            log.error("Checkpoint not found: %s", resume_path)
            sys.exit(1)
        ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)
        _load_model_state_compat(model, ckpt["model_state"])
        arcface.load_state_dict(ckpt["arcface_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        except (ValueError, KeyError):
            log.warning("Could not load scheduler state (may be incompatible with ReduceLROnPlateau)")
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_eer    = ckpt.get("best_eer", float("inf"))
        log.info(
            "Resumed from %s  (epoch %d, best EER=%.4f)",
            resume_path, start_epoch - 1, best_eer,
        )

    log.info(
        "\nHyperparameters"
        "\n  epochs        : %d"
        "\n  phys batch    : %d writers × %d slots = %d tensors"
        "\n  accum steps   : %d   →  effective %d writers per opt step"
        "\n  lr (peak)     : %.0e   wd: %.0e   AMP: %s"
        "\n  scheduler     : ReduceLROnPlateau (factor=0.5, patience=3, metric=val_eer)"
        "\n  triplet warmup: scales from 0 to 1 over 10 epochs"
        "\n  augmentation  : gentler rotation (10°, p=0.5), scale (0.9-1.1), shear (5°), erase (p=0.2)"
        "\n  ArcFace       : m=%.2f  s=%.0f  K=%d  classes=%d"
        "\n  HNM top-K     : %d  (hardest forgeries per writer in mining pool)"
        "\n  target metrics: Pos_Dist < 0.5  |  Neg_Dist > 1.5  |  TAR@.8 ≥ 80%%"
        "\n  embed dim     : %d"
        "\n  transformer   : 1 layer | 8 heads | FFN 2048 | Pre-LN",
        args.epochs,
        args.batch_size, _BUNDLE_SIZE, args.batch_size * _BUNDLE_SIZE,
        args.accum_steps, args.batch_size * args.accum_steps,
        args.lr, args.wd, "on" if device.type == "cuda" else "off",
        args.arcface_m, args.arcface_s, args.arcface_k, n_train_writers,
        _HN_TOP_K,
        args.embed_dim,
    )

    if args.hard_mining:
        log.info(
            "Hard-negative mining ENABLED — activates after epoch %d.",
            args.hard_mining_start,
        )

    _print_header()

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter()

        train_ds.set_epoch(epoch)

        trn_loss, trn_ap, trn_an = _run_train_epoch(
            model, arcface, train_loader, device,
            optimizer, scaler,
            wid_to_idx_lut,
            current_epoch=epoch,
            accumulation_steps = args.accum_steps,
        )

        val_loss, val_eer, forg_eer, rand_eer, val_ap, val_an, tar_at_08, writer_dists = (
            _run_val_epoch(model, val_loader, device)
        )

        # Step scheduler based on validation EER
        scheduler.step(val_eer)
        
        is_best = (val_eer < best_eer) 

        if is_best:
            best_eer = val_eer
            torch.save(
                {
                    "epoch":           epoch,
                    "model_state":     model.state_dict(),
                    "arcface_state":   arcface.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state":    scaler.state_dict(),
                    "best_eer":        best_eer,
                    "embed_dim":       args.embed_dim,
                    "arcface_k":       args.arcface_k,
                    "n_train_writers": n_train_writers,
                    "writer_to_idx":   writer_to_idx,
                    "manifest":        manifest,
                    "args":            vars(args),
                },
                str(BEST_CKPT_PATH),
            )
        current_lr = optimizer.param_groups[0]["lr"]
        _print_row(
            epoch, trn_loss, val_loss, val_eer,
            forg_eer, rand_eer,
            val_ap, val_an, tar_at_08, current_lr, is_best,
        )

        if tar_at_08 >= 0.80:
            _print_hint(
                f"Target TAR@0.8 = {tar_at_08:.2%} ✔  (≥ 80% of genuine pairs "
                f"score ≥ 0.80 cosine similarity)"
            )
        elif val_ap < 0.50:
            _print_hint(
                f"Pos_Dist = {val_ap:.4f} < 0.50 ✔  (target met)  "
                f"TAR@0.8 = {tar_at_08:.2%}"
            )

        if writer_dists and epoch % 5 == 0:
            dist_vals = np.array(list(writer_dists.values()))
            p75       = float(np.percentile(dist_vals, 75))
            outliers  = sorted(
                [(wid, d) for wid, d in writer_dists.items() if d > p75],
                key=lambda x: -x[1],
            )[:5]

        if args.hard_mining and epoch >= args.hard_mining_start:
            _print_hint(
                f"[HNM] Scoring {len(train_ds.writer_ids)} writers …"
            )
            hard_scores = compute_hard_forgery_scores(
                model, sample_map, train_ds.writer_ids, device,
                batch_size=args.batch_size * _BUNDLE_SIZE,
            )
            hard_count = sum(
                sum(1 for v in wsc.values() if v >= _HN_SIM_THRESHOLD)
                for wsc in hard_scores.values()
            )
            total_f = sum(len(wsc) for wsc in hard_scores.values())
            _print_hint(
                f"[HNM] Hard forgeries (sim≥{_HN_SIM_THRESHOLD:.2f}): "
                f"{hard_count}/{total_f}  ({100.0 * hard_count / max(total_f, 1):.1f}%)"
            )
            train_ds.update_forgery_scores(hard_scores)
            train_loader = DataLoader(
                train_ds,
                batch_size  = args.batch_size,
                shuffle     = True,
                num_workers = args.num_workers,
                drop_last   = True,
                **_loader_kw,
            )

    print(_SEP)
    log.info("Training complete.")
    log.info("Best val EER  : %.4f  (%.2f%%)", best_eer, best_eer * 100)
    log.info("Checkpoint    : %s", BEST_CKPT_PATH)
    log.info("Manifest      : %s", MANIFEST_PATH)


if __name__ == "__main__":
    main()