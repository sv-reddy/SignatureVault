"""
train_tavnet_ablation.py - Channel Ablation Training for TAV-Net
=================================================================

Trains TAV-Net variants from scratch to quantify contribution of each cue.

Variants:
  - baseline : channel [0]               (Shape only)
  - step1    : channels [0, 1]           (Shape + Pseudo-Pressure)
  - step2    : channels [0, 1, 2]        (Shape + Pseudo-Pressure + Stroke Angle)
  - full     : channels [0, 1, 2, 3]     (All cues)

This script does not modify existing scripts. It reuses the core training
utilities from train_tavnet.py and writes outputs to:
  backend/checkpoints/ablation/<variant>/

Examples:
  python train_tavnet_ablation.py --variant all
  python train_tavnet_ablation.py --variant step2 --epochs 50 --arcface-k 7
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import torchvision.models as tvm
from torchvision.models import ResNet50_Weights

_SCRIPT_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _SCRIPT_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import train_tavnet as base

_CKPT_ROOT = _BACKEND_DIR / "checkpoints" / "ablation"
_MANIFEST_PATH = _CKPT_ROOT / "manifest.json"


VARIANTS = {
    "baseline": {
        "channels": [0],
        "label": "Shape only",
    },
    "step1": {
        "channels": [0, 1],
        "label": "Shape + Pseudo-Pressure",
    },
    "step2": {
        "channels": [0, 1, 2],
        "label": "Shape + Pseudo-Pressure + Stroke Angle",
    },
    "full": {
        "channels": [0, 1, 2, 3],
        "label": "Shape + Pseudo-Pressure + Stroke Angle + Skeleton",
    },
}


class AblationTAVNet(nn.Module):
    _SEQ_LEN = 12 * 12
    _N_TOKENS = _SEQ_LEN

    def __init__(self, in_channels: int, embed_dim: int = 512) -> None:
        super().__init__()
        if in_channels < 1 or in_channels > 4:
            raise ValueError(f"in_channels must be in [1,4], got {in_channels}")

        base_model = tvm.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        old_w = base_model.conv1.weight.data
        new_conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        with torch.no_grad():
            if in_channels <= 3:
                new_conv1.weight[:, :in_channels, :, :] = old_w[:, :in_channels, :, :]
            else:
                new_conv1.weight[:, :3, :, :] = old_w
                new_conv1.weight[:, 3:4, :, :] = old_w.mean(dim=1, keepdim=True)
        base_model.conv1 = new_conv1

        self.stem = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
        )
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        for module in (self.stem, self.layer1, self.layer2):
            for param in module.parameters():
                param.requires_grad = False

        self.cbam = base.CBAM(channels=2048, reduction=16, spatial_k=7)

        self.pos_embed = nn.Parameter(torch.zeros(self._N_TOKENS, 1, 2048))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=2048,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=False,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=1, enable_nested_tensor=False
        )
        self.gem = base.GeM(p=3.0, eps=1e-6)

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
        b = x.shape[0]
        x = x.flatten(2)
        x = x.permute(2, 0, 1)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = x.permute(1, 2, 0).contiguous().view(b, 2048, 12, 12)
        x = self.gem(x).flatten(1)
        x = self.head(x)
        return F.normalize(x, p=2, dim=1)


class AblationAPNBundleDataset(base.APNBundleDataset):
    def __init__(
        self,
        writer_ids: list[int],
        sample_map: dict[int, dict[str, list[str]]],
        channel_indices: list[int],
        augment: bool = False,
    ) -> None:
        super().__init__(writer_ids=writer_ids, sample_map=sample_map, augment=augment)
        self.channel_indices = list(channel_indices)
        self._orig_to_local = {orig: i for i, orig in enumerate(self.channel_indices)}

    def _apply_channel_aware_morph(self, t: torch.Tensor) -> torch.Tensor:
        if random.random() >= base._MORPH_AUG.p:
            return t
        out = t.clone()
        # Morphology is meaningful for original channels 0 (shape) and 3 (skeleton).
        for orig_ch in (0, 3):
            if orig_ch not in self._orig_to_local:
                continue
            c = self._orig_to_local[orig_ch]
            ch = out[c:c + 1].unsqueeze(0)
            if random.random() < 0.5:
                morphed = F.max_pool2d(ch, kernel_size=3, stride=1, padding=1)
            else:
                morphed = -F.max_pool2d(-ch, kernel_size=3, stride=1, padding=1)
            out[c] = morphed.squeeze(0).squeeze(0)
        return out

    def _augment_selected(self, t: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.2:
            t = base._elastic_deform(t)
        t = self._apply_channel_aware_morph(t)
        if base._ROT_AUG is not None and random.random() < 0.5:
            t = base._ROT_AUG(t)
        if base._AFFINE_AUG is not None:
            t = base._AFFINE_AUG(t)
        if base._PERSP_AUG is not None and random.random() < 0.3:
            t = base._PERSP_AUG(t)
        if base._ERASE_AUG is not None:
            t = base._ERASE_AUG(t)
        if random.random() < 0.3:
            t = (t + torch.randn_like(t) * 0.05).clamp_(0.0, 1.0)
        return t

    def _load(self, path: str) -> torch.Tensor:
        arr = np.load(path).astype(np.float32) / 255.0
        arr = arr[self.channel_indices, :, :]
        t = torch.from_numpy(arr)
        if self.augment:
            t = self._augment_selected(t)
        return t


@torch.no_grad()
def compute_hard_forgery_scores_ablation(
    model: nn.Module,
    sample_map: dict[int, dict[str, list[str]]],
    writer_ids: list[int],
    channel_indices: list[int],
    device: torch.device,
    batch_size: int = 24,
) -> dict[int, dict[str, float]]:
    use_amp = device.type == "cuda"

    def _load_batch(paths: list[str]) -> torch.Tensor:
        arrays = [
            np.load(p).astype(np.float32)[channel_indices, :, :] / 255.0
            for p in paths
        ]
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

        g_embs = _embed_paths(genuine_paths)
        centroid = F.normalize(g_embs.mean(0), p=2, dim=0)
        f_embs = _embed_paths(forgery_paths)
        cos_sims = (f_embs @ centroid).clamp(min=0.0)
        all_scores[wid] = {
            fp: float(s) for fp, s in zip(forgery_paths, cos_sims.tolist())
        }

    return all_scores


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train ablation variants of TAV-Net from scratch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=["baseline", "step1", "step2", "full", "all"],
        help="Which ablation variant to train.",
    )
    p.add_argument("--epochs", type=int, default=15, help="Total training epochs.")
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Writers per physical batch (8 tensors each -> 64 tensors/batch).",
    )
    p.add_argument(
        "--accum-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )
    p.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate.")
    p.add_argument("--wd", type=float, default=5e-4, help="AdamW weight decay.")
    p.add_argument("--embed-dim", type=int, default=512, help="Embedding dimensionality.")
    p.add_argument("--arcface-m", type=float, default=0.55, help="ArcFace angular margin.")
    p.add_argument("--arcface-s", type=float, default=64.0, help="ArcFace logit scale.")
    p.add_argument("--arcface-k", type=int, default=7, help="ArcFace sub-centers per class.")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    p.add_argument("--seed", type=int, default=42, help="Global random seed.")
    p.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Override processed .npy directory.",
    )
    p.add_argument(
        "--hard-mining",
        action="store_true",
        help="Enable hard-negative mining.",
    )
    p.add_argument(
        "--hard-mining-start",
        type=int,
        default=base._HN_WARMUP_EPOCHS,
        help="Epoch after which hard-negative mining activates.",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="PATH",
        help="Resume this variant from checkpoint path.",
    )
    return p.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_processed_dir(arg_dir: str | None) -> Path:
    if arg_dir:
        return Path(arg_dir)
    if base._PROC_.exists() and any(base._PROC_.glob("*.npy")):
        return base._PROC_
    raise FileNotFoundError("No processed .npy directory found. Run extract_features.py first.")


def _load_or_create_split(
    sample_map: dict[int, dict[str, list[str]]],
    seed: int,
    log: logging.Logger,
) -> tuple[list[int], list[int], list[int], dict]:
    _CKPT_ROOT.mkdir(parents=True, exist_ok=True)

    if _MANIFEST_PATH.exists():
        with _MANIFEST_PATH.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        if base._manifest_has_target_split(manifest):
            log.info("Reusing ablation split manifest: %s", _MANIFEST_PATH)
            return (
                manifest["split"]["train"],
                manifest["split"]["val"],
                manifest["split"]["test"],
                manifest,
            )

    train_ids, val_ids, test_ids = base._make_split(sample_map, seed=seed)
    manifest = base._build_manifest(train_ids, val_ids, test_ids, sample_map)
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    manifest["ablation"] = {
        "variants": {
            k: v["channels"] for k, v in VARIANTS.items()
        }
    }
    with _MANIFEST_PATH.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    log.info("Ablation split manifest saved -> %s", _MANIFEST_PATH)
    return train_ids, val_ids, test_ids, manifest


def _train_single_variant(
    variant_name: str,
    channels: list[int],
    args: argparse.Namespace,
    sample_map: dict[int, dict[str, list[str]]],
    train_ids: list[int],
    val_ids: list[int],
    manifest: dict,
    device: torch.device,
    log: logging.Logger,
) -> float:
    _seed_everything(args.seed)

    variant_dir = _CKPT_ROOT / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = variant_dir / f"best_tavnet_{variant_name}.pt"

    train_writers = sorted(set(train_ids))
    writer_to_idx = {w: i for i, w in enumerate(train_writers)}
    n_train_writers = len(train_writers)

    max_wid = max(writer_to_idx.keys())
    wid_to_idx_lut = torch.full((max_wid + 1,), -1, dtype=torch.long, device=device)
    for wid, idx in writer_to_idx.items():
        wid_to_idx_lut[wid] = idx

    train_ds = AblationAPNBundleDataset(
        train_ids, sample_map, channel_indices=channels, augment=True
    )
    val_ds = AblationAPNBundleDataset(
        val_ids, sample_map, channel_indices=channels, augment=False
    )

    loader_kw = dict(
        collate_fn=base._collate_bundles,
        worker_init_fn=base._worker_init_fn,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        **loader_kw,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        **loader_kw,
    )

    model = AblationTAVNet(in_channels=len(channels), embed_dim=args.embed_dim).to(device)
    arcface = base.SubCenterArcFaceLoss(
        in_features=args.embed_dim,
        n_classes=n_train_writers,
        K=args.arcface_k,
        s=args.arcface_s,
        m=args.arcface_m,
    ).to(device)

    optimizer = AdamW(
        list(model.parameters()) + list(arcface.parameters()),
        lr=args.lr,
        weight_decay=args.wd,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    start_epoch = 1
    best_eer = float("inf")

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=False)
        arcface.load_state_dict(ckpt["arcface_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        except (ValueError, KeyError):
            log.warning("Scheduler state could not be restored for %s", variant_name)
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_eer = ckpt.get("best_eer", float("inf"))

    log.info(
        "\n[Variant %s] channels=%s (%s)",
        variant_name,
        channels,
        VARIANTS[variant_name]["label"],
    )
    log.info(
        "[Variant %s] train_writers=%d val_writers=%d",
        variant_name,
        len(train_ds),
        len(val_ds),
    )

    base._print_header()

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter()

        train_ds.set_epoch(epoch)
        trn_loss, _trn_ap, _trn_an = base._run_train_epoch(
            model,
            arcface,
            train_loader,
            device,
            optimizer,
            scaler,
            wid_to_idx_lut,
            current_epoch=epoch,
            accumulation_steps=args.accum_steps,
        )

        val_loss, val_eer, forg_eer, rand_eer, val_ap, val_an, tar_at_08, _writer_dists = (
            base._run_val_epoch(model, val_loader, device)
        )

        scheduler.step(val_eer)

        is_best = val_eer < best_eer
        if is_best:
            best_eer = val_eer
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "arcface_state": arcface.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "best_eer": best_eer,
                    "embed_dim": args.embed_dim,
                    "arcface_k": args.arcface_k,
                    "n_train_writers": n_train_writers,
                    "writer_to_idx": writer_to_idx,
                    "manifest": manifest,
                    "variant": variant_name,
                    "channels": channels,
                    "args": vars(args),
                },
                str(best_ckpt_path),
            )

        lr = optimizer.param_groups[0]["lr"]
        base._print_row(
            epoch,
            trn_loss,
            val_loss,
            val_eer,
            forg_eer,
            rand_eer,
            val_ap,
            val_an,
            tar_at_08,
            lr,
            is_best,
        )

        if args.hard_mining and epoch >= args.hard_mining_start:
            base._print_hint(f"[HNM:{variant_name}] Scoring {len(train_ds.writer_ids)} writers ...")
            hard_scores = compute_hard_forgery_scores_ablation(
                model,
                sample_map,
                train_ds.writer_ids,
                channel_indices=channels,
                device=device,
                batch_size=args.batch_size * base._BUNDLE_SIZE,
            )
            train_ds.update_forgery_scores(hard_scores)
            train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                drop_last=True,
                **loader_kw,
            )

        dt = time.perf_counter() - t0
        base._print_hint(f"epoch_time={dt:.1f}s")

    log.info("[Variant %s] Best val EER: %.4f (%.2f%%)", variant_name, best_eer, best_eer * 100.0)
    log.info("[Variant %s] Checkpoint: %s", variant_name, best_ckpt_path)

    return best_eer


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(message)s",
        handlers=[logging.StreamHandler()],
    )
    log = logging.getLogger("tavnet_ablation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    try:
        processed_dir = _resolve_processed_dir(args.processed_dir)
    except FileNotFoundError as exc:
        log.error(str(exc))
        sys.exit(1)

    log.info("Using processed tensors: %s", processed_dir)
    sample_map = base._scan_dir(processed_dir)
    if not sample_map:
        log.error("No .npy files found in %s", processed_dir)
        sys.exit(1)

    eligible = [
        uid
        for uid, v in sample_map.items()
        if len(v["G"]) >= base._MIN_GENUINE and len(v["F"]) >= base._MIN_FORGERY
    ]
    if len(eligible) < 6:
        log.error("Need >=6 eligible writers for 70/10/20 split. Found %d.", len(eligible))
        sys.exit(1)

    train_ids, val_ids, test_ids, manifest = _load_or_create_split(sample_map, args.seed, log)
    log.info(
        "Split writers -> train=%d val=%d test=%d",
        len(train_ids),
        len(val_ids),
        len(test_ids),
    )

    if args.variant == "all":
        run_list = ["baseline", "step1", "step2", "full"]
    else:
        run_list = [args.variant]

    summary: dict[str, float] = {}
    for vname in run_list:
        channels = VARIANTS[vname]["channels"]
        best_eer = _train_single_variant(
            variant_name=vname,
            channels=channels,
            args=args,
            sample_map=sample_map,
            train_ids=train_ids,
            val_ids=val_ids,
            manifest=manifest,
            device=device,
            log=log,
        )
        summary[vname] = best_eer

    summary_path = _CKPT_ROOT / "ablation_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "variants": {
                    k: {
                        "channels": VARIANTS[k]["channels"],
                        "label": VARIANTS[k]["label"],
                        "best_val_eer": summary.get(k),
                    }
                    for k in summary
                },
            },
            fh,
            indent=2,
        )

    print("\nAblation run complete.")
    for vname in run_list:
        print(
            f"  {vname:8s} channels={VARIANTS[vname]['channels']} "
            f"best_val_eer={summary[vname]:.4f}"
        )
    print(f"Summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
