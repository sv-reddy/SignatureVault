"""
train_tavnet_arch_ablation.py
=============================

Architectural ablation training for SignatureVault using full 4-channel tensors.

Variants:
  - resnet              : ResNet-50 backbone + GeM + projection head
  - resnet_cbam         : ResNet-50 + CBAM + GeM + projection head
  - resnet_transformer  : ResNet-50 + Transformer + GeM + projection head

Outputs:
  backend/checkpoints/arch_ablation/<variant>/best_tavnet_<variant>.pt
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
import torchvision.models as tvm
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights

_SCRIPT_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _SCRIPT_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import train_tavnet as base

_CKPT_ROOT = _BACKEND_DIR / "checkpoints" / "arch_ablation"
_MANIFEST_PATH = _CKPT_ROOT / "manifest.json"


ARCH_VARIANTS = {
    "resnet": {
        "use_cbam": False,
        "use_transformer": False,
        "label": "ResNet-50",
    },
    "resnet_cbam": {
        "use_cbam": True,
        "use_transformer": False,
        "label": "ResNet-50 + CBAM",
    },
    "resnet_transformer": {
        "use_cbam": False,
        "use_transformer": True,
        "label": "ResNet-50 + Transformer",
    },
}

# Backward-compat alias for scripts that import VARIANTS.
VARIANTS = ARCH_VARIANTS


class ArchitectureAblationTAVNet(nn.Module):
    _SEQ_LEN = 12 * 12
    _N_TOKENS = _SEQ_LEN

    def __init__(
        self,
        embed_dim: int = 512,
        use_cbam: bool = False,
        use_transformer: bool = False,
    ) -> None:
        super().__init__()
        self.use_cbam = use_cbam
        self.use_transformer = use_transformer

        base_model = tvm.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        old_w = base_model.conv1.weight.data
        new_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
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

        if self.use_cbam:
            self.cbam = base.CBAM(channels=2048, reduction=16, spatial_k=7)
        else:
            self.cbam = nn.Identity()

        if self.use_transformer:
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
                enc_layer,
                num_layers=1,
                enable_nested_tensor=False,
            )
        else:
            self.register_parameter("pos_embed", None)
            self.transformer = None

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

        if self.use_transformer:
            bsz = x.shape[0]
            x = x.flatten(2)
            x = x.permute(2, 0, 1)
            x = x + self.pos_embed
            x = self.transformer(x)
            x = x.permute(1, 2, 0).contiguous().view(bsz, 2048, 12, 12)

        x = self.gem(x).flatten(1)
        x = self.head(x)
        return F.normalize(x, p=2, dim=1)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train architectural ablation variants of TAV-Net.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=["resnet", "resnet_cbam", "resnet_transformer", "all"],
        help="Which architectural variant to train.",
    )
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--embed-dim", type=int, default=512)
    p.add_argument("--arcface-m", type=float, default=0.55)
    p.add_argument("--arcface-s", type=float, default=64.0)
    p.add_argument("--arcface-k", type=int, default=7)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--processed-dir", type=str, default=None, metavar="DIR")
    p.add_argument("--hard-mining", action="store_true")
    p.add_argument("--hard-mining-start", type=int, default=base._HN_WARMUP_EPOCHS)
    p.add_argument("--resume", type=str, default=None, metavar="PATH")
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
            log.info("Reusing architectural-ablation split manifest: %s", _MANIFEST_PATH)
            return (
                manifest["split"]["train"],
                manifest["split"]["val"],
                manifest["split"]["test"],
                manifest,
            )

    train_ids, val_ids, test_ids = base._make_split(sample_map, seed=seed)
    manifest = base._build_manifest(train_ids, val_ids, test_ids, sample_map)
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    manifest["arch_ablation"] = {
        "variants": {
            k: {
                "use_cbam": v["use_cbam"],
                "use_transformer": v["use_transformer"],
                "label": v["label"],
            }
            for k, v in ARCH_VARIANTS.items()
        }
    }
    with _MANIFEST_PATH.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    log.info("Architectural-ablation split manifest saved -> %s", _MANIFEST_PATH)
    return train_ids, val_ids, test_ids, manifest


def _train_single_variant(
    variant_name: str,
    variant_cfg: dict,
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

    train_ds = base.APNBundleDataset(train_ids, sample_map, augment=True)
    val_ds = base.APNBundleDataset(val_ids, sample_map, augment=False)

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

    model = ArchitectureAblationTAVNet(
        embed_dim=args.embed_dim,
        use_cbam=bool(variant_cfg["use_cbam"]),
        use_transformer=bool(variant_cfg["use_transformer"]),
    ).to(device)
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
        "\n[Variant %s] %s | cbam=%s transformer=%s",
        variant_name,
        variant_cfg["label"],
        variant_cfg["use_cbam"],
        variant_cfg["use_transformer"],
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
                    "variant_cfg": variant_cfg,
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
            hard_scores = base.compute_hard_forgery_scores(
                model,
                sample_map,
                train_ds.writer_ids,
                device,
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
    log = logging.getLogger("tavnet_arch_ablation")

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
        run_list = ["resnet", "resnet_cbam", "resnet_transformer"]
    else:
        run_list = [args.variant]

    summary: dict[str, float] = {}
    for vname in run_list:
        best_eer = _train_single_variant(
            variant_name=vname,
            variant_cfg=ARCH_VARIANTS[vname],
            args=args,
            sample_map=sample_map,
            train_ids=train_ids,
            val_ids=val_ids,
            manifest=manifest,
            device=device,
            log=log,
        )
        summary[vname] = best_eer

    summary_path = _CKPT_ROOT / "arch_ablation_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "variants": {
                    k: {
                        "label": ARCH_VARIANTS[k]["label"],
                        "use_cbam": ARCH_VARIANTS[k]["use_cbam"],
                        "use_transformer": ARCH_VARIANTS[k]["use_transformer"],
                        "best_val_eer": summary.get(k),
                    }
                    for k in summary
                },
            },
            fh,
            indent=2,
        )

    print("\nArchitectural ablation run complete.")
    for vname in run_list:
        print(
            f"  {vname:18s} best_val_eer={summary[vname]:.4f} "
            f"label={ARCH_VARIANTS[vname]['label']}"
        )
    print(f"Summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
