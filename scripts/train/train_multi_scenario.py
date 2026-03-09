"""
SpecMAE Multi-Scenario Joint Training.

Trains a single SpecMAE model jointly on multiple acoustic scenarios
(desert, forest, ocean) using a round-robin sampling strategy.

Strategy
--------
Each batch is sampled uniformly from the combined training pool.
The model learns a general acoustic representation that transfers
across environments, which improves anomaly detection in unseen scenarios
and provides the starting point for per-scenario fine-tuning.

Mask-ratio
----------
Pass --mask_ratio directly or point to a CV summary JSON from
train_cross_validation.py (the best mask_ratio is read automatically).
If multiple scenarios have CV results, the median best mask_ratio is used.

Usage
-----
    cd E:/model_train_example
    python Spec_Mae/scripts/train/train_multi_scenario.py \
        --scenarios desert forest ocean \
        --data_root Spec_Mae/data \
        --out_dir   Spec_Mae/results/multi_scenario \
        --mask_ratio 0.75

Output
------
    results/multi_scenario/
    ├── checkpoints/
    │   ├── best_model.pth
    │   └── ckpt_epoch_XXXX.pth
    ├── train_log.csv
    └── training_curve.png
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader

# ── Project path ─────────────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent
_SPEC    = _HERE.parent.parent
_PROJECT = _SPEC.parent
sys.path.insert(0, str(_PROJECT))

from Spec_Mae.models.specmae import specmae_vit_base_patch16
from Spec_Mae.scripts.utils.feature_extraction import AudioConfig, LogMelExtractor
from Spec_Mae.scripts.utils.data_loader import AudioDataset
from Spec_Mae.scripts.utils.device import (
    get_device, set_seed, should_pin_memory, supports_amp,
    make_grad_scaler, autocast_context, print_device_diagnostics,
    empty_device_cache,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Defaults
# ═══════════════════════════════════════════════════════════════════════════

EPOCHS:        int   = 200
BATCH_SIZE:    int   = 32
LR:            float = 1e-4
WEIGHT_DECAY:  float = 0.05
WARMUP_EPOCHS: int   = 10
MIN_LR:        float = 1e-6
GRAD_CLIP:     float = 1.0
VAL_FRAC:      float = 0.1
SAVE_EVERY:    int   = 10
PATIENCE:      int   = 30
SEED:          int   = 42


# ═══════════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════════

def cosine_lr(
    epoch:        int,
    total_epochs: int,
    base_lr:      float,
    min_lr:       float,
    warmup:       int,
) -> float:
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(total_epochs - warmup, 1)
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + np.cos(np.pi * progress))


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def load_best_mask_ratio_from_cv(cv_dirs: list[Path]) -> Optional[float]:
    """Read best mask_ratios from multiple CV summaries and return the median."""
    mrs: list[float] = []
    for cv_dir in cv_dirs:
        p = cv_dir / "cv_summary.json"
        if p.exists():
            with open(p) as f:
                s = json.load(f)
            mr = s.get("best_mask_ratio")
            if mr is not None:
                mrs.append(float(mr))
    return float(np.median(mrs)) if mrs else None


# ═══════════════════════════════════════════════════════════════════════════
#  Data
# ═══════════════════════════════════════════════════════════════════════════

def build_combined_loaders(
    scenarios:   list[str],
    data_root:   Path,
    cfg:         AudioConfig,
    batch_size:  int,
    num_workers: int,
    val_frac:    float,
    rng:         np.random.Generator,
    pin_memory:  bool = False,
) -> tuple[DataLoader, DataLoader, int]:
    """
    Build combined train/val DataLoaders from all scenarios.

    Returns (train_loader, val_loader, total_clips).
    """
    extractor = LogMelExtractor(cfg=cfg)

    train_datasets: list[AudioDataset] = []
    val_datasets:   list[AudioDataset] = []

    for scenario in scenarios:
        train_dir = data_root / scenario / "train" / "normal"
        if not train_dir.exists():
            print(f"  WARNING: {train_dir} not found — skipping {scenario}")
            continue

        all_files = sorted(train_dir.glob("*.wav"))
        if not all_files:
            print(f"  WARNING: no WAV files in {train_dir} — skipping {scenario}")
            continue

        n_val  = max(1, int(len(all_files) * val_frac))
        idx    = rng.permutation(len(all_files)).tolist()
        val_fs = [all_files[i] for i in idx[:n_val]]
        trn_fs = [all_files[i] for i in idx[n_val:]]

        # Wrap as simple datasets using a file-list approach
        class _FileDS(torch.utils.data.Dataset):
            def __init__(self, files: list[Path], ext: LogMelExtractor) -> None:
                self.files = files
                self.ext   = ext
            def __len__(self) -> int:
                return len(self.files)
            def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
                return self.ext(str(self.files[i])), 0

        train_datasets.append(_FileDS(trn_fs, extractor))
        val_datasets.append(_FileDS(val_fs, extractor))

        print(f"    {scenario:12s}: {len(trn_fs)} train, {len(val_fs)} val")

    if not train_datasets:
        raise FileNotFoundError("No training data found for any scenario.")

    combined_train = ConcatDataset(train_datasets)
    combined_val   = ConcatDataset(val_datasets)

    train_loader = DataLoader(
        combined_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        combined_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, len(combined_train)


# ═══════════════════════════════════════════════════════════════════════════
#  Training / eval
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model:        nn.Module,
    loader:       DataLoader,
    optimizer:    optim.Optimizer,
    scaler:       Optional[torch.amp.GradScaler],
    device:       torch.device,
    mask_ratio:   float,
    epoch:        int,
    total_epochs: int,
    base_lr:      float,
    min_lr:       float,
    warmup:       int,
) -> tuple[float, float]:
    model.train()
    lr = cosine_lr(epoch, total_epochs, base_lr, min_lr, warmup)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    total_loss = 0.0
    n_batches  = 0

    for specs, _ in loader:
        specs = specs.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:                      # AMP path (CUDA only)
            with torch.amp.autocast(device_type="cuda"):
                loss, _, _ = model(specs, mask_ratio=mask_ratio)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:                                        # Full-precision (MPS / CPU)
            loss, _, _ = model(specs, mask_ratio=mask_ratio)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1), lr


@torch.no_grad()
def evaluate(
    model:      nn.Module,
    loader:     DataLoader,
    device:     torch.device,
    mask_ratio: float,
) -> float:
    model.eval()
    losses: list[float] = []
    for specs, _ in loader:
        specs = specs.to(device, non_blocking=True)
        loss, _, _ = model(specs, mask_ratio=mask_ratio)
        losses.append(loss.item())
    return float(np.mean(losses))


def save_checkpoint(
    path:       Path,
    epoch:      int,
    model:      nn.Module,
    optimizer:  optim.Optimizer,
    scaler:     Optional[torch.amp.GradScaler],
    val_loss:   float,
    mask_ratio: float,
    cfg:        AudioConfig,
    scenarios:  list[str],
) -> None:
    payload: dict = {
        "epoch":                epoch,
        "mask_ratio":           mask_ratio,
        "val_loss":             val_loss,
        "scenarios":            scenarios,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "audio_cfg": {
            "sample_rate": cfg.sample_rate,
            "n_mels":      cfg.n_mels,
            "n_fft":       cfg.n_fft,
            "hop_length":  cfg.hop_length,
            "norm_mean":   cfg.norm_mean,
            "norm_std":    cfg.norm_std,
        },
    }
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    torch.save(payload, path)


def _save_curve(
    train_losses: list[float],
    val_records:  list[tuple[int, float]],
    mask_ratio:   float,
    scenarios:    list[str],
    out_dir:      Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(range(1, len(train_losses) + 1), train_losses,
                alpha=0.6, linewidth=0.8, label="train loss")
        if val_records:
            ep_v, v_vals = zip(*val_records)
            ax.plot(ep_v, v_vals, "o-", linewidth=1.5, markersize=4, label="val loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Reconstruction MSE")
        ax.set_title(
            f"SpecMAE-Base Multi-Scenario  mask={mask_ratio:.2f}  "
            f"[{', '.join(scenarios)}]"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "training_curve.png", dpi=150)
        plt.close(fig)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="SpecMAE multi-scenario joint training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scenarios",    nargs="+", default=["desert", "forest", "ocean"])
    p.add_argument("--data_root",    default="Spec_Mae/data",
                   help="Root directory; each scenario is a sub-folder")
    p.add_argument("--cv_root",      default=None,
                   help="Root dir for CV results (sub-folders: cv_<scenario>)")
    p.add_argument("--out_dir",      default="Spec_Mae/results/multi_scenario")
    p.add_argument("--mask_ratio",   type=float, default=None)
    p.add_argument("--epochs",       type=int,   default=EPOCHS)
    p.add_argument("--batch_size",   type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",           type=float, default=LR)
    p.add_argument("--min_lr",       type=float, default=MIN_LR)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--warmup_epochs",type=int,   default=WARMUP_EPOCHS)
    p.add_argument("--patience",     type=int,   default=PATIENCE)
    p.add_argument("--drop_path_rate",type=float,default=0.1)
    p.add_argument("--num_workers",  type=int,   default=0)
    p.add_argument("--amp",          action="store_true")
    p.add_argument("--seed",         type=int,   default=SEED)
    args = parse_args = p.parse_args()

    out_dir  = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    rng    = np.random.default_rng(args.seed)
    device = get_device(verbose=True)
    print_device_diagnostics()

    # ── Resolve mask_ratio ───────────────────────────────────────────────
    mask_ratio = args.mask_ratio
    if mask_ratio is None and args.cv_root is not None:
        cv_root = Path(args.cv_root)
        cv_dirs = [cv_root / f"cv_{sc}" for sc in args.scenarios]
        mask_ratio = load_best_mask_ratio_from_cv(cv_dirs)
    if mask_ratio is None:
        mask_ratio = 0.75
        print(f"  mask_ratio not specified; using default {mask_ratio}")

    print("=" * 66)
    print(f"SpecMAE Multi-Scenario Training")
    print(f"  Scenarios   : {args.scenarios}")
    print(f"  Device      : {device}")
    print(f"  mask_ratio  : {mask_ratio}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Output      : {out_dir}")
    print("=" * 66)

    cfg      = AudioConfig()
    data_root = Path(args.data_root)

    # ── Data ─────────────────────────────────────────────────────────────
    print("\n  Building datasets:")
    train_loader, val_loader, n_total = build_combined_loaders(
        scenarios=args.scenarios,
        data_root=data_root,
        cfg=cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_frac=VAL_FRAC,
        rng=rng,
        pin_memory=should_pin_memory(device),
    )
    print(f"  Combined training clips: {n_total}")
    print(f"  Batches/epoch: {len(train_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────
    set_seed(args.seed)
    model = specmae_vit_base_patch16(
        mask_ratio=mask_ratio,
        norm_pix_loss=True,
        drop_path_rate=args.drop_path_rate,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params/1e6:.1f}M")

    # ── Optimizer (no weight decay on 1-D params) ─────────────────────────
    decay_p, no_decay_p = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay_p.append(param)
        else:
            decay_p.append(param)

    optimizer = optim.AdamW(
        [
            {"params": decay_p,    "weight_decay": args.weight_decay},
            {"params": no_decay_p, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    use_amp = supports_amp(device) and args.amp
    scaler  = make_grad_scaler(device, enabled=args.amp)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val  = float("inf")
    no_improve = 0
    log_every  = max(1, args.epochs // 20)
    best_path  = ckpt_dir / "best_model.pth"
    train_losses: list[float]            = []
    val_records:  list[tuple[int, float]] = []
    t_start       = time.time()

    log_path = out_dir / "train_log.csv"
    log_fields = ["epoch", "lr", "train_loss", "val_loss", "elapsed_s"]
    csv_file = open(log_path, "w", newline="")
    csv_wr   = csv.DictWriter(csv_file, fieldnames=log_fields)
    csv_wr.writeheader()

    print(f"\n  Starting training — {args.epochs} epochs")

    for epoch in range(args.epochs):
        tr_loss, lr = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            mask_ratio, epoch, args.epochs, args.lr, args.min_lr, args.warmup_epochs,
        )
        train_losses.append(tr_loss)

        do_val = ((epoch + 1) % log_every == 0) or (epoch == args.epochs - 1)
        if do_val:
            val_loss = evaluate(model, val_loader, device, mask_ratio)
            val_records.append((epoch + 1, val_loss))
            is_best = val_loss < best_val

            if is_best:
                best_val   = val_loss
                no_improve = 0
                save_checkpoint(
                    best_path, epoch + 1, model, optimizer, scaler,
                    val_loss, mask_ratio, cfg, args.scenarios,
                )
            else:
                no_improve += log_every

            elapsed = time.time() - t_start
            print(
                f"  ep {epoch+1:4d}/{args.epochs} | "
                f"lr={lr:.2e} | tr={tr_loss:.5f} | val={val_loss:.5f} | "
                f"{'BEST ' if is_best else '     '}"
                f"[{fmt_time(elapsed)}]"
            )

            csv_wr.writerow({
                "epoch":      epoch + 1,
                "lr":         f"{lr:.2e}",
                "train_loss": f"{tr_loss:.6f}",
                "val_loss":   f"{val_loss:.6f}",
                "elapsed_s":  f"{elapsed:.1f}",
            })
            csv_file.flush()

            if no_improve >= args.patience:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % SAVE_EVERY == 0:
            snap = ckpt_dir / f"ckpt_epoch_{epoch+1:04d}.pth"
            save_checkpoint(
                snap, epoch + 1, model, optimizer, scaler,
                best_val, mask_ratio, cfg, args.scenarios,
            )

    csv_file.close()
    _save_curve(train_losses, val_records, mask_ratio, args.scenarios, out_dir)

    print(f"\n  Training complete in {fmt_time(time.time() - t_start)}")
    print(f"  Best val_loss = {best_val:.6f}")
    print(f"  Checkpoint    : {best_path}")
    print("Done.")


if __name__ == "__main__":
    main()
