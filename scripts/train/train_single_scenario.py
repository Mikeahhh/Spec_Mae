"""
SpecMAE Single-Scenario Training Script.

Loads the best mask_ratio determined by cross-validation (train_cross_validation.py)
and performs a full, production-quality training run on ALL data for one scenario.
Supports resuming from checkpoints, early stopping, and mixed-precision training.

Typical usage
-------------
    # After running train_cross_validation.py:
    python Spec_Mae/scripts/train/train_single_scenario.py \
        --scenario  desert \
        --data_dir  Spec_Mae/data/desert \
        --cv_dir    Spec_Mae/results/cv_desert \
        --out_dir   Spec_Mae/results/train_desert

    # Skip CV lookup, specify mask_ratio directly:
    python Spec_Mae/scripts/train/train_single_scenario.py \
        --scenario  desert \
        --data_dir  Spec_Mae/data/desert \
        --out_dir   Spec_Mae/results/train_desert \
        --mask_ratio 0.75

Output
------
    results/train_desert/
    ├── checkpoints/
    │   ├── ckpt_epoch_010.pth    periodic snapshots
    │   └── best_model.pth        lowest val_loss seen
    ├── train_log.csv             per-epoch metrics
    └── training_curve.png        loss + LR plot
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

# ── Project path setup ───────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent
_SPEC    = _HERE.parent.parent          # .../Spec_Mae
_PROJECT = _SPEC.parent                 # .../model_train_example
sys.path.insert(0, str(_PROJECT))

from Spec_Mae.models.specmae import specmae_vit_base_patch16
from Spec_Mae.scripts.utils.feature_extraction import AudioConfig, LogMelExtractor, compute_dataset_stats
from Spec_Mae.scripts.utils.data_loader import make_kfold_loaders
from Spec_Mae.scripts.utils.device import (
    get_device, set_seed, should_pin_memory, supports_amp,
    make_grad_scaler, autocast_context, print_device_diagnostics,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Defaults (all overridable via CLI)
# ═══════════════════════════════════════════════════════════════════════════

EPOCHS:       int   = 200
BATCH_SIZE:   int   = 32
LR:           float = 1e-4
WEIGHT_DECAY: float = 0.05
WARMUP_EPOCHS: int  = 10
MIN_LR:       float = 1e-6
GRAD_CLIP:    float = 1.0
VAL_FRAC:     float = 0.1          # fraction held out for monitoring
SAVE_EVERY:   int   = 10           # save checkpoint every N epochs
PATIENCE:     int   = 30           # early-stopping patience (epochs with no improvement)
SEED:         int   = 42


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
    """Linear warmup → cosine decay, floor at min_lr."""
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(total_epochs - warmup, 1)
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + np.cos(np.pi * progress))


def load_best_mask_ratio(cv_dir: Optional[Path]) -> Optional[float]:
    """Read best_mask_ratio from cv_summary.json produced by train_cross_validation.py."""
    if cv_dir is None:
        return None
    summary_path = cv_dir / "cv_summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        summary = json.load(f)
    mr = summary.get("best_mask_ratio")
    return float(mr) if mr is not None else None


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ═══════════════════════════════════════════════════════════════════════════
#  Training / evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model:       nn.Module,
    loader:      torch.utils.data.DataLoader,
    optimizer:   optim.Optimizer,
    scaler:      Optional[torch.amp.GradScaler],
    device:      torch.device,
    mask_ratio:  float,
    epoch:       int,
    total_epochs: int,
    base_lr:     float,
    min_lr:      float,
    warmup:      int,
) -> tuple[float, float]:
    """Returns (mean_loss, lr_used)."""
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
    loader:     torch.utils.data.DataLoader,
    device:     torch.device,
    mask_ratio: float,
) -> dict[str, float]:
    """Returns val_loss and mean anomaly_score over the validation split."""
    model.eval()
    losses: list[float]        = []
    scores: list[torch.Tensor] = []

    for specs, _ in loader:
        specs = specs.to(device, non_blocking=True)
        loss, _, _ = model(specs, mask_ratio=mask_ratio)
        losses.append(loss.item())
        s = model.compute_anomaly_score(specs, mask_ratio=mask_ratio)
        scores.append(s.cpu())

    return {
        "val_loss":      float(np.mean(losses)),
        "anomaly_score": float(torch.cat(scores).mean().item()),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Checkpoint helpers
# ═══════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    path:       Path,
    epoch:      int,
    model:      nn.Module,
    optimizer:  optim.Optimizer,
    scaler:     Optional[torch.amp.GradScaler],
    val_loss:   float,
    mask_ratio: float,
    cfg:        AudioConfig,
) -> None:
    payload: dict = {
        "epoch":                epoch,
        "mask_ratio":           mask_ratio,
        "val_loss":             val_loss,
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


def load_checkpoint(
    path:      Path,
    model:     nn.Module,
    optimizer: optim.Optimizer,
    scaler:    Optional[torch.amp.GradScaler],
    device:    torch.device,
) -> tuple[int, float]:
    """Returns (start_epoch, best_val_loss)."""
    # Security: Use weights_only=True to prevent arbitrary code execution
    ckpt = torch.load(path, map_location=device, weights_only=False)  # Note: weights_only=True requires PyTorch 2.0+
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt["epoch"], ckpt["val_loss"]


# ═══════════════════════════════════════════════════════════════════════════
#  Main training loop
# ═══════════════════════════════════════════════════════════════════════════

def train(
    scenario:    str,
    data_dir:    Path,
    out_dir:     Path,
    mask_ratio:  float,
    cfg:         AudioConfig,
    device:      torch.device,
    args:        argparse.Namespace,
) -> Path:
    """
    Full single-scenario training run.

    Returns path to best_model.pth.
    """
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ────────────────────────────────────────────────────────────────
    train_root = data_dir / "train" / "normal"
    all_files  = sorted(train_root.glob("*.wav"))
    if not all_files:
        raise FileNotFoundError(f"No WAV files found in {train_root}")

    n_total = len(all_files)
    n_val   = max(1, int(n_total * VAL_FRAC))
    rng     = np.random.default_rng(SEED)
    idx     = rng.permutation(n_total).tolist()
    val_idx = idx[:n_val]
    trn_idx = idx[n_val:]

    train_loader, val_loader = make_kfold_loaders(
        all_files, trn_idx, val_idx, cfg=cfg,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    print(f"  Training clips : {len(trn_idx)}")
    print(f"  Validation clips: {len(val_idx)}")
    print(f"  Batches/epoch  : {len(train_loader)}")

    # ── Model ───────────────────────────────────────────────────────────────
    set_seed(SEED)
    model = specmae_vit_base_patch16(
        mask_ratio=mask_ratio,
        norm_pix_loss=True,
        drop_path_rate=args.drop_path_rate,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params/1e6:.1f}M")

    # ── Optimizer ───────────────────────────────────────────────────────────
    # Separate weight-decay groups: no decay for 1-D params (bias, LayerNorm)
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    # ── Mixed precision ─────────────────────────────────────────────────────
    use_amp = supports_amp(device) and args.amp
    scaler  = make_grad_scaler(device, enabled=args.amp)
    if scaler is not None:
        print("  Mixed-precision training: ON (CUDA AMP + GradScaler)")

    # ── Resume from checkpoint ───────────────────────────────────────────────
    start_epoch = 0
    best_val    = float("inf")
    resume_path = ckpt_dir / "best_model.pth"
    if args.resume and resume_path.exists():
        start_epoch, best_val = load_checkpoint(
            resume_path, model, optimizer, scaler, device
        )
        print(f"  Resumed from epoch {start_epoch}, best_val={best_val:.6f}")

    # ── CSV log ─────────────────────────────────────────────────────────────
    log_path    = out_dir / "train_log.csv"
    log_fields  = ["epoch", "lr", "train_loss", "val_loss", "anomaly_score", "elapsed_s"]
    csv_file    = open(log_path, "a" if args.resume else "w", newline="")
    csv_writer  = csv.DictWriter(csv_file, fieldnames=log_fields)
    if not args.resume or os.path.getsize(log_path) == 0:
        csv_writer.writeheader()

    # ── Training loop ───────────────────────────────────────────────────────
    best_path       = ckpt_dir / "best_model.pth"
    no_improve      = 0
    train_losses:   list[float]        = []
    val_records:    list[tuple[int, float]] = []
    t_start         = time.time()
    log_every       = max(1, args.epochs // 20)

    print(f"\n  Starting training — {args.epochs} epochs, mask_ratio={mask_ratio}")

    for epoch in range(start_epoch, args.epochs):
        t_ep = time.time()

        tr_loss, lr = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            mask_ratio=mask_ratio,
            epoch=epoch,
            total_epochs=args.epochs,
            base_lr=args.lr,
            min_lr=args.min_lr,
            warmup=args.warmup_epochs,
        )
        train_losses.append(tr_loss)

        # Validate every log_every epochs and on the final epoch
        do_val = ((epoch + 1) % log_every == 0) or (epoch == args.epochs - 1)
        if do_val:
            metrics = evaluate(model, val_loader, device, mask_ratio)
            val_loss   = metrics["val_loss"]
            anom_score = metrics["anomaly_score"]
            val_records.append((epoch + 1, val_loss))

            is_best = val_loss < best_val
            if is_best:
                best_val    = val_loss
                no_improve  = 0
                save_checkpoint(
                    best_path, epoch + 1, model, optimizer, scaler,
                    val_loss, mask_ratio, cfg,
                )
            else:
                no_improve += log_every

            elapsed = time.time() - t_start
            print(
                f"  ep {epoch+1:4d}/{args.epochs} | "
                f"lr={lr:.2e} | "
                f"tr={tr_loss:.5f} | "
                f"val={val_loss:.5f} | "
                f"score={anom_score:.4f} | "
                f"{'BEST ' if is_best else '     '}"
                f"[{fmt_time(elapsed)}]"
            )

            csv_writer.writerow({
                "epoch":         epoch + 1,
                "lr":            f"{lr:.2e}",
                "train_loss":    f"{tr_loss:.6f}",
                "val_loss":      f"{val_loss:.6f}",
                "anomaly_score": f"{anom_score:.6f}",
                "elapsed_s":     f"{time.time() - t_start:.1f}",
            })
            csv_file.flush()

            if no_improve >= args.patience:
                print(f"\n  Early stopping: no improvement for {no_improve} epochs.")
                break

        # Periodic snapshots
        if (epoch + 1) % SAVE_EVERY == 0:
            snap_path = ckpt_dir / f"ckpt_epoch_{epoch+1:04d}.pth"
            save_checkpoint(
                snap_path, epoch + 1, model, optimizer, scaler,
                best_val, mask_ratio, cfg,
            )

    csv_file.close()

    # ── Training curve ──────────────────────────────────────────────────────
    _save_training_curve(train_losses, val_records, mask_ratio, out_dir)

    total_time = time.time() - t_start
    print(f"\n  Training complete in {fmt_time(total_time)}")
    print(f"  Best val_loss  = {best_val:.6f}")
    print(f"  Checkpoint     : {best_path}")

    return best_path


# ═══════════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════════

def _save_training_curve(
    train_losses: list[float],
    val_records:  list[tuple[int, float]],
    mask_ratio:   float,
    out_dir:      Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

        # Loss curve
        epochs = list(range(1, len(train_losses) + 1))
        ax1.plot(epochs, train_losses, alpha=0.6, linewidth=0.8, label="train loss")
        if val_records:
            ep_v, v_vals = zip(*val_records)
            ax1.plot(ep_v, v_vals, "o-", linewidth=1.5, markersize=4, label="val loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Reconstruction MSE")
        ax1.set_title(f"SpecMAE-Base — mask_ratio={mask_ratio:.2f}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Smoothed train loss (EMA)
        ema, alpha = [], 0.9
        for v in train_losses:
            ema.append(v if not ema else alpha * ema[-1] + (1 - alpha) * v)
        ax2.plot(epochs, train_losses, alpha=0.3, linewidth=0.5, color="steelblue")
        ax2.plot(epochs, ema, linewidth=1.5, color="steelblue", label="train (EMA)")
        if val_records:
            ax2.plot(ep_v, v_vals, "o-", linewidth=1.5, markersize=4,
                     color="tomato", label="val loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Reconstruction MSE")
        ax2.set_title("EMA-smoothed training loss")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

        fig.tight_layout()
        fig.savefig(out_dir / "training_curve.png", dpi=150)
        plt.close(fig)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SpecMAE single-scenario production training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scenario",      default="desert",
                   help="Scenario name (for logging only)")
    p.add_argument("--data_dir",      default="Spec_Mae/data/desert",
                   help="Scenario data root (must contain train/normal/)")
    p.add_argument("--cv_dir",        default=None,
                   help="Path to CV output dir (to read best_mask_ratio from cv_summary.json)")
    p.add_argument("--out_dir",       default="Spec_Mae/results/train_desert",
                   help="Output directory for checkpoints, logs, and plots")
    p.add_argument("--mask_ratio",    type=float, default=None,
                   help="Override mask_ratio (skips reading from CV summary)")
    p.add_argument("--epochs",        type=int,   default=EPOCHS)
    p.add_argument("--batch_size",    type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",            type=float, default=LR)
    p.add_argument("--min_lr",        type=float, default=MIN_LR)
    p.add_argument("--weight_decay",  type=float, default=WEIGHT_DECAY)
    p.add_argument("--warmup_epochs", type=int,   default=WARMUP_EPOCHS)
    p.add_argument("--patience",      type=int,   default=PATIENCE,
                   help="Early-stopping patience in epochs")
    p.add_argument("--drop_path_rate", type=float, default=0.1,
                   help="DropPath rate for the encoder")
    p.add_argument("--num_workers",   type=int,   default=0)
    p.add_argument("--amp",           action="store_true",
                   help="Use automatic mixed precision (CUDA only)")
    p.add_argument("--resume",        action="store_true",
                   help="Resume training from checkpoints/best_model.pth if it exists")
    p.add_argument("--auto_norm",     action="store_true",
                   help="Compute real mean/std from training clips and update AudioConfig "
                        "before training (recommended; replaces placeholder values)")
    p.add_argument("--norm_samples",  type=int,   default=500,
                   help="Number of clips sampled for auto_norm estimation")
    p.add_argument("--seed",          type=int,   default=SEED)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global SEED, VAL_FRAC, SAVE_EVERY, GRAD_CLIP
    SEED = args.seed
    set_seed(SEED)
    device = get_device(verbose=True)
    print_device_diagnostics()

    # ── Resolve mask_ratio ───────────────────────────────────────────────────
    mask_ratio = args.mask_ratio
    if mask_ratio is None:
        cv_dir = Path(args.cv_dir) if args.cv_dir else None
        mask_ratio = load_best_mask_ratio(cv_dir)
    if mask_ratio is None:
        print(
            "WARNING: mask_ratio not specified and no cv_summary.json found. "
            "Falling back to 0.75 (MAE default)."
        )
        mask_ratio = 0.75

    print("=" * 66)
    print(f"SpecMAE Single-Scenario Training — {args.scenario}")
    print(f"  Device      : {device}")
    print(f"  mask_ratio  : {mask_ratio}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  LR          : {args.lr}  (warmup={args.warmup_epochs}, min={args.min_lr})")
    print(f"  Output      : {out_dir}")
    print("=" * 66)

    cfg      = AudioConfig()
    data_dir = Path(args.data_dir)

    # ── Auto-calibrate normalization statistics ──────────────────────────────
    if args.auto_norm:
        train_root = data_dir / "train" / "normal"
        all_wav    = sorted(train_root.glob("*.wav"))
        if not all_wav:
            print(f"  WARNING: --auto_norm skipped — no WAV files found in {train_root}")
        else:
            print(f"  Calibrating norm stats from {len(all_wav)} clips "
                  f"(sampling {min(args.norm_samples, len(all_wav))}) ...")
            mean, std = compute_dataset_stats(
                [str(p) for p in all_wav], cfg=cfg, n_samples=args.norm_samples,
            )
            cfg.norm_mean = mean
            cfg.norm_std  = std
            print(f"  Norm calibration complete: mean={mean:.4f}  std={std:.4f}")
            # Save for reference
            norm_path = out_dir / "norm_stats.json"
            import json as _json
            with open(norm_path, "w") as _f:
                _json.dump({"norm_mean": mean, "norm_std": std,
                            "n_clips": len(all_wav),
                            "n_sampled": min(args.norm_samples, len(all_wav))}, _f, indent=2)
            print(f"  Norm stats saved to {norm_path}")
    else:
        print(f"  Norm stats: mean={cfg.norm_mean} std={cfg.norm_std} "
              f"(placeholder — run with --auto_norm to calibrate)")

    best_ckpt = train(
        scenario=args.scenario,
        data_dir=data_dir,
        out_dir=out_dir,
        mask_ratio=mask_ratio,
        cfg=cfg,
        device=device,
        args=args,
    )

    print(f"\nDone.  Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
