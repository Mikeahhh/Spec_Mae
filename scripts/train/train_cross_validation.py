"""
5-Fold Cross-Validation for SpecMAE Hyperparameter Search.

Searched hyperparameter
-----------------------
    mask_ratio ∈ {0.50, 0.60, 0.70, 0.75, 0.80, 0.90}

Protocol (following meeting notes, 2026-03-09)
----------------------------------------------
1. Split the training-set normal clips into 5 folds.
2. For each (mask_ratio, fold) pair:
   a. Train on 4 folds (normal only) for CV_EPOCHS epochs.
   b. Evaluate on the held-out fold: compute mean reconstruction loss
      and mean anomaly score (both should be low for normal samples).
3. Average results across 5 folds → one score per mask_ratio.
4. Select the mask_ratio with the lowest mean validation loss.
5. Re-train a fresh model on ALL training data with the best mask_ratio
   for FINAL_EPOCHS and save to checkpoints/.

Usage
-----
    cd E:/model_train_example
    python Spec_Mae/scripts/train/train_cross_validation.py \
        --scenario desert \
        --data_dir  Spec_Mae/data/desert \
        --out_dir   Spec_Mae/results/cv_desert

Output
------
    results/cv_desert/
    ├── cv_results.csv          per-fold results for every mask_ratio
    ├── cv_summary.json         mean +/- std per mask_ratio + best choice
    ├── best_model.pth          final model trained on all data
    └── training_curve.png      loss curves for the best mask_ratio
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

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── Project path setup ──────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent
_SPEC    = _HERE.parent.parent          # .../Spec_Mae
_PROJECT = _SPEC.parent                 # .../model_train_example
sys.path.insert(0, str(_PROJECT))

from Spec_Mae.models.specmae import specmae_vit_base_patch16
from Spec_Mae.scripts.utils.feature_extraction import AudioConfig, LogMelExtractor, compute_dataset_stats
from Spec_Mae.scripts.utils.data_loader import make_kfold_loaders
from Spec_Mae.scripts.utils.device import (
    get_device, set_seed, empty_device_cache, print_device_diagnostics,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Global defaults (overridable via CLI)
# ═══════════════════════════════════════════════════════════════════════════

MASK_RATIO_GRID: list[float] = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90]
N_FOLDS:      int   = 5
CV_EPOCHS:    int   = 30
FINAL_EPOCHS: int   = 100
BATCH_SIZE:   int   = 32
LR:           float = 1e-4
WEIGHT_DECAY: float = 0.05
WARMUP_EPOCHS: int  = 5
SEED:         int   = 42


# ═══════════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════════

def get_lr(epoch: int, n_epochs: int) -> float:
    """Linear warmup then cosine decay."""
    if epoch < WARMUP_EPOCHS:
        return LR * (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(n_epochs - WARMUP_EPOCHS, 1)
    return LR * 0.5 * (1.0 + np.cos(np.pi * progress))


# ═══════════════════════════════════════════════════════════════════════════
#  One epoch helpers
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model:      nn.Module,
    loader:     torch.utils.data.DataLoader,
    optimizer:  optim.Optimizer,
    device:     torch.device,
    mask_ratio: float,
    epoch:      int,
    n_epochs:   int,
) -> float:
    model.train()
    # Apply learning-rate schedule
    lr = get_lr(epoch, n_epochs)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    total_loss = 0.0
    for specs, _ in loader:
        specs = specs.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        loss, _, _ = model(specs, mask_ratio=mask_ratio)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model:      nn.Module,
    loader:     torch.utils.data.DataLoader,
    device:     torch.device,
    mask_ratio: float,
) -> dict[str, float]:
    """
    Evaluate on normal (held-out) samples.

    Returns:
        val_loss      — mean MSE on masked patches   (lower = better)
        anomaly_score — mean compute_anomaly_score()  (lower = better for normal)
    """
    model.eval()
    losses, scores = [], []
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
#  CV for one mask_ratio
# ═══════════════════════════════════════════════════════════════════════════

def run_cv_for_mask_ratio(
    mask_ratio:  float,
    all_files:   list[Path],
    cfg:         AudioConfig,
    device:      torch.device,
    n_epochs:    int,
    verbose:     bool = True,
) -> list[dict]:
    """
    Run N_FOLDS-fold CV for a single mask_ratio value.

    Returns list of dicts, one per fold:
        fold, mask_ratio, val_loss, anomaly_score
    """
    n       = len(all_files)
    indices = np.arange(n)
    rng     = np.random.default_rng(SEED)
    rng.shuffle(indices)
    folds   = np.array_split(indices, N_FOLDS)

    fold_results: list[dict] = []

    for k in range(N_FOLDS):
        val_idx   = folds[k].tolist()
        train_idx = [i for j, fold in enumerate(folds) if j != k
                     for i in fold.tolist()]

        train_loader, val_loader = make_kfold_loaders(
            all_files, train_idx, val_idx, cfg=cfg,
            batch_size=BATCH_SIZE, num_workers=0,
        )

        set_seed(SEED + k)
        model = specmae_vit_base_patch16(
            mask_ratio=mask_ratio, norm_pix_loss=True,
        ).to(device)

        optimizer = optim.AdamW(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
        )

        best_val  = float("inf")
        log_every = max(1, n_epochs // 6)

        for epoch in range(n_epochs):
            tr_loss = train_one_epoch(
                model, train_loader, optimizer, device,
                mask_ratio=mask_ratio, epoch=epoch, n_epochs=n_epochs,
            )
            if (epoch + 1) % log_every == 0 or epoch == n_epochs - 1:
                metrics  = evaluate(model, val_loader, device, mask_ratio)
                best_val = min(best_val, metrics["val_loss"])
                if verbose:
                    print(
                        f"    mask={mask_ratio:.2f} fold={k+1}/{N_FOLDS} "
                        f"ep={epoch+1:3d}/{n_epochs} "
                        f"lr={get_lr(epoch,n_epochs):.2e} "
                        f"tr={tr_loss:.4f} "
                        f"val={metrics['val_loss']:.4f} "
                        f"score={metrics['anomaly_score']:.4f}"
                    )

        # Record best val loss for this fold
        final = evaluate(model, val_loader, device, mask_ratio)
        fold_results.append({
            "fold":          k + 1,
            "mask_ratio":    mask_ratio,
            "val_loss":      final["val_loss"],
            "anomaly_score": final["anomaly_score"],
        })

        del model, optimizer
        empty_device_cache(device)

    return fold_results


# ═══════════════════════════════════════════════════════════════════════════
#  Final training on all data
# ═══════════════════════════════════════════════════════════════════════════

def train_final_model(
    best_mask_ratio: float,
    all_files:       list[Path],
    cfg:             AudioConfig,
    device:          torch.device,
    out_dir:         Path,
    n_epochs:        int,
    verbose:         bool = True,
) -> Path:
    """
    Train SpecMAE on all training data with the optimal mask_ratio.
    10 % of data held out as a monitoring-only validation split.
    Saves best checkpoint to out_dir/best_model.pth.
    """
    n_val   = max(1, int(len(all_files) * 0.1))
    trn_idx = list(range(len(all_files) - n_val))
    val_idx = list(range(len(all_files) - n_val, len(all_files)))

    train_loader, val_loader = make_kfold_loaders(
        all_files, trn_idx, val_idx, cfg=cfg,
        batch_size=BATCH_SIZE, num_workers=0,
    )

    set_seed(SEED)
    model = specmae_vit_base_patch16(
        mask_ratio=best_mask_ratio, norm_pix_loss=True,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
    )

    best_val  = float("inf")
    best_path = out_dir / "best_model.pth"
    train_losses: list[float]            = []
    val_log:      list[tuple[int, float]] = []

    log_every = max(1, n_epochs // 10)

    for epoch in range(n_epochs):
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            mask_ratio=best_mask_ratio, epoch=epoch, n_epochs=n_epochs,
        )
        train_losses.append(tr_loss)

        if (epoch + 1) % log_every == 0 or epoch == n_epochs - 1:
            metrics = evaluate(model, val_loader, device, best_mask_ratio)
            val_log.append((epoch + 1, metrics["val_loss"]))
            is_best = metrics["val_loss"] < best_val

            if is_best:
                best_val = metrics["val_loss"]
                torch.save(
                    {
                        "epoch":       epoch + 1,
                        "mask_ratio":  best_mask_ratio,
                        "val_loss":    best_val,
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
                    },
                    best_path,
                )

            if verbose:
                print(
                    f"  ep={epoch+1:4d}/{n_epochs} "
                    f"lr={get_lr(epoch,n_epochs):.2e} "
                    f"tr={tr_loss:.5f}  "
                    f"val={metrics['val_loss']:.5f}  "
                    f"{'<-- best' if is_best else ''}"
                )

    _save_curve(train_losses, val_log, best_mask_ratio, out_dir)
    print(f"\nBest val_loss = {best_val:.6f}")
    print(f"Checkpoint    : {best_path}")
    return best_path


def _save_curve(
    train_losses: list[float],
    val_log:      list[tuple[int, float]],
    mask_ratio:   float,
    out_dir:      Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(range(1, len(train_losses) + 1), train_losses,
                alpha=0.7, label="train loss")
        if val_log:
            ep_v, v_vals = zip(*val_log)
            ax.plot(ep_v, v_vals, "o-", label="val loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Reconstruction MSE")
        ax.set_title(f"SpecMAE-Base  mask_ratio={mask_ratio:.2f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
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
        description="SpecMAE 5-fold CV — mask_ratio hyperparameter search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scenario",     default="desert")
    p.add_argument("--data_dir",     default="Spec_Mae/data/desert",
                   help="Scenario data dir (must contain train/normal/)")
    p.add_argument("--out_dir",      default="Spec_Mae/results/cv_desert")
    p.add_argument("--cv_epochs",    type=int,   default=CV_EPOCHS)
    p.add_argument("--final_epochs", type=int,   default=FINAL_EPOCHS)
    p.add_argument("--batch_size",   type=int,   default=BATCH_SIZE)
    p.add_argument("--mask_ratios",  nargs="+",  type=float,
                   default=MASK_RATIO_GRID,
                   help="Mask ratios to search")
    p.add_argument("--skip_cv",      action="store_true",
                   help="Skip CV; use --best_mask_ratio directly")
    p.add_argument("--best_mask_ratio", type=float, default=None,
                   help="Force a specific mask_ratio (skips CV)")
    p.add_argument("--num_workers",  type=int,   default=0)
    p.add_argument("--auto_norm",    action="store_true",
                   help="Compute real mean/std from training clips and update AudioConfig "
                        "before CV (recommended; replaces placeholder values)")
    p.add_argument("--norm_samples", type=int,   default=500,
                   help="Number of clips sampled for auto_norm estimation")
    p.add_argument("--seed",         type=int,   default=SEED)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    args     = parse_args()
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global CV_EPOCHS, FINAL_EPOCHS, BATCH_SIZE, SEED
    CV_EPOCHS    = args.cv_epochs
    FINAL_EPOCHS = args.final_epochs
    BATCH_SIZE   = args.batch_size
    SEED         = args.seed

    set_seed(SEED)
    device   = get_device(verbose=True)
    print_device_diagnostics()
    data_dir = Path(args.data_dir)

    print("=" * 64)
    print(f"SpecMAE Cross-Validation — scenario: {args.scenario}")
    print(f"Device      : {device}")
    print(f"Mask grid   : {args.mask_ratios}")
    print(f"CV epochs   : {CV_EPOCHS}  |  Final epochs: {FINAL_EPOCHS}")
    print(f"Output      : {out_dir}")
    print("=" * 64)

    cfg = AudioConfig()

    # Gather all normal training clips
    train_dir = data_dir / "train" / "normal"
    all_files = sorted(train_dir.glob("*.wav"))
    if not all_files:
        raise FileNotFoundError(f"No WAV files found in {train_dir}")
    print(f"Training clips: {len(all_files)}")

    # ── Auto-calibrate normalization statistics ──────────────────────────────
    if args.auto_norm:
        print(f"\nCalibrating norm stats from {len(all_files)} clips "
              f"(sampling {min(args.norm_samples, len(all_files))}) ...")
        mean, std = compute_dataset_stats(
            [str(p) for p in all_files], cfg=cfg, n_samples=args.norm_samples,
        )
        cfg.norm_mean = mean
        cfg.norm_std  = std
        print(f"Norm calibration complete: mean={mean:.4f}  std={std:.4f}")
        norm_path = out_dir / "norm_stats.json"
        with open(norm_path, "w") as _f:
            import json as _json
            _json.dump({"norm_mean": mean, "norm_std": std,
                        "n_clips": len(all_files),
                        "n_sampled": min(args.norm_samples, len(all_files))}, _f, indent=2)
        print(f"Norm stats saved to {norm_path}")
    else:
        print(f"Norm stats: mean={cfg.norm_mean} std={cfg.norm_std} "
              f"(placeholder — run with --auto_norm to calibrate)")

    # ── Phase 1: Cross-validation ────────────────────────────────────────
    if args.best_mask_ratio is not None or args.skip_cv:
        best_mask_ratio = args.best_mask_ratio or 0.75
        print(f"\nSkipping CV  →  mask_ratio = {best_mask_ratio}")
        all_fold_results: list[dict] = []
    else:
        all_fold_results = []

        print(f"\nPhase 1 — 5-fold CV over {len(args.mask_ratios)} mask ratios")

        for mr in args.mask_ratios:
            print(f"\n  >>> mask_ratio = {mr:.2f}")
            t0 = time.time()

            fold_results = run_cv_for_mask_ratio(
                mask_ratio=mr,
                all_files=all_files,
                cfg=cfg,
                device=device,
                n_epochs=CV_EPOCHS,
                verbose=True,
            )
            all_fold_results.extend(fold_results)

            mean_loss  = np.mean([r["val_loss"]      for r in fold_results])
            mean_score = np.mean([r["anomaly_score"] for r in fold_results])
            print(
                f"  mask={mr:.2f}  "
                f"mean_val_loss={mean_loss:.4f}  "
                f"mean_score={mean_score:.4f}  "
                f"({time.time()-t0:.0f}s)"
            )

        # Save per-fold CSV
        csv_path = out_dir / "cv_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["mask_ratio", "fold", "val_loss", "anomaly_score"]
            )
            writer.writeheader()
            writer.writerows(all_fold_results)

        # Compute and print summary table
        summary: dict[str, dict] = {}
        for mr in args.mask_ratios:
            rows   = [r for r in all_fold_results if r["mask_ratio"] == mr]
            losses = [r["val_loss"]      for r in rows]
            scores = [r["anomaly_score"] for r in rows]
            summary[str(mr)] = {
                "mask_ratio":         mr,
                "mean_val_loss":      float(np.mean(losses)),
                "std_val_loss":       float(np.std(losses)),
                "mean_anomaly_score": float(np.mean(scores)),
                "std_anomaly_score":  float(np.std(scores)),
            }

        best_key        = min(summary, key=lambda k: summary[k]["mean_val_loss"])
        best_mask_ratio = float(best_key)
        summary["best_mask_ratio"] = best_mask_ratio

        with open(out_dir / "cv_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*64}")
        print("CV Summary:")
        print(f"  {'mask_ratio':>10}  {'mean_val_loss':>14}  {'std':>7}  {'mean_score':>11}")
        for k in sorted([k for k in summary if k != "best_mask_ratio"],
                        key=lambda k: summary[k].get("mean_val_loss", 99)):
            if k == "best_mask_ratio":
                continue
            v   = summary[k]
            tag = "  <-- BEST" if float(k) == best_mask_ratio else ""
            print(
                f"  {v['mask_ratio']:>10.2f}  "
                f"{v['mean_val_loss']:>14.5f}  "
                f"{v['std_val_loss']:>7.5f}  "
                f"{v['mean_anomaly_score']:>11.5f}"
                f"{tag}"
            )
        print(f"\nBest mask_ratio = {best_mask_ratio}")

    # ── Phase 2: Final training ──────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"Phase 2 — Final training (mask_ratio={best_mask_ratio}, "
          f"epochs={FINAL_EPOCHS})")
    print(f"{'='*64}")

    ckpt = train_final_model(
        best_mask_ratio=best_mask_ratio,
        all_files=all_files,
        cfg=cfg,
        device=device,
        out_dir=out_dir,
        n_epochs=FINAL_EPOCHS,
        verbose=True,
    )

    print(f"\nAll done.  Best model: {ckpt}")


if __name__ == "__main__":
    main()
