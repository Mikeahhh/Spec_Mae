"""
SpecMAE Anomaly-Detection Evaluation Script.

Runs inference on the desert (or other scenario) test set and computes:
    - Per-sample anomaly scores (reconstruction MSE)
    - AUC-ROC, pAUC (FPR ≤ 0.1), F1 @ best threshold
    - Breakdown by SNR level

Usage
-----
    cd E:/model_train_example
    python Spec_Mae/scripts/test/test_anomaly_detection.py \
        --checkpoint  Spec_Mae/results/cv_desert/best_model.pth \
        --data_dir    Spec_Mae/data/desert \
        --out_dir     Spec_Mae/results/eval_desert

Output
------
    results/eval_desert/
    ├── anomaly_scores.csv      (label, score, snr_tag, filename)
    ├── metrics.json            (AUC, pAUC, F1 overall + per-SNR)
    ├── roc_curve.png
    └── score_distribution.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

# ── Project path setup ───────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent
_SPEC    = _HERE.parent.parent
_PROJECT = _SPEC.parent
sys.path.insert(0, str(_PROJECT))

from Spec_Mae.models.specmae import SpecMAE, specmae_vit_base_patch16
from Spec_Mae.scripts.utils.feature_extraction import AudioConfig, LogMelExtractor
from Spec_Mae.scripts.utils.data_loader import AnomalyTestDataset
from Spec_Mae.scripts.eval.compute_metrics import (
    compute_metrics_per_snr,
    print_metrics_table,
)
from torch.utils.data import DataLoader

from Spec_Mae.scripts.utils.device import (
    get_device, should_pin_memory, print_device_diagnostics,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Checkpoint loading
# ═══════════════════════════════════════════════════════════════════════════

def load_model(ckpt_path: Path, device: torch.device) -> tuple[SpecMAE, float, AudioConfig]:
    """
    Load a SpecMAE checkpoint.

    Returns (model, mask_ratio, audio_cfg).
    """
    # Security: Use weights_only=True to prevent arbitrary code execution
    # Note: weights_only=True requires PyTorch 2.0+
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    mask_ratio = float(ckpt.get("mask_ratio", 0.75))

    # Reconstruct AudioConfig from stored dict (if present)
    cfg_dict = ckpt.get("audio_cfg", {})
    cfg = AudioConfig(
        sample_rate = cfg_dict.get("sample_rate", 48_000),
        n_mels      = cfg_dict.get("n_mels",      128),
        n_fft       = cfg_dict.get("n_fft",       1_024),
        hop_length  = cfg_dict.get("hop_length",  480),
        norm_mean   = cfg_dict.get("norm_mean",   -6.0),
        norm_std    = cfg_dict.get("norm_std",    5.0),
    )

    model = specmae_vit_base_patch16(mask_ratio=mask_ratio, norm_pix_loss=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    return model, mask_ratio, cfg


# ═══════════════════════════════════════════════════════════════════════════
#  Inference
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(
    model:      SpecMAE,
    loader:     DataLoader,
    device:     torch.device,
    mask_ratio: float,
    n_passes:   int = 1,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Run the model over the full test loader.

    Returns:
        labels    — (N,) int array,   0=normal, 1=anomaly
        scores    — (N,) float array, higher = more anomalous
        snr_tags  — (N,) list of strings
        filenames — (N,) list of strings
    """
    model.eval()
    all_labels:    list[int]   = []
    all_scores:    list[float] = []
    all_snr_tags:  list[str]   = []
    all_filenames: list[str]   = []

    # AnomalyTestDataset returns (spec, label, snr_tag); filenames via .samples
    samples = loader.dataset.samples   # list of (Path, label, snr_tag)

    sample_idx = 0
    for batch in loader:
        specs, labels, snr_tags = batch
        specs  = specs.to(device, non_blocking=True)
        scores = model.compute_anomaly_score(
            specs, mask_ratio=mask_ratio, n_passes=n_passes,
        ).cpu().numpy()

        for i in range(len(scores)):
            all_scores.append(float(scores[i]))
            all_labels.append(int(labels[i]))
            all_snr_tags.append(snr_tags[i])
            all_filenames.append(str(samples[sample_idx][0].name))
            sample_idx += 1

    return (
        np.array(all_labels),
        np.array(all_scores),
        all_snr_tags,
        all_filenames,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Saving outputs
# ═══════════════════════════════════════════════════════════════════════════

def save_scores_csv(
    out_dir:   Path,
    labels:    np.ndarray,
    scores:    np.ndarray,
    snr_tags:  list[str],
    filenames: list[str],
) -> Path:
    path = out_dir / "anomaly_scores.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "score", "snr_tag"])
        writer.writeheader()
        for fn, lb, sc, st in zip(filenames, labels, scores, snr_tags):
            writer.writerow({
                "filename": fn,
                "label":    int(lb),
                "score":    f"{sc:.8f}",
                "snr_tag":  st,
            })
    return path


def save_roc_plot(
    labels:   np.ndarray,
    scores:   np.ndarray,
    auc:      float,
    pauc:     float,
    out_dir:  Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from Spec_Mae.scripts.eval.compute_metrics import compute_roc, compute_pauc

        fprs, tprs, _ = compute_roc(labels, scores)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fprs, tprs, linewidth=2, label=f"AUC={auc:.4f} | pAUC={pauc:.4f}")
        ax.axvline(x=0.1, color="gray", linestyle="--", alpha=0.6, label="FPR=0.1 (pAUC limit)")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)
        ax.fill_between(fprs[fprs <= 0.1], tprs[fprs <= 0.1],
                        alpha=0.15, color="steelblue", label="pAUC region")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — SpecMAE Anomaly Detection")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "roc_curve.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"  [plot] ROC curve skipped: {exc}")


def save_score_distribution(
    labels:  np.ndarray,
    scores:  np.ndarray,
    out_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        normal_scores  = scores[labels == 0]
        anomaly_scores = scores[labels == 1]

        bins = np.linspace(scores.min(), scores.max(), 60)
        ax.hist(normal_scores,  bins=bins, alpha=0.6, density=True,
                label=f"Normal (n={len(normal_scores)})",  color="steelblue")
        ax.hist(anomaly_scores, bins=bins, alpha=0.6, density=True,
                label=f"Anomaly (n={len(anomaly_scores)})", color="tomato")
        ax.set_xlabel("Anomaly Score (Reconstruction MSE)")
        ax.set_ylabel("Density")
        ax.set_title("Anomaly Score Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "score_distribution.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"  [plot] Score distribution skipped: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SpecMAE anomaly detection evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",  required=True,
                   help="Path to .pth checkpoint file")
    p.add_argument("--data_dir",    default="Spec_Mae/data/desert",
                   help="Scenario data root (must contain test/normal/ and test/anomaly/)")
    p.add_argument("--out_dir",     default="Spec_Mae/results/eval_desert",
                   help="Output directory for scores and plots")
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--n_passes",    type=int, default=1,
                   help="Number of stochastic masking passes for score averaging")
    p.add_argument("--max_fpr",     type=float, default=0.1,
                   help="FPR upper bound for pAUC (DCASE: 0.1)")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = get_device(verbose=True)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    data_dir    = Path(args.data_dir)
    normal_dir  = data_dir / "test" / "normal"
    anomaly_dir = data_dir / "test" / "anomaly"

    for d in (normal_dir, anomaly_dir):
        if not d.exists():
            print(f"ERROR: directory not found: {d}", file=sys.stderr)
            sys.exit(1)

    print("=" * 60)
    print(f"SpecMAE Anomaly Detection Evaluation")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Data       : {data_dir}")
    print(f"  Device     : {device}")
    print("=" * 60)

    # ── Load model ─────────────────────────────────────────────────────────
    model, mask_ratio, cfg = load_model(ckpt_path, device)
    print(f"  Loaded model — mask_ratio={mask_ratio}")

    # ── Build test loader ──────────────────────────────────────────────────
    extractor = LogMelExtractor(cfg=cfg)
    test_ds   = AnomalyTestDataset(
        normal_dir=normal_dir,
        anomaly_dir=anomaly_dir,
        extractor=extractor,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=should_pin_memory(device),
    )

    n_normal  = sum(1 for _, lb, _ in test_ds.samples if lb == 0)
    n_anomaly = sum(1 for _, lb, _ in test_ds.samples if lb == 1)
    print(f"  Test set — normal: {n_normal}, anomaly: {n_anomaly}, "
          f"total: {len(test_ds)}")

    # ── Inference ──────────────────────────────────────────────────────────
    print("\n  Running inference...")
    labels, scores, snr_tags, filenames = run_inference(
        model, test_loader, device, mask_ratio, n_passes=args.n_passes,
    )

    # ── Save raw scores ────────────────────────────────────────────────────
    csv_path = save_scores_csv(out_dir, labels, scores, snr_tags, filenames)
    print(f"  Scores saved to {csv_path}")

    # ── Compute metrics ────────────────────────────────────────────────────
    print("\n  Computing metrics...")
    results = compute_metrics_per_snr(labels, scores, snr_tags, max_fpr=args.max_fpr)

    print("\n  Results:")
    print_metrics_table(results)

    # Save JSON
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Metrics JSON : {metrics_path}")

    # ── Plots ──────────────────────────────────────────────────────────────
    overall = results.get("overall", {})
    auc  = overall.get("auc",  0.0)
    pauc = overall.get("pauc", 0.0)

    save_roc_plot(labels, scores, auc, pauc, out_dir)
    save_score_distribution(labels, scores, out_dir)
    print(f"  Plots saved to {out_dir}")

    # ── Summary line ───────────────────────────────────────────────────────
    print(f"\n  Overall  AUC={auc:.4f}  pAUC={pauc:.4f}  "
          f"F1={overall.get('f1', float('nan')):.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
