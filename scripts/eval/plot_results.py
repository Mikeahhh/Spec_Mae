"""
SpecMAE Results Visualization.

Generates publication-quality figures from evaluation outputs:

    Fig A — Anomaly score curves vs SNR (ROC family)
    Fig B — Mask-ratio ablation: mean val_loss per mask_ratio (from CV summary)
    Fig C — SNR vs AUC / pAUC / F1 line plots
    Fig D — Score distributions (normal vs anomaly) per SNR

Usage
-----
    python Spec_Mae/scripts/eval/plot_results.py \
        --metrics_json  Spec_Mae/results/eval_desert/metrics.json \
        --scores_csv    Spec_Mae/results/eval_desert/anomaly_scores.csv \
        --cv_json       Spec_Mae/results/cv_desert/cv_summary.json \
        --out_dir       Spec_Mae/results/figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ── Matplotlib setup ─────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    _PLT = True
except ImportError:
    _PLT = False


# ── Project path ─────────────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent
_SPEC    = _HERE.parent.parent
_PROJECT = _SPEC.parent
sys.path.insert(0, str(_PROJECT))

from Spec_Mae.scripts.eval.compute_metrics import load_scores_csv, compute_roc


# ═══════════════════════════════════════════════════════════════════════════
#  Color palette
# ═══════════════════════════════════════════════════════════════════════════

SNR_COLORS = {
    "snr_-10dB": "#d62728",
    "snr_-5dB":  "#ff7f0e",
    "snr_+0dB":  "#bcbd22",
    "snr_+5dB":  "#2ca02c",
    "snr_+10dB": "#17becf",
    "snr_+15dB": "#1f77b4",
    "snr_+20dB": "#9467bd",
    "normal":    "#7f7f7f",
}


# ═══════════════════════════════════════════════════════════════════════════
#  Fig A — ROC curves per SNR
# ═══════════════════════════════════════════════════════════════════════════

def fig_roc_per_snr(
    labels:   np.ndarray,
    scores:   np.ndarray,
    snr_tags: list[str],
    out_dir:  Path,
) -> None:
    """One ROC curve per SNR level (anomaly vs the full normal pool)."""
    if not _PLT:
        return

    snr_arr    = np.array(snr_tags)
    normal_mask = snr_arr == "normal"
    normal_labels = labels[normal_mask]
    normal_scores = scores[normal_mask]

    snr_list = sorted({t for t in snr_tags if t != "normal"})

    fig, ax = plt.subplots(figsize=(7, 6))

    for snr in snr_list:
        mask  = snr_arr == snr
        c_lbl = np.concatenate([normal_labels, labels[mask]])
        c_sc  = np.concatenate([normal_scores, scores[mask]])
        if len(np.unique(c_lbl)) < 2:
            continue
        fprs, tprs, _ = compute_roc(c_lbl, c_sc)
        auc   = float(np.trapz(tprs, fprs))
        color = SNR_COLORS.get(snr, "gray")
        ax.plot(fprs, tprs, linewidth=1.8, color=color,
                label=f"{snr}  (AUC={auc:.3f})")

    ax.axvline(x=0.1, color="gray", linestyle="--", alpha=0.5, linewidth=0.9)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.25, linewidth=0.8)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves per SNR Level — SpecMAE", fontsize=13)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    fig.tight_layout()
    path = out_dir / "fig_roc_per_snr.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ═══════════════════════════════════════════════════════════════════════════
#  Fig B — Mask-ratio ablation
# ═══════════════════════════════════════════════════════════════════════════

def fig_mask_ratio_ablation(cv_summary: dict, out_dir: Path) -> None:
    """Bar chart of mean_val_loss +/- std across mask ratios from CV summary."""
    if not _PLT:
        return

    rows = [
        (float(k), v["mean_val_loss"], v["std_val_loss"])
        for k, v in cv_summary.items()
        if k != "best_mask_ratio" and isinstance(v, dict)
    ]
    if not rows:
        return

    rows.sort()
    mrs, means, stds = zip(*rows)
    best_mr = cv_summary.get("best_mask_ratio", None)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [
        "tomato" if (best_mr is not None and abs(mr - best_mr) < 1e-6) else "steelblue"
        for mr in mrs
    ]
    ax.bar(
        [f"{mr:.2f}" for mr in mrs], means,
        yerr=stds, capsize=4, color=colors,
        edgecolor="black", linewidth=0.6, alpha=0.85,
    )

    ax.set_xlabel("Mask Ratio", fontsize=12)
    ax.set_ylabel("5-Fold Mean Val Loss", fontsize=12)
    ax.set_title("Mask-Ratio Ablation (5-Fold CV)", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    if best_mr is not None:
        mrs_list = [mr for mr in mrs]
        if best_mr in mrs_list:
            best_idx = mrs_list.index(best_mr)
            ax.get_children()[best_idx].set_label(f"best={best_mr:.2f}")
            ax.legend(fontsize=9)
    fig.tight_layout()
    path = out_dir / "fig_mask_ratio_ablation.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ═══════════════════════════════════════════════════════════════════════════
#  Fig C — SNR vs AUC / pAUC / F1
# ═══════════════════════════════════════════════════════════════════════════

def _snr_value(tag: str) -> float:
    """Extract numeric dB value from tag like 'snr_-10dB' -> -10."""
    tag = tag.replace("snr_", "").replace("dB", "").replace("+", "")
    try:
        return float(tag)
    except ValueError:
        return float("nan")


def fig_snr_vs_metrics(metrics: dict, out_dir: Path) -> None:
    """Line plot: AUC, pAUC, F1 as functions of SNR."""
    if not _PLT:
        return

    snr_keys = sorted(
        [k for k in metrics if k != "overall" and "snr_" in k],
        key=_snr_value,
    )
    if not snr_keys:
        return

    db_vals = [_snr_value(k) for k in snr_keys]
    aucs    = [metrics[k].get("auc",  float("nan")) for k in snr_keys]
    paucs   = [metrics[k].get("pauc", float("nan")) for k in snr_keys]
    f1s     = [metrics[k].get("f1",   float("nan")) for k in snr_keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(db_vals, aucs,  "o-", linewidth=2, markersize=6, label="AUC",  color="#1f77b4")
    ax.plot(db_vals, paucs, "s-", linewidth=2, markersize=6, label="pAUC", color="#ff7f0e")
    ax.plot(db_vals, f1s,   "^-", linewidth=2, markersize=6, label="F1",   color="#2ca02c")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Score",    fontsize=12)
    ax.set_title("Anomaly Detection Performance vs SNR — SpecMAE", fontsize=13)
    ax.set_ylim([0.0, 1.05])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    path = out_dir / "fig_snr_vs_metrics.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ═══════════════════════════════════════════════════════════════════════════
#  Fig D — Score distributions per SNR
# ═══════════════════════════════════════════════════════════════════════════

def fig_score_distributions(
    labels:   np.ndarray,
    scores:   np.ndarray,
    snr_tags: list[str],
    out_dir:  Path,
) -> None:
    """Overlapping histograms of anomaly score, one subplot per SNR."""
    if not _PLT:
        return

    snr_list = sorted({t for t in snr_tags if t != "normal"})
    n_snr    = len(snr_list)
    if n_snr == 0:
        return

    ncols  = min(4, n_snr)
    nrows  = (n_snr + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), squeeze=False,
    )

    snr_arr       = np.array(snr_tags)
    normal_scores = scores[snr_arr == "normal"]
    bins          = np.linspace(scores.min(), scores.max(), 50)

    for i, snr in enumerate(snr_list):
        ax     = axes[i // ncols][i % ncols]
        anom_s = scores[snr_arr == snr]
        ax.hist(normal_scores, bins=bins, density=True, alpha=0.55,
                color="steelblue", label="normal")
        ax.hist(anom_s, bins=bins, density=True, alpha=0.6,
                color=SNR_COLORS.get(snr, "tomato"), label=snr)
        ax.set_title(snr, fontsize=9)
        ax.set_xlabel("Score", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)

    for j in range(n_snr, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle("Score Distributions: Normal vs Anomaly per SNR", fontsize=12)
    fig.tight_layout()
    path = out_dir / "fig_score_distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SpecMAE results visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--metrics_json", default=None,
                   help="metrics.json from test_anomaly_detection.py")
    p.add_argument("--scores_csv",   default=None,
                   help="anomaly_scores.csv from test_anomaly_detection.py")
    p.add_argument("--cv_json",      default=None,
                   help="cv_summary.json from train_cross_validation.py")
    p.add_argument("--out_dir",      default="Spec_Mae/results/figures")
    return p.parse_args()


def main() -> None:
    if not _PLT:
        print("ERROR: matplotlib is required for plot_results.py", file=sys.stderr)
        sys.exit(1)

    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("SpecMAE Results Visualization")
    print("=" * 55)

    metrics:   dict | None         = None
    cv_summary: dict | None        = None
    labels:    np.ndarray | None   = None
    scores:    np.ndarray | None   = None
    snr_tags:  list[str] | None    = None

    if args.metrics_json:
        path = Path(args.metrics_json)
        if path.exists():
            with open(path) as f:
                metrics = json.load(f)
            print(f"  Loaded metrics   : {path}")
        else:
            print(f"  WARNING: not found: {path}")

    if args.scores_csv:
        path = Path(args.scores_csv)
        if path.exists():
            labels, scores, snr_tags = load_scores_csv(path)
            print(f"  Loaded scores    : {path}  (N={len(labels)})")
        else:
            print(f"  WARNING: not found: {path}")

    if args.cv_json:
        path = Path(args.cv_json)
        if path.exists():
            with open(path) as f:
                cv_summary = json.load(f)
            print(f"  Loaded CV summary: {path}")
        else:
            print(f"  WARNING: not found: {path}")

    print()

    if labels is not None and scores is not None and snr_tags is not None:
        fig_roc_per_snr(labels, scores, snr_tags, out_dir)
        fig_score_distributions(labels, scores, snr_tags, out_dir)

    if cv_summary is not None:
        fig_mask_ratio_ablation(cv_summary, out_dir)

    if metrics is not None:
        fig_snr_vs_metrics(metrics, out_dir)

    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()
