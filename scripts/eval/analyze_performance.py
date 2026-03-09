"""
SpecMAE Performance Analysis Tool.

Reads evaluation outputs (metrics.json, anomaly_scores.csv, cv_summary.json,
localization_summary.json) and produces a structured report:

    - Anomaly detection summary table (AUC, pAUC, F1 per SNR)
    - Localization accuracy table (DOA MAE, within-5-deg rate per SNR)
    - Full-system simulation summary (detection rate, goal-reached, DOA error)
    - Cross-scenario comparison when multiple eval dirs are provided
    - Optional: export as Markdown or plain text report

Usage
-----
    # Single scenario
    python Spec_Mae/scripts/eval/analyze_performance.py \
        --eval_dir   Spec_Mae/results/eval_desert \
        --loc_dir    Spec_Mae/results/localization \
        --sys_dir    Spec_Mae/results/full_system \
        --cv_dir     Spec_Mae/results/cv_desert \
        --out_report Spec_Mae/results/report_desert.md

    # Multi-scenario comparison
    python Spec_Mae/scripts/eval/analyze_performance.py \
        --eval_dirs  Spec_Mae/results/eval_desert Spec_Mae/results/eval_forest \
        --labels     desert forest \
        --out_report Spec_Mae/results/comparison_report.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
#  JSON loaders
# ═══════════════════════════════════════════════════════════════════════════

def _load_json(path: Optional[Path]) -> Optional[dict]:
    if path is None or not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
#  Formatting helpers
# ═══════════════════════════════════════════════════════════════════════════

def _pct(v: float) -> str:
    return f"{v*100:.1f}%" if not np.isnan(v) else "N/A"

def _f(v: float, dec: int = 4) -> str:
    return f"{v:.{dec}f}" if not np.isnan(v) else "N/A"

def _snr_key(k: str) -> float:
    try:
        return float(k.replace("snr_", "").replace("dB", "").replace("+", ""))
    except ValueError:
        return float("inf")


# ═══════════════════════════════════════════════════════════════════════════
#  Section builders
# ═══════════════════════════════════════════════════════════════════════════

def section_anomaly_detection(metrics: dict, label: str = "") -> list[str]:
    lines: list[str] = []
    hdr = f"### Anomaly Detection{(' — ' + label) if label else ''}"
    lines.append(hdr)
    lines.append("")

    overall = metrics.get("overall", {})
    if overall and "error" not in overall:
        lines.append(
            f"**Overall**  AUC={_f(overall.get('auc', float('nan')))}  "
            f"pAUC={_f(overall.get('pauc', float('nan')))}  "
            f"F1={_f(overall.get('f1', float('nan')))}  "
            f"(normal={overall.get('n_normal','?')}, anomaly={overall.get('n_anomaly','?')})"
        )
        lines.append("")

    snr_keys = sorted(
        [k for k in metrics if k != "overall" and "snr_" in k],
        key=_snr_key,
    )
    if snr_keys:
        lines.append("| SNR | AUC | pAUC | F1 | Precision | Recall |")
        lines.append("|-----|-----|------|----|-----------|--------|")
        for k in snr_keys:
            m = metrics[k]
            lines.append(
                f"| {k} "
                f"| {_f(m.get('auc', float('nan')))} "
                f"| {_f(m.get('pauc', float('nan')))} "
                f"| {_f(m.get('f1', float('nan')))} "
                f"| {_f(m.get('precision', float('nan')))} "
                f"| {_f(m.get('recall', float('nan')))} |"
            )
    lines.append("")
    return lines


def section_localization(loc: dict, label: str = "") -> list[str]:
    lines: list[str] = []
    lines.append(f"### Localization{(' — ' + label) if label else ''}")
    lines.append("")
    snr_keys = sorted(loc.keys(), key=_snr_key)
    if snr_keys:
        lines.append("| SNR | DOA MAE (deg) | Within 5° | Within 10° | TDOA RMSE (samp) |")
        lines.append("|-----|--------------|-----------|------------|-----------------|")
        for k in snr_keys:
            v = loc[k]
            lines.append(
                f"| {k} "
                f"| {_f(v.get('doa_mae_deg', float('nan')), 2)} "
                f"| {_pct(v.get('within_5deg', float('nan')))} "
                f"| {_pct(v.get('within_10deg', float('nan')))} "
                f"| {_f(v.get('tdoa_rmse_samp', float('nan')), 2)} |"
            )
    lines.append("")
    return lines


def section_full_system(sys_summary: dict, label: str = "") -> list[str]:
    lines: list[str] = []
    lines.append(f"### Full System{(' — ' + label) if label else ''}")
    lines.append("")
    snr_keys = sorted(sys_summary.keys(), key=_snr_key)
    if snr_keys:
        lines.append("| SNR | Detection Rate | Goal Reached | Mean DOA Err | Final Dist (m) |")
        lines.append("|-----|---------------|--------------|--------------|----------------|")
        for k in snr_keys:
            v = sys_summary[k]
            lines.append(
                f"| {k} "
                f"| {_pct(v.get('detection_rate', float('nan')))} "
                f"| {_pct(v.get('goal_reached_rate', float('nan')))} "
                f"| {_f(v.get('mean_doa_err_deg', float('nan')), 1)}° "
                f"| {_f(v.get('mean_final_dist_m', float('nan')), 1)} |"
            )
    lines.append("")
    return lines


def section_cv_summary(cv: dict, label: str = "") -> list[str]:
    lines: list[str] = []
    lines.append(f"### Cross-Validation (Mask-Ratio Search){(' — ' + label) if label else ''}")
    lines.append("")
    best_mr = cv.get("best_mask_ratio", "?")
    lines.append(f"**Best mask_ratio**: {best_mr}")
    lines.append("")
    rows = [(k, v) for k, v in cv.items()
            if k != "best_mask_ratio" and isinstance(v, dict)]
    rows.sort(key=lambda x: float(x[0]))
    if rows:
        lines.append("| Mask Ratio | Mean Val Loss | Std Val Loss |")
        lines.append("|------------|--------------|--------------|")
        for k, v in rows:
            tag = " **<-- BEST**" if float(k) == float(best_mr) else ""
            lines.append(
                f"| {float(k):.2f} "
                f"| {_f(v.get('mean_val_loss', float('nan')))} "
                f"| {_f(v.get('std_val_loss', float('nan')))} |"
                f"{tag}"
            )
    lines.append("")
    return lines


def section_comparison(
    metrics_list: list[dict],
    labels:       list[str],
) -> list[str]:
    """Overall AUC/pAUC/F1 comparison table across scenarios."""
    lines: list[str] = []
    lines.append("### Cross-Scenario Comparison (Overall Metrics)")
    lines.append("")
    lines.append("| Scenario | AUC | pAUC | F1 |")
    lines.append("|----------|-----|------|----|")
    for m, lbl in zip(metrics_list, labels):
        ov = m.get("overall", {})
        lines.append(
            f"| {lbl} "
            f"| {_f(ov.get('auc', float('nan')))} "
            f"| {_f(ov.get('pauc', float('nan')))} "
            f"| {_f(ov.get('f1', float('nan')))} |"
        )
    lines.append("")
    return lines


# ═══════════════════════════════════════════════════════════════════════════
#  Report assembly
# ═══════════════════════════════════════════════════════════════════════════

def build_report(
    eval_dirs:    list[Path],
    labels:       list[str],
    loc_dirs:     list[Optional[Path]],
    sys_dirs:     list[Optional[Path]],
    cv_dirs:      list[Optional[Path]],
) -> str:
    from datetime import date
    lines: list[str] = [
        "# SpecMAE Performance Report",
        "",
        f"*Generated: {date.today().isoformat()}*",
        "",
    ]

    metrics_list: list[dict] = []

    for i, (eval_dir, label) in enumerate(zip(eval_dirs, labels)):
        metrics = _load_json(eval_dir / "metrics.json")
        loc     = _load_json(loc_dirs[i] / "localization_summary.json") if loc_dirs[i] else None
        sys_s   = _load_json(sys_dirs[i] / "full_system_summary.json")  if sys_dirs[i] else None
        cv      = _load_json(cv_dirs[i]  / "cv_summary.json")           if cv_dirs[i] else None

        lines.append(f"## Scenario: {label}")
        lines.append("")

        if cv:
            lines.extend(section_cv_summary(cv, label))

        if metrics:
            metrics_list.append(metrics)
            lines.extend(section_anomaly_detection(metrics, label))

        if loc:
            lines.extend(section_localization(loc, label))

        if sys_s:
            lines.extend(section_full_system(sys_s, label))

    if len(metrics_list) > 1:
        lines.append("---")
        lines.append("")
        lines.extend(section_comparison(metrics_list, labels))

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SpecMAE performance analysis and report generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Single-scenario convenience
    p.add_argument("--eval_dir",   default=None,
                   help="Single evaluation output dir (contains metrics.json)")
    p.add_argument("--loc_dir",    default=None)
    p.add_argument("--sys_dir",    default=None)
    p.add_argument("--cv_dir",     default=None)

    # Multi-scenario
    p.add_argument("--eval_dirs",  nargs="+", default=None)
    p.add_argument("--loc_dirs",   nargs="+", default=None)
    p.add_argument("--sys_dirs",   nargs="+", default=None)
    p.add_argument("--cv_dirs",    nargs="+", default=None)
    p.add_argument("--labels",     nargs="+", default=None)

    p.add_argument("--out_report", default=None,
                   help="Save Markdown report to this path (default: print to stdout)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Normalise to lists
    if args.eval_dirs:
        eval_dirs = [Path(d) for d in args.eval_dirs]
        labels    = args.labels or [d.name for d in eval_dirs]
        loc_dirs  = [Path(d) if d else None for d in (args.loc_dirs  or [None]*len(eval_dirs))]
        sys_dirs  = [Path(d) if d else None for d in (args.sys_dirs  or [None]*len(eval_dirs))]
        cv_dirs   = [Path(d) if d else None for d in (args.cv_dirs   or [None]*len(eval_dirs))]
    elif args.eval_dir:
        eval_dirs = [Path(args.eval_dir)]
        labels    = [args.labels[0] if args.labels else Path(args.eval_dir).name]
        loc_dirs  = [Path(args.loc_dir)  if args.loc_dir  else None]
        sys_dirs  = [Path(args.sys_dir)  if args.sys_dir  else None]
        cv_dirs   = [Path(args.cv_dir)   if args.cv_dir   else None]
    else:
        print("ERROR: provide --eval_dir or --eval_dirs", file=sys.stderr)
        sys.exit(1)

    report = build_report(eval_dirs, labels, loc_dirs, sys_dirs, cv_dirs)

    if args.out_report:
        Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_report, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report written to {args.out_report}")
    else:
        print(report)


if __name__ == "__main__":
    main()
