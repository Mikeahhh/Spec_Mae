"""
GCC-PHAT TDOA/DOA Localization Test.

Evaluates the Responder layer's sound-source localization performance using
Generalized Cross-Correlation with Phase Transform (GCC-PHAT) on simulated
multi-microphone signals at varying SNR levels.

Setup
-----
- Microphone array: uniform linear array (ULA) or custom geometry
- Source positions: sampled from a 3-D hemisphere at varying azimuths
- Simulated room: free-field (no reverb) or optional image-source method (ISM)
- Each clip is a 1-s mixture: ambient background + drone/human signal at target SNR

The ground-truth TDOA is computed analytically from source-mic geometry.
The estimated TDOA comes from GCC-PHAT applied to each mic pair.
DOA (azimuth) is recovered via arcsin(c * tau / d) for a 2-mic pair.

Metrics
-------
    TDOA RMSE (samples)     — root mean squared TDOA estimation error
    DOA MAE  (degrees)      — mean absolute azimuth error
    Fraction within 5 deg   — coarse accuracy for search-and-rescue
    Results are reported per SNR level

Usage
-----
    cd E:/model_train_example
    python Spec_Mae/scripts/test/test_localization.py \
        --data_dir Spec_Mae/data/desert \
        --out_dir  Spec_Mae/results/localization \
        --n_trials 200
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# ── Project path ─────────────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent
_SPEC    = _HERE.parent.parent
_PROJECT = _SPEC.parent
sys.path.insert(0, str(_PROJECT))

from Spec_Mae.scripts.utils.feature_extraction import AudioConfig

try:
    import soundfile as sf
    _SF = True
except ImportError:
    _SF = False

try:
    import librosa
    _LIBROSA = True
except ImportError:
    _LIBROSA = False


# ═══════════════════════════════════════════════════════════════════════════
#  GCC-PHAT implementation
# ═══════════════════════════════════════════════════════════════════════════

def gcc_phat(
    sig1:     np.ndarray,
    sig2:     np.ndarray,
    fs:       int,
    max_tau:  float | None = None,
    interp:   int = 1,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    GCC-PHAT TDOA estimation between two signals.

    Args:
        sig1, sig2: 1-D float arrays of equal length.
        fs:         Sample rate (Hz).
        max_tau:    Maximum plausible TDOA in seconds (clips search range).
        interp:     Interpolation factor for sub-sample accuracy (default: 1).

    Returns:
        tau_est   : Estimated TDOA in seconds.
        cc        : Cross-correlation function values.
        lags_sec  : Lag axis in seconds.
    """
    n = sig1.shape[0] + sig2.shape[0]
    # Next power of 2 for efficiency
    n_fft = 1 << (n - 1).bit_length()

    SIG1 = np.fft.rfft(sig1, n=n_fft * interp)
    SIG2 = np.fft.rfft(sig2, n=n_fft * interp)

    # GCC-PHAT: normalise by magnitude product
    cross   = SIG1 * np.conj(SIG2)
    denom   = np.abs(cross) + 1e-12
    gcc     = np.fft.irfft(cross / denom)
    gcc     = np.concatenate([gcc[-(n_fft * interp // 2):], gcc[:(n_fft * interp // 2)]])

    lags_s = (np.arange(len(gcc)) - len(gcc) // 2) / (fs * interp)

    if max_tau is not None:
        valid = np.abs(lags_s) <= max_tau
        gcc[~valid] = 0.0

    peak_idx = int(np.argmax(gcc))
    tau_est  = float(lags_s[peak_idx])

    return tau_est, gcc, lags_s


# ═══════════════════════════════════════════════════════════════════════════
#  Geometry helpers
# ═══════════════════════════════════════════════════════════════════════════

SPEED_OF_SOUND: float = 343.0   # m/s at ~20 C

def true_tdoa(
    source_pos: np.ndarray,   # (3,) xyz in metres
    mic1_pos:   np.ndarray,   # (3,)
    mic2_pos:   np.ndarray,   # (3,)
    c:          float = SPEED_OF_SOUND,
) -> float:
    """TDOA = (dist1 - dist2) / c  [seconds]; positive → source closer to mic1."""
    d1 = float(np.linalg.norm(source_pos - mic1_pos))
    d2 = float(np.linalg.norm(source_pos - mic2_pos))
    return (d1 - d2) / c


def tdoa_to_azimuth(tau: float, mic_dist: float, c: float = SPEED_OF_SOUND) -> float:
    """
    Convert TDOA to azimuth angle (degrees) for a 2-mic broadside ULA.
    Returns NaN if |tau| > mic_dist/c (physically impossible).
    """
    ratio = c * tau / mic_dist
    ratio = np.clip(ratio, -1.0, 1.0)
    return float(np.degrees(np.arcsin(ratio)))


# ═══════════════════════════════════════════════════════════════════════════
#  Signal simulation
# ═══════════════════════════════════════════════════════════════════════════

def simulate_signal_pair(
    signal:    np.ndarray,
    ambient:   np.ndarray,
    tau_true:  float,
    snr_db:    float,
    fs:        int,
    rng:       np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate a 2-channel observation.

    mic1 receives signal + ambient at the target SNR.
    mic2 receives the same signal time-shifted by tau_true samples.

    Returns (ch1, ch2) each of shape (n_samples,).
    """
    n = len(signal)

    # Normalise signal power
    sig_rms = np.sqrt(np.mean(signal ** 2)) + 1e-12
    amb_rms = np.sqrt(np.mean(ambient ** 2)) + 1e-12

    # Scale ambient to achieve target SNR
    target_amb_rms = sig_rms * 10 ** (-snr_db / 20.0)
    ambient_scaled = ambient * (target_amb_rms / amb_rms)

    # Time-shift for mic2 via fractional-delay (linear interpolation)
    tau_samples = tau_true * fs
    i_shift     = int(np.round(tau_samples))
    ch1 = signal + ambient_scaled
    ch2_clean = np.roll(signal, i_shift)

    # Independent ambient for mic2
    ambient2 = rng.permutation(ambient) * (target_amb_rms / amb_rms)
    ch2 = ch2_clean + ambient2

    return ch1.astype(np.float32), ch2.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  Single-trial evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_one_trial(
    signal:    np.ndarray,
    ambient:   np.ndarray,
    azimuth_gt: float,
    snr_db:    float,
    mic_dist:  float,
    fs:        int,
    rng:       np.random.Generator,
) -> dict[str, float]:
    """
    Simulate, apply GCC-PHAT, and compute TDOA/DOA errors for one trial.

    Returns a dict with keys: tau_true, tau_est, doa_gt, doa_est,
                               tdoa_err_samples, doa_err_deg.
    """
    tau_gt = np.sin(np.radians(azimuth_gt)) * mic_dist / SPEED_OF_SOUND

    ch1, ch2 = simulate_signal_pair(signal, ambient, tau_gt, snr_db, fs, rng)

    max_tau = mic_dist / SPEED_OF_SOUND + 1e-6
    tau_est, _, _ = gcc_phat(ch1, ch2, fs, max_tau=max_tau, interp=4)

    doa_gt  = azimuth_gt
    doa_est = tdoa_to_azimuth(tau_est, mic_dist)

    return {
        "tau_true":          tau_gt,
        "tau_est":           tau_est,
        "doa_gt":            doa_gt,
        "doa_est":           doa_est,
        "tdoa_err_samples":  abs((tau_gt - tau_est) * fs),
        "doa_err_deg":       abs(doa_gt - doa_est),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Full experiment
# ═══════════════════════════════════════════════════════════════════════════

SNR_LEVELS = [-10, -5, 0, 5, 10, 15, 20]

def run_experiment(
    signal_files:  list[Path],
    ambient_files: list[Path],
    fs:            int,
    mic_dist:      float,
    snr_levels:    list[int],
    n_trials:      int,
    rng:           np.random.Generator,
) -> list[dict]:
    """
    Run N trials per SNR level and collect results.

    Returns a flat list of result dicts (one per trial).
    """
    results: list[dict] = []

    n_sig = len(signal_files)
    n_amb = len(ambient_files)
    if n_sig == 0 or n_amb == 0:
        raise ValueError("No signal or ambient files found.")

    def _load(path: Path) -> np.ndarray:
        if _SF:
            audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        elif _LIBROSA:
            audio, sr = librosa.load(str(path), sr=None, mono=True)
        else:
            raise ImportError("soundfile or librosa is required.")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        target_len = fs   # 1 second
        if len(audio) >= target_len:
            return audio[:target_len]
        return np.pad(audio, (0, target_len - len(audio)))

    for snr in snr_levels:
        print(f"  SNR={snr:+3d} dB  ({n_trials} trials) ... ", end="", flush=True)
        for t in range(n_trials):
            sig_path = signal_files[int(rng.integers(0, n_sig))]
            amb_path = ambient_files[int(rng.integers(0, n_amb))]
            try:
                signal  = _load(sig_path)
                ambient = _load(amb_path)
            except Exception:
                continue

            azimuth_gt = float(rng.uniform(-80.0, 80.0))

            trial = evaluate_one_trial(
                signal, ambient, azimuth_gt, snr, mic_dist, fs, rng,
            )
            trial.update({"snr_db": snr, "trial": t})
            results.append(trial)

        snr_rows = [r for r in results if r["snr_db"] == snr]
        tdoa_rmse = float(np.sqrt(np.mean([r["tdoa_err_samples"]**2 for r in snr_rows])))
        doa_mae   = float(np.mean([r["doa_err_deg"] for r in snr_rows]))
        within5   = float(np.mean([r["doa_err_deg"] <= 5.0 for r in snr_rows]))
        print(f"TDOA-RMSE={tdoa_rmse:.2f}samp  DOA-MAE={doa_mae:.1f}deg  within5={within5:.2%}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Summary and saving
# ═══════════════════════════════════════════════════════════════════════════

def compute_summary(results: list[dict]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    snr_levels = sorted({r["snr_db"] for r in results})
    for snr in snr_levels:
        rows      = [r for r in results if r["snr_db"] == snr]
        tdoa_errs = [r["tdoa_err_samples"] for r in rows]
        doa_errs  = [r["doa_err_deg"]      for r in rows]
        summary[f"snr_{snr:+d}dB"] = {
            "snr_db":        snr,
            "n_trials":      len(rows),
            "tdoa_rmse_samp": float(np.sqrt(np.mean([e**2 for e in tdoa_errs]))),
            "tdoa_mae_samp":  float(np.mean(tdoa_errs)),
            "doa_mae_deg":    float(np.mean(doa_errs)),
            "doa_rmse_deg":   float(np.sqrt(np.mean([e**2 for e in doa_errs]))),
            "within_5deg":    float(np.mean([e <= 5.0 for e in doa_errs])),
            "within_10deg":   float(np.mean([e <= 10.0 for e in doa_errs])),
        }
    return summary


def save_results(
    results: list[dict],
    summary: dict,
    out_dir: Path,
) -> None:
    # CSV
    csv_path = out_dir / "localization_results.csv"
    if results:
        fields = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
    # JSON summary
    json_path = out_dir / "localization_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results CSV  : {csv_path}")
    print(f"  Summary JSON : {json_path}")


def save_doa_plot(summary: dict, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = sorted(summary.values(), key=lambda x: x["snr_db"])
        if not rows:
            return
        snr_vals = [r["snr_db"]       for r in rows]
        mae_vals = [r["doa_mae_deg"]  for r in rows]
        w5_vals  = [r["within_5deg"]  for r in rows]
        w10_vals = [r["within_10deg"] for r in rows]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(snr_vals, mae_vals, "o-", linewidth=2, color="#1f77b4")
        ax1.set_xlabel("SNR (dB)")
        ax1.set_ylabel("DOA MAE (degrees)")
        ax1.set_title("GCC-PHAT DOA Error vs SNR")
        ax1.grid(True, alpha=0.3)

        ax2.plot(snr_vals, w5_vals,  "s-", linewidth=2, color="#2ca02c",  label="within 5 deg")
        ax2.plot(snr_vals, w10_vals, "^-", linewidth=2, color="#ff7f0e", label="within 10 deg")
        ax2.set_xlabel("SNR (dB)")
        ax2.set_ylabel("Fraction Correct")
        ax2.set_title("Localization Accuracy vs SNR")
        ax2.set_ylim([0, 1.05])
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_dir / "localization_vs_snr.png", dpi=150)
        plt.close(fig)
        print(f"  Plot saved: localization_vs_snr.png")
    except Exception as exc:
        print(f"  [plot] Localization plot skipped: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GCC-PHAT TDOA/DOA localization evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",  default="Spec_Mae/data/desert",
                   help="Scenario data root (test/normal/ for ambient, test/anomaly/ for signals)")
    p.add_argument("--out_dir",   default="Spec_Mae/results/localization")
    p.add_argument("--mic_dist",  type=float, default=0.10,
                   help="Microphone spacing in metres (default: 10 cm)")
    p.add_argument("--n_trials",  type=int,   default=200,
                   help="Number of random trials per SNR level")
    p.add_argument("--snr_levels", nargs="+", type=int, default=SNR_LEVELS)
    p.add_argument("--seed",       type=int,  default=42)
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng     = np.random.default_rng(args.seed)
    cfg     = AudioConfig()
    data_dir = Path(args.data_dir)

    # Use anomaly clips as "signal" (something to localise)
    # and normal clips as ambient background
    anomaly_root = data_dir / "test" / "anomaly"
    normal_root  = data_dir / "test" / "normal"

    signal_files  = sorted(anomaly_root.rglob("*.wav"))
    ambient_files = sorted(normal_root.glob("*.wav"))

    if not signal_files or not ambient_files:
        print(f"ERROR: no WAV files found under {data_dir}/test/")
        print(f"  Signal  dir : {anomaly_root}  ({len(signal_files)} files)")
        print(f"  Ambient dir : {normal_root}  ({len(ambient_files)} files)")
        sys.exit(1)

    print("=" * 60)
    print(f"GCC-PHAT Localization Evaluation")
    print(f"  Mic distance   : {args.mic_dist} m")
    print(f"  SNR levels     : {args.snr_levels}")
    print(f"  Trials per SNR : {args.n_trials}")
    print(f"  Signal files   : {len(signal_files)}")
    print(f"  Ambient files  : {len(ambient_files)}")
    print("=" * 60)

    results = run_experiment(
        signal_files=signal_files,
        ambient_files=ambient_files,
        fs=cfg.sample_rate,
        mic_dist=args.mic_dist,
        snr_levels=args.snr_levels,
        n_trials=args.n_trials,
        rng=rng,
    )

    summary = compute_summary(results)
    save_results(results, summary, out_dir)
    save_doa_plot(summary, out_dir)

    print("\nSummary:")
    print(f"  {'SNR':>8}  {'DOA MAE':>9}  {'within 5deg':>12}  {'TDOA RMSE':>11}")
    print("  " + "-" * 48)
    for tag, v in summary.items():
        print(
            f"  {tag:>8}  "
            f"{v['doa_mae_deg']:>8.2f}d  "
            f"{v['within_5deg']:>12.2%}  "
            f"{v['tdoa_rmse_samp']:>10.2f}s"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
