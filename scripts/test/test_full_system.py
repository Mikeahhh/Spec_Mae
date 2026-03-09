"""
End-to-End UAV Acoustic System Simulation.

Simulates the complete Sound-UAV pipeline:

    1. Sentinel (SpecMAE on single mic) detects acoustic anomaly
    2. Responder triggered: 2-channel GCC-PHAT estimates DOA
    3. UAV navigates toward estimated direction (simple proportional controller)
    4. Metrics: detection latency, final DOA error, navigation error

Scenario
--------
- UAV starts at a random 3-D position within a cube
- A stationary human distress signal is placed at a fixed target
- The drone flies along a simulated trajectory (waypoints at 1 s intervals)
- At each 1-s window the Sentinel SpecMAE scores the ambient + signal mix
- When score > threshold, Responder activates and estimates DOA
- DOA is updated each second until the UAV reaches within 2 m of the target

Usage
-----
    cd E:/model_train_example
    python Spec_Mae/scripts/test/test_full_system.py \
        --checkpoint Spec_Mae/results/cv_desert/best_model.pth \
        --data_dir   Spec_Mae/data/desert \
        --out_dir    Spec_Mae/results/full_system \
        --n_episodes 100
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

# ── Project path ─────────────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent
_SPEC    = _HERE.parent.parent
_PROJECT = _SPEC.parent
sys.path.insert(0, str(_PROJECT))

from Spec_Mae.models.specmae import SpecMAE, specmae_vit_base_patch16
from Spec_Mae.scripts.utils.feature_extraction import AudioConfig, LogMelExtractor
from Spec_Mae.scripts.test.test_localization import gcc_phat, tdoa_to_azimuth, SPEED_OF_SOUND
from Spec_Mae.scripts.utils.device import get_device, print_device_diagnostics

try:
    import soundfile as sf
    _SF = True
except ImportError:
    _SF = False


# ═══════════════════════════════════════════════════════════════════════════
#  Model loading (reuse from test_anomaly_detection logic)
# ═══════════════════════════════════════════════════════════════════════════

def load_model(
    ckpt_path: Path,
    device:    torch.device,
) -> tuple[SpecMAE, float, AudioConfig]:
    ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
    mask_ratio = float(ckpt.get("mask_ratio", 0.75))
    cfg_dict   = ckpt.get("audio_cfg", {})
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
#  Audio helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_wav(path: Path, fs: int) -> np.ndarray:
    if _SF:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    else:
        raise ImportError("soundfile is required.")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != fs:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=fs)
        except ImportError:
            pass
    n = fs
    return audio[:n] if len(audio) >= n else np.pad(audio, (0, n - len(audio)))


def mix_at_snr(signal: np.ndarray, ambient: np.ndarray, snr_db: float) -> np.ndarray:
    sig_rms = np.sqrt(np.mean(signal ** 2)) + 1e-12
    amb_rms = np.sqrt(np.mean(ambient ** 2)) + 1e-12
    scale   = sig_rms / (amb_rms * 10 ** (snr_db / 20.0))
    return signal + ambient * scale


# ═══════════════════════════════════════════════════════════════════════════
#  Sentinel: anomaly detection wrapper
# ═══════════════════════════════════════════════════════════════════════════

class Sentinel:
    """Wraps SpecMAE anomaly scorer for 1-s audio windows."""

    def __init__(
        self,
        model:      SpecMAE,
        extractor:  LogMelExtractor,
        mask_ratio: float,
        threshold:  float,
        device:     torch.device,
    ) -> None:
        self.model      = model
        self.extractor  = extractor
        self.mask_ratio = mask_ratio
        self.threshold  = threshold
        self.device     = device

    @torch.no_grad()
    def score(self, audio: np.ndarray) -> float:
        spec  = self.extractor.extract(audio).unsqueeze(0).to(self.device)
        score = self.model.compute_anomaly_score(spec, mask_ratio=self.mask_ratio)
        return float(score.item())

    def detect(self, audio: np.ndarray) -> tuple[bool, float]:
        s = self.score(audio)
        return s > self.threshold, s


# ═══════════════════════════════════════════════════════════════════════════
#  Responder: GCC-PHAT DOA estimator
# ═══════════════════════════════════════════════════════════════════════════

class Responder:
    """2-mic GCC-PHAT DOA estimator."""

    def __init__(self, mic_dist: float, fs: int) -> None:
        self.mic_dist = mic_dist
        self.fs       = fs

    def estimate_doa(
        self,
        ch1: np.ndarray,
        ch2: np.ndarray,
    ) -> float:
        max_tau = self.mic_dist / SPEED_OF_SOUND + 1e-6
        tau_est, _, _ = gcc_phat(ch1, ch2, self.fs, max_tau=max_tau, interp=4)
        return tdoa_to_azimuth(tau_est, self.mic_dist)


# ═══════════════════════════════════════════════════════════════════════════
#  Navigation: simple proportional controller in 2-D
# ═══════════════════════════════════════════════════════════════════════════

class Navigator:
    """
    Simulated UAV navigator.

    The UAV moves toward the estimated DOA direction each time step.
    Position is 2-D (x, y).  Azimuth 0 = +x axis, 90 = +y axis.
    """

    def __init__(self, start: np.ndarray, speed: float = 5.0) -> None:
        """
        Args:
            start: (2,) array, initial position in metres.
            speed: UAV speed in m/s.
        """
        self.pos   = np.array(start, dtype=float)
        self.speed = speed

    def step(self, doa_deg: float, dt: float = 1.0) -> np.ndarray:
        """Move one time step toward doa_deg. Returns new position."""
        heading  = np.radians(doa_deg)
        delta    = np.array([np.cos(heading), np.sin(heading)]) * self.speed * dt
        self.pos = self.pos + delta
        return self.pos.copy()

    def distance_to(self, target: np.ndarray) -> float:
        return float(np.linalg.norm(self.pos - target))


# ═══════════════════════════════════════════════════════════════════════════
#  Single-episode simulation
# ═══════════════════════════════════════════════════════════════════════════

def run_episode(
    sentinel:       Sentinel,
    responder:      Responder,
    signal_files:   list[Path],
    ambient_files:  list[Path],
    fs:             int,
    snr_db:         float,
    mic_dist:       float,
    rng:            np.random.Generator,
    max_steps:      int = 30,
    goal_radius:    float = 2.0,
) -> dict:
    """
    Simulate one episode.

    Returns a dict with:
        detected        — bool: was anomaly detected?
        detection_step  — int:  step at which detection occurred (-1 if never)
        final_dist_m    — float: distance to target at end
        goal_reached    — bool: UAV came within goal_radius
        n_steps         — int:  total steps taken
        snr_db          — float
        azimuth_gt      — float: ground-truth source azimuth
        doa_errors      — list[float]: |doa_est - doa_gt| per Responder step
    """
    # Random target azimuth (gt)
    azimuth_gt = float(rng.uniform(-75.0, 75.0))

    # UAV starts 50 m from origin; target at origin
    start_angle  = float(rng.uniform(0, 360))
    start_dist   = float(rng.uniform(30.0, 80.0))
    uav_start    = np.array([
        start_dist * np.cos(np.radians(start_angle)),
        start_dist * np.sin(np.radians(start_angle)),
    ])
    target_pos = np.zeros(2)

    nav = Navigator(start=uav_start, speed=5.0)

    # Current true azimuth from UAV to target
    def current_azimuth() -> float:
        diff   = target_pos - nav.pos
        return float(np.degrees(np.arctan2(diff[1], diff[0])))

    sig_path = signal_files[int(rng.integers(0, len(signal_files)))]
    signal   = load_wav(sig_path, fs)

    detection_step = -1
    detected       = False
    doa_errors: list[float] = []

    for step in range(max_steps):
        amb_path = ambient_files[int(rng.integers(0, len(ambient_files)))]
        ambient  = load_wav(amb_path, fs)

        # Mix for Sentinel (single channel)
        mixed = mix_at_snr(signal, ambient, snr_db)
        is_anom, score = sentinel.detect(mixed)

        if is_anom and not detected:
            detected       = True
            detection_step = step

        if detected:
            # Generate 2-ch signal for Responder
            tau_gt = np.sin(np.radians(current_azimuth())) * mic_dist / SPEED_OF_SOUND
            tau_s  = int(round(tau_gt * fs))
            ch1    = mixed
            ch2    = np.roll(signal, tau_s) + ambient * (np.sqrt(np.mean(ambient**2)) /
                     (np.sqrt(np.mean(signal**2)) + 1e-12) * 10 ** (-snr_db / 20.0))
            doa_est = responder.estimate_doa(ch1, ch2)
            doa_err = abs(current_azimuth() - doa_est)
            doa_errors.append(doa_err)

            nav.step(doa_est, dt=1.0)
        else:
            # Blind fly toward center (fallback)
            nav.step(current_azimuth(), dt=1.0)

        if nav.distance_to(target_pos) <= goal_radius:
            break

    return {
        "detected":       detected,
        "detection_step": detection_step,
        "final_dist_m":   nav.distance_to(target_pos),
        "goal_reached":   nav.distance_to(target_pos) <= goal_radius,
        "n_steps":        step + 1,
        "snr_db":         snr_db,
        "azimuth_gt":     azimuth_gt,
        "mean_doa_err":   float(np.mean(doa_errors)) if doa_errors else float("nan"),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Full experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_full_system_test(
    sentinel:      Sentinel,
    responder:     Responder,
    signal_files:  list[Path],
    ambient_files: list[Path],
    fs:            int,
    mic_dist:      float,
    snr_levels:    list[int],
    n_episodes:    int,
    rng:           np.random.Generator,
) -> list[dict]:
    results: list[dict] = []

    for snr in snr_levels:
        print(f"  SNR={snr:+3d} dB  ({n_episodes} episodes) ...", end="", flush=True)
        snr_results: list[dict] = []

        for _ in range(n_episodes):
            ep = run_episode(
                sentinel, responder, signal_files, ambient_files,
                fs, float(snr), mic_dist, rng,
            )
            snr_results.append(ep)
            results.append(ep)

        det_rate  = np.mean([r["detected"]     for r in snr_results])
        goal_rate = np.mean([r["goal_reached"] for r in snr_results])
        doa_errs  = [r["mean_doa_err"] for r in snr_results if not np.isnan(r["mean_doa_err"])]
        doa_str   = f"{np.mean(doa_errs):.1f}" if doa_errs else "N/A"
        print(
            f"  det={det_rate:.0%}  goal={goal_rate:.0%}  doa_err={doa_str}deg"
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Output helpers
# ═══════════════════════════════════════════════════════════════════════════

def summarize(results: list[dict]) -> dict:
    snr_levels = sorted({r["snr_db"] for r in results})
    summary: dict = {}
    for snr in snr_levels:
        rows       = [r for r in results if r["snr_db"] == snr]
        doa_errs   = [r["mean_doa_err"] for r in rows if not np.isnan(r["mean_doa_err"])]
        det_steps  = [r["detection_step"] for r in rows if r["detected"]]
        summary[f"snr_{int(snr):+d}dB"] = {
            "snr_db":             snr,
            "n_episodes":         len(rows),
            "detection_rate":     float(np.mean([r["detected"]     for r in rows])),
            "goal_reached_rate":  float(np.mean([r["goal_reached"] for r in rows])),
            "mean_final_dist_m":  float(np.mean([r["final_dist_m"] for r in rows])),
            "mean_detection_step": float(np.mean(det_steps)) if det_steps else float("nan"),
            "mean_doa_err_deg":   float(np.mean(doa_errs)) if doa_errs else float("nan"),
        }
    return summary


def save_outputs(results: list[dict], summary: dict, out_dir: Path) -> None:
    if results:
        csv_path = out_dir / "full_system_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
    json_path = out_dir / "full_system_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results: {out_dir / 'full_system_results.csv'}")
    print(f"  Summary: {json_path}")


def plot_system_summary(summary: dict, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = sorted(summary.values(), key=lambda x: x["snr_db"])
        if not rows:
            return
        snr_vals  = [r["snr_db"]             for r in rows]
        det_rates = [r["detection_rate"]      for r in rows]
        goal_rates= [r["goal_reached_rate"]   for r in rows]
        doa_errs  = [r["mean_doa_err_deg"]    for r in rows]
        dist_vals = [r["mean_final_dist_m"]   for r in rows]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(snr_vals, det_rates,  "o-", color="#1f77b4", linewidth=2, label="Detection rate")
        axes[0].plot(snr_vals, goal_rates, "s-", color="#2ca02c", linewidth=2, label="Goal reached")
        axes[0].set_xlabel("SNR (dB)")
        axes[0].set_ylabel("Rate")
        axes[0].set_title("Detection & Navigation Success")
        axes[0].set_ylim([0, 1.05])
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(snr_vals, [d if not np.isnan(d) else 0 for d in doa_errs],
                     "^-", color="#ff7f0e", linewidth=2)
        axes[1].set_xlabel("SNR (dB)")
        axes[1].set_ylabel("DOA Error (degrees)")
        axes[1].set_title("Mean DOA Error vs SNR")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(snr_vals, dist_vals, "D-", color="#9467bd", linewidth=2)
        axes[2].set_xlabel("SNR (dB)")
        axes[2].set_ylabel("Final Distance (m)")
        axes[2].set_title("Navigation Final Distance vs SNR")
        axes[2].grid(True, alpha=0.3)

        fig.suptitle("End-to-End UAV Acoustic System Performance", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / "full_system_summary.png", dpi=150)
        plt.close(fig)
        print(f"  Plot: full_system_summary.png")
    except Exception as exc:
        print(f"  [plot] Skipped: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end UAV acoustic system simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--data_dir",    default="Spec_Mae/data/desert")
    p.add_argument("--out_dir",     default="Spec_Mae/results/full_system")
    p.add_argument("--threshold",   type=float, default=None,
                   help="Anomaly score threshold (default: auto from 95th pct of normal scores)")
    p.add_argument("--mic_dist",    type=float, default=0.10)
    p.add_argument("--n_episodes",  type=int,   default=100)
    p.add_argument("--snr_levels",  nargs="+",  type=int,
                   default=[-10, -5, 0, 5, 10, 15, 20])
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = get_device(verbose=True)
    rng     = np.random.default_rng(args.seed)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    model, mask_ratio, cfg = load_model(ckpt_path, device)
    extractor = LogMelExtractor(cfg=cfg)

    data_dir     = Path(args.data_dir)
    normal_root  = data_dir / "test" / "normal"
    anomaly_root = data_dir / "test" / "anomaly"

    signal_files  = sorted(anomaly_root.rglob("*.wav"))
    ambient_files = sorted(normal_root.glob("*.wav"))

    if not signal_files or not ambient_files:
        print("ERROR: missing test audio files.")
        sys.exit(1)

    # ── Auto-threshold from normal scores ────────────────────────────────
    threshold = args.threshold
    if threshold is None:
        print("  Estimating anomaly threshold from normal clips...")
        normal_scores: list[float] = []
        with torch.no_grad():
            for p in ambient_files[:min(50, len(ambient_files))]:
                try:
                    audio = load_wav(p, cfg.sample_rate)
                    spec  = extractor.extract(audio).unsqueeze(0).to(device)
                    s     = model.compute_anomaly_score(spec, mask_ratio=mask_ratio)
                    normal_scores.append(float(s.item()))
                except Exception:
                    continue
        if normal_scores:
            threshold = float(np.percentile(normal_scores, 95))
            print(f"  Auto threshold (95th pct of normal): {threshold:.6f}")
        else:
            threshold = 1.0
            print(f"  WARNING: could not compute threshold, using {threshold}")

    sentinel  = Sentinel(model, extractor, mask_ratio, threshold, device)
    responder = Responder(mic_dist=args.mic_dist, fs=cfg.sample_rate)

    print("=" * 60)
    print(f"Full System Test")
    print(f"  Device     : {device}")
    print(f"  Threshold  : {threshold:.6f}")
    print(f"  Mic dist   : {args.mic_dist} m")
    print(f"  Episodes   : {args.n_episodes} per SNR")
    print("=" * 60)

    results = run_full_system_test(
        sentinel, responder,
        signal_files, ambient_files,
        fs=cfg.sample_rate,
        mic_dist=args.mic_dist,
        snr_levels=args.snr_levels,
        n_episodes=args.n_episodes,
        rng=rng,
    )

    summary = summarize(results)
    save_outputs(results, summary, out_dir)
    plot_system_summary(summary, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
