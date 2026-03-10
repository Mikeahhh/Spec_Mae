"""
Generate synthetic forest ambient audio clips.

Produces 1-second WAV clips at 48 kHz that simulate a forest soundscape:
  - Brown noise base (wind through foliage)
  - Random chirp bursts (bird-like calls)
  - Continuous high-frequency buzz (insect sounds)
  - Leaf-rustling transients (filtered noise bursts)

Each clip is unique due to random parameters.

Usage:
    python Spec_Mae/scripts/data/generate_forest_ambient.py \
        --out_dir Spec_Mae/data/processed/forest_ambient \
        --n_clips 200
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf


SR = 48_000
DURATION = 1.0
N_SAMPLES = int(SR * DURATION)


def brown_noise(n: int) -> np.ndarray:
    """Generate brown noise via cumulative sum of white noise."""
    white = np.random.randn(n)
    brown = np.cumsum(white)
    # High-pass to remove DC drift
    brown -= np.mean(brown)
    # Normalize
    mx = np.abs(brown).max()
    if mx > 0:
        brown /= mx
    return brown


def chirp_burst(n: int, f_lo: float, f_hi: float, dur_samples: int) -> np.ndarray:
    """Generate a single frequency-modulated chirp burst (bird-like)."""
    t = np.arange(dur_samples) / SR
    freq = np.linspace(f_lo, f_hi, dur_samples)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    chirp = np.sin(phase)
    # Apply Hann envelope
    chirp *= np.hanning(dur_samples)
    # Embed in full-length signal
    out = np.zeros(n)
    start = np.random.randint(0, max(1, n - dur_samples))
    out[start:start + dur_samples] = chirp
    return out


def add_bird_calls(n: int, n_calls: int = 5) -> np.ndarray:
    """Add multiple random bird-like chirps."""
    out = np.zeros(n)
    for _ in range(n_calls):
        f_lo = np.random.uniform(2000, 4000)
        f_hi = np.random.uniform(f_lo, min(f_lo + 3000, 8000))
        dur_ms = np.random.uniform(30, 150)
        dur_samples = int(SR * dur_ms / 1000)
        amplitude = np.random.uniform(0.05, 0.25)
        out += amplitude * chirp_burst(n, f_lo, f_hi, dur_samples)
    return out


def insect_buzz(n: int) -> np.ndarray:
    """Generate continuous high-frequency insect buzz."""
    t = np.arange(n) / SR
    # Multiple harmonics around 4-8 kHz
    freq = np.random.uniform(4000, 6000)
    buzz = np.zeros(n)
    for harmonic in range(1, 4):
        amp = 0.02 / harmonic
        f = freq * harmonic
        if f > SR / 2:
            break
        buzz += amp * np.sin(2 * np.pi * f * t + np.random.uniform(0, 2 * np.pi))
    # Amplitude modulation (slow flutter)
    mod_freq = np.random.uniform(5, 20)
    mod = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t + np.random.uniform(0, 2 * np.pi))
    buzz *= mod
    return buzz


def leaf_rustling(n: int, n_events: int = 8) -> np.ndarray:
    """Generate intermittent leaf-rustling transients."""
    out = np.zeros(n)
    for _ in range(n_events):
        dur_ms = np.random.uniform(20, 100)
        dur_samples = int(SR * dur_ms / 1000)
        # Bandpass noise (1-6 kHz range)
        noise = np.random.randn(dur_samples)
        # Simple low-pass via moving average
        kernel_size = max(1, int(SR / 6000))
        if kernel_size > 1:
            kernel = np.ones(kernel_size) / kernel_size
            noise = np.convolve(noise, kernel, mode='same')
        noise *= np.hanning(dur_samples)
        amplitude = np.random.uniform(0.02, 0.10)
        noise *= amplitude
        start = np.random.randint(0, max(1, n - dur_samples))
        end = min(start + dur_samples, n)
        out[start:end] += noise[:end - start]
    return out


def generate_one_clip(rng_seed: int | None = None) -> np.ndarray:
    """Generate a single 1-second forest ambient clip."""
    if rng_seed is not None:
        np.random.seed(rng_seed)

    # Brown noise base (wind through trees)
    wind = brown_noise(N_SAMPLES)
    wind_level = np.random.uniform(0.15, 0.35)
    signal = wind * wind_level

    # Bird calls (random count 2-8)
    n_birds = np.random.randint(2, 9)
    signal += add_bird_calls(N_SAMPLES, n_calls=n_birds)

    # Insect buzz (sometimes present)
    if np.random.random() > 0.3:
        signal += insect_buzz(N_SAMPLES)

    # Leaf rustling
    n_events = np.random.randint(3, 12)
    signal += leaf_rustling(N_SAMPLES, n_events=n_events)

    # Add subtle white noise floor
    signal += 0.005 * np.random.randn(N_SAMPLES)

    # Normalize to prevent clipping
    mx = np.abs(signal).max()
    if mx > 0.99:
        signal = signal * 0.95 / mx

    return signal.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic forest ambient audio")
    parser.add_argument("--out_dir", default="Spec_Mae/data/processed/forest_ambient",
                        help="Output directory")
    parser.add_argument("--n_clips", type=int, default=200,
                        help="Number of 1-second clips to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed for reproducibility")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n_clips} forest ambient clips → {out_dir}")

    for i in range(args.n_clips):
        clip = generate_one_clip(rng_seed=args.seed + i)
        fname = f"forest_ambient_{i:04d}.wav"
        sf.write(str(out_dir / fname), clip, SR)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{args.n_clips}")

    print(f"Done. {args.n_clips} clips saved to {out_dir}")


if __name__ == "__main__":
    main()
