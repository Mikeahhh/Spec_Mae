"""
Desert scenario: generate training and test mixtures.

Training/Validation (normal only):
    drone_noise + desert_ambient  →  data/desert/train/ & val/

Test (normal + anomaly):
    normal:  drone + ambient      →  data/desert/test/normal/
    anomaly: drone + ambient + human_voice at SNR ∈ {-10,-5,0,5,10,15,20} dB
                                  →  data/desert/test/anomaly/snr_{X}dB/

Mixing levels:
    - Drone  : normalize to -20 dBFS
    - Ambient: normalize to -25 dBFS  (≈ 5 dB below drone, realistic background)
    - Human  : added at specified SNR relative to the final background (drone+ambient)
"""

import os
import random
import numpy as np
import soundfile as sf
from pathlib import Path

# ── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────
PROC_DIR = Path("E:/model_train_example/Spec_Mae/data/processed")
OUT_DIR  = Path("E:/model_train_example/Spec_Mae/data/desert")

DRONE_DIR   = PROC_DIR / "drone_noise"
AMBIENT_DIR = PROC_DIR / "desert_ambient"
VOICE_DIRS  = [
    PROC_DIR / "human_voice" / "Child_Cry_400_600Hz",
    PROC_DIR / "human_voice" / "Male_Rescue_100_300Hz",
]

# ── Config ─────────────────────────────────────────────────────────────────
TARGET_SR = 48000
TARGET_SAMPLES = 48000

# Split sizes
N_TRAIN = 1000
N_VAL   = 200
N_TEST_NORMAL  = 200
N_TEST_ANOMALY_PER_SNR = 30
TEST_SNRS = [-10, -5, 0, 5, 10, 15, 20]  # dB


# ── Audio utilities ────────────────────────────────────────────────────────

def load(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path))
    assert sr == TARGET_SR and len(audio) == TARGET_SAMPLES
    return audio.astype(np.float32)


def rms(audio: np.ndarray) -> float:
    """Global RMS over the entire clip. Used for continuous signals (drone, ambient)."""
    r = np.sqrt(np.mean(audio ** 2))
    return r if r > 1e-10 else 1e-10


def active_rms(audio: np.ndarray, frame_len: int = 512, top_frac: float = 0.3) -> float:
    """
    Active-segment RMS: compute RMS over the most energetic frames only.

    For signals with significant silence (e.g. short human voice clips in a
    1-second window), global RMS under-estimates the power of the speech-active
    portion.  If a clip contains 0.2 s of speech and 0.8 s of silence, global
    RMS is ~7 dB lower than the true active-speech level, making the SNR label
    7 dB optimistic.

    This function splits the clip into frames of `frame_len` samples, ranks
    them by energy, and returns the RMS of the top `top_frac` fraction.
    For a continuous noise signal (all frames active) the result converges to
    the global RMS, so it is safe to use on any signal type.

    Args:
        audio:     1-D float32 array.
        frame_len: Frame size in samples (default 512 ≈ 10.7 ms @ 48 kHz).
        top_frac:  Fraction of highest-energy frames to include (default 0.3).

    Returns:
        Active-segment RMS as a positive float.
    """
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return rms(audio)

    frames     = audio[:n_frames * frame_len].reshape(n_frames, frame_len)
    frame_rms  = np.sqrt(np.mean(frames ** 2, axis=1))   # (n_frames,)
    n_active   = max(1, int(n_frames * top_frac))
    active     = np.sort(frame_rms)[-n_active:]           # top energetic frames
    r = float(np.sqrt(np.mean(active ** 2)))
    return r if r > 1e-10 else 1e-10


def normalize_to_dbfs(audio: np.ndarray, target_dbfs: float) -> np.ndarray:
    """Scale audio so its global RMS equals target_dbfs (dBFS, e.g. -20.0)."""
    target_rms = 10 ** (target_dbfs / 20.0)
    return audio * (target_rms / rms(audio))


def mix_snr(background: np.ndarray, signal: np.ndarray,
            snr_db: float) -> np.ndarray:
    """
    Add `signal` to `background` at a given SNR.

    SNR is defined relative to the **active-segment** power of the signal,
    not the global RMS.  This ensures that the SNR label reflects the
    perceptual loudness of the signal during its active portion rather than
    being diluted by surrounding silence.

    Background (drone + ambient) is continuous, so its power is measured with
    the standard global RMS.  Signal (e.g. human voice) may contain silence,
    so its power is measured with active_rms().

    SNR = 20 * log10( active_RMS(signal_scaled) / RMS(background) )
    """
    bg_rms         = rms(background)
    sig_active_rms = active_rms(signal)
    target_sig_rms = bg_rms * (10 ** (snr_db / 20.0))
    signal_scaled  = signal * (target_sig_rms / sig_active_rms)
    return background + signal_scaled


def peak_normalize(audio: np.ndarray, headroom_db: float = -1.0) -> np.ndarray:
    """Peak-normalize to avoid clipping."""
    peak = np.max(np.abs(audio))
    if peak < 1e-10:
        return audio
    limit = 10 ** (headroom_db / 20.0)
    if peak > limit:
        audio = audio * (limit / peak)
    return audio


def save(audio: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, TARGET_SR, subtype="PCM_16")


# ── Core mix functions ─────────────────────────────────────────────────────

def make_background(drone_files, ambient_files) -> np.ndarray:
    """Pick one drone + one ambient, mix at fixed levels."""
    drone   = normalize_to_dbfs(load(random.choice(drone_files)),   -20.0)
    ambient = normalize_to_dbfs(load(random.choice(ambient_files)), -25.0)
    return drone + ambient          # combined background


def generate_normal(drone_files, ambient_files, out_dir: Path, n: int, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        bg = peak_normalize(make_background(drone_files, ambient_files))
        save(bg, out_dir / f"normal_{i:05d}.wav")
    print(f"  {tag}: {n} files → {out_dir}")


def generate_anomaly(drone_files, ambient_files, voice_files,
                     out_dir: Path, snr_db: float, n: int, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        bg    = make_background(drone_files, ambient_files)
        voice = load(random.choice(voice_files))
        mixed = mix_snr(bg, voice, snr_db)
        mixed = peak_normalize(mixed)
        save(mixed, out_dir / f"anomaly_snr{snr_db:+.0f}dB_{i:05d}.wav")
    print(f"  {tag} SNR={snr_db:+.0f}dB: {n} files → {out_dir}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Desert Scenario: Generating Mixtures")
    print("=" * 60)

    # Load file lists
    drone_files   = sorted(DRONE_DIR.glob("*.wav"))
    ambient_files = sorted(AMBIENT_DIR.glob("*.wav"))
    voice_files   = []
    for vd in VOICE_DIRS:
        voice_files += sorted(vd.glob("*.wav"))

    print(f"\nSource files:")
    print(f"  drone  : {len(drone_files)}")
    print(f"  ambient: {len(ambient_files)}")
    print(f"  voice  : {len(voice_files)}")

    # ── Training split ──────────────────────────────────────────────────
    print(f"\n[1/3] Training set ({N_TRAIN} normal)")
    generate_normal(drone_files, ambient_files,
                    OUT_DIR / "train" / "normal", N_TRAIN, "train/normal")

    # ── Validation split ────────────────────────────────────────────────
    print(f"\n[2/3] Validation set ({N_VAL} normal)")
    generate_normal(drone_files, ambient_files,
                    OUT_DIR / "val" / "normal", N_VAL, "val/normal")

    # ── Test split ──────────────────────────────────────────────────────
    print(f"\n[3/3] Test set")
    generate_normal(drone_files, ambient_files,
                    OUT_DIR / "test" / "normal", N_TEST_NORMAL, "test/normal")

    for snr in TEST_SNRS:
        tag = f"snr_{snr:+.0f}dB".replace("+", "+").replace("-", "-")
        generate_anomaly(
            drone_files, ambient_files, voice_files,
            OUT_DIR / "test" / "anomaly" / f"snr_{snr:+.0f}dB",
            snr_db=snr,
            n=N_TEST_ANOMALY_PER_SNR,
            tag="test/anomaly"
        )

    # ── Summary ─────────────────────────────────────────────────────────
    total = sum(1 for _ in OUT_DIR.rglob("*.wav"))
    print(f"\n{'=' * 60}")
    print(f"Done. Total files: {total}")
    print(f"\nDataset structure:")
    for split in ["train", "val", "test"]:
        sp = OUT_DIR / split
        if sp.exists():
            counts = {}
            for f in sp.rglob("*.wav"):
                key = str(f.parent.relative_to(sp))
                counts[key] = counts.get(key, 0) + 1
            for k, v in sorted(counts.items()):
                print(f"  desert/{split}/{k}: {v} files")
    print("=" * 60)


if __name__ == "__main__":
    main()
