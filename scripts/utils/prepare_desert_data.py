"""
Desert scenario data preparation script.
Processes all raw audio to standard: 48kHz, 1.0s, mono WAV.

Output structure:
  data/processed/desert_ambient/   - 1s clips from desert background
  data/processed/drone_noise/      - drone noise resampled to 48kHz
  data/processed/human_voice/      - human voice clips at 48kHz
"""

import os
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────
TARGET_SR = 48000
TARGET_DURATION = 1.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)  # 48000

RAW_DIR = Path("E:/model_train_example/Spec_Mae/data/raw")
OUT_DIR = Path("E:/model_train_example/Spec_Mae/data/processed")


# ── Helpers ────────────────────────────────────────────────────────────────

def load_and_normalize(path: str) -> np.ndarray:
    """Load any audio file to mono float32 at TARGET_SR."""
    audio, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    return audio.astype(np.float32)


def save_wav(audio: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, TARGET_SR, subtype="PCM_16")


def slice_audio(audio: np.ndarray, min_length_ratio: float = 0.5):
    """
    Slice audio into TARGET_SAMPLES chunks.
    Last chunk: kept if >= min_length_ratio * TARGET_SAMPLES, zero-padded to full length.
    Returns list of numpy arrays, each of length TARGET_SAMPLES.
    """
    chunks = []
    total = len(audio)
    start = 0
    while start < total:
        end = start + TARGET_SAMPLES
        chunk = audio[start:end]
        if len(chunk) == TARGET_SAMPLES:
            chunks.append(chunk)
        elif len(chunk) >= int(TARGET_SAMPLES * min_length_ratio):
            # Pad with zeros
            padded = np.zeros(TARGET_SAMPLES, dtype=np.float32)
            padded[:len(chunk)] = chunk
            chunks.append(padded)
        # else: too short, discard
        start = end
    return chunks


def process_directory(raw_path: Path, out_path: Path, recursive: bool = False,
                       tag: str = ""):
    """
    Process all WAV/MP3 files in raw_path.
    Each file is loaded, resampled, then sliced into 1s clips.
    """
    out_path.mkdir(parents=True, exist_ok=True)

    if recursive:
        files = sorted(raw_path.rglob("*.wav")) + sorted(raw_path.rglob("*.mp3"))
    else:
        files = sorted(raw_path.glob("*.wav")) + sorted(raw_path.glob("*.mp3"))

    total_clips = 0
    skipped = 0

    for fp in files:
        try:
            audio = load_and_normalize(str(fp))
            chunks = slice_audio(audio)
            stem = fp.stem
            for i, chunk in enumerate(chunks):
                out_name = f"{stem}_{i:04d}.wav" if len(chunks) > 1 else f"{stem}.wav"
                save_wav(chunk, out_path / out_name)
                total_clips += 1
        except Exception as e:
            print(f"  [SKIP] {fp.name}: {e}")
            skipped += 1

    print(f"  {tag}: {len(files)} files → {total_clips} clips"
          + (f" ({skipped} skipped)" if skipped else ""))
    return total_clips


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Desert Data Preparation")
    print(f"Target: {TARGET_SR}Hz, {TARGET_DURATION}s, mono, PCM16")
    print("=" * 60)

    # 1. Desert ambient ────────────────────────────────────────────────────
    print("\n[1/3] Desert ambient noise")
    n = process_directory(
        raw_path=RAW_DIR / "desert_ambient",
        out_path=OUT_DIR / "desert_ambient",
        tag="desert_ambient"
    )

    # 2. Drone noise ───────────────────────────────────────────────────────
    print("\n[2/3] Drone noise")
    n = process_directory(
        raw_path=RAW_DIR / "drone_noise_original",
        out_path=OUT_DIR / "drone_noise",
        tag="drone_noise"
    )

    # 3. Human voice (anomaly) ─────────────────────────────────────────────
    print("\n[3/3] Human voice (anomaly signals)")
    total_voice = 0
    for sub in ["Child_Cry_400_600Hz", "Male_Rescue_100_300Hz"]:
        n = process_directory(
            raw_path=RAW_DIR / "human_voice" / sub,
            out_path=OUT_DIR / "human_voice" / sub,
            tag=sub
        )
        total_voice += n
    print(f"  Human voice total clips: {total_voice}")

    # 4. Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Processing complete. Output directory:")
    print(f"  {OUT_DIR}")
    for sub in OUT_DIR.iterdir():
        if sub.is_dir():
            count = sum(1 for _ in sub.rglob("*.wav"))
            print(f"  {sub.name:25s}: {count} clips")
    print("=" * 60)


if __name__ == "__main__":
    main()
