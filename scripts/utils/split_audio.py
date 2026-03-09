"""
切分长音频为1秒片段
用于处理从FreeSound下载的长音频文件
"""

import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm


def split_audio_to_segments(
    input_file: str,
    output_dir: str,
    segment_length: float = 1.0,
    target_sr: int = 48000,
    overlap: float = 0.0,
    min_segment_length: float = 0.5
):
    """
    将长音频切分成固定长度的片段

    Args:
        input_file: 输入音频文件路径
        output_dir: 输出目录
        segment_length: 片段长度（秒）
        target_sr: 目标采样率
        overlap: 重叠比例（0-1之间）
        min_segment_length: 最小片段长度（秒），小于此长度的片段会被丢弃
    """
    # 加载音频
    print(f"Loading audio: {input_file}")
    audio, orig_sr = librosa.load(input_file, sr=None, mono=True)

    # 重采样
    if orig_sr != target_sr:
        print(f"Resampling from {orig_sr} Hz to {target_sr} Hz...")
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    # 计算参数
    total_duration = len(audio) / target_sr
    segment_samples = int(segment_length * target_sr)
    hop_samples = int(segment_samples * (1 - overlap))

    print(f"Total duration: {total_duration:.2f} seconds")
    print(f"Segment length: {segment_length} seconds")
    print(f"Overlap: {overlap * 100:.1f}%")
    print(f"Hop size: {hop_samples / target_sr:.2f} seconds")

    # 计算片段数量
    n_segments = (len(audio) - segment_samples) // hop_samples + 1
    print(f"Expected segments: {n_segments}")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 获取基础文件名
    base_name = Path(input_file).stem

    # 切分音频
    saved_count = 0
    for i in tqdm(range(n_segments), desc="Splitting"):
        start = i * hop_samples
        end = start + segment_samples

        # 检查是否超出范围
        if end > len(audio):
            # 最后一个片段
            segment = audio[start:]

            # 检查是否太短
            if len(segment) / target_sr < min_segment_length:
                print(f"Skipping last segment (too short: {len(segment)/target_sr:.2f}s)")
                break

            # 填充到目标长度
            if len(segment) < segment_samples:
                segment = np.pad(segment, (0, segment_samples - len(segment)), mode='constant')
        else:
            segment = audio[start:end]

        # 保存
        output_file = output_path / f"{base_name}_seg{i:04d}.wav"
        sf.write(output_file, segment, target_sr)
        saved_count += 1

    print(f"\nDone! Saved {saved_count} segments to {output_dir}")
    return saved_count


def batch_split_directory(
    input_dir: str,
    output_dir: str,
    segment_length: float = 1.0,
    target_sr: int = 48000,
    overlap: float = 0.0,
    pattern: str = "*.wav"
):
    """
    批量切分目录中的所有音频文件

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        segment_length: 片段长度（秒）
        target_sr: 目标采样率
        overlap: 重叠比例
        pattern: 文件匹配模式
    """
    input_path = Path(input_dir)
    audio_files = list(input_path.glob(pattern))

    if len(audio_files) == 0:
        print(f"No files found matching {pattern} in {input_dir}")
        return

    print(f"Found {len(audio_files)} files to process")
    print("-" * 60)

    total_segments = 0

    for audio_file in audio_files:
        print(f"\nProcessing: {audio_file.name}")
        print("-" * 60)

        try:
            n_segments = split_audio_to_segments(
                input_file=str(audio_file),
                output_dir=output_dir,
                segment_length=segment_length,
                target_sr=target_sr,
                overlap=overlap
            )
            total_segments += n_segments
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")

    print("\n" + "=" * 60)
    print(f"Total segments created: {total_segments}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split long audio files into fixed-length segments"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file or directory"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory"
    )

    parser.add_argument(
        "--segment_length",
        type=float,
        default=1.0,
        help="Segment length in seconds (default: 1.0)"
    )

    parser.add_argument(
        "--target_sr",
        type=int,
        default=48000,
        help="Target sample rate (default: 48000)"
    )

    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="Overlap ratio between segments (0-1, default: 0.0)"
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all files in input directory"
    )

    args = parser.parse_args()

    if args.batch:
        # 批量处理目录
        batch_split_directory(
            input_dir=args.input,
            output_dir=args.output,
            segment_length=args.segment_length,
            target_sr=args.target_sr,
            overlap=args.overlap
        )
    else:
        # 处理单个文件
        split_audio_to_segments(
            input_file=args.input,
            output_dir=args.output,
            segment_length=args.segment_length,
            target_sr=args.target_sr,
            overlap=args.overlap
        )
