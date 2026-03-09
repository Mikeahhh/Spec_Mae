"""
批量转换音频采样率
将16kHz的无人机噪音转换为48kHz
"""

import os
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import argparse


def convert_sample_rate(
    input_file: str,
    output_file: str,
    target_sr: int = 48000,
    verbose: bool = False
):
    """
    转换单个音频文件的采样率

    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        target_sr: 目标采样率
        verbose: 是否打印详细信息
    """
    try:
        # 加载音频
        audio, orig_sr = librosa.load(input_file, sr=None, mono=True)

        if verbose:
            print(f"Original SR: {orig_sr} Hz, Duration: {len(audio)/orig_sr:.3f}s")

        # 如果采样率已经是目标值，直接复制
        if orig_sr == target_sr:
            if verbose:
                print(f"Already at {target_sr} Hz, copying...")
            audio_resampled = audio
        else:
            # 重采样
            audio_resampled = librosa.resample(
                audio,
                orig_sr=orig_sr,
                target_sr=target_sr
            )
            if verbose:
                print(f"Resampled to {target_sr} Hz, Duration: {len(audio_resampled)/target_sr:.3f}s")

        # 保存
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        sf.write(output_file, audio_resampled, target_sr)

        if verbose:
            print(f"Saved: {output_file}")

        return True

    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return False


def batch_convert_sample_rate(
    input_dir: str,
    output_dir: str,
    target_sr: int = 48000,
    pattern: str = "*.wav"
):
    """
    批量转换目录中的所有音频文件

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        target_sr: 目标采样率
        pattern: 文件匹配模式
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 获取所有音频文件
    audio_files = list(input_path.glob(pattern))

    if len(audio_files) == 0:
        print(f"No files found matching {pattern} in {input_dir}")
        return

    print(f"Found {len(audio_files)} files to convert")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target sample rate: {target_sr} Hz")
    print("-" * 60)

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 批量转换
    success_count = 0
    fail_count = 0

    for input_file in tqdm(audio_files, desc="Converting"):
        # 构建输出文件路径
        output_file = output_path / input_file.name

        # 转换
        if convert_sample_rate(str(input_file), str(output_file), target_sr):
            success_count += 1
        else:
            fail_count += 1

    print("-" * 60)
    print(f"Conversion complete!")
    print(f"Success: {success_count} files")
    print(f"Failed: {fail_count} files")
    print(f"Output saved to: {output_dir}")


def check_sample_rates(directory: str, pattern: str = "*.wav"):
    """
    检查目录中所有音频文件的采样率

    Args:
        directory: 目录路径
        pattern: 文件匹配模式
    """
    dir_path = Path(directory)
    audio_files = list(dir_path.glob(pattern))

    if len(audio_files) == 0:
        print(f"No files found in {directory}")
        return

    print(f"Checking {len(audio_files)} files in {directory}")
    print("-" * 60)

    sample_rates = {}

    for audio_file in tqdm(audio_files, desc="Checking"):
        try:
            _, sr = librosa.load(str(audio_file), sr=None, duration=0.1)
            sample_rates[sr] = sample_rates.get(sr, 0) + 1
        except Exception as e:
            print(f"Error reading {audio_file}: {e}")

    print("-" * 60)
    print("Sample rate distribution:")
    for sr, count in sorted(sample_rates.items()):
        print(f"  {sr} Hz: {count} files ({count/len(audio_files)*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert audio sample rate"
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["convert", "check"],
        help="Mode: convert or check"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory (for convert mode)"
    )

    parser.add_argument(
        "--target_sr",
        type=int,
        default=48000,
        help="Target sample rate (default: 48000)"
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*.wav",
        help="File pattern (default: *.wav)"
    )

    args = parser.parse_args()

    if args.mode == "check":
        check_sample_rates(args.input_dir, args.pattern)

    elif args.mode == "convert":
        if args.output_dir is None:
            print("Error: --output_dir is required for convert mode")
            exit(1)

        batch_convert_sample_rate(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            target_sr=args.target_sr,
            pattern=args.pattern
        )
