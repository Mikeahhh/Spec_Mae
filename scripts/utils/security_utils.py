"""
Security utilities for Sound-UAV SpecMAE project.

Provides functions for:
- File size validation
- File integrity verification (SHA256 hashing)
- Checkpoint integrity verification
- Path sanitization (防止目录遍历攻击)
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
#  File size validation
# ═══════════════════════════════════════════════════════════════════════════

def verify_file_size(
    path: Path | str,
    max_size_mb: int = 100,
    raise_error: bool = True,
) -> bool:
    """
    验证文件大小是否在允许范围内。

    Args:
        path:        文件路径
        max_size_mb: 最大允许大小（MB）
        raise_error: 如果超过大小限制是否抛出异常

    Returns:
        bool: 文件大小是否合法

    Raises:
        ValueError: 如果文件过大且 raise_error=True
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    size_bytes = path.stat().st_size
    max_bytes = max_size_mb * 1024 * 1024

    if size_bytes > max_bytes:
        msg = (
            f"File size {size_bytes / 1024 / 1024:.2f} MB exceeds "
            f"maximum allowed size of {max_size_mb} MB: {path}"
        )
        if raise_error:
            raise ValueError(msg)
        else:
            print(f"⚠️  Warning: {msg}")
            return False

    return True


# ═══════════════════════════════════════════════════════════════════════════
#  File integrity verification
# ═══════════════════════════════════════════════════════════════════════════

def compute_file_hash(path: Path | str, algorithm: str = "sha256") -> str:
    """
    计算文件的哈希值。

    Args:
        path:      文件路径
        algorithm: 哈希算法 (sha256, sha512, md5)

    Returns:
        str: 十六进制哈希字符串

    Example:
        >>> hash_value = compute_file_hash("model.pth")
        >>> print(f"SHA256: {hash_value}")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    hash_func = hashlib.new(algorithm)

    with open(path, "rb") as f:
        # Read file in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)

    return hash_func.hexdigest()


def verify_checkpoint_integrity(
    path: Path | str,
    expected_hash: str,
    algorithm: str = "sha256",
) -> bool:
    """
    验证 checkpoint 文件的完整性。

    Args:
        path:          Checkpoint 文件路径
        expected_hash: 期望的哈希值
        algorithm:     哈希算法

    Returns:
        bool: 文件是否完整

    Example:
        >>> expected = "abc123..."
        >>> is_valid = verify_checkpoint_integrity("best_model.pth", expected)
        >>> if not is_valid:
        ...     print("⚠️  Checkpoint file may be corrupted!")
    """
    actual_hash = compute_file_hash(path, algorithm)
    return actual_hash.lower() == expected_hash.lower()


def save_checkpoint_with_hash(
    checkpoint_path: Path | str,
    hash_path: Optional[Path | str] = None,
    algorithm: str = "sha256",
) -> str:
    """
    保存 checkpoint 后计算并保存其哈希值。

    Args:
        checkpoint_path: Checkpoint 文件路径
        hash_path:       哈希值保存路径（默认为 checkpoint_path + ".sha256"）
        algorithm:       哈希算法

    Returns:
        str: 计算得到的哈希值

    Example:
        >>> torch.save(model.state_dict(), "model.pth")
        >>> hash_val = save_checkpoint_with_hash("model.pth")
        >>> print(f"Checkpoint saved with hash: {hash_val}")
    """
    checkpoint_path = Path(checkpoint_path)

    # Compute hash
    hash_value = compute_file_hash(checkpoint_path, algorithm)

    # Save hash to file
    if hash_path is None:
        hash_path = checkpoint_path.with_suffix(f".{algorithm}")
    else:
        hash_path = Path(hash_path)

    with open(hash_path, "w") as f:
        f.write(f"{hash_value}  {checkpoint_path.name}\n")

    return hash_value


def load_and_verify_checkpoint(
    checkpoint_path: Path | str,
    hash_path: Optional[Path | str] = None,
    algorithm: str = "sha256",
) -> Optional[str]:
    """
    加载并验证 checkpoint 完整性。

    Args:
        checkpoint_path: Checkpoint 文件路径
        hash_path:       哈希文件路径
        algorithm:       哈希算法

    Returns:
        str: 哈希值（如果验证通过），否则返回 None

    Raises:
        ValueError: 如果哈希值不匹配
    """
    checkpoint_path = Path(checkpoint_path)

    if hash_path is None:
        hash_path = checkpoint_path.with_suffix(f".{algorithm}")
    else:
        hash_path = Path(hash_path)

    if not hash_path.exists():
        print(f"⚠️  Warning: Hash file not found: {hash_path}")
        print("    Skipping integrity verification")
        return None

    # Read expected hash
    with open(hash_path, "r") as f:
        line = f.read().strip()
        expected_hash = line.split()[0]

    # Verify
    if verify_checkpoint_integrity(checkpoint_path, expected_hash, algorithm):
        print(f"✅ Checkpoint integrity verified: {checkpoint_path.name}")
        return expected_hash
    else:
        raise ValueError(
            f"⚠️  Checkpoint integrity check FAILED!\n"
            f"   File: {checkpoint_path}\n"
            f"   Expected: {expected_hash}\n"
            f"   Actual:   {compute_file_hash(checkpoint_path, algorithm)}"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Path sanitization (防止目录遍历攻击)
# ═══════════════════════════════════════════════════════════════════════════

def sanitize_path(
    path: Path | str,
    allowed_parent: Path | str,
    raise_error: bool = True,
) -> Optional[Path]:
    """
    验证路径是否在允许的父目录下，防止目录遍历攻击。

    Args:
        path:           要验证的路径
        allowed_parent: 允许的父目录
        raise_error:    是否在路径不合法时抛出异常

    Returns:
        Path: 解析后的绝对路径（如果合法），否则返回 None

    Raises:
        ValueError: 如果路径不在允许的目录下且 raise_error=True

    Example:
        >>> sanitize_path("data/desert/test.wav", "data")  # OK
        >>> sanitize_path("../../../etc/passwd", "data")   # ERROR
    """
    path = Path(path).resolve()
    allowed_parent = Path(allowed_parent).resolve()

    # Check if path is under allowed_parent
    try:
        path.relative_to(allowed_parent)
    except ValueError:
        msg = (
            f"⚠️  Security: Path {path} is outside allowed directory {allowed_parent}"
        )
        if raise_error:
            raise ValueError(msg)
        else:
            print(msg)
            return None

    return path


def sanitize_filename(filename: str) -> str:
    """
    清理文件名，移除危险字符。

    Args:
        filename: 原始文件名

    Returns:
        str: 清理后的安全文件名

    Example:
        >>> sanitize_filename("../../etc/passwd")
        'etcpasswd'
        >>> sanitize_filename("normal_file.wav")
        'normal_file.wav'
    """
    # Remove directory separators
    filename = filename.replace("/", "").replace("\\", "")

    # Remove other dangerous characters
    dangerous_chars = [".", "~", "*", "?", "<", ">", "|", ":", '"']
    for char in dangerous_chars:
        if char == "." and filename.endswith((".wav", ".pth", ".npy", ".csv")):
            # Keep file extension dots
            parts = filename.rsplit(".", 1)
            if len(parts) == 2:
                filename = parts[0].replace(".", "") + "." + parts[1]
        else:
            filename = filename.replace(char, "")

    # Limit length
    if len(filename) > 255:
        filename = filename[:255]

    return filename


# ═══════════════════════════════════════════════════════════════════════════
#  Audio file validation
# ═══════════════════════════════════════════════════════════════════════════

def validate_audio_file(
    path: Path | str,
    max_size_mb: int = 50,
    allowed_formats: tuple[str, ...] = (".wav", ".flac", ".mp3"),
) -> bool:
    """
    验证音频文件的安全性。

    Args:
        path:            音频文件路径
        max_size_mb:     最大文件大小（MB）
        allowed_formats: 允许的文件格式

    Returns:
        bool: 文件是否合法

    Raises:
        ValueError: 如果文件不合法
    """
    path = Path(path)

    # Check existence
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # Check format
    if path.suffix.lower() not in allowed_formats:
        raise ValueError(
            f"Invalid audio format: {path.suffix}. "
            f"Allowed: {allowed_formats}"
        )

    # Check size
    verify_file_size(path, max_size_mb=max_size_mb, raise_error=True)

    return True


# ═══════════════════════════════════════════════════════════════════════════
#  Usage examples
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Security Utilities - Usage Examples")
    print("=" * 60)

    # Example 1: File size validation
    print("\n1. File size validation:")
    print("   verify_file_size('large_file.pth', max_size_mb=100)")

    # Example 2: Compute hash
    print("\n2. Compute file hash:")
    print("   hash = compute_file_hash('model.pth')")
    print("   print(f'SHA256: {hash}')")

    # Example 3: Save checkpoint with hash
    print("\n3. Save checkpoint with integrity check:")
    print("   torch.save(model.state_dict(), 'model.pth')")
    print("   save_checkpoint_with_hash('model.pth')")

    # Example 4: Load and verify checkpoint
    print("\n4. Load and verify checkpoint:")
    print("   load_and_verify_checkpoint('model.pth')")
    print("   ckpt = torch.load('model.pth', weights_only=False)")

    # Example 5: Path sanitization
    print("\n5. Path sanitization:")
    print("   safe_path = sanitize_path('data/test.wav', 'data')")

    print("\n" + "=" * 60)
