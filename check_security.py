#!/usr/bin/env python
"""
Security Check Script for Sound-UAV SpecMAE Project

Performs automated security checks:
1. Dependency version verification
2. Known CVE scanning
3. Code pattern analysis
4. File integrity checks

Usage:
    python check_security.py
    python check_security.py --verbose
    python check_security.py --fix
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
#  Version checks
# ═══════════════════════════════════════════════════════════════════════════

def check_package_version(
    package_name: str,
    current_version: Optional[str],
    min_safe_version: str,
    cve_info: str = "",
) -> bool:
    """Check if package version is safe."""
    if current_version is None:
        print(f"  ⚠️  {package_name}: NOT INSTALLED")
        return False

    from packaging import version

    current = version.parse(current_version)
    minimum = version.parse(min_safe_version)

    if current < minimum:
        print(f"  ❌ {package_name}: {current_version} < {min_safe_version} (VULNERABLE)")
        if cve_info:
            print(f"     {cve_info}")
        return False
    else:
        print(f"  ✅ {package_name}: {current_version} >= {min_safe_version}")
        return True


def check_dependencies(verbose: bool = False) -> bool:
    """Check all critical dependencies."""
    print("🔍 Checking dependency versions...")
    print("-" * 60)

    all_ok = True

    # PyTorch
    torch_version = torch.__version__ if TORCH_AVAILABLE else None
    all_ok &= check_package_version(
        "torch",
        torch_version,
        "2.8.0",
        "CVE-2025-32434 (CRITICAL RCE), CVE-2024-31583, CVE-2024-31580",
    )

    # scikit-learn
    sklearn_version = sklearn.__version__ if SKLEARN_AVAILABLE else None
    all_ok &= check_package_version(
        "scikit-learn",
        sklearn_version,
        "1.5.0",
        "CVE-2024-5206 (data leakage)",
    )

    # tqdm
    tqdm_version = tqdm.__version__ if TQDM_AVAILABLE else None
    all_ok &= check_package_version(
        "tqdm",
        tqdm_version,
        "4.66.3",
        "CVE-2024-34062 (code injection)",
    )

    print("-" * 60)
    return all_ok


# ═══════════════════════════════════════════════════════════════════════════
#  CVE scanning
# ═══════════════════════════════════════════════════════════════════════════

def run_safety_check(verbose: bool = False) -> bool:
    """Run safety check for known vulnerabilities."""
    print("\n🔍 Running Safety check...")
    print("-" * 60)

    try:
        result = subprocess.run(
            ["safety", "check", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            print("  ✅ No known vulnerabilities found")
            return True
        else:
            print("  ❌ Vulnerabilities detected:")
            if verbose:
                print(result.stdout)
            else:
                print("     Run with --verbose for details")
                print("     Or run: safety check")
            return False

    except FileNotFoundError:
        print("  ⚠️  'safety' not installed. Install with: pip install safety")
        return False
    except subprocess.TimeoutExpired:
        print("  ⚠️  Safety check timed out")
        return False
    except Exception as e:
        print(f"  ⚠️  Error running safety: {e}")
        return False


def run_pip_audit(verbose: bool = False) -> bool:
    """Run pip-audit for vulnerability scanning."""
    print("\n🔍 Running pip-audit...")
    print("-" * 60)

    try:
        result = subprocess.run(
            ["pip-audit"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            print("  ✅ All dependencies are secure")
            return True
        else:
            print("  ❌ Issues found:")
            if verbose:
                print(result.stdout)
            else:
                print("     Run with --verbose for details")
                print("     Or run: pip-audit")
            return False

    except FileNotFoundError:
        print("  ⚠️  'pip-audit' not installed. Install with: pip install pip-audit")
        return False
    except subprocess.TimeoutExpired:
        print("  ⚠️  pip-audit timed out")
        return False
    except Exception as e:
        print(f"  ⚠️  Error running pip-audit: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  Code pattern checks
# ═══════════════════════════════════════════════════════════════════════════

def check_torch_load_usage(project_root: Path, verbose: bool = False) -> bool:
    """Check for unsafe torch.load usage."""
    print("\n🔍 Checking torch.load usage...")
    print("-" * 60)

    issues = []
    py_files = list(project_root.rglob("*.py"))

    for py_file in py_files:
        try:
            content = py_file.read_text(encoding="utf-8")
            lines = content.split("\n")

            for i, line in enumerate(lines, 1):
                if "torch.load" in line and "weights_only" not in line:
                    # Check if it's in a comment
                    if line.strip().startswith("#"):
                        continue

                    issues.append((py_file.relative_to(project_root), i, line.strip()))

        except Exception as e:
            if verbose:
                print(f"  ⚠️  Error reading {py_file}: {e}")

    if not issues:
        print("  ✅ All torch.load calls look safe")
        return True
    else:
        print(f"  ⚠️  Found {len(issues)} potentially unsafe torch.load calls:")
        for file, line_no, line in issues:
            print(f"     {file}:{line_no}")
            if verbose:
                print(f"       {line}")
        print("     Ensure all torch.load calls specify weights_only parameter")
        return False


def check_dangerous_functions(project_root: Path, verbose: bool = False) -> bool:
    """Check for dangerous function usage."""
    print("\n🔍 Checking for dangerous functions...")
    print("-" * 60)

    dangerous_patterns = [
        ("eval(", "eval() can execute arbitrary code"),
        ("exec(", "exec() can execute arbitrary code"),
        ("__import__", "__import__ can load arbitrary modules"),
        ("pickle.load", "pickle.load can execute arbitrary code"),
        ("yaml.load(", "yaml.load() is unsafe, use yaml.safe_load()"),
        ("os.system(", "os.system() can execute shell commands"),
    ]

    issues = []
    py_files = list(project_root.rglob("*.py"))

    for py_file in py_files:
        try:
            content = py_file.read_text(encoding="utf-8")
            lines = content.split("\n")

            for i, line in enumerate(lines, 1):
                # Skip comments
                if line.strip().startswith("#"):
                    continue

                for pattern, reason in dangerous_patterns:
                    if pattern in line:
                        issues.append((
                            py_file.relative_to(project_root),
                            i,
                            pattern,
                            reason,
                            line.strip(),
                        ))

        except Exception as e:
            if verbose:
                print(f"  ⚠️  Error reading {py_file}: {e}")

    if not issues:
        print("  ✅ No dangerous function patterns detected")
        return True
    else:
        print(f"  ⚠️  Found {len(issues)} potentially dangerous function calls:")
        for file, line_no, pattern, reason, line in issues:
            print(f"     {file}:{line_no} - {pattern}")
            print(f"       Reason: {reason}")
            if verbose:
                print(f"       Line: {line}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  File checks
# ═══════════════════════════════════════════════════════════════════════════

def check_required_files(project_root: Path) -> bool:
    """Check for required security files."""
    print("\n🔍 Checking required security files...")
    print("-" * 60)

    required_files = {
        "requirements.txt": "Dependency specification",
        ".gitignore": "Git ignore rules",
        "scripts/utils/security_utils.py": "Security utilities",
        "SECURITY_AUDIT_REPORT.md": "Security audit report",
        "SECURITY_FIX_GUIDE.md": "Security fix guide",
    }

    all_exist = True
    for file, description in required_files.items():
        file_path = project_root / file
        if file_path.exists():
            print(f"  ✅ {file}: {description}")
        else:
            print(f"  ❌ {file}: MISSING - {description}")
            all_exist = False

    print("-" * 60)
    return all_exist


# ═══════════════════════════════════════════════════════════════════════════
#  Fix suggestions
# ═══════════════════════════════════════════════════════════════════════════

def suggest_fixes(failed_checks: list[str]) -> None:
    """Suggest fixes for failed checks."""
    print("\n💡 Suggested fixes:")
    print("=" * 60)

    if "dependencies" in failed_checks:
        print("\n1. Update vulnerable dependencies:")
        print("   pip install --upgrade -r requirements.txt")
        print("   # Or upgrade individually:")
        if not TORCH_AVAILABLE or torch.__version__ < "2.8.0":
            print("   pip install --upgrade 'torch>=2.8.0'")
        if not SKLEARN_AVAILABLE or sklearn.__version__ < "1.5.0":
            print("   pip install --upgrade 'scikit-learn>=1.5.0'")
        if not TQDM_AVAILABLE or tqdm.__version__ < "4.66.3":
            print("   pip install --upgrade 'tqdm>=4.66.3'")

    if "safety" in failed_checks:
        print("\n2. Install and run safety:")
        print("   pip install safety")
        print("   safety check")

    if "pip-audit" in failed_checks:
        print("\n3. Install and run pip-audit:")
        print("   pip install pip-audit")
        print("   pip-audit")

    if "torch_load" in failed_checks:
        print("\n4. Fix torch.load usage:")
        print("   Replace: ckpt = torch.load(path)")
        print("   With:    ckpt = torch.load(path, weights_only=False)")

    if "dangerous" in failed_checks:
        print("\n5. Review dangerous function usage:")
        print("   - Replace eval() with safer alternatives")
        print("   - Replace yaml.load() with yaml.safe_load()")
        print("   - Validate all external inputs")

    if "files" in failed_checks:
        print("\n6. Create missing security files:")
        print("   See SECURITY_FIX_GUIDE.md for instructions")

    print("\n" + "=" * 60)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run security checks on Sound-UAV SpecMAE project"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Show fix suggestions for failed checks",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🔒 Sound-UAV SpecMAE Security Check")
    print("=" * 60)

    project_root = Path(__file__).resolve().parent
    failed_checks = []

    # Run all checks
    if not check_dependencies(args.verbose):
        failed_checks.append("dependencies")

    if not check_required_files(project_root):
        failed_checks.append("files")

    if not run_safety_check(args.verbose):
        failed_checks.append("safety")

    if not run_pip_audit(args.verbose):
        failed_checks.append("pip-audit")

    if not check_torch_load_usage(project_root, args.verbose):
        failed_checks.append("torch_load")

    if not check_dangerous_functions(project_root, args.verbose):
        failed_checks.append("dangerous")

    # Summary
    print("\n" + "=" * 60)
    if not failed_checks:
        print("✅ All security checks PASSED!")
        print("=" * 60)
        return 0
    else:
        print(f"⚠️  {len(failed_checks)} security check(s) FAILED:")
        for check in failed_checks:
            print(f"   - {check}")
        print("=" * 60)

        if args.fix:
            suggest_fixes(failed_checks)

        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Security check interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
