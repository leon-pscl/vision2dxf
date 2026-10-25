"""Compute the Maintainability Index for the design-2 code.

Usage:
    python backend/tests/compute_maintainability_index.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE.parents[0] / "app"


def main() -> int:
    cmd = [sys.executable, "-m", "radon", "mi", "-s", str(TARGET)]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
