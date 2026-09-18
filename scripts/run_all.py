#!/usr/bin/env python3
import subprocess
import sys


def run(cmd: str, label: str) -> bool:
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n  FAILED: {label}")
        return False
    return True


def main():
    steps = [
        ("pytest -v", "Running Tests"),
        ("python scripts/run_evaluation.py", "Running Evaluation"),
        ("python scripts/run_sensitivity.py", "Running Sensitivity Analysis"),
    ]

    all_ok = True
    for cmd, label in steps:
        if not run(cmd, label):
            all_ok = False
            print(f"\nStopping early at: {label}")
            break

    if all_ok:
        print(f"\n{'='*60}\n  ALL STEPS COMPLETE\n{'='*60}")
    else:
        print(f"\n{'='*60}\n  SOME STEPS FAILED\n{'='*60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
