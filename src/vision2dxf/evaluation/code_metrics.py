from __future__ import annotations
from pathlib import Path
import subprocess
import json


def measure_code_metrics(design_paths: list[str]) -> list[dict]:
    results = []
    for dpath in design_paths:
        path = Path(dpath)
        if not path.exists():
            results.append({"path": dpath, "error": "path not found"})
            continue

        py_files = [f for f in path.rglob("*.py") if f.name != "__init__.py"]
        if not py_files:
            results.append({"path": dpath, "error": "no Python files found"})
            continue

        try:
            proc = subprocess.run(
                ["radon", "cc", "-j", str(path)],
                capture_output=True, text=True, timeout=30,
            )
            cc_data = json.loads(proc.stdout) if proc.stdout.strip() else {}

            vol_proc = subprocess.run(
                ["radon", "hal", "-j", str(path)],
                capture_output=True, text=True, timeout=30,
            )
            vol_data = json.loads(vol_proc.stdout) if vol_proc.stdout.strip() else {}

            mi_proc = subprocess.run(
                ["radon", "mi", "-j", str(path)],
                capture_output=True, text=True, timeout=30,
            )
            mi_data = json.loads(mi_proc.stdout) if mi_proc.stdout.strip() else {}

            total_volume = 0
            total_mi = 0
            mi_count = 0

            for fname, file_data in vol_data.items():
                if isinstance(file_data, dict) and "total" in file_data:
                    total_volume += file_data["total"].get("volume", 0)

            for fname, file_data in mi_data.items():
                if isinstance(file_data, dict) and "mi" in file_data:
                    total_mi += file_data["mi"]
                    mi_count += 1

            avg_mi = total_mi / mi_count if mi_count > 0 else 0

            results.append({
                "path": dpath,
                "halstead_volume": round(total_volume, 4),
                "maintainability_index": round(avg_mi, 4),
                "cyclomatic_complexity": cc_data,
                "file_count": len(py_files),
            })
        except Exception as e:
            results.append({"path": dpath, "error": str(e)})

    return results
