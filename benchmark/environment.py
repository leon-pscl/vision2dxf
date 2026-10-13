"""Environment record and configuration fingerprint for the Design 3 benchmark.

Everything here is stdlib. `radon`, `scipy` and friends are reported if present and null
if not, because an absent optional library is a fact about the environment, not a reason
to guess. The fingerprints are what tie a result file to the exact configuration and
commit that produced it.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Every configuration file whose contents can change a generated pattern. Hashing these
# is what makes "the configuration did not change during the benchmark" checkable.
CONFIG_FILES = (
    "benchmark/config-design-3.yaml",
    "fixtures/garments/d3-basic-torso.yaml",
    "backend/app/config/reconstruction.yaml",
    "backend/app/config/drafting.yaml",
    "backend/app/config/measurement-semantics.yaml",
    "backend/app/config/measurements.yaml",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_fingerprint() -> dict:
    """Hash of every frozen configuration file, plus one combined hash."""
    per_file = {name: sha256_file(ROOT / name) for name in CONFIG_FILES}
    combined = hashlib.sha256(
        "\n".join(f"{name}:{digest}" for name, digest in sorted(per_file.items())).encode()
    ).hexdigest()
    return {"perFile": per_file, "combined": combined}


def configuration_version() -> str:
    import yaml

    return str(yaml.safe_load(
        (ROOT / "benchmark" / "config-design-3.yaml").read_text(encoding="utf-8")
    )["configuration_version"])


def git_state() -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                                  text=True, timeout=15).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    dirty = run("status", "--porcelain")
    return {
        "commitHash": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirtyWorkingTree": bool(dirty),
        "dirtyPaths": dirty.splitlines()[:20],
    }


def cpu_info() -> dict:
    model = platform.processor()
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            model = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    elif Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    return {
        "model": model or None,
        "machine": platform.machine(),
        "logicalCores": os.cpu_count(),
        "physicalCores": _physical_cores(),
    }


def _physical_cores() -> int | None:
    try:
        import psutil  # noqa: PLC0415

        return psutil.cpu_count(logical=False)
    except Exception:
        pass
    if sys.platform == "win32":
        return None  # stdlib cannot enumerate physical cores; not worth guessing
    try:
        ids = set()
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith(("physical id", "core id")):
                ids.add(line.strip())
        return len(ids) or None
    except OSError:
        return None


def installed_ram_bytes() -> int | None:
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class Status(ctypes.Structure):
                # MEMORYSTATUSEX: two DWORDs, then eight SIZE_Ts. Dropping
                # `memoryLoad` misaligns every later field, so keep the order exact.
                _fields_ = [("length", wintypes.DWORD),
                            ("memoryLoad", wintypes.DWORD),
                            ("totalPhys", ctypes.c_ulonglong),
                            ("availPhys", ctypes.c_ulonglong),
                            ("totalPageFile", ctypes.c_ulonglong),
                            ("availPageFile", ctypes.c_ulonglong),
                            ("totalVirtual", ctypes.c_ulonglong),
                            ("availVirtual", ctypes.c_ulonglong),
                            ("availExtended", ctypes.c_ulonglong)]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            status = Status()
            status.length = ctypes.sizeof(Status)
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.totalPhys)
        except Exception:
            return None
        return None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def library_versions() -> dict:
    """Version of every library the results may depend on. Absent means null."""
    from importlib import metadata

    names = {
        "numpy": "numpy", "scipy": "scipy", "trimesh": "trimesh", "shapely": "shapely",
        "ezdxf": "ezdxf", "radon": "radon", "pandas": "pandas", "pydantic": "pydantic",
        "PyYAML": "PyYAML", "reportlab": "reportlab", "Pillow": "Pillow",
    }
    out = {}
    for label, distribution in names.items():
        try:
            out[label] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            out[label] = None
    return out


def blas_info() -> dict | None:
    """BLAS/LAPACK the NumPy build uses, when it will say."""
    try:
        import numpy

        config = numpy.show_config(mode="dicts")
    except Exception:
        return None
    blas = config.get("Build Dependencies", {}).get("blas", {})
    lapack = config.get("Build Dependencies", {}).get("lapack", {})
    return {
        "blas": blas.get("name"),
        "blasVersion": blas.get("version"),
        "lapack": lapack.get("name"),
        "lapackVersion": lapack.get("version"),
    }


THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def environment_record() -> dict:
    fingerprint = config_fingerprint()
    return {
        "recordedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "operatingSystem": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
        },
        "cpu": cpu_info(),
        "installedRamBytes": installed_ram_bytes(),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "libraries": library_versions(),
        "blas": blas_info(),
        "threadEnvironment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "git": git_state(),
        "configurationVersion": configuration_version(),
        "configurationHash": fingerprint["combined"],
        "configurationHashesPerFile": fingerprint["perFile"],
    }


def main() -> int:
    print(json.dumps(environment_record(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())