"""Cell 3 — environment, paths, imports.

WHAT — Resolve the working/export/artifacts paths, install only missing packages,
record the environment, and import every third-party symbol the notebook uses.

WHY
- [Ref] Pimentel et al. (2021): a notebook whose environment is not recorded is not
  reproducible. This block is the source for requirements.txt in section 9.
- [Ref] Brief section 1: no segmentation model is trained here, so the only
  image-processing dependencies are torch/torchvision, PIL and scipy.
- [Assumption] The Kaggle image ships a recent CUDA-enabled torch.
  Tested by: gpu_report(), which reports availability, name, compute capability
  and memory rather than assuming one.

OUTPUT — PATHS (dict), ENV (dict), plus every imported symbol in the namespace.

CHECK — every required package imports; DataLoader is bound (fix A3: the
  training and sensitivity cells both call DataLoader directly).
"""

import os, sys, json, time, math, random, platform, textwrap, hashlib, zipfile, shutil, warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

t_notebook_start = time.time()
PHASE_TIMES: Dict[str, float] = {}


def mark_phase(name: str) -> float:
    """Record elapsed wall-clock seconds for a pipeline phase.

    Parameters
    ----------
    name : str
        Phase label used as key in :data:`PHASE_TIMES`.

    Returns
    -------
    float
        Seconds elapsed since the previous call.
    """
    now = time.time()
    last = PHASE_TIMES.get("_last", t_notebook_start)
    PHASE_TIMES[name] = round(now - last, 1)
    PHASE_TIMES["_last"] = now
    return PHASE_TIMES[name]


# --------------------------------------------------------------------------------------
# Paths. Kaggle mounts read-only inputs at /kaggle/input and a writable dir at /kaggle/working.
# Locally we fall back to the repo layout so the notebook is runnable during development.
# --------------------------------------------------------------------------------------
def resolve_paths() -> Dict[str, Path]:
    """Return {working, src, export, artifacts} for Kaggle or a local checkout.

    Returns
    -------
    Dict[str, Path]
        Absolute paths. ``working`` is created if missing.
    """
    if Path("/kaggle/working").is_dir():
        working = Path("/kaggle/working")
    else:
        working = Path.cwd()
    src = working / "src"
    export = working / "export"
    artifacts = working / "artifacts"
    for p in (src, export, artifacts):
        p.mkdir(parents=True, exist_ok=True)
    return {"working": working, "src": src, "export": export, "artifacts": artifacts}


PATHS = resolve_paths()
print("PATHS:", json.dumps({k: str(v) for k, v in PATHS.items()}, indent=2))

sys.path.insert(0, str(PATHS["src"]))
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# --- install only what is missing; never assume a clean image ------------------------------
import importlib.util, subprocess

REQUIRED = {
    "numpy": "numpy",
    "pandas": "pandas",
    "PIL": "pillow",
    "matplotlib": "matplotlib",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "yaml": "pyyaml",
    "torch": "torch",
    "torchvision": "torchvision",
    "tqdm": "tqdm",
    "torch.utils.data": None,   # always part of torch; imported explicitly below
}

missing = [pkg for mod, pkg in REQUIRED.items() if pkg and importlib.util.find_spec(mod) is None]
if missing:
    print("installing:", missing)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=False)
else:
    print("all required packages present")

# Optional: ONNX export verification. Not fatal if absent.
for opt in ("onnx", "onnxruntime"):
    if importlib.util.find_spec(opt) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", opt], check=False)

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import scipy
import yaml
import torch
import torchvision
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from scipy import ndimage as ndi
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

# Display figures inline as well as saving them (fix D6). Without this, matplotlib's
# Agg backend silently suppresses every rendered figure.
try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass

print(f"DataLoader bound: {DataLoader.__module__}.{DataLoader.__name__}")
assert DataLoader is not None, "torch.utils.data.DataLoader failed to import (fix A3)"

matplotlib.use("Agg")
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25,
                     "figure.facecolor": "white", "font.size": 9})


def gpu_report() -> Dict[str, Any]:
    """Describe the visible CUDA device.

    Returns
    -------
    dict
        ``available`` (bool), ``name`` (str), ``count`` (int), ``capability`` (str),
        ``total_gb`` (float).
    """
    info = {"available": torch.cuda.is_available(), "name": "cpu", "count": 0,
            "capability": "n/a", "total_gb": 0.0}
    if info["available"]:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            if i == 0:
                info.update(name=p.name, count=torch.cuda.device_count(),
                            capability=f"{p.major}.{p.minor}", total_gb=round(p.total_memory / 2**30, 1))
    return info


def internet_report(timeout: float = 4.0) -> bool:
    """Return True if an outbound HTTPS connection to pypi succeeds within ``timeout``.

    Parameters
    ----------
    timeout : float
        Socket timeout in seconds.

    Returns
    -------
    bool
        True when the network is reachable.
    """
    import socket
    for host, port in (("pypi.org", 443), ("download.pytorch.org", 443)):
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            continue
    return False


ENV = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "executable": sys.executable,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "matplotlib": matplotlib.__version__,
    "scikit_learn": sklearn.__version__,
    "scipy": scipy.__version__,
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "pillow": Image.__version__ if hasattr(Image, "__version__") else __import__("PIL").__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu": gpu_report(),
    "internet": internet_report(),
    "kaggle_env": bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")),
}
print(json.dumps(ENV, indent=2))
print("mark_phase('setup') =", mark_phase("setup"), "s")