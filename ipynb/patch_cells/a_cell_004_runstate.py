# %%writefile src/run_state.py
"""Resumable phase state for a Kaggle notebook with "Files only" persistence.

Why this module exists
----------------------
Kaggle's *Files only* persistence guarantees that files in ``/kaggle/working`` survive between
sessions; it guarantees nothing about Python state. A committed "Save & Run All" can also start
with an empty ``/kaggle/working``. So a notebook that passes objects between cells cannot be
resumed, and a session that dies at hour 9 of 12 loses everything.

This module turns each phase into an **artefact contract**: a phase writes its outputs to disk,
records them in ``run/manifest.json`` alongside the hash of the configuration that produced them,
and a later session can reload the phase instead of recomputing it. Three conditions must all hold
for a phase to be considered reusable:

1. its manifest status is ``done``;
2. its recorded ``config_hash`` matches the current configuration; and
3. every artefact it listed still exists on disk.

Condition 2 is what stops a cached EDA figure from being reused after ``smoke_test`` was turned off,
and condition 3 catches a half-deleted working directory.

Design notes
------------
* Writes are **atomic** (temp file + :func:`os.replace`) everywhere, because a session killed
  mid-``torch.save`` would otherwise leave a truncated checkpoint that looks valid on reload.
* ``config_hash`` excludes only keys that provably cannot change a result. ``smoke_test`` and
  ``run_final_eval`` are **included**: both change what is computed.
* A ``partial`` phase (interrupted by the session time guard) is never ``done``, so it reruns.
* :func:`maybe_skip` returns a three-state result. The ``BLOCKED`` sentinel is what lets a
  downstream cell define its variables without recomputing, instead of raising ``NameError``.

References
----------
Pimentel, J. F., Murta, L., Braganholo, V., & Freire, J. (2021). Understanding and improving the
quality and reproducibility of Jupyter notebooks. Empirical Software Engineering, 26, 65.
https://doi.org/10.1007/s10664-021-09961-9
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

MANIFEST_VERSION = 1

#: Statuses a phase entry may hold.
STATUSES = ("done", "partial", "blocked", "failed", "running")

#: Sentinel distinguishing "a dependency is not available" from "nothing cached".
BLOCKED = object()

_REQUIRED_MANIFEST_KEYS = {"version", "created", "updated", "config_hash", "phases"}
_REQUIRED_PHASE_KEYS = {"status", "config_hash", "started", "depends_on", "artefacts"}


# =========================================================================================
# Config hashing
# =========================================================================================
def canonical_config(cfg: Dict[str, Any], exclude: Sequence[str] = ()) -> str:
    """Render a configuration deterministically, excluding non-result keys.

    Parameters
    ----------
    cfg : dict
        The notebook configuration.
    exclude : sequence of str
        Keys to drop before hashing. Only keys that cannot change any computed value belong
        here.

    Returns
    -------
    str
        Sorted-key YAML text, safe to hash.
    """
    import yaml

    return yaml.safe_dump({k: v for k, v in sorted(cfg.items()) if k not in set(exclude)},
                          sort_keys=True, default_flow_style=False)


def config_hash(cfg: Dict[str, Any], exclude: Sequence[str] = ()) -> str:
    """Short, stable hash of the result-affecting configuration.

    Parameters
    ----------
    cfg : dict
        The notebook configuration.
    exclude : sequence of str, default ()
        Keys that do not affect results and are therefore excluded.

    Returns
    -------
    str
        First 16 hex characters of the sha256 of :func:`canonical_config`.
    """
    return hashlib.sha256(canonical_config(cfg, exclude).encode("utf-8")).hexdigest()[:16]


# =========================================================================================
# Atomic writes
# =========================================================================================
def atomic_write_bytes(path: Path | str, data: bytes) -> Path:
    """Write bytes to ``path`` atomically.

    Parameters
    ----------
    path : path-like
        Destination.
    data : bytes
        Payload.

    Returns
    -------
    Path
        The destination.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, p)
    return p


def atomic_save(obj: Any, path: Path | str, kind: str = "torch") -> Path:
    """Persist an object atomically.

    Parameters
    ----------
    obj : object
        Object to save.
    path : path-like
        Destination. The suffix is added when missing, from ``kind``.
    kind : {'torch', 'npz', 'json', 'parquet'}, default 'torch'
        Serialisation backend.

    Returns
    -------
    Path
        The written path.

    Raises
    ------
    ValueError
        If ``kind`` is unknown.
    """
    p = Path(path)
    if p.suffix == "":
        p = p.with_suffix({"torch": ".pt", "npz": ".npz", "json": ".json",
                           "parquet": ".parquet"}[kind])
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + f".tmp{os.getpid()}")
    if kind == "torch":
        import torch
        torch.save(obj, tmp)
    elif kind == "npz":
        # numpy appends '.npz' to a *path* whose name lacks it, which would leave the temp file
        # behind and make os.replace fail. Hand it an open file object instead.
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **obj)
    elif kind == "json":
        tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    elif kind == "parquet":
        obj.to_parquet(tmp, index=False)
    else:
        raise ValueError(f"unknown save kind {kind!r}")
    os.replace(tmp, p)
    return p


# =========================================================================================
# Payload round-tripping
# =========================================================================================
def save_payload(directory: Path | str, name: str, payload: Dict[str, Any]) -> List[str]:
    """Write a phase payload to disk, one file per entry.

    Parameters
    ----------
    directory : path-like
        Destination directory (usually ``run/results/<phase>``).
    name : str
        Phase name; becomes a subdirectory.
    payload : dict
        ``{key: value}``. DataFrames are stored as parquet, ndarrays as npz, everything else as
        JSON. Values whose key starts with ``_`` are treated as private and not written.

    Returns
    -------
    list of str
        Artefact paths relative to the run directory.
    """
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for key, value in payload.items():
        if key.startswith("_"):
            continue
        target = d / f"{key}"
        if isinstance(value, pd.DataFrame):
            written.append(str(atomic_save(value, target, "parquet")))
        elif isinstance(value, np.ndarray):
            written.append(str(atomic_save({key: value}, target, "npz")))
        else:
            written.append(str(atomic_save(value, target, "json")))
    return written


def load_payload(directory: Path | str) -> Dict[str, Any]:
    """Load every artefact written by :func:`save_payload`.

    Parameters
    ----------
    directory : path-like
        Directory previously passed to :func:`save_payload`.

    Returns
    -------
    dict
        ``{key: value}``, reconstructed to the types that were saved. Returns an empty dict when
        the directory does not exist, so a first run is not an error.
    """
    d = Path(directory)
    if not d.is_dir():
        return {}
    out: Dict[str, Any] = {}
    for p in sorted(d.iterdir()):
        if p.is_dir() or ".tmp" in p.name:
            continue
        suf = p.suffix
        if suf == ".parquet":
            try:
                out[p.stem] = pd.read_parquet(p)
            except Exception as exc:
                out[p.stem] = f"<unreadable parquet: {exc}>"
        elif suf == ".npz":
            with np.load(p, allow_pickle=True) as z:
                # a single-array npz round-trips to the array itself, which is what the caller
                # passed in; multiple arrays come back as a dict
                out[p.stem] = z[z.files[0]] if len(z.files) == 1 else {k: z[k] for k in z.files}
        elif suf == ".json":
            out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        elif suf in (".pt", ".npy", ".yaml", ".csv"):
            out[p.stem] = p          # a path; the caller decides how to read it
        else:
            out[p.stem] = p
    return out


# =========================================================================================
# Manifest
# =========================================================================================
class RunState:
    """Owns ``run/manifest.json`` and the phase artefacts under ``run/``.

    Parameters
    ----------
    root : path-like
        The run directory, normally ``/kaggle/working/run``.
    cfg : dict
        Notebook configuration, hashed to decide reusability.
    hash_exclude : sequence of str
        Config keys excluded from the hash.
    phases : dict
        Registry ``name -> {"deps": [...], "requires_gpu": bool, "artefacts": [...]}``.
    verbose : bool, default True
        Print lifecycle messages.
    """

    def __init__(self, root: Path | str, cfg: Dict[str, Any],
                 hash_exclude: Sequence[str] = (),
                 phases: Optional[Dict[str, Dict[str, Any]]] = None,
                 verbose: bool = True) -> None:
        self.root = Path(root)
        self.cfg = cfg
        self.hash_exclude = list(hash_exclude)
        self.verbose = verbose
        self.phases: Dict[str, Dict[str, Any]] = phases or {}
        self.ckpt_root = self.root / "ckpt"
        self.results_root = self.root / "results"
        for d in (self.root, self.ckpt_root, self.results_root):
            d.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.hash = config_hash(cfg, self.hash_exclude)
        self.manifest: Dict[str, Any] = self._load_manifest()
        self.blocked_reasons: Dict[str, str] = {}
        # Write the manifest immediately so it exists from the first cell, even if the run dies
        # before any phase finishes. It also records the current config hash, which is what the
        # status table and any later session compare against.
        self.save_manifest()

    # -- manifest io ---------------------------------------------------------------------
    def _load_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.is_file():
            return {"version": MANIFEST_VERSION, "created": _now(), "updated": _now(),
                    "config_hash": self.hash, "phases": {}}
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            # A truncated manifest means the previous session died mid-write. Rather than
            # abort, start a fresh manifest and keep the artefacts on disk; phases will simply
            # rerun. Losing an hour of cached work is better than losing the run.
            self._log(f"WARNING: manifest.json is unreadable ({exc}); starting a fresh manifest. "
                      "Cached phases will rerun.")
            return {"version": MANIFEST_VERSION, "created": _now(), "updated": _now(),
                    "config_hash": self.hash, "phases": {}}

    def save_manifest(self) -> None:
        """Write ``manifest.json`` atomically."""
        self.manifest["updated"] = _now()
        self.manifest["config_hash"] = self.hash
        atomic_write_bytes(self.manifest_path,
                           json.dumps(self.manifest, indent=2, default=str).encode("utf-8"))

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # -- validation ----------------------------------------------------------------------
    def validate(self) -> List[str]:
        """Check the manifest against the expected schema.

        Returns
        -------
        list of str
            Problems found. Empty when the manifest is valid. A missing manifest is reported as
            a single informational entry rather than a schema violation.
        """
        m = self.manifest
        problems: List[str] = []
        missing = _REQUIRED_MANIFEST_KEYS - set(m)
        if missing:
            return [f"manifest.json missing top-level keys: {sorted(missing)}"]
        if m["version"] != MANIFEST_VERSION:
            problems.append(f"manifest version {m['version']} != {MANIFEST_VERSION}")
        if not isinstance(m["phases"], dict):
            return problems + ["manifest 'phases' is not an object"]
        for name, entry in m["phases"].items():
            if not isinstance(entry, dict):
                problems.append(f"phase {name}: entry is not an object")
                continue
            miss = _REQUIRED_PHASE_KEYS - set(entry)
            if miss:
                problems.append(f"phase {name}: missing keys {sorted(miss)}")
            # keep checking after a missing-key report: reporting only the first problem per
            # phase would hide the rest, and this list is what the user reads to fix the file
            if "status" in entry and entry["status"] not in STATUSES:
                problems.append(f"phase {name}: status {entry['status']!r} not in {list(STATUSES)}")
            if miss:
                continue
            if not isinstance(entry["depends_on"], list):
                problems.append(f"phase {name}: depends_on is not a list")
            if not isinstance(entry["artefacts"], list):
                problems.append(f"phase {name}: artefacts is not a list")
            for dep in entry["depends_on"]:
                if dep not in m["phases"] and dep in self.phases:
                    problems.append(f"phase {name}: depends on {dep}, which has no entry")
        return problems

    # -- status -------------------------------------------------------------------------
    def entry(self, name: str) -> Dict[str, Any]:
        """Return the manifest entry for a phase, or a synthetic not-run entry.

        Parameters
        ----------
        name : str
            Phase name.

        Returns
        -------
        dict
            The stored entry, or one describing a phase that has never run.
        """
        return self.manifest["phases"].get(name, {
            "status": "missing", "config_hash": None, "started": None,
            "finished": None, "depends_on": [], "artefacts": [], "meta": {}})

    def artefacts_present(self, name: str) -> List[str]:
        """Which of a phase's recorded artefacts are missing from disk.

        Parameters
        ----------
        name : str
            Phase name.

        Returns
        -------
        list of str
            Relative paths that do not exist. Empty when all are present.
        """
        missing = []
        for rel in self.entry(name).get("artefacts", []):
            if not (self.root / rel).is_file():
                missing.append(rel)
        return missing

    def phase_done(self, name: str) -> bool:
        """Whether a phase can be reused.

        Parameters
        ----------
        name : str
            Phase name.

        Returns
        -------
        bool
            True only when status is ``done``, the config hash matches, and every artefact
            exists.
        """
        e = self.entry(name)
        if e["status"] != "done":
            return False
        if e["config_hash"] != self.hash:
            return False
        return not self.artefacts_present(name)

    def hash_mismatch(self, name: str) -> bool:
        """Whether a cached phase was produced by a different configuration.

        Parameters
        ----------
        name : str
            Phase name.

        Returns
        -------
        bool
        """
        e = self.entry(name)
        return e["status"] != "missing" and e["config_hash"] not in (None, self.hash)

    def status_table(self) -> pd.DataFrame:
        """Tabulate every registered phase.

        Returns
        -------
        pd.DataFrame
            Columns ``phase``, ``status``, ``hash_match``, ``artefacts``, ``requires_gpu``,
            ``depends_on``.
        """
        rows = []
        for name, spec in self.phases.items():
            e = self.entry(name)
            missing = self.artefacts_present(name)
            rows.append({
                "phase": name,
                "status": e["status"],
                "hash_match": ("n/a" if e["status"] == "missing"
                               else ("yes" if e["config_hash"] == self.hash
                                     else "NO -> rerun")),
                "artefacts": ("ok" if e["status"] == "missing" or not missing
                              else f"MISSING {len(missing)}"),
                "requires_gpu": bool(spec.get("requires_gpu", False)),
                "depends_on": ",".join(spec.get("deps", [])) or "-",
            })
        return pd.DataFrame(rows)

    # -- lifecycle ----------------------------------------------------------------------
    def deps_ready(self, name: str) -> Tuple[bool, str]:
        """Whether every dependency of a phase is ``done``.

        Parameters
        ----------
        name : str
            Phase name.

        Returns
        -------
        (ok, reason) : tuple of (bool, str)
            ``reason`` names the first unsatisfied dependency and what state it is in.
        """
        for dep in self.phases.get(name, {}).get("deps", []):
            if dep not in self.phases:
                continue
            if not self.phase_done(dep):
                st = self.entry(dep)["status"]
                extra = []
                if self.hash_mismatch(dep):
                    extra.append("config hash changed")
                miss = self.artefacts_present(dep)
                if miss:
                    extra.append(f"{len(miss)} artefact(s) missing")
                why = "; ".join(extra) or f"status={st}"
                return False, f"dependency {dep} is not done ({why})"
        return True, ""

    def mark_done(self, name: str, artefacts: Optional[Sequence[str]] = None,
                  meta: Optional[Dict[str, Any]] = None, status: str = "done") -> None:
        """Record a phase as finished, or as partial/blocked.

        Parameters
        ----------
        name : str
            Phase name.
        artefacts : sequence of str, optional
            Paths relative to the run directory that must exist for reuse.
        meta : dict, optional
            Extra provenance (epoch counts, step counts, epochs trained, and so on).
        status : str, default 'done'
            One of :data:`STATUSES`.

        Raises
        ------
        ValueError
            If ``status`` is not recognised, or a listed artefact does not exist.
        """
        if status not in STATUSES:
            raise ValueError(f"status must be one of {list(STATUSES)}, got {status!r}")
        rels = [str(Path(a).resolve().relative_to(self.root.resolve()))
                if Path(a).is_absolute() else str(a) for a in (artefacts or [])]
        missing = [r for r in rels if not (self.root / r).is_file()]
        if missing:
            raise ValueError(f"mark_done({name!r}) listed artefacts that do not exist: {missing}")
        prev = self.entry(name)
        self.manifest["phases"][name] = {
            "status": status,
            "config_hash": self.hash,
            "started": prev.get("started") or _now(),
            "finished": _now(),
            "depends_on": list(self.phases.get(name, {}).get("deps", [])),
            "artefacts": rels,
            "meta": {**(prev.get("meta") or {}), **(meta or {})},
        }
        self.save_manifest()
        self._log(f"  [{name}] marked {status}"
                  + (f" ({len(rels)} artefacts)" if rels else ""))

    def mark_failed(self, name: str, error: str) -> None:
        """Record a phase as failed, with the error text.

        Parameters
        ----------
        name : str
            Phase name.
        error : str
            Error message or traceback summary.
        """
        prev = self.entry(name)
        self.manifest["phases"][name] = {
            "status": "failed",
            "config_hash": self.hash,
            "started": prev.get("started") or _now(),
            "finished": _now(),
            "depends_on": list(self.phases.get(name, {}).get("deps", [])),
            "artefacts": [],
            "meta": {"error": error[:2000]},
        }
        self.save_manifest()
        self._log(f"  [{name}] marked FAILED: {error.splitlines()[-1][:160]}")

    def maybe_skip(self, name: str, deps: Optional[Sequence[str]] = None
                   ) -> Tuple[Any, Optional[str]]:
        """Decide whether a phase should run, be skipped, or is blocked.

        Parameters
        ----------
        name : str
            Phase name.
        deps : sequence of str, optional
            Overrides the registry's dependency list.

        Returns
        -------
        (payload, reason) : tuple
            * ``(BLOCKED, reason)`` - an upstream dependency is unavailable, so this phase
              cannot run and the caller must define its variables without them.
            * ``(payload_dict, "cached")`` - the phase is reusable and its payload was loaded.
            * ``(None, None)`` - the phase must run.
        """
        if name in set(self.cfg.get("force_rerun", []) or []):
            self._log(f"[{name}] in force_rerun -> running despite any cache")
            return None, None

        want = list(deps) if deps is not None else list(self.phases.get(name, {}).get("deps", []))
        for dep in want:
            if dep not in self.phases:
                continue
            if not self.phase_done(dep):
                st = self.entry(dep)["status"]
                why = []
                if self.hash_mismatch(dep):
                    why.append("config hash changed")
                miss = self.artefacts_present(dep)
                if miss:
                    why.append(f"{len(miss)} artefact(s) missing")
                reason = (f"dependency {dep} unavailable (status={st}"
                          + (", " + ", ".join(why) if why else "") + ")")
                self.blocked_reasons[name] = reason
                self._log(f"BLOCKED {name}: {reason}")
                return BLOCKED, reason

        if self.hash_mismatch(name):
            self._log(f"WARNING: cached {name} was produced with a different config "
                      f"(cached {self.entry(name)['config_hash']}, current {self.hash}); rerunning")
        if self.phase_done(name):
            payload = self.load_phase(name)
            self._log(f"SKIPPED {name} (cached, hash={self.hash})")
            return payload, "cached"
        if self.entry(name)["status"] == "missing":
            self.manifest["phases"][name] = {
                "status": "running", "config_hash": self.hash, "started": _now(),
                "finished": None, "depends_on": want, "artefacts": [], "meta": {}}
            self.save_manifest()
        return None, None

    def load_phase(self, name: str) -> Dict[str, Any]:
        """Load a phase's saved artefacts.

        Parameters
        ----------
        name : str
            Phase name.

        Returns
        -------
        dict
            The payload written by :func:`save_payload`, plus ``_meta`` from the manifest.
        """
        payload = load_payload(self.results_root / name)
        payload["_meta"] = self.entry(name).get("meta", {})
        payload["_artefacts"] = list(self.entry(name).get("artefacts", []))
        return payload

    def results_dir(self, name: str) -> Path:
        """Directory holding a phase's results.

        Parameters
        ----------
        name : str
            Phase name.

        Returns
        -------
        Path
        """
        d = self.results_root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def ckpt_dir(self, variant: str) -> Path:
        """Checkpoint directory for one training variant.

        Parameters
        ----------
        variant : str
            Variant name, e.g. ``'vhw'``.

        Returns
        -------
        Path
        """
        d = self.ckpt_root / variant
        d.mkdir(parents=True, exist_ok=True)
        return d


# =========================================================================================
# Budget and device guards
# =========================================================================================
def working_usage(root: Path | str) -> Tuple[int, float, List[Tuple[str, float]]]:
    """Count files and bytes under a directory, with the largest files named.

    Parameters
    ----------
    root : path-like
        Directory to measure, normally ``/kaggle/working``.

    Returns
    -------
    (n_files, total_gb, largest) : tuple
        ``largest`` is a list of ``(relative_path, size_bytes)`` for the ten biggest files.
    """
    r = Path(root)
    sizes: List[Tuple[str, float]] = []
    total = 0
    for p in r.rglob("*"):
        if p.is_file():
            try:
                sz = p.stat().st_size
            except OSError:
                continue
            total += sz
            sizes.append((str(p.relative_to(r)), sz))
    sizes.sort(key=lambda kv: -kv[1])
    return len(sizes), total / 2 ** 30, sizes[:10]


def check_working_budget(root: Path | str, max_files: int, max_gb: float,
                         strict: bool = True) -> pd.DataFrame:
    """Measure ``/kaggle/working`` and enforce the Kaggle persistence caps.

    Kaggle allows roughly 500 files and 20 GB under ``/kaggle/working``; this notebook budgets
    for less so that a later phase cannot push a completed run over the edge.

    Parameters
    ----------
    root : path-like
        Directory to measure.
    max_files : int
        Maximum permitted file count.
    max_gb : float
        Maximum permitted size, GiB.
    strict : bool, default True
        Raise when a limit is exceeded. Set False to report only.

    Returns
    -------
    pd.DataFrame
        One row per top-level entry with its file count and size.

    Raises
    ------
    AssertionError
        If ``strict`` and either limit is exceeded.
    """
    r = Path(root)
    n, gb, largest = working_usage(r)
    rows = []
    for child in sorted(r.iterdir()) if r.is_dir() else []:
        if child.is_dir():
            cn, cgb, _ = working_usage(child)
        else:
            cn, cgb = 1, child.stat().st_size / 2 ** 30
        rows.append({"entry": child.name, "files": cn, "gb": round(cgb, 3)})
    df = pd.DataFrame(rows)
    print(f"/kaggle/working: {n} files, {gb:.2f} GiB "
          f"(limits: {max_files} files, {max_gb} GiB)")
    if len(df):
        print(df.sort_values("gb", ascending=False).to_string(index=False))
    if largest:
        print("largest files:")
        for rel, sz in largest:
            print(f"    {rel:56s} {sz / 2**20:9.1f} MiB")
    if strict:
        assert n < max_files, (
            f"/kaggle/working holds {n} files, at or above the {max_files} budget. Kaggle caps "
            f"the directory at about 500 and a completed run must be re-zippable; prune figures "
            "or write regenerable caches to /kaggle/temp.")
        assert gb < max_gb, (
            f"/kaggle/working holds {gb:.2f} GiB, at or above the {max_gb} GiB budget.")
    return df


@contextmanager
def cpu_guard(phase: str = ""):
    """Trip on any attempt to move a tensor to CUDA inside this block.

    This is the executable form of "phases 0-3 must run on CPU": rather than printing a device
    string and hoping, it makes an accidental ``.cuda()`` or ``.to('cuda')`` raise, so a phase
    that silently needed the GPU fails immediately and visibly instead of after an hour of
    EDA work on a CPU-only accelerator.

    Parameters
    ----------
    phase : str
        Phase name, used in the error message.

    Yields
    ------
    None
    """
    import torch

    label = phase or "this phase"
    orig_to = torch.Tensor.to
    orig_cuda = torch.Tensor.cuda
    orig_device = torch.device

    def _blocked(*a, **kw):
        dev = None
        for arg in a[1:]:
            if isinstance(arg, (str, torch.device)):
                dev = arg
        if dev is None:
            dev = kw.get("device")
        if isinstance(dev, str) and dev.startswith("cuda"):
            raise AssertionError(
                f"CUDA was requested during {label}, which must run on CPU. Either the phase "
                "needs the GPU (mark it requires_gpu in the phase registry) or a tensor was "
                "left on the default device by mistake.")
        return orig_to(*a, **kw)

    def _blocked_cuda(self, *a, **kw):
        raise AssertionError(
            f"Tensor.cuda() was called during {label}, which must run on CPU. Mark the phase "
            "requires_gpu in the registry if it genuinely needs an accelerator.")

    torch.Tensor.to = _blocked
    torch.Tensor.cuda = _blocked_cuda
    try:
        yield
    finally:
        torch.Tensor.to = orig_to
        torch.Tensor.cuda = orig_cuda
        torch.device = orig_device


# =========================================================================================
# Resume source discovery (addendum A3)
# =========================================================================================
def find_resume_source(working: Path | str, inputs: Sequence[Path | str] = ()) -> Dict[str, Any]:
    """Locate previous run state to resume from.

    Search order, per the addendum:

    1. ``<working>/run/manifest.json`` - an interactive session that kept its working directory;
    2. ``/kaggle/input/*/run/manifest.json`` - a previous committed version's output, attached by
       the user as an input dataset.

    Parameters
    ----------
    working : path-like
        The notebook's working directory.
    inputs : sequence of path-like
        Candidate input directories, normally ``/kaggle/input/*``.

    Returns
    -------
    dict
        ``{'source': 'working'|'input'|'fresh', 'path': Path|None, 'note': str}``.
    """
    working = Path(working)
    local = working / "run" / "manifest.json"
    found_inputs: List[Path] = []
    for base in inputs:
        b = Path(base)
        if not b.is_dir():
            continue
        for m in sorted(b.glob("run/manifest.json")):
            found_inputs.append(m.parent)
        for m in sorted(b.glob("*/run/manifest.json")):
            found_inputs.append(m.parent)

    if local.is_file() and found_inputs:
        return {"source": "working", "path": working / "run",
                "note": f"found run state in BOTH {working/'run'} and {len(found_inputs)} "
                        "attached input(s); preferring the working directory"}
    if local.is_file():
        return {"source": "working", "path": working / "run",
                "note": "resuming from the working directory's own run state"}
    if found_inputs:
        return {"source": "input", "path": found_inputs[0],
                "note": f"no local run state; copying from attached input {found_inputs[0]}"}
    return {"source": "fresh", "path": None,
            "note": "no previous run state found; starting fresh"}


def adopt_resume_source(working: Path | str, inputs: Sequence[Path | str] = (),
                        verbose: bool = True) -> Dict[str, Any]:
    """Copy an attached run directory into the working directory when appropriate.

    Parameters
    ----------
    working : path-like
        The notebook's working directory.
    inputs : sequence of path-like
        Candidate input directories.
    verbose : bool, default True
        Print what was found and done.

    Returns
    -------
    dict
        Whatever :func:`find_resume_source` returned, with ``adopted`` added.
    """
    info = find_resume_source(working, inputs)
    dest = Path(working) / "run"
    if info["source"] == "input" and info["path"] is not None:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(info["path"], dest)
        info["adopted"] = True
        if verbose:
            n = len(list(dest.rglob("*")))
            print(f"  adopted run state from {info['path']} ({n} entries)")
    else:
        info["adopted"] = False
    if verbose:
        print(f"  resume source: {info['source']} - {info['note']}")
    return info


def _now() -> str:
    """UTC timestamp, second resolution.

    Returns
    -------
    str
    """
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# =========================================================================================
# Test-set lock (addendum A5)
# =========================================================================================
def evaluate_test_lock(results_dir: Path | str, current_hashes: Dict[str, str],
                       allow_reeval: bool = False) -> Dict[str, Any]:
    """Decide whether the single-use test evaluation may run.

    Parameters
    ----------
    results_dir : path-like
        Directory holding ``final_eval.json``.
    current_hashes : dict
        ``{variant: sha256-of-best.pt}`` for the models about to be evaluated.
    allow_reeval : bool, default False
        Explicit user override from ``config.yaml: allow_reeval_after_change``.

    Returns
    -------
    dict
        ``{'action': 'run'|'reuse'|'blocked', 'reason': str, 'prior': dict, 'reeval': bool}``.
        ``action == 'reuse'`` means the cached evaluation is valid for these exact weights and
        the test sets must not be read again.
    """
    f = Path(results_dir) / "final_eval.json"
    if not f.is_file():
        return {"action": "run", "reason": "no previous final_eval.json",
                "prior": {}, "reeval": False}

    try:
        prior = json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"action": "run", "reason": f"previous final_eval.json is unreadable ({exc})",
                "prior": {}, "reeval": False}

    prior_hashes = prior.get("evaluated_best_hash", {}) or {}
    if not prior_hashes:
        return {"action": "run", "reason": "previous final_eval.json records no weight hashes",
                "prior": prior, "reeval": False}

    if prior_hashes == current_hashes:
        return {"action": "reuse",
                "reason": f"final_eval.json was produced from these exact weights "
                          f"({current_hashes}); re-reading Test-A/Test-B would spend the "
                          "single-use budget twice",
                "prior": prior, "reeval": False}

    changed = {k: {"was": prior_hashes.get(k), "now": v}
               for k, v in current_hashes.items() if prior_hashes.get(k) != v}
    if allow_reeval:
        return {"action": "run",
                "reason": f"weights changed ({changed}) but allow_reeval_after_change=true; "
                          "re-evaluating and stamping the record",
                "prior": prior, "reeval": True}
    return {"action": "blocked",
            "reason": (
                "Test-A/Test-B were already evaluated from different weights "
                f"({changed}). The brief allows each to be used exactly once, and evaluating a "
                "new model on them would be selection on the test set. Set "
                "allow_reeval_after_change: true in config.yaml to override, and be aware the "
                "model card will record that the override was used."),
            "prior": prior, "reeval": False}

# %%writefile src/phase_wrap.py
"""Phase wrapper: the pattern every phase cell follows.

Why a wrapper rather than a decorator
-------------------------------------
Each phase in this notebook needs to *define its own variables*, and those variables differ per
phase (fitted models, DataFrames, prediction arrays). A decorator cannot inject them. So the
pattern is explicit and uniform:

    _res, _why = PH("p05_b1")
    if _res is R.BLOCKED:
        b1 = b1_metrics = None
    elif _res is not None:
        <reload from _res>
    else:
        <compute>
        PH_done("p05_b1", artefacts=[...], meta={...})

Three helpers below keep the shape identical everywhere:

* :func:`begin` - the ``maybe_skip`` call plus a printed decision.
* :func:`done` - save a payload and mark the phase.
* :func:`guard` - a context manager combining the CPU tripwire with a forced device.

The ``BLOCKED`` branch is the part that matters most. When a dependency is unavailable (an
interrupted training phase, a blocked test-set lock), the dependent cell must still define its
variables, or every later cell raises ``NameError``. Assigning ``None`` makes the failure legible
at the point where the value is used, rather than as an unrelated crash three phases later.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

import run_state as R

#: Set when any phase has been blocked, so the final summary can say so loudly.
BLOCKED_PHASES: Dict[str, str] = {}

#: Every phase that was skipped because a dependency was unavailable, and why.
UNAVAILABLE: Dict[str, str] = {}


class PhaseUnavailable(RuntimeError):
    """Raised when a cell's inputs cannot exist because an upstream phase did not finish.

    A notebook cell that raises stops only that cell; the following cells still run. That is the
    behaviour we want here. The alternative — letting ``predict_index(None, ...)`` fail with an
    ``AttributeError`` three phases later — hides which decision caused the problem.

    The exception is caught by the final summary cell, which reports every unavailable phase in
    one place instead of leaving a trail of tracebacks.
    """


def begin(name: str, deps: Optional[Sequence[str]] = None) -> Tuple[Any, str]:
    """Ask the run state whether a phase should run, be skipped, or is blocked.

    Parameters
    ----------
    name : str
        Phase name from the registry.
    deps : sequence of str, optional
        Overrides the registry's dependency list.

    Returns
    -------
    (payload, reason) : tuple
        * ``(R.BLOCKED, reason)`` - a dependency is unavailable; the cell must define its
          variables as ``None`` and skip its body.
        * ``(payload, 'cached')`` - the phase is reusable; reload from ``payload``.
        * ``(None, 'run')`` - compute the phase, then call :func:`done`.
    """
    payload, reason = RUNSTATE.maybe_skip(name, deps)
    if payload is R.BLOCKED:
        BLOCKED_PHASES[name] = str(reason)
        return R.BLOCKED, str(reason)
    if reason == "cached":
        meta = payload.get("_meta", {})
        extra = ""
        if meta.get("epochs_completed") is not None:
            extra = (f" [{meta.get('epochs_completed')}/{meta.get('epochs_configured')} epochs, "
                     f"best {meta.get('best_val_mae_mm')} mm, "
                     f"stop={meta.get('stop_reason')}]")
        print(f"  loaded {name} from {RUNSTATE.results_dir(name)}"
              + (f", checkpoint {RUNSTATE.ckpt_dir(name.replace('p09_', '').replace('p08_', '')
                                                        .replace('p06_', ''))}"
                 if "train" in name else "")
              + extra)
        return payload, "cached"
    print(f"  running {name}")
    return None, "run"


def done(name: str, payload: Dict[str, Any], status: str = "done",
         extra_artefacts: Sequence[str] = (), meta: Optional[Dict[str, Any]] = None) -> List[str]:
    """Persist a phase payload and record it in the manifest.

    Parameters
    ----------
    name : str
        Phase name.
    payload : dict
        Values to write. DataFrames become parquet, ndarrays npz, everything else JSON.
    status : str, default 'done'
        One of ``run_state.STATUSES``. Use ``'partial'`` for an interrupted phase.
    extra_artefacts : sequence of str, optional
        Additional files the phase depends on, such as a checkpoint. Included in the manifest so
        their deletion also invalidates the phase.
    meta : dict, optional
        Provenance recorded alongside the status.

    Returns
    -------
    list of str
        Every artefact now listed for this phase.
    """
    arts = R.save_payload(RUNSTATE.results_dir(name), payload)
    arts = list(arts) + [str(a) for a in extra_artefacts]
    RUNSTATE.mark_done(name, artefacts=arts, meta=meta or {}, status=status)
    return arts


@contextmanager
def guard(name: str, requires_gpu: Optional[bool] = None):
    """Run a block under the right device policy for its phase.

    CPU-only phases are wrapped in :func:`run_state.cpu_guard`, which turns any attempt to move a
    tensor to CUDA into an immediate, named error rather than a silent dependence on an
    accelerator that may not be present.

    Parameters
    ----------
    name : str
        Phase name.
    requires_gpu : bool, optional
        Overrides the registry entry.

    Yields
    ------
    torch.device
        The device the phase should use.
    """
    need_gpu = (PHASES.get(name, {}).get("requires_gpu", False)
                if requires_gpu is None else requires_gpu)
    if need_gpu:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            print(f"  WARNING: {name} requires a GPU but none is available; running on CPU. "
                  "Expect this to be very slow.")
        yield dev
    else:
        with R.cpu_guard(name):
            yield torch.device("cpu")


def blocked_reason(name: str) -> str:
    """Why a phase was blocked, or an empty string.

    Parameters
    ----------
    name : str
        Phase name.

    Returns
    -------
    str
    """
    return BLOCKED_PHASES.get(name, "")


def any_blocked() -> Dict[str, str]:
    """Every blocked phase and its reason.

    Returns
    -------
    dict
    """
    return dict(BLOCKED_PHASES)


def require(*names: str) -> None:
    """Stop the current cell if any named phase did not produce usable output.

    Used at the top of every cell that consumes a trained network. A cell that raises stops only
    itself, so the rest of the notebook continues; letting ``predict_index(None, ...)`` fail
    instead would surface three phases later as an ``AttributeError`` that says nothing about the
    real cause.

    Parameters
    ----------
    *names : str
        Phase names whose outputs this cell needs.

    Raises
    ------
    PhaseUnavailable
        If any of them is unavailable. The message names the phase and the reason.
    """
    missing = [n for n in names
               if RUNSTATE.entry(n)["status"] in ("partial", "blocked", "failed", "missing")]
    if not missing:
        return
    lines = []
    for n in missing:
        st = RUNSTATE.entry(n)
        why = (st.get("meta") or {}).get("stop_reason") or st["status"]
        lines.append(f"  {n}: status={st['status']} ({why})")
        UNAVAILABLE[n] = f"{st['status']}: {why}"
    print("=" * 78)
    print("THIS CELL SKIPPED - a phase it depends on did not finish:")
    print("\n".join(lines))
    print("Fix that phase first (add its name to config.yaml force_rerun, or attach the previous")
    print("version's output as an input dataset) and run the notebook again.")
    print("=" * 78)
    raise PhaseUnavailable("; ".join(missing))