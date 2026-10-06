"""Discovery, integrity checks and tabular assembly for the BodyM dataset.

References
----------
Ruiz, N., Bellver, M., Bolkart, T., Arora, A., Lin, M. C., Romero, J., & Bala, R. (2022).
Human body measurement estimation with adversarial augmentation (arXiv:2210.05667).
https://arxiv.org/abs/2210.05667
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

SPLITS: Tuple[str, ...] = ("train", "testA", "testB")

#: Subject counts and silhouette counts published with BodyM (Ruiz et al., 2022).
EXPECTED: Dict[str, Tuple[int, int]] = {"train": (2018, 6134), "testA": (87, 1684),
                                        "testB": (400, 1160)}

#: The 14 measurement columns, in the order they appear in ``measurements.csv``.
MEASUREMENT_COLUMNS: Tuple[str, ...] = (
    "ankle", "arm-length", "bicep", "calf", "chest", "forearm", "height", "hip",
    "leg-length", "shoulder-breadth", "shoulder-to-crotch", "thigh", "waist", "wrist",
)

#: Ground-truth definition, identical for every column (fix D4). Ruiz et al. (2022) release
#: the BodyM measurements as vertex-path lengths on an SMPL-registered mesh and do not
#: publish the vertex path behind each column name, so a per-column anatomical description
#: would be invention. See DECISIONS.md.
DEFINITION_TEMPLATE: str = (
    "BodyM measurement; vertex-path length on an SMPL-registered mesh "
    "(Ruiz et al., 2022). Exact path not published. Not an ISO 20685 tape measurement."
)

DEFINITIONS: Dict[str, str] = {c: DEFINITION_TEMPLATE for c in MEASUREMENT_COLUMNS}


def find_bodym_root(search_roots: Sequence[Path | str]) -> Path:
    """Locate the ``bodym/`` root that contains the three splits.

    The search is structural: a candidate must contain ``train``, ``testA`` and ``testB``,
    each with ``measurements.csv`` plus ``mask/`` and ``mask_left/`` directories.

    Parameters
    ----------
    search_roots : sequence of path-like
        Directories to scan recursively, typically ``["/kaggle/input", "."]``.

    Returns
    -------
    Path
        Path to the directory that holds the three split folders.

    Raises
    ------
    FileNotFoundError
        If no structural candidate is found, or if more than one is found (ambiguous input).
    """
    candidates: List[Path] = []
    for base in (Path(r) for r in search_roots):
        if not base.exists():
            continue
        for csv_path in sorted(base.rglob("measurements.csv")):
            split_dir = csv_path.parent
            if split_dir.name not in SPLITS:
                continue
            root = split_dir.parent
            if not all((root / s / "mask").is_dir() and (root / s / "mask_left").is_dir()
                       for s in SPLITS):
                continue
            if root not in candidates:
                candidates.append(root)
    if not candidates:
        raise FileNotFoundError(
            "No BodyM root found. Expected a directory containing train/testA/testB, each with "
            "measurements.csv, hwg_metadata.csv, subject_to_photo_map.csv, mask/, mask_left/. "
            f"Searched: {[str(r) for r in search_roots]}")
    if len(candidates) > 1:
        raise FileNotFoundError(f"Ambiguous BodyM roots: {[str(c) for c in candidates]}")
    return candidates[0]


def load_split(root: Path | str, split: str) -> Dict[str, pd.DataFrame]:
    """Load the three CSV tables of one split.

    Parameters
    ----------
    root : path-like
        BodyM root (the directory containing ``train``/``testA``/``testB``).
    split : {'train', 'testA', 'testB'}
        Split name.

    Returns
    -------
    dict
        Keys ``measurements``, ``hwg``, ``photos``.

    Raises
    ------
    ValueError
        If ``split`` is not one of the three known splits.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    d = Path(root) / split
    return {
        "measurements": pd.read_csv(d / "measurements.csv"),
        "hwg": pd.read_csv(d / "hwg_metadata.csv"),
        "photos": pd.read_csv(d / "subject_to_photo_map.csv"),
    }


def bmi_band(bmi_kg_m2, edges: Sequence[float]) -> pd.Series:
    """Bin BMI into the five clinical bands used throughout this notebook.

    Parameters
    ----------
    bmi_kg_m2 : array-like of float
        BMI in kg/m^2.
    edges : sequence of float
        Strictly increasing band edges, e.g. ``[0, 18.5, 25, 30, 40, 100]``.

    Returns
    -------
    pd.Series
        Band index per row, as int.
    """
    return pd.cut(pd.Series(np.asarray(bmi_kg_m2, dtype=float)), bins=list(edges),
                  labels=False, include_lowest=True).astype("Int64")


def verify_integrity(root: Path | str, verbose: bool = True) -> pd.DataFrame:
    """Validate BodyM against the published counts and cross-table consistency.

    Checks performed
    ----------------
    1. Subject and silhouette counts per split equal :data:`EXPECTED` (Ruiz et al., 2022).
    2. The three tables of a split describe the same subject set.
    3. No missing values in measurements or height/weight/sex.
    4. Every ``photo_id`` resolves to both a front and a side mask file.
    5. Warning-level: identical photo ids or subject ids shared across splits.

    Parameters
    ----------
    root : path-like
        BodyM root.
    verbose : bool, default True
        Print a per-check report.

    Returns
    -------
    pandas.DataFrame
        One row per check: ``check``, ``status`` in {ok, warn, fail}, ``detail``.

    Raises
    ------
    AssertionError
        If any check has status ``fail``.
    """
    root = Path(root)
    rows: List[dict] = []

    def add(check: str, status: str, detail: str) -> None:
        rows.append({"check": check, "status": status, "detail": detail})
        if verbose:
            print(f"[{status.upper():4s}] {check}: {detail}")

    tables = {s: load_split(root, s) for s in SPLITS}

    for split, tab in tables.items():
        exp_subj, exp_photo = EXPECTED[split]
        n_subj = tab["measurements"]["subject_id"].nunique()
        n_photo = len(tab["photos"])
        if (n_subj, n_photo) == (exp_subj, exp_photo):
            add(f"counts/{split}", "ok", f"{n_subj} subjects / {n_photo} silhouettes "
                                        f"(expected {exp_subj}/{exp_photo})")
        else:
            add(f"counts/{split}", "fail", f"{n_subj} subjects / {n_photo} silhouettes "
                                          f"(expected {exp_subj}/{exp_photo})")

        sets = {k: set(v["subject_id"]) for k, v in tab.items()}
        if sets["measurements"] == sets["hwg"] == sets["photos"]:
            add(f"subject_sets/{split}", "ok", f"{n_subj} ids identical across the 3 tables")
        else:
            add(f"subject_sets/{split}", "fail",
                f"mismatch: meas-hwg {len(sets['measurements'] - sets['hwg'])}, "
                f"meas-photos {len(sets['measurements'] - sets['photos'])}")

        nan_m = int(tab["measurements"].isna().sum().sum())
        nan_h = int(tab["hwg"].isna().sum().sum())
        add(f"missing_values/{split}", "ok" if nan_m + nan_h == 0 else "fail",
            f"{nan_m} null in measurements, {nan_h} null in hwg")

        photos = set(tab["photos"]["photo_id"])
        front = {p.stem for p in (root / split / "mask").glob("*.png")}
        side = {p.stem for p in (root / split / "mask_left").glob("*.png")}
        if photos <= front and photos <= side and photos == front == side:
            add(f"mask_pairs/{split}", "ok",
                f"{len(photos)} photo ids, all with a front and a side mask; no orphans")
        else:
            add(f"mask_pairs/{split}", "fail",
                f"missing front {len(photos - front)}, missing side {len(photos - side)}, "
                f"orphan front {len(front - photos)}, orphan side {len(side - photos)}")

    # cross-split leakage warnings
    for a, b in (("train", "testA"), ("train", "testB"), ("testA", "testB")):
        shared_s = (set(tables[a]["measurements"]["subject_id"])
                    & set(tables[b]["measurements"]["subject_id"]))
        shared_p = (set(tables[a]["photos"]["photo_id"])
                    & set(tables[b]["photos"]["photo_id"]))
        status = "ok" if not shared_s and not shared_p else "warn"
        add(f"leakage/{a}->{b}", status,
            f"{len(shared_s)} shared subject ids, {len(shared_p)} shared photo ids"
            + (f" -> {sorted(shared_s)[:3]}" if shared_s else ""))

    # height column appears twice; quantify the disagreement instead of silently choosing
    for split, tab in tables.items():
        m = tab["measurements"].merge(tab["hwg"], on="subject_id")
        d = (m["height"] - m["height_cm"]).abs()
        add(f"height_sources/{split}", "warn",
            f"measurements.height vs hwg.height_cm: mean|diff| {d.mean():.2f} cm, "
            f"max {d.max():.2f} cm, corr {m['height'].corr(m['height_cm']):.4f}. "
            "Two different height definitions ship with BodyM; see DECISIONS.md.")

    report = pd.DataFrame(rows)
    if verbose:
        print()
        print(report["status"].value_counts().to_string())
    fails = report.loc[report["status"] == "fail", "check"].tolist()
    assert not fails, f"BodyM integrity check FAILED for: {fails}"
    return report


@dataclass(frozen=True)
class BodyMIndex:
    """One assembled BodyM view: a subject's metadata plus its photos."""

    split: str
    subjects: pd.DataFrame   # one row per subject: sex, height_cm, weight_kg, bmi, bmi_band, targets
    photos: pd.DataFrame     # one row per silhouette: subject_id, photo_id, split, mask paths
    root: Path

    def subjects_in(self, subject_ids) -> "BodyMIndex":
        """Return a new index restricted to ``subject_ids``.

        Parameters
        ----------
        subject_ids : iterable of str
            Subject ids to keep.

        Returns
        -------
        BodyMIndex
            Filtered copy (same dataclass, no copying of arrays).
        """
        keep = set(subject_ids)
        return BodyMIndex(self.split, self.subjects[self.subjects.subject_id.isin(keep)].copy(),
                          self.photos[self.photos.subject_id.isin(keep)].copy(), self.root)


def build_index(root: Path | str, split: str) -> BodyMIndex:
    """Assemble the per-subject and per-silhouette tables for one split.

    Adds to the raw CSVs: ``bmi``, ``bmi_band``, ``sex`` as an int, and absolute paths to the
    front/side masks.

    Parameters
    ----------
    root : path-like
        BodyM root.
    split : {'train', 'testA', 'testB'}
        Split name.

    Returns
    -------
    BodyMIndex
    """
    root = Path(root)
    tab = load_split(root, split)
    targets = list(MEASUREMENT_COLUMNS)

    subs = (tab["measurements"]
            .merge(tab["hwg"], on="subject_id", how="inner", validate="one_to_one"))
    subs["bmi"] = subs["weight_kg"] / (subs["height_cm"] / 100.0) ** 2
    subs["sex"] = subs["gender"].str.lower().map({"female": 0, "male": 1})
    assert subs["sex"].notna().all(), "unexpected gender label(s) in hwg_metadata.csv"

    photos = tab["photos"].copy()
    photos["split"] = split
    photos["front_mask"] = [str(root / split / "mask" / f"{p}.png") for p in photos["photo_id"]]
    photos["side_mask"] = [str(root / split / "mask_left" / f"{p}.png") for p in photos["photo_id"]]
    return BodyMIndex(split, subs, photos, root)


def target_columns(cfg: Dict[str, Any]) -> List[str]:
    """Return the model output names, honouring ``cfg['include_height_as_target']``.

    Parameters
    ----------
    cfg : dict
        Notebook configuration.

    Returns
    -------
    list of str
        The 14 measurement columns, or the 13 non-height columns when height is excluded.
    """
    cols = [c for c in MEASUREMENT_COLUMNS if c != "height"]
    return list(MEASUREMENT_COLUMNS) if cfg.get("include_height_as_target", True) else cols
