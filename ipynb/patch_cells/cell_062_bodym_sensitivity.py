"""Boundary-sensitivity sweep: how much measurement error does a segmentation error cause?"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm          # fix A3: used in sweep_b2

from bodym_perturb import apply_perturbation
from bodym_baselines import B1Geometric


def build_grid(cfg: Dict) -> List[Tuple[str, float]]:
    """Enumerate every (family, magnitude) pair to sweep.

    Parameters
    ----------
    cfg : dict
        Notebook configuration (``erosion_px``, ``jitter_sigma_px``, ``downsample_factors``).

    Returns
    -------
    list of (str, float)
        Pairs including the unperturbed identity case ``('none', 0.0)`` first.
    """
    grid: List[Tuple[str, float]] = [("none", 0.0)]
    grid += [("morph", float(k)) for k in range(-cfg["erosion_px"], cfg["erosion_px"] + 1)]
    grid += [("jitter", float(s)) for s in cfg["jitter_sigma_px"] if s > 0]
    grid += [("downsample", float(f)) for f in cfg["downsample_factors"]]
    return grid


def sweep_b1(index, subject_ids: Sequence[str], targets: Sequence[str], cfg: Dict,
             grid: Sequence[Tuple[str, float]], seed: int = 0,
             fit_subject_ids: Optional[Sequence[str]] = None,
             model_name: str = "B1") -> pd.DataFrame:
    """Run the B1 geometric model across the perturbation grid.

    B1 is cheap enough to **re-fit** under every perturbation, which is the more informative
    choice: the per-measurement linear calibration absorbs any *global* scale shift, so what
    survives in its dMAE is the effect of local boundary change. Fitting uses the fit split,
    evaluated on the validation subjects.

    Parameters
    ----------
    index : BodyMIndex
        Assembled index.
    subject_ids : sequence of str
        Validation subjects to evaluate, one photo each.
    targets : sequence of str
        Target names.
    cfg : dict
        Notebook configuration.
    grid : sequence of (str, float)
        Perturbation grid from :func:`build_grid`.
    seed : int, default 0
        Seed for the jitter RNG.
    fit_subject_ids : sequence of str, optional
        Subjects used to fit the calibration under each perturbation. Defaults to
        ``subject_ids`` (self-calibration), which isolates the perturbation effect.
    model_name : str, default "B1"
        Value written to the ``model`` column.

    Returns
    -------
    pd.DataFrame
        One row per (family, magnitude, measurement) with ``mae_mm`` and ``delta_mae_mm``.
    """
    fit_ids = sorted(set(fit_subject_ids)) if fit_subject_ids else sorted(set(subject_ids))
    val_ids = sorted(set(subject_ids))
    photos = (index.photos.sort_values("photo_id").groupby("subject_id", as_index=False).first()
              .set_index("subject_id"))
    meta = index.subjects.set_index("subject_id")
    height = meta["height_cm"]
    Y_fit = meta.loc[fit_ids, list(targets)].to_numpy(float)
    Y_val = meta.loc[val_ids, list(targets)].to_numpy(float)

    def _features(ids: Sequence[str], fam: str, mag: float, rng_seed: int) -> np.ndarray:
        """Perturb both views of every subject and extract B1 features.

        Parameters
        ----------
        ids : sequence of str
            Subjects to process.
        fam : str
            Perturbation family.
        mag : float
            Perturbation magnitude.
        rng_seed : int
            Seed; a distinct seed per view set keeps front and side perturbations independent
            while remaining reproducible.

        Returns
        -------
        ndarray, shape (len(ids), n_features)
        """
        rng = np.random.default_rng(rng_seed)
        out = []
        for sid in ids:
            row = photos.loc[sid]
            f = apply_perturbation(np.array(Image.open(row.front_mask)), fam, mag, rng)
            s = apply_perturbation(np.array(Image.open(row.side_mask)), fam, mag, rng)
            out.append(B1Geometric(threshold=cfg["mask_threshold"]).features(
                f, s, float(height[sid])))
        return np.asarray(out)

    rows: List[dict] = []
    for fam, mag in grid:
        F_fit = _features(fit_ids, fam, mag, seed)
        F_val = _features(val_ids, fam, mag, seed + 1)
        model = B1Geometric(threshold=cfg["mask_threshold"]).fit(F_fit, Y_fit, targets)
        mae_mm = np.abs(model.predict(F_val) - Y_val).mean(axis=0) * 10.0
        rows.extend({"model": model_name, "family": fam, "magnitude": mag, "measurement": t,
                     "mae_mm": float(m)} for t, m in zip(targets, mae_mm))
    tbl = pd.DataFrame(rows)
    base = (tbl[tbl.family == "none"].set_index("measurement")["mae_mm"])
    tbl["delta_mae_mm"] = tbl.apply(lambda r: r.mae_mm - base[r.measurement], axis=1)
    return tbl


def sweep_model(model, index, subject_ids: Sequence[str], targets: Sequence[str], cfg: Dict,
                grid: Sequence[Tuple[str, float]], device, seed: int = 0,
                model_name: str = "B2", max_pairs_per_subject: Optional[int] = None
                ) -> pd.DataFrame:
    """Run a trained CNN across the perturbation grid, per photo pair (fix B5, B9).

    Metrics are computed over **one row per photo pair**, matching deployment. The previous
    version averaged a subject's photos before scoring, so a subject with many photos
    contributed a much lower-variance (and smaller) error than a subject with one - which is
    not what ``infer.py`` will do at capture time.

    Parameters
    ----------
    model : nn.Module
        Trained network with a ``use_weight`` attribute.
    index : BodyMIndex
        Assembled index.
    subject_ids : sequence of str
        Subjects to evaluate (validation only).
    targets : sequence of str
        Target names.
    cfg : dict
        Notebook configuration.
    grid : sequence of (str, float)
        Perturbation grid from :func:`build_grid`.
    device : torch.device
        Compute device.
    seed : int, default 0
        Seed for the jitter RNG.
    model_name : str, default "B2"
        Value written to the ``model`` column.
    max_pairs_per_subject : int, optional
        Cap on photo pairs per subject, for cost control.

    Returns
    -------
    pd.DataFrame
        One row per (family, magnitude, measurement) with ``mae_mm`` and ``delta_mae_mm``.
    """
    import torch
    from bodym_cnn import to_tensor

    photos = index.photos[index.photos.subject_id.isin(set(subject_ids))]
    if max_pairs_per_subject:
        photos = (photos.sort_values("photo_id")
                  .groupby("subject_id", head=0, as_index=False)
                  .head(max_pairs_per_subject))
    photos = photos.reset_index(drop=True)
    meta = index.subjects.set_index("subject_id")
    height, weight = meta["height_cm"].to_dict(), meta["weight_kg"].to_dict()
    Y_by_subject = {s: meta.loc[s, list(targets)].to_numpy(float) for s in meta.index}

    use_w = getattr(model, "use_weight", True)   # fix A4: local is use_w, not use_weight
    model.eval()
    rows: List[dict] = []
    with torch.no_grad():
        for fam, mag in grid:
            rng = np.random.default_rng(seed)
            preds: List[np.ndarray] = []
            trues: List[np.ndarray] = []
            for row in tqdm(photos.itertuples(), total=len(photos),
                            desc=f"{model_name} {fam}={mag}", leave=False):
                f = apply_perturbation(np.array(Image.open(row.front_mask)), fam, mag, rng)
                s = apply_perturbation(np.array(Image.open(row.side_mask)), fam, mag, rng)
                x = to_tensor(f, s, height[row.subject_id], weight[row.subject_id],
                              cfg["img_height"], cfg["img_width"], cfg["mask_threshold"],
                              use_weight=use_w)
                xb = torch.from_numpy(x).unsqueeze(0).to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    p = model(xb).float().cpu().numpy()[0]
                preds.append(p)
                trues.append(Y_by_subject[row.subject_id])
            mae_mm = np.abs(np.stack(preds) - np.stack(trues)).mean(axis=0) * 10.0
            rows.extend({"model": model_name, "family": fam, "magnitude": mag, "measurement": t,
                         "mae_mm": float(m)} for t, m in zip(targets, mae_mm))

    tbl = pd.DataFrame(rows)
    base = tbl[tbl.family == "none"].set_index("measurement")["mae_mm"]
    tbl["delta_mae_mm"] = tbl.apply(lambda r: r.mae_mm - base[r.measurement], axis=1)
    return tbl


def sweep_b2(model, index, subject_ids: Sequence[str], targets: Sequence[str], cfg: Dict,
             grid: Sequence[Tuple[str, float]], device, seed: int = 0) -> pd.DataFrame:
    """Backwards-compatible alias for :func:`sweep_model` with ``model_name='B2'``.

    Parameters
    ----------
    model : nn.Module
        Trained network.
    index : BodyMIndex
        Assembled index.
    subject_ids : sequence of str
        Subjects to evaluate.
    targets : sequence of str
        Target names.
    cfg : dict
        Notebook configuration.
    grid : sequence of (str, float)
        Perturbation grid.
    device : torch.device
        Compute device.
    seed : int, default 0
        Seed for the jitter RNG.

    Returns
    -------
    pd.DataFrame
        Sweep output.
    """
    return sweep_model(model, index, subject_ids, targets, cfg, grid, device, seed, "B2")


def _passes(tbl: pd.DataFrame, tolerance_mm: float, mode: str) -> pd.Series:
    """Boolean mask of rows whose error is inside ``tolerance_mm``.

    Parameters
    ----------
    tbl : DataFrame
        Sensitivity rows for one (model, measurement, family, side).
    tolerance_mm : float
        Threshold, mm.
    mode : {'total', 'delta'}
        ``'total'`` tests ``mae_mm <= tolerance`` - the error the caller would actually
        observe. ``'delta'`` tests ``delta_mae_mm <= tolerance``, i.e. the degradation
        attributable to the perturbation alone.

    Returns
    -------
    pd.Series of bool
        One flag per row, indexed by magnitude.
    """
    col = "mae_mm" if mode == "total" else "delta_mae_mm"
    return tbl[col] <= tolerance_mm


def max_tolerable_boundary_error(sensitivity: pd.DataFrame, key_measurements: Sequence[str],
                                 tolerances_mm: Sequence[float], mm_per_px: float,
                                 mode: str = "total") -> pd.DataFrame:
    """Largest boundary perturbation whose error stays inside each tolerance band (fix B8).

    Four corrections to the previous definition, each of which changed the answer materially:

    1. **Criterion.** The previous version tested ``delta_mae_mm <= tolerance`` - the
       *additional* error caused by the perturbation, ignoring the model's baseline error. A
       model already at 20 mm MAE on the chest would be reported as tolerating unlimited
       boundary error at a 25 mm threshold, because its delta was small. The default here is
       ``mode='total'``: ``mae_mm <= tolerance``, i.e. the error the caller actually sees.
       ``mode='delta'`` is also computed and labelled, so the old figure remains visible.
    2. **Contiguity.** The previous version took ``max(|magnitude|)`` over *all* passing rows,
       so a non-monotone curve could report ``k=6`` as tolerable when ``k=3`` failed. This
       version walks outward from 0 on each side and stops at the first failure.
    3. **Downsample.** A factor is not a distance. Converting it to millimetres via
       ``mm_per_px`` is meaningless, so the downsample row reports the largest passing
       **factor** and its ``unit`` is ``'factor'``, never ``'px'``/``'mm'``.
    4. **Nothing passes.** The previous version emitted ``0.0``, which is indistinguishable
       from "only k=0 passes". This version emits the string ``'none tolerable'``.

    Parameters
    ----------
    sensitivity : DataFrame
        Sweep output with columns ``model``, ``family``, ``magnitude``, ``measurement``,
        ``mae_mm``, ``delta_mae_mm``.
    key_measurements : sequence of str
        Measurements to report.
    tolerances_mm : sequence of float
        Tolerance thresholds, e.g. (9, 15, 25).
    mm_per_px : float
        Nominal millimetres per pixel from the height-derived scale.
    mode : {'total', 'delta'}, default 'total'
        Which error definition is authoritative; the other is emitted alongside, labelled.

    Returns
    -------
    pd.DataFrame
        One row per (model, measurement, tolerance, family, side) with ``max_tolerable_px``,
        ``max_tolerable_mm``, ``unit``, ``criterion`` and ``binding_measurement``.
    """
    out: List[dict] = []
    sensitivity = sensitivity.copy()
    if "delta_mae_mm" not in sensitivity.columns:
        # derive it from the identity row rather than requiring the caller to have computed it
        _base = (sensitivity[sensitivity.family == "none"]
                 .set_index(["model", "measurement"])["mae_mm"])
        if len(_base) == 0:
            raise ValueError("sensitivity frame has no 'none' row to difference against")
        sensitivity["delta_mae_mm"] = (
            sensitivity["mae_mm"]
            - pd.MultiIndex.from_arrays([sensitivity["model"], sensitivity["measurement"]])
            .map(_base).to_numpy())
    for model, msub in sensitivity.groupby("model"):
        for meas in key_measurements:
            if meas not in set(msub.measurement):
                continue
            for fam, fsub in msub[msub.measurement == meas].groupby("family"):
                if fam == "none":
                    continue
                # erosion and dilation are separate budgets: a segmenter biased loose and one
                # biased tight are different failure modes with different consequences.
                if fam == "morph":
                    sides = {"erosion": fsub[fsub.magnitude < 0].sort_values("magnitude",
                                                                             ascending=False),
                             "dilation": fsub[fsub.magnitude > 0].sort_values("magnitude")}
                else:
                    sides = {"both": fsub.sort_values("magnitude")}

                for side, sub in sides.items():
                    if sub.empty:
                        continue
                    for tol in tolerances_mm:
                        for criterion, use_mode in (("total", mode), ("delta", "delta")):
                            ok = _passes(sub, tol, use_mode)
                            val, unit = _contiguous_limit(sub, ok, fam, mm_per_px)
                            is_px = unit == "px"
                            num = float(val) if isinstance(val, (int, float, np.floating)) \
                                else np.nan
                            out.append({
                                "model": model, "measurement": meas, "tolerance_mm": tol,
                                "family": fam, "side": side, "criterion": criterion,
                                # `max_tolerable` always carries the answer, in `unit`; it is the
                                # string 'none tolerable' when even magnitude 0 fails
                                "max_tolerable": val,
                                "unit": unit,
                                # px/mm exist only for the pixel families; a downsample factor
                                # has no distance equivalent and is never converted
                                "max_tolerable_px": num if is_px else np.nan,
                                "max_tolerable_mm": (num * mm_per_px) if is_px else np.nan,
                                "authoritative": criterion == mode,
                            })
    df = pd.DataFrame(out)
    if df.empty:
        return df
    return df.sort_values(["model", "measurement", "family", "side", "criterion",
                           "tolerance_mm"]).reset_index(drop=True)


def _contiguous_limit(sub: pd.DataFrame, ok: pd.Series, fam: str, mm_per_px: float):
    """Walk outward from zero and stop at the first failing magnitude (fix B8 contiguity).

    Parameters
    ----------
    sub : DataFrame
        Rows for one (model, measurement, family, side), sorted by magnitude.
    ok : pd.Series of bool
        Pass/fail per magnitude, index-aligned with ``sub``.
    fam : str
        Perturbation family.
    mm_per_px : float
        Nominal millimetres per pixel.

    Returns
    -------
    (value, unit) : tuple
        ``value`` is the largest contiguous magnitude from 0 that passes, or the string
        ``'none tolerable'`` when magnitude 0 itself fails. ``unit`` is ``'px'`` or ``'factor'``.
    """
    sub = sub.reset_index(drop=True)
    ok = pd.Series(ok).reset_index(drop=True).to_numpy(dtype=bool)
    mags = sub["magnitude"].to_numpy(dtype=float)

    # start from the entry closest to zero and walk outward in |magnitude|
    order = np.argsort(np.abs(mags), kind="stable")
    limit = None
    for i in order:
        if not ok[i]:
            break
        limit = mags[i]
    if limit is None:
        return "none tolerable", "factor" if fam == "downsample" else "px"
    if fam == "downsample":
        return float(limit), "factor"
    return float(abs(limit)), "px"


def augmentation_magnitudes(sensitivity: pd.DataFrame, targets: Sequence[str], cfg: Dict,
                            model: str = "B2") -> Dict[str, object]:
    """Derive the training augmentation magnitudes from the sweep (fix B10).

    The previous rule took the largest magnitude at which *any* measurement passed the
    loosest tolerance, while the surrounding markdown claimed the magnitudes were the largest
    at which *all* measurements passed. Those are very different numbers: one generous
    measurement can carry a magnitude that eleven other measurements fail.

    Parameters
    ----------
    sensitivity : DataFrame
        Sweep output.
    targets : sequence of str
        All target names.
    cfg : dict
        Notebook configuration (``tolerance_mm``, ``downsample_factors``,
        ``augmentation_exclude_targets``).
    model : str, default "B2"
        Which model's sweep to read. B2 is used because the sweep must precede training.

    Returns
    -------
    dict
        ``erosion_px``, ``jitter_sigma_px``, ``downsample_factor``, ``binding_measurement``
        and ``rotation_deg`` / ``scale_jitter`` / ``translate_frac``.
    """
    tol = max(cfg["tolerance_mm"])
    excluded = set(cfg.get("augmentation_exclude_targets", ["height"]))
    required = [t for t in targets if t not in excluded]
    sub = sensitivity[sensitivity.model == model]

    def largest_contiguous(fam: str, mags: Sequence[float]) -> Tuple[float, Optional[str]]:
        """Largest magnitude from 0 where every required measurement passes.

        Parameters
        ----------
        fam : str
            Perturbation family.
        mags : sequence of float
            Candidate magnitudes, ordered by increasing |magnitude|.

        Returns
        -------
        (value, binding) : tuple
            ``value`` is 0.0 when nothing passes; ``binding`` names the measurement that
            failed first, or None.
        """
        best, binding = 0.0, None
        for m in mags:
            blk = sub[(sub.family == fam) & (sub.magnitude == m)]
            failed = [t for t in required
                      if t in set(blk.measurement)
                      and float(blk[blk.measurement == t].mae_mm.iloc[0]) > tol]
            if failed:
                binding = failed[0]
                break
            best = float(abs(m))
        return best, binding

    morph_mags = sorted(sub[sub.family == "morph"].magnitude.unique(),
                        key=lambda v: (abs(v), v))
    erosion_mags = sorted([m for m in morph_mags if m < 0], key=lambda v: (abs(v), v))
    dil_mags = sorted([m for m in morph_mags if m > 0], key=lambda v: (abs(v), v))
    erosion, b_ero = largest_contiguous("morph", erosion_mags)
    dilation, b_dil = largest_contiguous("morph", dil_mags)
    jit_mags = sorted(sub[sub.family == "jitter"].magnitude.unique())
    jitter, b_jit = largest_contiguous("jitter", jit_mags)

    binding = next((b for b in (b_ero, b_dil, b_jit) if b), None)
    return {
        "erosion_px": int(erosion),
        "dilation_px": int(dilation),
        "jitter_sigma_px": float(jitter),
        # capped at the smallest swept factor: x8 is a catastrophic mask and training on it
        # teaches the network to expect holes it will never see in deployment
        "downsample_factor": int(min(cfg["downsample_factors"])),
        "tolerance_mm_used": tol,
        "required_targets": required,
        "excluded_targets": sorted(excluded),
        "binding_measurement": binding,
        "rotation_deg": 3.0,
        "scale_jitter": 0.03,
        "translate_frac": 0.03,
    }