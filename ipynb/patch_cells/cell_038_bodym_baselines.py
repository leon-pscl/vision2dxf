"""Non-visual (B0) and geometric (B1) baselines.

Both answer one question: does the silhouette add information beyond height, weight and sex?
Neither uses a neural network, so neither can memorise a subject.

References
----------
Ramanujan, S. (1914). Modular equations and approximations to pi. Quarterly Journal of
  Mathematics, 45, 350-372.   (the ellipse-perimeter approximation used below)
Bland, J. M., & Altman, D. G. (1986). Statistical methods for assessing agreement between two
  methods of clinical measurement. The Lancet, 1(8476), 307-310.   (metric suite)

The slice-height table in :data:`SLICE_HEIGHTS_FROM_FLOOR` is an [Assumption]: proportional
anthropometric heuristics. It is NOT BodyM ground truth and no published source is claimed
for it (fix D5 removed the unverifiable Drillis & Contini citation). Its residual bias is
reported per measurement by the per-measurement calibration.

Units
-----
Heights and perimeters are millimetres (mm) internally; stature is centimetres (cm).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage as ndi


# =========================================================================================
# B0 - non-visual
# =========================================================================================
def b0_features(subjects: pd.DataFrame) -> np.ndarray:
    """Build the B0 design matrix.

    Parameters
    ----------
    subjects : DataFrame
        Per-subject table containing ``height_cm``, ``weight_kg``, ``sex`` (0 female, 1 male).

    Returns
    -------
    ndarray, shape (n, 5)
        ``[height_cm, weight_kg, sex, bmi, height_cm * weight_kg]``. BMI and the interaction
        term are included because girth is not linear in either variable alone.
    """
    h = subjects["height_cm"].to_numpy(float)
    w = subjects["weight_kg"].to_numpy(float)
    s = subjects["sex"].to_numpy(float)
    bmi = w / (h / 100.0) ** 2
    return np.column_stack([h, w, s, bmi, h * w])


class B0NonVisual:
    """Ridge and gradient-boosting baseline over height, weight and sex.

    Ridge is the interpretable closed-form model; gradient boosting captures the nonlinear
    height-weight-girth coupling. Both are fitted so the ridge-vs-boosting gap quantifies how
    nonlinear the relation is.

    Parameters
    ----------
    ridge_alpha : float, default 10.0
        L2 penalty for the ridge model.
    n_estimators : int, default 300
        Boosting iterations.
    random_state : int, default 42
        Seed forwarded to scikit-learn.
    """

    def __init__(self, ridge_alpha: float = 10.0, n_estimators: int = 300,
                 random_state: int = 42) -> None:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import Ridge

        self.ridge_alpha = ridge_alpha
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.ridge: Dict[str, Ridge] = {}
        self.gbm: Dict[str, HistGradientBoostingRegressor] = {}
        self.columns: List[str] = []

    def fit(self, X: np.ndarray, Y: np.ndarray, columns: Sequence[str]) -> "B0NonVisual":
        """Fit one ridge and one boosting model per measurement.

        Parameters
        ----------
        X : ndarray, shape (n, d)
            Design matrix from :func:`b0_features`.
        Y : ndarray, shape (n, m)
            Targets, cm.
        columns : sequence of str
            Target names.

        Returns
        -------
        B0NonVisual
            ``self``, fitted.
        """
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import Ridge

        self.columns = list(columns)
        for j, col in enumerate(self.columns):
            self.ridge[col] = Ridge(alpha=self.ridge_alpha).fit(X, Y[:, j])
            self.gbm[col] = HistGradientBoostingRegressor(
                max_iter=self.n_estimators, random_state=self.random_state,
                early_stopping=False).fit(X, Y[:, j])
        return self

    def predict(self, X: np.ndarray, which: str = "gbm") -> np.ndarray:
        """Predict all measurements.

        Parameters
        ----------
        X : ndarray, shape (n, d)
            Design matrix from :func:`b0_features`.
        which : {'gbm', 'ridge'}, default 'gbm'
            Which sub-model to predict with.

        Returns
        -------
        ndarray, shape (n, m)
            Predictions, cm.

        Raises
        ------
        ValueError
            If ``which`` is unknown or the model is not fitted.
        """
        if which not in ("gbm", "ridge"):
            raise ValueError(f"which must be 'gbm' or 'ridge', got {which!r}")
        if not self.columns:
            raise ValueError("B0NonVisual.predict called before fit")
        models = self.gbm if which == "gbm" else self.ridge
        return np.column_stack([np.asarray(models[c].predict(X)).ravel() for c in self.columns])


# =========================================================================================
# B1 - geometric, per-slice ellipse
# =========================================================================================
#: [Assumption] Anatomical slice heights as a fraction of stature measured from the FLOOR
#: (0 = floor, 1 = top of head). Proportional heuristics, NOT BodyM ground truth, and no
#: published source is claimed. The residual bias after calibration is reported per
#: measurement. Image row indices increase downward, so f=0 maps to the *largest* row index
#: (fix B3: the previous mapping was ``y1 - (1 - f) * span``, which mirrored every slice).
SLICE_HEIGHTS_FROM_FLOOR: Dict[str, float] = {
    "ankle": 0.039,      # just above the malleoli
    "calf": 0.165,       # maximum calf girth
    "knee": 0.285,
    "crotch": 0.470,
    "hip": 0.530,        # maximum hip girth, below the trochanters
    "waist": 0.620,
    "elbow": 0.630,
    "chest": 0.720,
    "bicep": 0.800,      # upper arm just below the shoulder
    "shoulder": 0.818,   # acromion
    "wrist": 0.485,
}


def ramanujan_perimeter(a_mm: float, b_mm: float) -> float:
    """Ramanujan's ellipse perimeter approximation.

    Parameters
    ----------
    a_mm, b_mm : float
        Semi-axes, mm.

    Returns
    -------
    float
        Approximate perimeter, mm. Reduces to the circle perimeter when ``a == b``.
    """
    a, b = abs(float(a_mm)), abs(float(b_mm))
    return float(np.pi * (3.0 * (a + b) - np.sqrt((3.0 * a + b) * (a + 3.0 * b))))


def mask_row_bounds(mask: np.ndarray, threshold: int = 127) -> Tuple[int, int]:
    """Inclusive top and bottom rows of the person in a mask.

    Parameters
    ----------
    mask : ndarray, shape (H, W)
        Raw mask image.
    threshold : int, default 127
        A pixel is foreground when ``mask > threshold``.

    Returns
    -------
    (y0, y1) : tuple of int
        Top and bottom rows of the foreground bounding box.

    Raises
    ------
    ValueError
        If the mask has no foreground.
    """
    fg = np.asarray(mask) > threshold
    if not fg.any():
        raise ValueError("empty silhouette mask")
    rows = np.nonzero(fg.any(axis=1))[0]
    return int(rows.min()), int(rows.max())


def foreground_runs(line: np.ndarray) -> List[Tuple[int, int]]:
    """Contiguous inclusive ``(x0, x1)`` runs of True in a 1-D boolean row.

    Parameters
    ----------
    line : ndarray of bool, shape (W,)
        One image row.

    Returns
    -------
    list of (int, int)
        One tuple per run, left to right.
    """
    padded = np.concatenate(([False], np.asarray(line, dtype=bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(edges[i]), int(edges[i + 1]) - 1) for i in range(0, len(edges) - 1, 2)]


def torso_run_width(line: np.ndarray, midline: int) -> Tuple[float, Tuple[int, int]]:
    """Width of the single foreground run containing the body midline (fix B4).

    Summing every foreground pixel in a row measured arms and legs along with the torso:
    at chest height the "front width" was arm-tip to arm-tip, and at thigh height it was
    outer-thigh to outer-thigh. Girths derived from those numbers were systematically too
    large. The midline column is the median x of the mask's foreground bounding box, which
    passes through the trunk in an A-pose front view.

    Parameters
    ----------
    line : ndarray of bool, shape (W,)
        One image row.
    midline : int
        Column index of the body's vertical midline.

    Returns
    -------
    (width_px, run) : tuple
        ``width_px`` is the inclusive width of the midline run; ``run`` is ``(x0, x1)``.
        Returns ``(0.0, (-1, -1))`` when no run covers the midline.
    """
    runs = foreground_runs(line)
    for a, b in runs:
        if a <= midline <= b:
            return float(b - a + 1), (a, b)
    return 0.0, (-1, -1)


def leg_run_width(line: np.ndarray, midline: int) -> Tuple[float, Tuple[int, int]]:
    """Width of the single leg run on one side of the midline (fix B4).

    For thigh and calf the pose puts two legs side by side with a gap between them, so the
    midline run is empty or is the gap itself. The leg is then the run *nearest* the
    midline, taken from the right-hand side so the choice is deterministic.

    Parameters
    ----------
    line : ndarray of bool, shape (W,)
        One image row.
    midline : int
        Column index of the body's vertical midline.

    Returns
    -------
    (width_px, run) : tuple
        ``width_px`` is the inclusive width of the chosen run; ``run`` is ``(x0, x1)``.
    """
    runs = foreground_runs(line)
    right = [r for r in runs if r[0] >= midline]
    left = [r for r in runs if r[1] < midline]
    pool = right if right else left
    if not pool:
        return 0.0, (-1, -1)
    a, b = min(pool, key=lambda r: abs((r[0] + r[1]) / 2 - midline))
    return float(b - a + 1), (a, b)


class B1Geometric:
    """Per-slice ellipse model with per-measurement linear calibration.

    Pipeline: ``mm_per_px = height_cm * 10 / person_pixel_height``; horizontal slices at
    heuristic anatomical heights; the *torso* foreground run (or one leg run) gives the front
    width and the side-view run gives the depth; Ramanujan perimeter for the girths; true row
    differences x mm_per_px for the lengths; stature echoed back as an input passthrough; one
    ridge per derived measurement on the fit split.

    Exactly two learned numbers per measurement, so the model cannot memorise a subject and
    its coefficients stay inspectable.

    Standardisation is not optional here. The feature vector mixes perimeters of ~1000 mm
    with a scale term of ~2 mm/px, so without it the ridge penalty is applied to features
    whose natural scales differ by three orders of magnitude and the fit is dominated by
    whichever feature has the largest raw magnitude.
    """

    #: Vertical (length) measurements as ``(upper_slice, lower_slice)``.
    VERTICAL_PAIRS: Dict[str, Tuple[str, str]] = {
        "shoulder-to-crotch": ("shoulder", "crotch"),
        "leg-length": ("crotch", "ankle"),
        "arm-length": ("shoulder", "wrist"),
    }

    #: Girth measurements and the slice each is read from.
    PERIMETER_SLICES: Dict[str, str] = {
        "ankle": "ankle", "bicep": "bicep", "calf": "calf", "chest": "chest",
        "hip": "hip", "thigh": "thigh", "waist": "waist", "wrist": "wrist",
    }

    #: Slices that are not in the canonical heuristic table. ``thigh`` is read mid-thigh and
    #: ``forearm`` at the elbow-to-wrist midpoint, because neither girth peaks at a canonical
    #: height. Declaring them explicitly keeps the feature matrix auditable instead of aliasing
    #: a girth onto a slice name it does not belong to.
    EXTRA_SLICES: Dict[str, float] = {"thigh": 0.380, "forearm": 0.560}

    #: Slices whose width must be read from ONE leg rather than the torso (fix B4).
    LEG_SLICES: frozenset = frozenset({"thigh", "calf", "knee", "ankle"})

    def __init__(self, threshold: int = 127, ridge_alpha: float = 1.0) -> None:
        """
        Parameters
        ----------
        threshold : int, default 127
            Foreground threshold for the uint8 PNG masks.
        ridge_alpha : float, default 1.0
            L2 penalty for the per-measurement calibration ridge, applied to **standardised**
            features.
        """
        self.threshold = threshold
        self.ridge_alpha = ridge_alpha
        self.calibration_: Dict[str, np.ndarray] = {}
        self.models_: Dict[str, object] = {}
        self.columns: List[str] = []
        self.scaler_mean_: Optional[np.ndarray] = None
        self.scaler_scale_: Optional[np.ndarray] = None

    # ---- standardisation ------------------------------------------------------------------
    def _fit_scaler(self, F: np.ndarray) -> None:
        """Record the feature mean and scale from the fit split.

        Parameters
        ----------
        F : ndarray, shape (n, d)
            Feature matrix from :meth:`features`.

        Returns
        -------
        None
        """
        self.scaler_mean_ = F.mean(axis=0)
        self.scaler_scale_ = np.where(F.std(axis=0) > 1e-9, F.std(axis=0), 1.0)

    def _standardise(self, F: np.ndarray) -> np.ndarray:
        """Apply the fitted mean/scale.

        Parameters
        ----------
        F : ndarray, shape (n, d)
            Raw feature matrix.

        Returns
        -------
        ndarray, shape (n, d)
            Standardised features.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self.scaler_mean_ is None:
            raise RuntimeError("B1Geometric._standardise called before fit")
        return (F - self.scaler_mean_) / self.scaler_scale_

    def coefficients(self, measurement: str) -> np.ndarray:
        """Return a measurement's calibration weights, in standardised-feature units.

        Parameters
        ----------
        measurement : str
            Measurement name.

        Returns
        -------
        ndarray
            One coefficient per feature, or an empty array for a passthrough measurement.

        Raises
        ------
        KeyError
            If the measurement is unknown or unsupported.
        """
        return self.calibration_[measurement]

    def intercept(self, measurement: str) -> float:
        """Return a measurement's calibration intercept (fix A1).

        Parameters
        ----------
        measurement : str
            Measurement name.

        Returns
        -------
        float
            The fitted Ridge intercept, or 0.0 for a passthrough measurement.
        """
        model = self.models_.get(measurement)
        return float(getattr(model, "intercept_", 0.0)) if model is not None else 0.0

    # ---- geometry -----------------------------------------------------------------------
    def slice_heights(self) -> Dict[str, float]:
        """All slice heights as a fraction of stature from the floor.

        Returns
        -------
        dict
            Canonical heuristics merged with :attr:`EXTRA_SLICES`.
        """
        return {**SLICE_HEIGHTS_FROM_FLOOR, **self.EXTRA_SLICES}

    def mm_per_px(self, front: np.ndarray, height_cm: float) -> float:
        """Millimetres per pixel implied by stature.

        Parameters
        ----------
        front : ndarray, shape (H, W)
            Front mask.
        height_cm : float
            Subject stature, cm.

        Returns
        -------
        float
            ``height_cm * 10 / person_pixel_height``.
        """
        y0, y1 = mask_row_bounds(front, self.threshold)
        return float(height_cm) * 10.0 / float(y1 - y0 + 1)

    def _row_for(self, frac_from_floor: float, y0: int, y1: int) -> int:
        """Absolute row index for an anatomical height (fix B3).

        Image row indices increase downward, so a height fraction measured *from the floor*
        maps to ``y1 - f * span``: f = 0 is the bottom row and f = 1 the top row. The previous
        expression ``y1 - (1 - f) * span`` inverted both, placing the "chest" slice at
        upper-thigh height and the "ankle" slice at the crown.

        Parameters
        ----------
        frac_from_floor : float
            Height as a fraction of stature; 0 = floor, 1 = top of head.
        y0, y1 : int
            Foreground row bounds of the view being sliced.

        Returns
        -------
        int
            Absolute row index.
        """
        span = float(y1 - y0 + 1)
        return int(round(y1 - float(frac_from_floor) * span))

    def midline(self, fg: np.ndarray) -> int:
        """Column index of the body's vertical midline.

        Parameters
        ----------
        fg : ndarray of bool, shape (H, W)
            Foreground mask.

        Returns
        -------
        int
            Midpoint of the foreground bounding box.

        Raises
        ------
        ValueError
            If the mask is empty.
        """
        xs = np.nonzero(fg.any(axis=0))[0]
        if xs.size == 0:
            raise ValueError("empty silhouette mask")
        return int(round((int(xs.min()) + int(xs.max())) / 2))

    def measure(self, front: np.ndarray, side: np.ndarray, height_cm: float) -> Dict[str, float]:
        """Measure every slice for one silhouette pair.

        Parameters
        ----------
        front, side : ndarray, shape (H, W)
            Front and side masks for the same ``photo_id``.
        height_cm : float
            Subject stature, cm.

        Returns
        -------
        dict
            ``<slice>_width_mm``, ``<slice>_depth_mm``, ``<slice>_perim_mm``, ``<slice>_row``,
            ``vert_<length>_mm`` for each of :attr:`VERTICAL_PAIRS``, plus ``mm_per_px`` and
            ``height_cm``. Widths come from the torso run (or one leg run) at the midline, not
            from summing the whole row.

        Raises
        ------
        ValueError
            If either mask is empty.
        """
        s = self.mm_per_px(front, height_cm)
        f = np.asarray(front) > self.threshold
        sd = np.asarray(side) > self.threshold
        y0f, y1f = mask_row_bounds(front, self.threshold)
        y0s, y1s = mask_row_bounds(side, self.threshold)
        mid_f = self.midline(f)
        mid_s = self.midline(sd)

        out: Dict[str, float] = {}
        rows: Dict[str, int] = {}
        for name, frac in self.slice_heights().items():
            rf = int(np.clip(self._row_for(frac, y0f, y1f), 0, f.shape[0] - 1))
            rs = int(np.clip(self._row_for(frac, y0s, y1s), 0, sd.shape[0] - 1))
            if name in self.LEG_SLICES:
                wf_px, _ = leg_run_width(f[rf], mid_f)
            else:
                wf_px, _ = torso_run_width(f[rf], mid_f)
            wd_px, _ = torso_run_width(sd[rs], mid_s)

            wf = max(float(wf_px) * s, 1e-3)
            wd = max(float(wd_px) * s, 1e-3)
            a, b = wf / 2.0, wd / 2.0
            out[f"{name}_width_mm"] = wf
            out[f"{name}_depth_mm"] = wd
            out[f"{name}_perim_mm"] = ramanujan_perimeter(a, b)
            out[f"{name}_row"] = float(rf)
            rows[name] = rf

        for meas, (upper, lower) in self.VERTICAL_PAIRS.items():
            out[f"vert_{meas}_mm"] = abs(rows[upper] - rows[lower]) * s

        out["mm_per_px"] = s
        out["height_cm"] = float(height_cm)
        return out

    def measure_with_runs(self, front: np.ndarray, side: np.ndarray, height_cm: float
                          ) -> Tuple[Dict[str, float], Dict[str, List[Tuple[int, int]]]]:
        """:meth:`measure`, plus the per-slice foreground runs that were actually measured.

        Used only for the overlay figure (fix B4's check); the hot path is :meth:`measure`.

        Parameters
        ----------
        front, side : ndarray, shape (H, W)
            Front and side masks for one subject.
        height_cm : float
            Subject stature, cm.

        Returns
        -------
        (values, runs) : tuple
            ``values`` as per :meth:`measure`; ``runs`` maps slice name to the list of
            ``(x0, x1)`` runs present on that row.
        """
        vals = self.measure(front, side, height_cm)
        f = np.asarray(front) > self.threshold
        y0f, y1f = mask_row_bounds(front, self.threshold)
        runs: Dict[str, List[Tuple[int, int]]] = {}
        for name, frac in self.slice_heights().items():
            rf = int(np.clip(self._row_for(frac, y0f, y1f), 0, f.shape[0] - 1))
            runs[name] = foreground_runs(f[rf])
        return vals, runs

    def features(self, front: np.ndarray, side: np.ndarray, height_cm: float) -> np.ndarray:
        """Fixed-order feature vector for one silhouette pair.

        Parameters
        ----------
        front, side : ndarray, shape (H, W)
            Front and side masks for the same ``photo_id``.
        height_cm : float
            Subject stature, cm.

        Returns
        -------
        ndarray, shape (n_features,)
            Slice perimeters (mm), slice widths (mm), the three vertical lengths (mm),
            ``mm_per_px`` and ``height_cm``.
        """
        m = self.measure(front, side, height_cm)
        names = self.slice_heights()
        return np.array(
            [m[f"{n}_perim_mm"] for n in names]
            + [m[f"{n}_width_mm"] for n in names]
            + [m[f"vert_{k}_mm"] for k in self.VERTICAL_PAIRS]
            + [m["mm_per_px"], m["height_cm"]], float)

    def feature_names(self) -> List[str]:
        """Feature names matching the :meth:`features` order.

        Returns
        -------
        list of str
        """
        names = self.slice_heights()
        return ([f"{n}_perim_mm" for n in names] + [f"{n}_width_mm" for n in names]
                + [f"vert_{k}_mm" for k in self.VERTICAL_PAIRS] + ["mm_per_px", "height_cm"])

    def features_for(self, measurement: str) -> List[int]:
        """Indices of the features a measurement's calibration is allowed to use (fix B4).

        Each measurement is regressed on its own perimeter, its own width, the scale terms,
        and - for the three lengths - its own row difference. Feeding every feature to every
        measurement let the ridge borrow the chest perimeter to predict the wrist, which is
        how B1 silently converged on B0.

        Parameters
        ----------
        measurement : str
            Measurement name.

        Returns
        -------
        list of int
            Column indices into the :meth:`features` vector.

        Raises
        ------
        KeyError
            If ``measurement`` is not a known slice, girth or length.
        """
        names = list(self.slice_heights())
        idx: List[int] = []
        if measurement in self.PERIMETER_SLICES:
            n = self.PERIMETER_SLICES[measurement]
            idx += [names.index(n), len(names) + names.index(n)]
        if measurement in self.VERTICAL_PAIRS:
            idx += [2 * len(names) + list(self.VERTICAL_PAIRS).index(measurement)]
        idx += [2 * len(names) + len(self.VERTICAL_PAIRS), 2 * len(names) + len(self.VERTICAL_PAIRS) + 1]
        return sorted(set(idx))

    # ---- calibration --------------------------------------------------------------------
    def unsupported(self) -> List[str]:
        """Targets B1 does not predict.

        Returns
        -------
        list of str
            ``['height']``: stature is a model *input*, so echoing it is not a prediction.
        """
        return ["height"]

    def derived(self, columns: Sequence[str]) -> List[str]:
        """Targets B1 does estimate.

        Parameters
        ----------
        columns : sequence of str
            Target names.

        Returns
        -------
        list of str
        """
        return [c for c in columns if c not in self.unsupported()]

    def fit(self, F: np.ndarray, Y: np.ndarray, columns: Sequence[str]) -> "B1Geometric":
        """Standardise the features, then fit one ridge per derived measurement.

        Parameters
        ----------
        F : ndarray, shape (n, d)
            Feature matrix from :meth:`features`.
        Y : ndarray, shape (n, m)
            Targets, cm.
        columns : sequence of str
            Target names.

        Returns
        -------
        B1Geometric
            ``self``, fitted.

        Notes
        -----
        The fitted :class:`sklearn.linear_model.Ridge` object is stored per measurement
        (fix A1). The previous version kept only ``coef_`` and predicted ``Z @ coef``,
        discarding every intercept: for measurements whose standardised features average
        near zero that is close to harmless, but for ``height_cm``-dominated features the
        intercept *is* the model, so B1 lost roughly a stature's worth of accuracy on
        every length measurement.
        """
        from sklearn.linear_model import Ridge

        self.columns = list(columns)
        self._fit_scaler(np.asarray(F, dtype=float))
        Z = self._standardise(np.asarray(F, dtype=float))
        for j, col in enumerate(self.columns):
            if col in self.unsupported():
                continue
            cols = self.features_for(col)                    # fix B4: own features only
            model = Ridge(alpha=self.ridge_alpha).fit(Z[:, cols], Y[:, j])
            self.models_[col] = model
            full = np.zeros(Z.shape[1])
            full[cols] = model.coef_
            self.calibration_[col] = full                    # full-width view for the probe
        return self

    def predict(self, F: np.ndarray) -> np.ndarray:
        """Predict all targets from a feature matrix.

        Parameters
        ----------
        F : ndarray, shape (n, d)
            Feature matrix from :meth:`features`.

        Returns
        -------
        ndarray, shape (n, m)
            Predictions, cm.
        """
        F = np.asarray(F, dtype=float)
        Z = self._standardise(F)
        cols_out = []
        for col in self.columns:
            if col in self.unsupported():
                cols_out.append(F[:, -1])                    # stature passthrough
                continue
            model = self.models_[col]
            cols_out.append(np.asarray(model.predict(Z[:, self.features_for(col)])).ravel())
        return np.column_stack(cols_out)
