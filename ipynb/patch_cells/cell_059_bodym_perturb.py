"""Silhouette boundary perturbations for the sensitivity experiment and for training.

All operations act on a uint8 0/255 mask and return uint8 0/255, so they compose.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi
from PIL import Image


def binarise(mask: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Binarise a mask to uint8 {0, 255}.

    Parameters
    ----------
    mask : ndarray
        Raw mask.
    threshold : int, default 127
        Foreground when ``> threshold``.

    Returns
    -------
    ndarray of uint8
    """
    return ((np.asarray(mask) > threshold).astype(np.uint8) * 255)


def morph(mask: np.ndarray, k_px: int) -> np.ndarray:
    """Erode (k < 0) or dilate (k > 0) a mask by ``|k|`` pixels.

    A 3x3 structuring element is iterated ``|k|`` times, so the radius in pixels is
    approximately ``|k|``.

    Parameters
    ----------
    mask : ndarray
        Mask to transform.
    k_px : int
        Signed pixel magnitude. Negative erodes, positive dilates, 0 is identity.

    Returns
    -------
    ndarray of uint8
        Transformed mask, same shape as the input.
    """
    if k_px == 0:
        return binarise(mask)
    fg = np.asarray(mask) > 127
    st = np.ones((3, 3), dtype=bool)
    out = ndi.binary_erosion(fg, structure=st, iterations=abs(int(k_px))) if k_px < 0 \
        else ndi.binary_dilation(fg, structure=st, iterations=abs(int(k_px)))
    if not out.any():                       # never return an empty silhouette
        out = fg
    return (out.astype(np.uint8) * 255)


def boundary_jitter(mask: np.ndarray, sigma_px: float, rng: np.random.Generator) -> np.ndarray:
    """Displace the silhouette boundary by a smooth zero-mean random field (fix B6).

    The boundary is moved *in both directions*: the new foreground is the set of pixels whose
    signed distance (negative inside) minus the displaced field is negative. Where the field
    is positive the boundary moves inward and the body shrinks; where it is negative the
    boundary moves outward and the body grows.

    The previous implementation was dilation-only - it seeded near the boundary, dilated the
    seed, then OR-ed the original foreground back in with ``grown |= fg``, so no pixel could
    ever be removed. Both ``np.where`` branches were also identical, making the displacement
    a no-op before the OR. The result was that ``jitter`` and ``dilate`` measured the same
    thing, and the sweep's jitter axis was really re-measuring the dilation axis.

    Parameters
    ----------
    mask : ndarray
        Input mask.
    sigma_px : float
        Standard deviation of the boundary displacement, px. 0 returns the mask unchanged.
    rng : numpy.random.Generator
        Source of randomness, seeded for reproducibility.

    Returns
    -------
    ndarray of uint8
        Jittered mask, same shape.
    """
    if sigma_px <= 0:
        return binarise(mask)
    fg = np.asarray(mask) > 127
    h, w = fg.shape
    noise = rng.standard_normal((h, w)).astype(np.float32)
    field = ndi.gaussian_filter(noise, sigma=max(1.0, 2.0 * sigma_px), mode="nearest")
    field /= max(field.std(), 1e-6)

    # Signed distance from the boundary: negative inside, positive outside.
    dist_in = ndi.distance_transform_edt(fg)
    dist_out = ndi.distance_transform_edt(~fg)
    sd = np.where(fg, -dist_in, dist_out)                # (H, W), float

    # Displace the iso-contour: where `move` is positive the threshold `sd - move < 0` is met
    # further out, so the silhouette grows; where it is negative it shrinks. The field is
    # zero-mean, so over a whole mask the area change is zero-mean - which the notebook CHECK
    # asserts, and which a dilation-only implementation cannot satisfy.
    move = np.clip(field * float(sigma_px), -3.0 * sigma_px, 3.0 * sigma_px)
    moved = sd - move
    out = moved < 0.0                                    # (H, W) bool, both directions
    if not out.any():
        return binarise(mask)
    return (out.astype(np.uint8) * 255)


def downsample_upsample(mask: np.ndarray, factor: int) -> np.ndarray:
    """Simulate a low-resolution prototype mask: box-downsample then nearest-upsample.

    Parameters
    ----------
    mask : ndarray
        Input mask.
    factor : int
        Downsample factor; 2, 4 or 8 in this notebook.

    Returns
    -------
    ndarray of uint8
        Same shape as the input.
    """
    fg = (np.asarray(mask) > 127).astype(np.uint8) * 255
    h, w = fg.shape
    small = np.asarray(Image.fromarray(fg).resize((max(1, w // factor), max(1, h // factor)),
                                           Image.BILINEAR))
    back = np.asarray(Image.fromarray((small > 127).astype(np.uint8) * 255).resize((w, h),
                                                                                  Image.NEAREST))
    return back


def apply_perturbation(mask: np.ndarray, family: str, magnitude: float,
                       rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Apply one named perturbation at one magnitude.

    Parameters
    ----------
    mask : ndarray
        Input mask.
    family : {'none', 'morph', 'jitter', 'downsample'}
        Perturbation family.
    magnitude : float
        Signed pixels for ``morph``, sigma px for ``jitter``, integer factor for
        ``downsample``. Ignored for ``none``.
    rng : numpy.random.Generator, optional
        Required for ``jitter``.

    Returns
    -------
    ndarray of uint8
        Perturbed mask.

    Raises
    ------
    ValueError
        If ``family`` is unknown or ``jitter`` is requested without an ``rng``.
    """
    if family == "none":
        return binarise(mask)
    if family == "morph":
        return morph(mask, int(magnitude))
    if family == "jitter":
        if rng is None:
            raise ValueError("boundary_jitter requires an rng")
        return boundary_jitter(mask, float(magnitude), rng)
    if family == "downsample":
        return downsample_upsample(mask, int(magnitude))
    raise ValueError(f"unknown perturbation family {family!r}")


class WorkerRNG:
    """Per-worker, per-epoch RNG for the training augmenter (fix B11).

    The previous implementation built one ``np.random.default_rng(seed)`` in the enclosing
    scope and captured it in the closure. A ``DataLoader`` with ``num_workers > 0`` forks the
    dataset object into every worker, so each worker inherited an *identical copy of the same
    generator state*. Every worker then produced the same augmentation sequence, which both
    wastes the augmentation and biases the run: the same erosion/dilation offsets were applied
    to every sample in a batch.

    This class defers construction of the generator until first use inside the worker and
    derives its seed from the worker id, the epoch and the master seed, so the streams differ
    across workers and across epochs while staying reproducible.

    Parameters
    ----------
    seed : int
        Master seed from the notebook config.
    """

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self.epoch = 0
        self._rng: Optional[np.random.Generator] = None
        self._key: Optional[Tuple[int, int]] = None

    def set_epoch(self, epoch: int) -> None:
        """Advance the epoch counter so the next draw uses a fresh stream.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index.
        """
        self.epoch = int(epoch)
        self._key = None

    def generator(self) -> np.random.Generator:
        """Return this worker's generator, (re)creating it if the epoch changed.

        Returns
        -------
        numpy.random.Generator
        """
        import torch

        try:
            info = torch.utils.data.get_worker_info()
            wid = info.id if info is not None else 0
        except Exception:                       # pragma: no cover - torch always provides this
            wid = 0
        key = (wid, self.epoch)
        if self._rng is None or self._key != key:
            self._rng = np.random.default_rng((self.seed, wid, self.epoch))
            self._key = key
        return self._rng


def training_augmenter(erosion_px: int, jitter_sigma_px: float, downsample_factor: int,
                       rotation_deg: float = 3.0, scale_jitter: float = 0.03,
                       translate_frac: float = 0.03, seed: int = 42):
    """Build the per-view training augmentation callable for :class:`SilhouetteDataset`.

    Perturbation magnitudes are *not* invented: ``erosion_px`` and ``jitter_sigma_px`` are
    taken from the Phase-4 results, which is the whole point of running the sensitivity
    experiment first.

    Parameters
    ----------
    erosion_px : int
        Symmetric erosion/dilation magnitude, px.
    jitter_sigma_px : float
        Boundary jitter sigma, px.
    downsample_factor : int
        Downsample -> upsample factor. 0 or 1 disables it.
    rotation_deg : float, default 3.0
        Maximum absolute rotation, degrees.
    scale_jitter : float, default 0.03
        Relative scale jitter, as a fraction.
    translate_frac : float, default 0.03
        Maximum translation, as a fraction of image size.
    seed : int, default 42
        Master seed. The per-worker stream is derived from this plus the worker id and epoch.

    Returns
    -------
    (callable, WorkerRNG)
        ``f(front, side) -> (front_aug, side_aug)``, applied independently per view, and the
        :class:`WorkerRNG` handle so the training loop can call ``set_epoch`` each epoch.
    """
    worker_rng = WorkerRNG(seed)

    def _augment(front: np.ndarray, side: np.ndarray):
        rng = worker_rng.generator()
        outs = []
        for mask in (front, side):           # INDEPENDENTLY per view
            m = binarise(mask)
            k = int(rng.integers(-erosion_px, erosion_px + 1)) if erosion_px > 0 else 0
            m = morph(m, k)
            m = boundary_jitter(m, jitter_sigma_px, rng)
            if downsample_factor > 1:
                m = downsample_upsample(m, downsample_factor)
            if scale_jitter > 0 or translate_frac > 0 or rotation_deg > 0:
                h, w = m.shape
                ang = float(rng.uniform(-rotation_deg, rotation_deg))
                sc = 1.0 + float(rng.uniform(-scale_jitter, scale_jitter))
                tx = float(rng.uniform(-translate_frac, translate_frac)) * w
                ty = float(rng.uniform(-translate_frac, translate_frac)) * h
                # zoom about the image centre, then rotate, then translate
                zoomed = ndi.zoom(m.astype(np.float32) / 255.0, sc, order=1)
                zh, zw = zoomed.shape
                cy, cx = h // 2, w // 2
                y0, x0 = max(0, cy - zh // 2), max(0, cx - zw // 2)
                canvas = np.zeros((h, w), np.float32)
                hh, ww = min(h - y0, zh), min(w - x0, zw)
                canvas[y0:y0 + hh, x0:x0 + ww] = zoomed[:hh, :ww]
                rot = ndi.rotate(canvas, ang, reshape=False, order=1, mode="constant")
                shifted = ndi.shift(rot, (ty, tx), order=1, mode="constant")
                m = ((shifted > 0.5).astype(np.uint8) * 255)
                if not m.any():
                    m = binarise(mask)
            outs.append(m)
        return outs[0], outs[1]

    return _augment, worker_rng


def jitter_area_stats(masks: Sequence[np.ndarray], sigma_px: float, seed: int = 0
                      ) -> Dict[str, float]:
    """Foreground-area statistics of the jitter perturbation across many masks (fix B6 check).

    Parameters
    ----------
    masks : sequence of ndarray
        Input masks.
    sigma_px : float
        Jitter sigma, px.
    seed : int, default 0
        Seed for the per-mask RNG.

    Returns
    -------
    dict
        ``n``, ``mean_pct_change``, ``min_pct_change``, ``max_pct_change``,
        ``n_shrank``, ``n_grew``. The sign counts are the point: a dilation-only jitter would
        report ``n_shrank == 0``.
    """
    deltas, shrank, grew = [], 0, 0
    for i, m in enumerate(masks):
        rng = np.random.default_rng(seed + i)
        before = float((np.asarray(m) > 127).sum())
        after = float((boundary_jitter(m, sigma_px, rng) > 127).sum())
        pct = 100.0 * (after - before) / max(before, 1.0)
        deltas.append(pct)
        shrank += int(pct < -0.01)
        grew += int(pct > 0.01)
    return {"n": len(deltas),
            "mean_pct_change": float(np.mean(deltas)) if deltas else 0.0,
            "min_pct_change": float(np.min(deltas)) if deltas else 0.0,
            "max_pct_change": float(np.max(deltas)) if deltas else 0.0,
            "n_shrank": shrank,
            "n_grew": grew}