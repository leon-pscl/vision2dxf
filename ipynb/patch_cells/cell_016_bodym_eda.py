"""Small plotting/aggregation helpers used by the EDA section."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def artefact_score(mask: np.ndarray) -> float:
    """Score a binary mask by how badly it looks segmented. Higher is worse.

    Four additive terms, each a different failure mode:

    1. ``holes / area`` - enclosed background regions (the classic mask failure).
    2. ``excess components / area`` - connected fragments beyond the first, i.e. detached
       limbs or background blobs that survived thresholding.
    3. ``|compactness - 1|`` - perimeter/area against the circle's value, which detects a
       ragged or frayed boundary. This is what discriminates clean masks from ragged ones:
       hole and component counts are usually zero, so without this term the score is nearly
       constant across a clean dataset.
    4. ``tiny-component area / area`` - speckle that a component count alone under-weights.

    Parameters
    ----------
    mask : ndarray of bool, shape (H, W)
        Foreground = True.

    Returns
    -------
    float
        Score, 0 for an ideal compact solid silhouette. There is no absolute threshold:
        what counts as clean is dataset-specific, so compare the printed percentiles in
        section 2 rather than against a fixed number.
    """
    from scipy import ndimage as ndi

    fg = np.asarray(mask, dtype=bool)
    area = float(fg.sum())
    if area == 0:
        return 1.0

    holes_lab, n_holes = ndi.label(~fg)
    if n_holes:
        border = {0}
        border |= set(holes_lab[0]) | set(holes_lab[-1])
        border |= set(holes_lab[:, 0]) | set(holes_lab[:, -1])
        n_holes = len(set(holes_lab.ravel()) - border)
    hole_term = n_holes / area

    comp_lab, n_comp = ndi.label(fg)
    if n_comp > 1:
        comp_area = np.bincount(comp_lab.ravel())[1:]
        comp_area = comp_area[comp_area > 0]
        excess = n_comp - 1
        speckle = float(comp_area[comp_area < 0.001 * area].sum()) / area
    else:
        excess, speckle = 0, 0.0

    # compactness: perimeter^2 / (4*pi*area), 1.0 for a disc, larger for a ragged outline
    perim = float(ndi.binary_dilation(fg).sum() - fg.sum())
    compact = perim / np.sqrt(max(4.0 * np.pi * area, 1.0))
    ragged = abs(compact - 1.0)

    return float(hole_term + excess / area + ragged + speckle)


def mask_panel(pairs: Sequence[tuple], out_path: Path, title: str = "", max_rows: int = 10) -> None:
    """Save a grid of front/side mask pairs.

    Parameters
    ----------
    pairs : sequence of (str, str, str)
        ``(photo_id, front_png_path, side_png_path)`` triples.
    out_path : Path
        Destination PNG.
    title : str, default ''
        Figure suptitle.
    max_rows : int, default 10
        Maximum number of pairs drawn.
    """
    pairs = list(pairs)[:max_rows]
    if not pairs:
        return
    n = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(4.2, 2.1 * n), squeeze=False)
    for r, (pid, fp, sp) in enumerate(pairs):
        for c, path in enumerate((fp, sp)):
            ax = axes[r][c]
            ax.imshow(np.array(Image.open(path)), cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            ax.set_title(f"{pid[:12]}  {'front' if c == 0 else 'side'}", fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.show()          # fix D6: display as well as save
    plt.close(fig)


def slice_overlay(front_mask: np.ndarray, rows: Sequence[Tuple[str, int]], out_path: Path,
                  threshold: int = 127, title: str = "") -> Tuple[dict, dict]:
    """Draw every B1 slice row on one front mask and return the measured runs.

    This is the visual check for fixes B3 (slice rows were vertically mirrored) and B4
    (slice widths included both arms / both legs). A correct model puts the ankle row
    *below* the chest row in image coordinates, and the drawn runs are the connected
    foreground runs the geometry actually measured - not the full row width.

    Parameters
    ----------
    front_mask : ndarray, shape (H, W)
        Raw front-view mask.
    rows : sequence of (str, int)
        ``(slice_name, absolute_row_index)`` pairs.
    out_path : Path
        Destination PNG.
    threshold : int, default 127
        A pixel is foreground when ``mask > threshold``.
    title : str, default ''
        Figure suptitle.

    Returns
    -------
    (row_runs, row_extent) : tuple of dict
        ``row_runs`` maps slice name to the list of ``(x0, x1)`` foreground runs actually
        measured on that row; ``row_extent`` maps slice name to ``(x_first, x_last)`` of
        the measured runs.
    """
    import numpy as np
    from matplotlib.patches import Rectangle

    fg = np.asarray(front_mask) > threshold
    ys, xs = np.nonzero(fg)
    y0, y1 = int(ys.min()), int(ys.max())
    midline = int(round((xs.min() + xs.max()) / 2))

    row_runs: dict = {}
    row_extent: dict = {}
    colours = plt.cm.tab10(np.linspace(0, 1, max(len(rows), 1)))

    fig, ax = plt.subplots(figsize=(5.2, 7.4))
    ax.imshow(fg, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.axvline(midline, ls=":", c="cyan", lw=0.8)
    ax.text(midline + 2, y0, " mid", color="cyan", fontsize=6)

    for (name, r), col in zip(rows, colours):
        r = int(np.clip(r, 0, fg.shape[0] - 1))
        line = fg[r]
        # connected runs of foreground on this row
        padded = np.concatenate(([False], line, [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        runs = [(int(edges[i]), int(edges[i + 1]) - 1) for i in range(0, len(edges) - 1, 2)]
        row_runs[name] = runs
        if runs:
            row_extent[name] = (runs[0][0], runs[-1][1])
            for k, (a, b) in enumerate(runs):
                ax.add_patch(Rectangle((a, r - 0.5), b - a + 1, 1.0,
                                       facecolor=col if k == 0 else "none",
                                       edgecolor=col, lw=0.7, alpha=0.35))
            ax.text(runs[0][0], r, f" {name} (row {r}, {len(runs)} run"
                                    f"{'s' if len(runs) != 1 else ''})",
                    color=col, fontsize=6, va="bottom")
        else:
            row_extent[name] = (0, 0)
            ax.text(midline, r, f" {name} (row {r}, EMPTY)", color="red", fontsize=6,
                    va="bottom")

    ax.set_ylim(fg.shape[0], 0)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(title or "B1 slice rows and measured foreground runs")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.show()          # fix D6
    plt.close(fig)
    return row_runs, row_extent
