"""Convert every GarmageSet (Style3D) pattern JSON under pattern_templates/ into
sewing-pattern images (PDF or JPG) in pattern_template_output/<format>/.

Usage: python pattern_generator.py
"""

from __future__ import annotations

import gc
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LABELS_EN = {
    "荷叶边": "ruffle/flounce",
    "衣身前中": "bodice front",
    "衣身后中": "bodice back (half)",
    "裙前中": "skirt front",
    "裙后中": "skirt back (half)",
    "领": "collar/band",
    "袖片": "sleeve",
    "袖笼弧线": "armhole curve",
    "袖山弧线": "sleeve-cap curve",
    "领窝线": "neckline",
    "底摆线": "hem line",
    "袖口线": "cuff line",
}


# ---------------------------------------------------------------- curves --
def polyline(pts: np.ndarray, n: int = 0) -> np.ndarray:
    return pts


def catmull_rom(pts: np.ndarray, samples: int = 24, alpha: float = 0.5) -> np.ndarray:
    if len(pts) < 3:
        return pts
    p = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    out = [pts[0]]
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]

        def tj(ti, a, b):
            return ti + max(np.linalg.norm(b - a), 1e-9) ** alpha

        t0 = 0.0
        t1 = tj(t0, p0, p1)
        t2 = tj(t1, p1, p2)
        t3 = tj(t2, p2, p3)
        for t in np.linspace(t1, t2, samples)[1:]:
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    return np.array(out)


def bezier(pts: np.ndarray, samples: int = 200) -> np.ndarray:
    if len(pts) < 3:
        return pts
    n = len(pts) - 1
    t = np.linspace(0, 1, samples)[:, None]
    coeff = [math.comb(n, k) for k in range(n + 1)]
    return sum(coeff[k] * (1 - t) ** (n - k) * t**k * pts[k] for k in range(n + 1))


INTERPRETATIONS = {"A_polyline": polyline, "B_through_points": catmull_rom, "C_bezier": bezier}


def arc_length(poly: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(poly, axis=0), axis=1)))


# ---------------------------------------------------------------- model --
@dataclass
class Edge:
    id: str
    label: str
    pts: np.ndarray

    def curve(self, interp: str) -> np.ndarray:
        return INTERPRETATIONS[interp](self.pts)


@dataclass
class Panel:
    id: str
    label: str
    center: np.ndarray
    edges: list[Edge] = field(default_factory=list)  # every loop, in order
    outer: int = 0  # how many leading edges form the visible contour

    def contour(self, interp: str) -> np.ndarray:
        parts = [e.curve(interp)[:-1] for e in self.edges[: self.outer]]
        return np.vstack(parts + [parts[0][:1]])


def load(path: Path) -> tuple[dict[str, Panel], list]:
    data = json.loads(path.read_text(encoding="utf-8"))
    panels = {}
    for p in data["panels"]:
        loops = p["seqEdges"]
        if len(loops) != 1:
            name = LABELS_EN.get(p["label"], p["label"])
            print(f"warning: panel {name} has {len(loops)} loops (holes?)")
        # all loops are loaded so stitches referencing a hole edge resolve
        edges = [
            Edge(e["id"], e["label"], np.array(e["controlPoints"], dtype=float)[:, :2])
            for loop in loops
            for e in loop["edges"]
        ]
        outer = len(loops[0]["edges"])
        panels[p["id"]] = Panel(p["id"], p["label"], np.array(p["center"][:2]), edges, outer)
    return panels, data["stitches"]


# -------------------------------------------------------------- stitches --
def seg_length(panels: dict[str, Panel], seg: dict, interp: str) -> float:
    panel = panels[seg["start"]["clothPieceId"]]
    ids = [e.id for e in panel.edges]
    lens = [arc_length(e.curve(interp)) for e in panel.edges]
    n = len(ids)

    def pos(ref):
        i = ids.index(ref["edgeId"])
        return sum(lens[:i]) + ref["param"] * lens[i]

    total = sum(lens)
    s, e = pos(seg["start"]), pos(seg["end"])
    fwd = (e - s) % total
    bwd = (s - e) % total
    return min(fwd, bwd) if n else 0.0


def seam_report(panels, stitches, interp):
    rows = []
    for k, (a, b) in enumerate(stitches):
        la, lb = seg_length(panels, a, interp), seg_length(panels, b, interp)
        pa = panels[a["start"]["clothPieceId"]].label
        pb = panels[b["start"]["clothPieceId"]].label
        rows.append((k, pa, pb, la, lb, la - lb))
    return rows


# ------------------------------------------------------------- validity --
def self_intersects(poly: np.ndarray) -> bool:
    segs = list(zip(poly[:-1], poly[1:]))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    for i in range(len(segs)):
        for j in range(i + 2, len(segs)):
            if i == 0 and j == len(segs) - 1:
                continue
            p1, p2 = segs[i]
            q1, q2 = segs[j]
            d1, d2 = cross(q1, q2, p1), cross(q1, q2, p2)
            d3, d4 = cross(p1, p2, q1), cross(p1, p2, q2)
            if d1 * d2 < 0 and d3 * d4 < 0:
                return True
    return False


def signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


# ------------------------------------------------------------------ DXF --
def write_dxf_r12(path: Path, panels: dict[str, Panel], interp: str) -> None:
    out = ["0", "SECTION", "2", "HEADER", "9", "$ACADVER", "1", "AC1009",
           "9", "$INSUNITS", "70", "4", "0", "ENDSEC",
           "0", "SECTION", "2", "ENTITIES"]
    for idx, p in enumerate(panels.values()):
        layer = f"PANEL_{idx:02d}"
        poly = p.contour(interp) + p.center
        out += ["0", "POLYLINE", "8", layer, "66", "1", "70", "1"]
        for x, y in poly[:-1]:
            out += ["0", "VERTEX", "8", layer, "10", f"{x:.4f}", "20", f"{y:.4f}"]
        out += ["0", "SEQEND", "8", layer]
        name = LABELS_EN.get(p.label, p.label)
        out += ["0", "TEXT", "8", layer, "10", f"{p.center[0]:.2f}", "20",
                f"{p.center[1]:.2f}", "40", "20", "1", name]
        for e in p.edges:
            for x, y in e.pts + p.center:
                out += ["0", "POINT", "8", "STORED_POINTS", "10", f"{x:.4f}", "20", f"{y:.4f}"]
    out += ["0", "ENDSEC", "0", "EOF"]
    path.write_text("\n".join(out) + "\n", encoding="ascii")


# --------------------------------------------------------------- render --
def _render(path: Path, panels: dict[str, Panel], interp: str, fmt: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16.54, 11.69))
    for p in panels.values():
        poly = p.contour(interp) + p.center
        ax.plot(poly[:, 0], poly[:, 1], "k-", lw=0.8)
        ax.text(*p.center, LABELS_EN.get(p.label, p.label),
                ha="center", va="center", fontsize=7)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.5)
    save_kw = {"format": fmt}
    if fmt == "jpg":
        save_kw.update(format="jpeg", dpi=200, facecolor="white")
    fig.savefig(path, **save_kw)
    plt.close(fig)


# ----------------------------------------------------------------- CLI --
def best_interp(panels: dict[str, Panel], stitches: list) -> str:
    if not stitches:
        return "B_through_points"
    best_key, best_med = "", float("inf")
    for interp in INTERPRETATIONS:
        rows = seam_report(panels, stitches, interp)
        med = float(np.median([abs(r[5]) for r in rows]))
        if med < best_med:
            best_key, best_med = interp, med
    return best_key


def convert(src: Path, out_dir: Path, fmt: str) -> None:
    out_path = out_dir / (src.stem + f"_pattern.{fmt}")
    if out_path.exists():
        return
    panels, stitches = load(src)
    interp = best_interp(panels, stitches)
    _render(out_path, panels, interp, fmt)
    gc.collect()
    print(f"{src.name}: {len(panels)} panels, {interp} -> {out_path.name}")


def main() -> None:
    sys.stdout.reconfigure(errors="replace")  # Windows consoles reject CJK labels

    choice = input("Export format — [1] PDF  [2] JPG: ").strip()
    fmt = "jpg" if choice == "2" else "pdf"

    root = Path(__file__).resolve().parent
    out_dir = root / "pattern_template_output" / fmt
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted((root / "pattern_templates").rglob("*.json"))
    if not files:
        sys.exit("No JSON files found in pattern_templates/")
    print(f"Converting {len(files)} pattern file(s) to {fmt.upper()}...\n")
    for src in files:
        try:
            convert(src, out_dir, fmt)
        except Exception as e:  # one bad file must not stop the batch
            print(f"{src.name}: FAILED ({type(e).__name__}: {e})")
    print(f"\nDone: {len(files)} file(s) processed -> {out_dir}")


if __name__ == "__main__":
    main()
