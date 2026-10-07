"""Shared pattern geometry for the step 8 and 9 mocks.

Templates are half-outlines in normalised units: x is a fraction of the piece
half-width, y a fraction of the piece length. They get scaled by the step 6
measurements, so changing an interval visibly changes the shape of the piece.

Everything is drawn in centimetres inside the SVG viewBox and then transformed
to a pixel space, so the numbers in the pattern are true numbers.
"""

from __future__ import annotations

from contracts import GarmentType

# --------------------------------------------------------------------------
# templates: (x, y, smoothing) where smoothing 0 = hard corner, 1 = round
# --------------------------------------------------------------------------

_BODICE_FRONT = [
    (0.00, 0.000, 0),  # centre neck
    (0.16, 0.028, 1),
    (0.30, 0.048, 0),
    (0.50, 0.080, 1),  # shoulder point
    (0.478, 0.180, 1),  # armhole
    (0.440, 0.268, 0),
    (0.400, 0.330, 1),  # underarm
    (0.392, 0.520, 1),  # waist
    (0.402, 0.740, 1),
    (0.412, 1.000, 0),  # hem
]

_BODICE_BACK = [
    (0.00, 0.030, 0),  # centre neck, higher than front
    (0.18, 0.050, 1),
    (0.32, 0.062, 0),
    (0.50, 0.092, 1),  # shoulder point
    (0.492, 0.185, 1),  # shallower armhole
    (0.452, 0.260, 0),
    (0.410, 0.330, 1),
    (0.398, 0.520, 1),
    (0.408, 0.740, 1),
    (0.418, 1.000, 0),
]

_SLEEVE = [
    (0.00, 0.000, 0),  # sleeve head centre
    (0.22, 0.018, 1),
    (0.44, 0.062, 1),  # sleeve head cap
    (0.50, 0.075, 0),
    (0.44, 0.230, 1),  # underarm
    (0.30, 0.330, 1),
    (0.275, 0.700, 1),
    (0.255, 1.000, 0),  # cuff edge
]

_SLACKS_FRONT = [
    (0.00, 0.00, 0),  # centre front waist
    (0.30, 0.00, 0),
    (0.46, 0.02, 1),  # waistband corner
    (0.50, 0.10, 1),  # hip
    (0.50, 0.30, 1),  # upper thigh
    (0.46, 0.55, 1),
    (0.30, 0.86, 1),  # into the crotch
    (0.16, 0.985, 0),  # inside hem
    (0.00, 1.00, 1),  # inside seam knee
    (0.05, 0.62, 1),
    (0.10, 0.30, 1),  # rise curve
    (0.10, 0.10, 1),  # fly
    (0.00, 0.085, 0),  # fly top
]

_SLACKS_BACK = [
    (0.00, 0.00, 0),
    (0.30, 0.00, 0),
    (0.47, 0.02, 1),
    (0.52, 0.10, 1),
    (0.52, 0.32, 1),
    (0.47, 0.56, 1),
    (0.31, 0.86, 1),
    (0.17, 0.985, 0),
    (0.00, 1.00, 1),
    (0.06, 0.62, 1),
    (0.11, 0.30, 1),  # seat curve
    (0.11, 0.10, 1),
    (0.00, 0.085, 0),
]


def _fmt(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _path(points: list[tuple[float, float, int]], sx: float, sy: float, ox: float, oy: float) -> str:
    """Scale a template to cm and emit an SVG path.

    ``sx``/``sy`` are the piece half-width and length in cm; ``ox``/``oy`` move
    the piece into the viewBox.
    """
    pts = [(ox + x * sx, oy + y * sy) for x, y, _ in points]
    smooth = [s for _, _, s in points]

    d = [f"M{_fmt(pts[0][0])},{_fmt(pts[0][1])}"]
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        s = smooth[i]
        if s:
            # quadratic through the midpoint: reads as a smooth seam without
            # needing real curve fitting
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            d.append(f"Q{_fmt(x1)},{_fmt(y1)} {_fmt(mx)},{_fmt(my)}")
        else:
            d.append(f"L{_fmt(x1)},{_fmt(y1)}")
    d.append("Z")
    return " ".join(d)


# --------------------------------------------------------------------------
# piece definitions
# --------------------------------------------------------------------------


def _top_pieces(garment: GarmentType, m: dict[str, float], ease: dict[str, float]) -> list[dict]:
    def g(name: str, fallback: float) -> float:
        return float(m.get(name, fallback))

    def e(name: str) -> float:
        return float(ease.get(name, 0.0))

    chest = g("chest", 96.0) + e("chest")
    back_len = g("back_length", 42.0) + 3.0
    # half of the pattern width: chest/2 + quarter sleeve set into the armhole
    half_w = chest / 2 + 3.0
    neck_w = g("neck", 0) or (chest * 0.23)

    pieces = [
        {
            "name": "Front bodice",
            "path": _path(_BODICE_FRONT, half_w, back_len, half_w, 0),
            "box": (0, 0, half_w * 2, back_len),
        },
        {
            "name": "Back bodice",
            "path": _path(_BODICE_BACK, half_w, back_len, half_w * 2 + 1.5, 0),
            "box": (half_w * 2 + 1.5, 0, half_w * 2, back_len),
        },
    ]

    long_sleeve = garment is not GarmentType.SHORT_SLEEVE_POLO
    sleeve_len = g("sleeve_length", 60.0) if long_sleeve else 20.0
    sleeve_w = g("bicep", 0) or chest * 0.155
    pieces.append(
        {
            "name": "Sleeve" + (" (long)" if long_sleeve else " (short)"),
            "path": _path(_SLEEVE, sleeve_w, sleeve_len, 1.5, back_len + 2.0),
            "box": (0, back_len + 2.0, sleeve_w * 2, sleeve_len),
        }
    )

    collar_w = chest * 0.23
    collar_depth = 7.5 if garment in (GarmentType.POLO_SHIRT, GarmentType.LONG_SLEEVE_POLO) else 4.0
    pieces.append(
        {
            "name": "Collar",
            "path": _path(
                [(0.0, 0.0, 0), (1.0, 0.0, 1), (0.88, 1.0, 1), (0.12, 1.0, 1)],
                collar_w,
                collar_depth,
                0,
                back_len + 2.0 + sleeve_len + 2.0,
            ),
            "box": (0, back_len + 2.0 + sleeve_len + 2.0, collar_w, collar_depth),
        }
    )

    if garment in (GarmentType.POLO_SHIRT, GarmentType.LONG_SLEEVE_POLO):
        pl_w = 5.5 + float(ease.get("placket_overlap", 1.8))
        pieces.append(
            {
                "name": "Placket",
                "path": _path(
                    [(0.0, 0.0, 0), (1.0, 0.0, 0), (1.0, 1.0, 0), (0.0, 1.0, 0)],
                    pl_w,
                    16.0,
                    0,
                    back_len + 4.0 + sleeve_len + 2.0 + collar_depth + 2.0,
                ),
                "box": (0, back_len + 4.0 + sleeve_len + 2.0 + collar_depth + 2.0, pl_w, 16.0),
            }
        )

    return pieces


def _slacks_pieces(m: dict[str, float], ease: dict[str, float]) -> list[dict]:
    def g(name: str, fallback: float) -> float:
        return float(m.get(name, fallback))

    def e(name: str) -> float:
        return float(ease.get(name, 0.0))

    outseam = g("outseam", 103.0) + e("outseam")
    hip = g("hip", 98.0) + e("hip")
    waist = g("waist", 82.0) + e("waist")
    # pattern width: waist/2 + hip ease on each half, plus a side seam
    half_w = max(hip / 2 + 3.0, waist / 2 + 3.0)

    return [
        {
            "name": "Front leg",
            "path": _path(_SLACKS_FRONT, half_w, outseam, 0, 0),
            "box": (0, 0, half_w, outseam),
        },
        {
            "name": "Back leg",
            "path": _path(_SLACKS_BACK, half_w, outseam, half_w + 2.0, 0),
            "box": (half_w + 2.0, 0, half_w, outseam),
        },
        {
            "name": "Waistband",
            "path": _path(
                [(0.0, 0.0, 0), (1.0, 0.0, 0), (1.0, 1.0, 0), (0.0, 1.0, 0)],
                half_w * 2,
                3.5,
                0,
                outseam + 2.0,
            ),
            "box": (0, outseam + 2.0, half_w * 2, 3.5),
        },
    ]


def build_pieces(garment: GarmentType, measurements: dict[str, float], ease: dict[str, float]) -> list[dict]:
    """Return named pieces with an SVG path, cm units and a bounding box."""
    if garment is GarmentType.SLACKS:
        return _slacks_pieces(measurements, ease)
    return _top_pieces(garment, measurements, ease)


def render_svg(
    pieces: list[dict],
    *,
    seam_allowance_cm: float | None = None,
    margin: float = 6.0,
) -> tuple[str, float, float, list[dict]]:
    """Compose the pieces into one SVG.

    Returns (svg, width_cm, height_cm, notches). With ``seam_allowance_cm`` set
    (step 9) each piece gets a dashed allowance line and notch ticks, and the
    notch positions come back so the contract can carry them.
    """
    empty = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"></svg>'
    if not pieces:
        return empty, 10.0, 10.0, []

    xs0 = min(p["box"][0] for p in pieces)
    ys0 = min(p["box"][1] for p in pieces)
    xs1 = max(p["box"][0] + p["box"][2] for p in pieces)
    ys1 = max(p["box"][1] + p["box"][3] for p in pieces)
    w, h = xs1 - xs0 + margin * 2, ys1 - ys0 + margin * 2

    body: list[str] = []
    notch_list: list[dict] = []
    for p in pieces:
        dx, dy = margin - xs0, margin - ys0
        transform = f'transform="translate({_fmt(dx)},{_fmt(dy)})"'
        label = p["name"]
        lx = p["box"][0] + p["box"][2] / 2
        ly = p["box"][1] + p["box"][3] / 2

        # grainline
        gx = p["box"][0] + p["box"][2] * 0.12
        gy1 = p["box"][1] + p["box"][3] * 0.25
        gy2 = p["box"][1] + p["box"][3] * 0.75

        body.append(
            f'<g {transform}>'
            f'<path d="{p["path"]}" class="seam"/>'
            f'<line x1="{_fmt(gx)}" y1="{_fmt(gy1)}" x2="{_fmt(gx)}" y2="{_fmt(gy2)}" '
            f'class="grain"/>'
            f'<text x="{_fmt(lx)}" y="{_fmt(ly)}" class="label" text-anchor="middle">{label}</text>'
            f"</g>"
        )

        if seam_allowance_cm:
            sa = float(seam_allowance_cm)
            marks, spots = _notches_for(p, sa)
            body.append(
                f'<g {transform}>'
                f'<path d="{p["path"]}" class="sa"/>'
                f'<g class="notch">{marks}</g>'
                f"</g>"
            )
            # report notches in final SVG cm, not piece-local cm
            for px, py in spots:
                notch_list.append(
                    {
                        "piece": p["name"],
                        "x_cm": round(px + margin - xs0, 2),
                        "y_cm": round(py + margin - ys0, 2),
                    }
                )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_fmt(w)} {_fmt(h)}" '
        f'width="100%" class="pattern">'
        f"<style>"
        f".seam{{fill:none;stroke:#D6336C;stroke-width:0.3}}"
        f".sa{{fill:none;stroke:#F26FA5;stroke-width:0.4;stroke-dasharray:0.7 0.5;opacity:0.8}}"
        f".grain{{stroke:#9aa4b2;stroke-width:0.2;stroke-dasharray:1.2 0.6}}"
        f".label{{fill:#8a8f98;font-size:2.2px;font-family:sans-serif;letter-spacing:0.08em}}"
        f".notch{{stroke:#1f6feb;stroke-width:0.25;fill:none}}"
        f"</style>"
        + "".join(body)
        + "</svg>"
    )
    return svg, w, h, notch_list


# fraction-of-piece positions for a notch. Offsets land on the seam line edge.
_NOTCH_SPOTS = [(0.5, 0.0), (0.5, 1.0), (1.0, 0.33), (1.0, 0.67), (0.0, 0.33), (0.0, 0.67)]


def _notches_for(piece: dict, sa: float) -> tuple[str, list[tuple[float, float]]]:
    """Notch ticks just outside the seam line. Returns (svg markup, positions)."""
    bw, bh = piece["box"][2], piece["box"][3]
    depth = max(0.4, sa)
    ticks: list[str] = []
    spots: list[tuple[float, float]] = []

    for fx, fy in _NOTCH_SPOTS:
        px = piece["box"][0] + fx * bw
        py = piece["box"][1] + fy * bh
        # push the tick outward, away from the piece centre
        cx = piece["box"][0] + bw / 2
        cy = piece["box"][1] + bh / 2
        dx = 0.0 if abs(px - cx) < bw * 0.02 else (1 if px > cx else -1)
        dy = 0.0 if abs(py - cy) < bh * 0.02 else (1 if py > cy else -1)

        x1, y1 = px, py
        x2, y2 = px + dx * depth, py + dy * depth
        if dx and dy:  # corner tick: shorten to an L
            x2, y2 = px + dx * depth, py
            y2 = py + dy * depth
        ticks.append(
            f'<line x1="{_fmt(x1)}" y1="{_fmt(y1)}" x2="{_fmt(x2)}" y2="{_fmt(y2)}"/>'
        )
        spots.append((px, py))

    return "".join(ticks), spots
