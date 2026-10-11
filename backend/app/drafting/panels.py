"""Panel assembly: front and back bodice built from the traced landmark values."""
from __future__ import annotations

from ..config import drafting_config
from .ease import Settings, allowance_for
from .geometry import CubicBezier, Dart, Notch, PatternDocument, PatternEdge, PatternPanel, Point2D


def _pt(values: dict[str, float], x_name: str, y_name: str) -> Point2D:
    return Point2D(values[x_name], values[y_name])


def build_panel(
    side: str,
    values: dict[str, float],
    settings: Settings,
    ease,
) -> PatternPanel:
    p = side
    v = values
    label_prefix = "Front" if side == "front" else "Back"

    cf_neck = _pt(v, f"lm.{p}.cf_neck_x", f"lm.{p}.cf_neck_y")
    cf_waist = _pt(v, f"lm.{p}.cf_waist_x", f"lm.{p}.cf_waist_y")
    snp = _pt(v, f"lm.{p}.neck_shoulder_x", f"lm.{p}.neck_shoulder_y")
    st = _pt(v, f"lm.{p}.shoulder_tip_x", f"lm.{p}.shoulder_tip_y")
    ua = _pt(v, f"lm.{p}.underarm_x", f"lm.{p}.underarm_y")
    ws = _pt(v, f"lm.{p}.waist_side_x", f"lm.{p}.waist_side_y")

    neck_c1 = _pt(v, f"lm.{p}.neckline_c1_x", f"lm.{p}.neckline_c1_y")
    neck_c2 = _pt(v, f"lm.{p}.neckline_c2_x", f"lm.{p}.neckline_c2_y")
    arm_c1 = _pt(v, f"lm.{p}.armhole_c1_x", f"lm.{p}.armhole_c1_y")
    arm_c2 = _pt(v, f"lm.{p}.armhole_c2_x", f"lm.{p}.armhole_c2_y")

    overrides = settings.seam_allowances or None

    def sa(role: str) -> float:
        return allowance_for(role, overrides)

    notches = [
        Notch(at=0.0, kind="double", role="side_seam"),
        Notch(at=0.25, kind="single", role="balance"),
    ]

    edges = [
        PatternEdge(
            role="center_front" if side == "front" else "center_back",
            start=cf_waist, end=cf_neck, seam_allowance_mm=sa("center_front" if side == "front" else "center_back"),
            notches=[Notch(at=0.0, kind="double", role="centre_front")],
            label=f"{label_prefix} centre", key=f"{side}.center", rule_id="lm.origin",
        ),
        PatternEdge(
            role="neckline", start=cf_neck, end=snp,
            curve=CubicBezier(cf_neck, neck_c1, neck_c2, snp),
            seam_allowance_mm=sa("neckline"), notches=list(notches),
            label=f"{label_prefix} neckline", key=f"{side}.neckline", rule_id="lm.{p}.neckline_c1",
        ),
        PatternEdge(
            role="shoulder", start=snp, end=st, seam_allowance_mm=sa("shoulder"),
            notches=[Notch(at=0.5, kind="single", role="shoulder")],
            label=f"{label_prefix} shoulder", key=f"{side}.shoulder", rule_id="lm.{p}.shoulder_tip",
        ),
        PatternEdge(
            role="armscye", start=st, end=ua,
            curve=CubicBezier(st, arm_c1, arm_c2, ua),
            seam_allowance_mm=sa("armscye"),
            notches=[Notch(at=1.0 / 3.0, kind="single", role="armscye")],
            label=f"{label_prefix} armscye", key=f"{side}.armscye", rule_id="lm.{p}.armhole_c1",
        ),
        PatternEdge(
            role="side_seam", start=ua, end=ws, seam_allowance_mm=sa("side_seam"),
            notches=[Notch(at=0.5, kind="single", role="side_seam")],
            label=f"{label_prefix} side seam", key=f"{side}.side_seam", rule_id="lm.{p}.side_seam_length",
        ),
        PatternEdge(
            role="waist", start=ws, end=cf_waist, seam_allowance_mm=sa("waist"),
            notches=[Notch(at=0.0, kind="double", role="side_seam"),
                     Notch(at=0.5, kind="single", role="centre")],
            label=f"{label_prefix} waist", key=f"{side}.waist", rule_id="lm.{p}.waist_side",
        ),
    ]

    darts: list[Dart] = []
    if side == "front" and v["der.dart_intake"] > 1e-6:
        apex = Point2D(v["lm.front.bust_apex_x"], v["lm.front.bust_apex_y"])
        half = v["lm.front.dart_half_intake"]
        top_y = v["lm.front.dart_top_y"]
        bot_y = v["lm.front.dart_bottom_y"]
        darts.append(
            Dart(
                id=f"{side}.dart",
                apex=apex,
                legs=[
                    (apex, Point2D(apex.x - half, top_y), Point2D(apex.x + half, top_y)),
                    (apex, Point2D(apex.x - half, bot_y), Point2D(apex.x + half, bot_y)),
                ],
                intake_mm=v["der.dart_intake"],
                rule_id="der.dart_intake",
            )
        )

    # Construction lines are explicit segments, not a polygon: centre, neck line,
    # shoulder line, bust line and waist line.
    construction = [
        (cf_neck, cf_waist),
        (cf_neck, Point2D(snp.x, cf_neck.y)),
        (snp, Point2D(st.x + 40.0, st.y)),
        (Point2D(0.0, v["neck_to_bust_level"]), Point2D(v["der.front_bust_half_width"], v["neck_to_bust_level"])),
        (cf_waist, ws),
    ]

    landmarks = {
        "centre_neck": cf_neck, "centre_waist": cf_waist, "neck_shoulder": snp,
        "shoulder_tip": st, "underarm": ua, "waist_side": ws,
        "neck_dip": Point2D(v[f"lm.{p}.neck_dip_x"], v[f"lm.{p}.neck_dip_y"]),
    }
    if darts:
        landmarks["bust_apex"] = darts[0].apex

    panel = PatternPanel(key=side, label=f"{label_prefix} bodice", edges=edges, darts=darts,
                         construction=construction, landmarks=landmarks)
    for e in panel.edges:
        e.ensure_id()
    return panel


def build_document(values: dict[str, float], settings: Settings, ease) -> PatternDocument:
    from .stitches import build_stitches

    panels = [build_panel(side, values, settings, ease) for side in ("front", "back")]
    doc = PatternDocument(panels=panels, stitches=[])
    doc.stitches = build_stitches(doc, panels)
    place_panels(doc, drafting_config()["paper"]["gap_mm"])
    return doc


def place_panels(doc: PatternDocument, gap_mm: float) -> None:
    """Shift every panel so its local bbox centre is the origin, then lay them out.

    The fixture stores panel-local geometry centred on its own bounding box and puts
    the workspace placement in `center`, so the exported coordinates stay local.
    """
    cursor = 0.0
    for panel in doc.panels:
        x0, y0, x1, y1 = panel.bbox()
        local_centre = Point2D((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        shift = Point2D(-local_centre.x, -local_centre.y)
        for edge in panel.edges:
            edge.start = _shift(edge.start, shift)
            edge.end = _shift(edge.end, shift)
            if edge.curve is not None:
                edge.curve = CubicBezier(*[_shift(p, shift) for p in (
                    edge.curve.p0, edge.curve.c1, edge.curve.c2, edge.curve.p3)])
        panel.construction = [(_shift(a, shift), _shift(b, shift)) for a, b in panel.construction]
        panel.landmarks = {k: _shift(p, shift) for k, p in panel.landmarks.items()}
        for dart in panel.darts:
            dart.apex = _shift(dart.apex, shift)
            dart.legs = [(_shift(apex, shift), _shift(a, shift), _shift(b, shift))
                         for apex, a, b in dart.legs]
        panel.center = Point2D(cursor, 0.0)
        cursor += (x1 - x0) + gap_mm
        panel.ensure_id()


def _shift(p: Point2D, s: Point2D) -> Point2D:
    return Point2D(p.x + s.x, p.y + s.y)


def world_point(panel: PatternPanel, p: Point2D) -> Point2D:
    return Point2D(p.x + panel.center.x, p.y + panel.center.y)
