"""Anatomical cross-sections.

Each level gets one superellipse (or ellipse). Which of the three solve paths a section
takes depends on what the input actually measures at that level:

  * `depth`        circumference and a width whose anatomical definition matches the
                   section: the depth is solved numerically and nothing is assumed.
  * `scale`        circumference only: the depth-to-width ratio is assumed and the
                   section is scaled to the circumference exactly.
  * `width_ratio`  width only, no circumference: both the ratio and the resulting
                   circumference are assumed.

Every section reports its target perimeter, its achieved numerical perimeter and the
residual between them.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..assumptions import AssumptionRegistry
from ..config import reconstruction_config
from .perimeter_solver import SectionError, perimeter, solve_depth, solve_scale

# Which drafting measurement feeds which section, and whether its width is defined at the
# same anatomical level as the circumference. `back_width` is an across-back breadth at
# the chest level, so it sizes the upper chest, not the bust.
WIDTH_SOURCE = {
    "upper_chest": ("back_width_full", "back_width"),
    "waist": ("waist_back_width_full", "waist_back_width"),
    "hip": ("hip_back_width_full", "hip_back_width"),
}
GIRTH_SOURCE = {
    "neck": ("neck_circumference_from_half", "neck_w"),
    "bust": ("bust_circumference", "bust"),
    "underbust": ("underbust_circumference", "underbust"),
    "waist": ("waist_circumference", "waist"),
    "hip": ("hip_circumference", "hips"),
}


@dataclass(frozen=True)
class CrossSection:
    name: str
    y_mm: float
    a_mm: float          # half width, along x
    b_mm: float          # half depth, along z
    exponent: float
    target_perimeter_mm: float | None
    achieved_perimeter_mm: float
    residual_mm: float
    solved_for: str
    iterations: int
    source_fields: list[str]

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "yMm": round(self.y_mm, 4),
            "halfWidthMm": round(self.a_mm, 4),
            "halfDepthMm": round(self.b_mm, 4),
            "widthMm": round(2 * self.a_mm, 4),
            "depthMm": round(2 * self.b_mm, 4),
            "exponent": self.exponent,
            "targetPerimeterMm": (None if self.target_perimeter_mm is None
                                  else round(self.target_perimeter_mm, 4)),
            "achievedPerimeterMm": round(self.achieved_perimeter_mm, 4),
            "residualMm": round(self.residual_mm, 6),
            "solvedFor": self.solved_for,
            "iterations": self.iterations,
            "sourceFields": self.source_fields,
        }


def exponent_for(model: str) -> float:
    cfg = reconstruction_config()["cross_section"]
    if model == "ellipse":
        return 2.0
    return float(cfg["exponent"])


def build_sections(
    m,
    levels: dict,
    model: str,
    registry: AssumptionRegistry,
    exponent_override: float | None = None,
) -> dict[str, CrossSection]:
    """One section per anatomical level, in level order."""
    cfg = reconstruction_config()["cross_section"]
    n = float(exponent_override) if exponent_override else exponent_for(model)
    ratios = cfg["depth_to_width"]
    residual_tol = float(cfg["solver"]["residual_tolerance_mm"])

    if model == "superellipse":
        registry.add(
            "section.exponent", "Superellipse exponent", n, None,
            """A torso section is neither an ellipse nor a rectangle. The exponent was
            chosen by eye on the supplied profiles; no measurement constrains it. Set the
            section model to `ellipse` for the exact n = 2 case.""",
            [], True,
        )
    if not cfg["front_back_asymmetry"]:
        registry.add(
            "section.symmetry", "Front/back symmetry", "front depth == back depth", None,
            """Every section is symmetric about its own width axis. `bust_points` is an
            apex protrusion from the chest wall, not a half-depth, so turning it into a
            front/back distribution would be a guess. It is recorded as unused.""",
            ["bust_points", "bum_points"], True,
        )

    girths = _girths(m)
    widths = _widths(m)
    sections: dict[str, CrossSection] = {}
    errors: list[dict] = []

    for name in ("neck", "upper_chest", "bust", "underbust", "waist", "hip"):
        level = levels.get(name)
        if level is None:
            continue
        girth = girths.get(name)
        width = widths.get(name)
        ratio = float(ratios[name])
        sources: list[str] = []
        try:
            if girth is not None and width is not None:
                b, iterations = solve_depth(girth, width / 2.0, n)
                a, solved = width / 2.0, "depth"
                sources = [GIRTH_SOURCE[name][1], WIDTH_SOURCE[name][1]]
            elif girth is not None:
                a, b = solve_scale(girth, ratio, n)
                iterations, solved = 0, "scale"
                sources = [GIRTH_SOURCE[name][1]]
                _record_ratio(registry, name, ratio, "circumference")
            elif width is not None:
                a, b = width / 2.0, ratio * width / 2.0
                iterations, solved = 0, "width_ratio"
                sources = [WIDTH_SOURCE[name][1]]
                _record_ratio(registry, name, ratio, "width")
            else:
                continue
        except SectionError as exc:
            errors.extend({**e, "field": name} for e in exc.errors)
            continue

        achieved = perimeter(a, b, n)
        target = girth
        residual = 0.0 if target is None else abs(achieved - target)
        if target is not None and residual > residual_tol:
            errors.append({
                "field": name, "code": "SECTION_RESIDUAL_ABOVE_TOLERANCE",
                "message": (f"The {name} section misses its target circumference by "
                            f"{residual:.3f} mm; the limit is {residual_tol:.3f} mm."),
                "suggestedAction": "Raise the residual tolerance or change the section model.",
                "severity": "blocking",
            })
        sections[name] = CrossSection(
            name=name, y_mm=level.y_mm, a_mm=a, b_mm=b, exponent=n,
            target_perimeter_mm=target, achieved_perimeter_mm=achieved,
            residual_mm=residual, solved_for=solved, iterations=iterations,
            source_fields=sources,
        )

    if errors:
        raise SectionError(errors)

    _fix_upper_chest_depth(sections, registry)
    return sections


def _girths(m) -> dict[str, float]:
    """Target circumferences in mm, keyed by level."""
    out: dict[str, float] = {
        "bust": m.bust_circumference,
        "waist": m.waist_circumference,
    }
    # neck_width_half is the confirmed half neck girth, so the neck circumference is twice
    # it. The alternative candidate (full girth) is already normalised to the half by the
    # semantics layer, so this holds either way.
    out["neck"] = 2.0 * m.neck_width_half
    if m.underbust_circumference:
        out["underbust"] = m.underbust_circumference
    if m.hip_circumference:
        out["hip"] = m.hip_circumference
    return out


def _widths(m) -> dict[str, float]:
    out: dict[str, float] = {
        "upper_chest": m.back_width_full,
        "waist": m.waist_back_width_full,
    }
    if m.hip_back_width_full:
        out["hip"] = m.hip_back_width_full
    return out


def _record_ratio(registry: AssumptionRegistry, name: str, ratio: float, known: str) -> None:
    if name == "upper_chest":
        return  # recorded by _fix_upper_chest_depth, which replaces this ratio
    registry.add(
        f"section.ratio.{name}", f"{name.replace('_', ' ')} depth/width ratio", ratio, None,
        f"""The {name.replace('_', ' ')} level supplies a {known} but not the other
        dimension, so the section's depth-to-width ratio is assumed. This is the single
        largest assumption in the reconstructed body shape.""",
        [], True,
    )


def _fix_upper_chest_depth(sections: dict[str, CrossSection], registry) -> None:
    """Give the upper chest the bust section's shape rather than its own ratio.

    The upper chest has a measured width and no circumference. Taking its depth from the
    bust section's own solved ratio ties it to a measurement of the same body instead of
    to a table value, and keeps the loft from stepping between the two.
    """
    chest, bust = sections.get("upper_chest"), sections.get("bust")
    if chest is None or bust is None or chest.solved_for != "width_ratio":
        return
    ratio = bust.b_mm / bust.a_mm
    b = ratio * chest.a_mm
    sections["upper_chest"] = CrossSection(
        name=chest.name, y_mm=chest.y_mm, a_mm=chest.a_mm, b_mm=b, exponent=chest.exponent,
        target_perimeter_mm=None, achieved_perimeter_mm=perimeter(chest.a_mm, b, chest.exponent),
        residual_mm=0.0, solved_for="width_and_bust_ratio", iterations=0,
        source_fields=[*chest.source_fields, "bust"],
    )
    registry.add(
        "section.ratio.upper_chest", "Upper-chest depth/width ratio", round(ratio, 4), None,
        """The upper chest has a measured breadth but no circumference. Its depth is taken
        from the depth-to-width ratio the bust section solved for on the same body, which
        is closer to this person than a table value, but still an assumption: the input
        does not measure chest depth.""",
        ["back_width", "bust"], True,
    )
