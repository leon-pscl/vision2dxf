"""Step 8 mock: drafting.

Both routes (block and garmentcode) run the same geometry here; they differ in
the seam line treatment, which is enough to show the toggle works before any
real drafting engine exists.
"""

from __future__ import annotations

from contracts import DraftResult, DraftRoute, GarmentSpec, MeasurementResult, PatternPiece

from . import pattern_svg


class MockDrafting:
    def run(
        self, spec: GarmentSpec, measurements: MeasurementResult, route: str = DraftRoute.BLOCK.value
    ) -> DraftResult:
        chosen = DraftRoute(route)
        values = {k: m.value_cm for k, m in measurements.measurements.items()}

        pieces = pattern_svg.build_pieces(spec.garment_type, values, spec.ease_cm)
        svg, w, h, _ = pattern_svg.render_svg(pieces, seam_allowance_cm=None)

        # block drafts carry the block margin inside the seam line; garmentcode
        # drafts to the seam line only, which is what a pattern package wants
        note = (
            "garmentcode route: seam line only, block margins stripped"
            if chosen is DraftRoute.GARMENTCODE
            else "block route: seam line with the 1 cm block margin inside it"
        )

        out = [
            PatternPiece(name=p["name"], svg=_wrap(p["path"], chosen)) for p in pieces
        ]

        return DraftResult(
            is_mock=True,
            route=chosen,
            pieces=out,
            warnings=[f"mock drafting, {note}", f"pattern sheet {w:.1f} x {h:.1f} cm"],
        )


def _wrap(path: str, route: DraftRoute) -> str:
    """One piece as a standalone SVG.

    Block keeps the seam line plus a solid inner block margin. GarmentCode drops
    the margin and dashes the seam line instead, so the two routes are visibly
    different before any real drafting engine exists.
    """
    if route is DraftRoute.GARMENTCODE:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 130" '
            'preserveAspectRatio="xMidYMid meet" class="piece">'
            "<style>.s{fill:none;stroke:#F26FA5;stroke-width:0.35;stroke-dasharray:1 0.6}</style>"
            f'<path d="{path}" class="s" vector-effect="non-scaling-stroke"/>'
            "</svg>"
        )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 130" '
        'preserveAspectRatio="xMidYMid meet" class="piece">'
        "<style>"
        ".s{fill:none;stroke:#D6336C;stroke-width:0.35}"
        # block margin, inset inside the seam line
        ".m{fill:none;stroke:#D6336C;stroke-width:0.2;stroke-dasharray:0.7 0.5;opacity:0.7}"
        "</style>"
        f'<path d="{path}" class="s" vector-effect="non-scaling-stroke" '
        f'style="transform:scale(0.955);transform-origin:50%50%"/>'
        f'<path d="{path}" class="m" vector-effect="non-scaling-stroke"/>'
        "</svg>"
    )
