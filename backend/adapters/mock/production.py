"""Step 9 mock: production pieces.

Same geometry as step 8 plus the seam allowance line and notch marks, which is
what a cutter needs on the actual pattern.
"""

from __future__ import annotations

from contracts import GarmentSpec, MeasurementResult, PatternPiece, ProductionResult

from . import pattern_svg
from ._common import cfg


class MockProduction:
    def run(self, spec: GarmentSpec, measurements: MeasurementResult) -> ProductionResult:
        sa = float(cfg()["drafting"]["seam_allowance_cm"])
        values = {k: m.value_cm for k, m in measurements.measurements.items()}

        pieces = pattern_svg.build_pieces(spec.garment_type, values, spec.ease_cm)
        _, w, h, notches = pattern_svg.render_svg(pieces, seam_allowance_cm=sa)

        # notches belong to the piece they sit on
        by_piece: dict[str, list[dict]] = {}
        for n in notches:
            by_piece.setdefault(n["piece"], []).append(n)

        out = [
            PatternPiece(
                name=p["name"],
                svg=_production_svg(p["name"], p["path"], sa),
                seam_allowance_cm=sa,
                notches=by_piece.get(p["name"], []),
            )
            for p in pieces
        ]

        return ProductionResult(
            is_mock=True,
            pieces=out,
            export_formats=["svg"],
            warnings=[
                f"mock production pieces, {sa:.1f} cm allowance",
                f"sheet {w:.1f} x {h:.1f} cm",
            ],
        )


def _production_svg(name: str, path: str, sa: float) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 130" '
        f'preserveAspectRatio="xMidYMid meet" class="piece">'
        f"<style>"
        f".s{{fill:none;stroke:#D6336C;stroke-width:0.3}}"
        f".a{{fill:none;stroke:#F26FA5;stroke-width:0.5;stroke-dasharray:1.4 1;opacity:0.85}}"
        f"</style>"
        f'<path d="{path}" class="s" vector-effect="non-scaling-stroke"/>'
        f'<path d="{path}" class="a" vector-effect="non-scaling-stroke"/>'
        f"</svg>"
    )
