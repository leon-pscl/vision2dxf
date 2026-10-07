"""Step 10 mock: drape placeholder and toile checklist."""

from __future__ import annotations

from contracts import DrapeResult, GarmentSpec

from ._common import cfg, placeholder_png, seeded


class MockDrape:
    def run(self, spec: GarmentSpec) -> DrapeResult:
        checklist = list(cfg()["drape"]["toile_checklist"])
        rnd = seeded("drape", spec.garment_type.value)

        return DrapeResult(
            is_mock=True,
            placeholder_image=placeholder_png(360, 480, "drape"),
            toile_checklist=checklist,
            warnings=[
                "mock drape, no simulation was run",
                f"first toile expected in {rnd.randint(3, 7)} days of lead time",
            ],
        )
