"""Step 1 mock: the garment catalogue.

A static table, not a random number. One source of truth for which
measurements each garment needs and its default ease.
"""

from __future__ import annotations

from contracts import GarmentSpec, GarmentType

# A shared set that every top needs, plus what makes each one distinct.
_TOP_CORE = ["chest", "waist", "hip", "shoulder_width", "back_length", "armhole_depth"]
_SLACKS = ["waist", "hip", "outseam", "inseam", "rise", "thigh"]

_EASE_TOP = {
    "chest": 8.0,
    "waist": 8.0,
    "hip": 8.0,
    "sleeve_length": 2.5,
    "armhole_depth": 1.0,
}

CATALOGUE: dict[GarmentType, dict] = {
    GarmentType.POLO_SHIRT: {
        "required_measurements": _TOP_CORE + ["sleeve_length"],
        "ease_cm": _EASE_TOP,
        "design_rules": {
            "collar": "two-piece, 7.5 cm stand, sits flat on the neckline",
            "placket": "5.5 cm wide, 16 cm long, 3 buttons, reinforced top",
            "placket_overlap": 1.8,
            "sleeve": "short set-in with a 2.5 cm turnback cuff",
            "cuff_width": 6.0,
            "side_seam": "straight, no shaping",
            "shoulder": "no drop, 0 cm slope at the shoulder point",
            "back": "0.5 cm waist suppression, no back yoke",
            "hem": "straight, 3 cm above the hip bone",
        },
    },
    GarmentType.LONG_SLEEVE_POLO: {
        "required_measurements": _TOP_CORE + ["sleeve_length"],
        "ease_cm": _EASE_TOP,
        "design_rules": {
            "collar": "two-piece, 7.5 cm stand, sits flat on the neckline",
            "placket": "5.5 cm wide, 16 cm long, 3 buttons",
            "placket_overlap": 1.8,
            "sleeve": "long set-in with a 2.5 cm turnback cuff",
            "cuff_width": 6.5,
            "elbow_ease": 3.0,
            "side_seam": "straight",
            "shoulder": "1.5 cm drop at the shoulder point",
            "back": "0.5 cm waist suppression",
            "hem": "straight, at the hip bone",
        },
    },
    GarmentType.SHORT_SLEEVE_POLO: {
        "required_measurements": _TOP_CORE,
        "ease_cm": {k: v for k, v in _EASE_TOP.items() if k != "sleeve_length"},
        "design_rules": {
            "collar": "flat rib collar, 4 cm, no stand",
            "placket": "none, pullover",
            "sleeve": "short set-in, open hem, no cuff",
            "armhole_binding": "2 cm self bias",
            "side_seam": "straight",
            "shoulder": "no drop",
            "back": "0.5 cm waist suppression",
            "hem": "straight, 3 cm above the hip bone",
        },
    },
    GarmentType.SLACKS: {
        "required_measurements": _SLACKS,
        "ease_cm": {
            "waist": 2.5,
            "hip": 7.0,
            "thigh": 6.0,
            "inseam": 0.0,
            "outseam": 1.0,
            "rise": 1.5,
        },
        "design_rules": {
            "waistband": "3.5 cm, straight top edge, no contour",
            "fly": "J-stitch, 16 cm",
            "front_pocket": "slant, 15 cm, no dart",
            "back_pocket": "welt, 12 cm, single pleat",
            "leg": "straight, no taper below the knee",
            "side_seam": "2.5 cm, pressed open",
            "hem": "4 cm blind hem, original hem allowance kept",
        },
    },
}


class MockGarmentSpec:
    """Deterministic: same garment, same spec, every time."""

    def run(self, garment_type: GarmentType) -> GarmentSpec:
        entry = CATALOGUE[garment_type]
        return GarmentSpec(
            is_mock=True,
            garment_type=garment_type,
            required_measurements=list(entry["required_measurements"]),
            ease_cm=dict(entry["ease_cm"]),
            design_rules=dict(entry["design_rules"]),
            warnings=["mock garment catalogue, not a pattern library"],
        )
