"""Step 5 mock: landmarks.

Positions are fractions of the image box, so they scale to any upload. Mixes
model and heuristic sources so the colour coding has something to show. The
user's drags are applied by main.py, which flips the source to "user".
"""

from __future__ import annotations

from contracts import LandmarkPoint, LandmarkResult, LandmarksSource

from ._common import seeded

# name -> (x fraction, y fraction, source)
# x/y are measured from the top-left of the image.
LAYOUT: dict[str, tuple[float, float, LandmarksSource]] = {
    "nose": (0.500, 0.075, LandmarksSource.MODEL),
    "neck_left": (0.452, 0.155, LandmarksSource.MODEL),
    "neck_right": (0.548, 0.155, LandmarksSource.MODEL),
    "shoulder_left": (0.300, 0.205, LandmarksSource.MODEL),
    "shoulder_right": (0.700, 0.205, LandmarksSource.MODEL),
    "elbow_left": (0.215, 0.395, LandmarksSource.MODEL),
    "elbow_right": (0.785, 0.395, LandmarksSource.MODEL),
    "wrist_left": (0.165, 0.575, LandmarksSource.MODEL),
    "wrist_right": (0.835, 0.575, LandmarksSource.MODEL),
    "bust": (0.500, 0.255, LandmarksSource.MODEL),
    "chest": (0.500, 0.230, LandmarksSource.HEURISTIC),
    "waist": (0.500, 0.375, LandmarksSource.HEURISTIC),
    "hip": (0.500, 0.480, LandmarksSource.HEURISTIC),
    "crotch": (0.500, 0.520, LandmarksSource.HEURISTIC),
    "knee_left": (0.455, 0.740, LandmarksSource.MODEL),
    "knee_right": (0.545, 0.740, LandmarksSource.MODEL),
    "ankle_left": (0.440, 0.945, LandmarksSource.MODEL),
    "ankle_right": (0.560, 0.945, LandmarksSource.MODEL),
}

# which measurements each landmark feeds, shown as tooltips in the UI
GROUPS: dict[str, list[str]] = {
    "chest": ["shoulder_left", "shoulder_right"],
    "waist": ["waist"],
    "hip": ["hip"],
    "back_length": ["neck_left", "waist"],
    "armhole_depth": ["shoulder_left", "neck_left"],
    "sleeve_length": ["shoulder_left", "elbow_left", "wrist_left"],
    "outseam": ["waist", "ankle_left"],
    "inseam": ["crotch", "ankle_left"],
    "rise": ["waist", "crotch"],
    "thigh": ["hip", "knee_left"],
}


class MockLandmarks:
    def run(self, image: str, size: dict[str, int]) -> LandmarkResult:
        w = float(size.get("front_width") or size.get("width") or 640)
        h = float(size.get("front_height") or size.get("height") or 960)
        rnd = seeded("lm", len(image), int(w), int(h))

        points: dict[str, LandmarkPoint] = {}
        for name, (fx, fy, source) in LAYOUT.items():
            # a few px of jitter so the mock does not look like a template
            jx = fx * w + rnd.uniform(-4, 4)
            jy = fy * h + rnd.uniform(-4, 4)
            conf = 0.99 if source is LandmarksSource.MODEL else 0.72
            points[name] = LandmarkPoint(
                x=round(jx, 1),
                y=round(jy, 1),
                confidence=round(conf - rnd.random() * 0.06, 3),
                source=source,
            )

        return LandmarkResult(
            is_mock=True,
            points=points,
            confirmed_by_user=False,
            warnings=["mock landmarks, derived from image geometry only"],
        )
