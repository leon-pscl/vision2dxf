# Pose gate leniency and the wrist cross-section fix

Date: 2026-10-11. Branch: `iteration1-pascual2`.

Context: the three photos in `web/pascual_sample/` are a realistic hand-held
capture of one subject (front, left side, right side). Every one of them was
rejected by the pose gate, and once they were accepted the measure step failed
with a 422. The target garments are topwear - dresses, long-sleeve polos,
polo shirts - so the gate was tuned to what those patterns actually need rather
than to an ideal studio capture.

## 1. What the samples measured

Numbers taken by running the real seg and pose models over the three photos, not
estimated:

| | front | side left | side right |
|---|---|---|---|
| frame | 808 x 1280 | 446 x 1280 | 422 x 1280 |
| clear above the head | 3.4% | 9.5% | 10.8% |
| clear below the feet | 26.6% | 2.9% | 3.3% |
| body span of frame height | 69.9% | 87.7% | 85.9% |
| centre offset | 0.2% | 8.7% | 4.0% |
| arm angle from torso | 19.7 / 35.2 deg | - | - |
| ankle confidence | 0.05 / 0.07 | 0.96 / 0.98 | 0.93 / 0.99 |

Every rejection came from a threshold the subject missed by a hair, or from a
keypoint nothing downstream reads.

## 2. What these files already carried

Two earlier passes shaped the gate before this one. Both are listed here because
section 4 reads as a reversal of one of them and is not.

**`b5757b4` - height input replaces the checkerboard (committed).** Dropping the
floor mark took `floor_row_tol_px` out of `app.yaml` and put `body_height_frac`
and `body_centre_band_frac` in: with no calibrated floor row to check the feet
against, the gate had to protect the whole body being in frame at a usable size
instead. That same commit raised `frame_margin_frac` from **0.02 to 0.08**. The
change in section 4 puts it back to 0.02 - not a new guess, the value the gate
ran on before the floor mark went away.

**Reason wording (uncommitted, already in the working tree).** The too-short
silhouette used to say *"Stand further away so the whole body fits in the
frame"*, which told the operator to do the opposite of what fixes it. It now says
*"Body too small in the frame: step closer, or crop an uploaded photo"*, and the
assertion in `test_a_body_too_short_in_the_frame_is_rejected` matches, with a
docstring paragraph spelling out why the fix is to come closer. That wording is
what the rejection screenshots show; it is not part of this pass.

## 3. Front gate: only the measured keypoints

`gate_front` required all 17 COCO keypoints at confidence >= 0.5. Nothing outside
`posegate.py` reads the face, the knees or the ankles - the scale comes from the
mask's head-to-floor span, not from keypoints, and the leg girths are cut from the
silhouette. The front photo's ankles scored 0.05 and 0.07 (dark trousers against a
dark floor), so a usable capture was thrown away over data the engine never opens.

`MEASURED_KEYPOINTS` in `src/bodymeasure/posegate.py` now names the ten that are
actually read - both ears, shoulders, elbows, wrists and hips - and the front gate
checks only those. The legs still have to be inside the mask; that is what the
framing checks are for.

## 4. Loosened thresholds (`config/app.yaml`)

| threshold | was | now | why |
|---|---|---|---|
| `frame_margin_frac` | 0.08 | 0.02 | The margin exists to prove the body is not clipped, because a cut-off foot corrupts the height scale. Two per cent of 1280 px is ~26 px: enough proof, and it stops punishing a photo framed tight to the feet. |
| `body_height_frac` | 0.70 | 0.60 | The front sample filled 69.9%. At 0.60 of a 1280 px frame the body is still ~770 px tall. |
| `arm_angle_min_deg` | 20 | 15 | The subject's left arm sat at 19.7 deg. People asked to hold an A-pose do not hold it to the degree. |
| `arm_angle_max_deg` | 45 | 60 | Same reason, other end. |

Left alone on purpose: `body_centre_band_frac` (0.30) and
`side_shoulder_ratio_max` (0.25). All three samples pass both, and the side-on
check guards the depth that every girth is built from.

The on-screen guide in `web/app.js` still draws the old 8% / middle-third box. It
is now a comfortable frame rather than the gate's minimum: stand inside it and the
capture passes with room to spare.

## 5. The 422: "arm not found near the wrist keypoint"

With the gate passing, `measure` raised from `narrowest_cross_section`
(`src/bodymeasure/landmarks.py`). The left wrist keypoint landed **9 px outside the
mask** - the segmentation stopped short of the hand, while pose still scored the
wrist at 0.96. The search walked the elbow-wrist line looking for a foreground run
that *contained* the sample centre, and the centre was never on the arm, so every
step was skipped and the function raised. The right wrist was inside its mask,
which is why only one arm failed.

The fix takes the run **nearest** the sample centre within `search_px` - the same
+/- 4 cm already allowed along the arm - instead of requiring it to contain the
centre. Nearest, not any, so a hand held near the body cannot make the torso win;
the A-pose check in the gate already keeps the wrist off the torso.

`tests/test_landmarks.py::test_the_cross_section_survives_a_wrist_keypoint_off_the_arm`
puts the wrist one arm-thickness off the silhouette. It fails on the old code and
passes on the new.

## 6. Known limitation: the side rows collapse

All three samples now pass the gate and the full flow runs, but the measurements
are **not trustworthy yet**. Measuring the samples (at a placeholder 170 cm):

```
bust 117.1   underbust 117.1   waist 117.2   hips 117.1
```

The landmark dump shows why: on `side_left`, `bust_row`, `underbust_row` and
`hip_row` all land on row 692, the waist row. On `side_right` they separate
properly (610 / 670 / 733). One side view is therefore feeding a single depth into
every girth.

The likely driver is the pose: the arms hang against the body in both side photos
(3-4 deg from the torso), so the profile at chest and hip includes the forearm.
Not chased to the root. Two places to look:

- the depth search in `find_side_landmarks` (`landmarks.py`), where `bust`,
  `underbust` and `hip` are picked by max/min front and back depth;
- `gate_side`, which has no A-pose check at all. For topwear that may be the real
  gap - bust depth is exactly what a dress or polo pattern needs.

## 7. Scope

Changed: `src/bodymeasure/posegate.py`, `src/bodymeasure/landmarks.py`,
`config/app.yaml`, `tests/test_posegate.py`, `tests/test_landmarks.py`,
`docs/AGENT_BRIEF.md` (section 5.2), one comment in `web/app.js`. The reason
wording in section 2 was already in the working tree, uncommitted, and is
unchanged by this pass.

The framing tests used to hardcode 0.08 and 0.70; they now derive their numbers
from the thresholds, so the next tuning pass does not break them.

209 passed, 1 skipped, `ruff check` clean. No measurement has been checked against
a tape.
