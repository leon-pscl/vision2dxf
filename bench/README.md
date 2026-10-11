# Bench: the GarmentCode reference bodies

Five bodies ship with GarmentCode, each with a YAML of its true measurements.
This renders them, runs the engine, and reports the error. **The thresholds in
`config/app.yaml` are not tuned to these bodies** (brief M4b): five bodies cannot
justify a threshold, and fitting them would only hide the real error.

Each body's **published `height` is fed in as the scale**, because that is what
the app does: the engine turns the head-to-floor pixel span into millimetres with
the operator's measured height. The five bodies are 165-179 cm, so the bench
exercises five different scales rather than one.

## Running it

```bash
pip install -r requirements-bench.txt
git clone https://github.com/maria-korosteleva/GarmentCode vendor/GarmentCode
PYTHONPATH=src python bench/run_bench.py vendor/GarmentCode/assets/bodies
```

Results land in `bench/results.csv`; `bench/out/` holds each body's export.
`pygarment` comes from the clone, not from PyPI: the published wheel installs its
modules under different top-level names, does not ship the `BodyParameters`
subclass, and pins a numpy older than opencv-python 5 accepts.

## Results

All five exports load through GarmentCode's `BodyParameters`, which recomputes
all seven derived underscore keys — so the output is accepted by Objective 1.2
unchanged.

Mean absolute error over the five bodies, in cm (degrees for the three angles):

| Good (< 3 cm) | | Fair (3-8 cm) | | Poor (> 8 cm) | |
|---|---|---|---|---|---|
| height | 0.0 | neck_w | 3.2 | hips | 10.8 |
| back_width | 1.0 | underbust | 3.1 | shoulder_w | 10.8 |
| bust_points | 1.5 | wrist | 2.4 | shoulder_incl | 11.8 |
| arm_pose_angle | 1.5 | hip_inclination | 2.8 | bust_line | 12.0 |
| armscye_depth | 1.6 | hip_back_width | 3.4 | waist_over_bust_line | 12.2 |
| bum_points | 1.8 | bust | 2.2 | hips_line | 13.5 |
| | | leg_circ | 4.1 | arm_length | 14.4 |
| | | head_l | 4.6 | waist_line | 17.7 |
| | | vert_bust_line | 7.3 | crotch_hip_diff | 22.7 |
| | | waist | 2.9 | | |

`height` errs by exactly zero, and that is not a result: since height-based
scaling it is an **input**, not a measurement, so it is as accurate as the number
the operator typed. The girths the method exists to produce - bust, underbust,
waist - still land within about 3 cm. The poor column is dominated by three
causes, none of them the scale.

### 1. The curvature search picks the wrong corner

`shoulder_w`, `shoulder_incl`, `waist_line` and `vert_bust_line` all depend on a
point found as the maximum turning angle over +/-15 contour points (brief 5.5).
On a real body several corners compete:

- the shoulder point was landing on the neck-shoulder junction. Reading "outer
  contour" as *lateral to the neck* fixed that, and `shoulder_w` on `mean_all`
  went from 25.8 cm to 37.4 cm against a true 36.5 cm. It still under-reaches on
  the other four bodies.
- the nape still lands about 7 cm high, at the base of the skull rather than the
  neck-to-back junction, which is why `waist_line` reads ~18 cm long on every
  body: the path starts too far up.

A curvature search over a 3 cm baseline is simply not selective on a smooth
body. This is the single change most likely to move the numbers, and it needs a
definition the brief does not currently give.

### 2. The crotch row is a silhouette artifact

`crotch_hip_diff` is 16-30 cm high on every body, and `hips_line` is 9-23 cm
low. Both follow from two rows:

- the **crotch row** is "the first row below the hip keypoints where background
  appears between the legs". The thighs of these meshes touch well below the
  anatomical crotch, so the row lands ~9 cm low. The rule is the brief's own;
  the limitation is inherent to a silhouette.
- the **hip row** is the row of maximum back depth between the waist and crotch
  rows. It lands ~12 cm above the true hip level, which also makes `hips` read
  about 12 cm small.

### 3. Two reference values look like a different convention

`f_smpl_average_A40` and `m_smpl_average_A40` both publish `arm_length: 80.0`
on a ~170 cm body, where the three `mean_*` bodies publish 52-57. Measured
against the `mean_*` bodies alone, `arm_length` errs by 0.0, 4.6 and 5.5 cm.
The 80 cm figures are what make its mean error 14.4, and they are worth
checking before treating that number as real.

## What this does not tell you

The renderings are noiseless, perfectly lit, exactly frontal and exactly side-on.
A real capture has none of that. These errors are a floor, not an estimate.
