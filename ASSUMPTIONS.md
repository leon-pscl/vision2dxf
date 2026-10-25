# Assumption Registry (static companion to `generation-manifest.json`)

This file documents the inferred values that are not present in
`body_measurements.yaml` and are therefore recorded as `SurfaceAssumption`
entries at runtime. The runtime record lives in `generation-manifest.json`; this
file is the human-readable source list. Values and reasons are versioned in
`backend/app/config/reconstruction.yaml` (config version `3.0.0`).

## Surface

- **Inferred torso depth (`depth_to_width`)** — neck 0.85, upper chest 0.70, bust 0.72,
  underbust 0.70, waist 0.74, hip 0.75. Used when a level has a circumference but no
  matching width measurement. Reason: anthropometric ratios inferred from fixture profiles;
  no input measures depth directly.
- **Superellipse exponent `n = 2.3`** (range 2.0–4.0). Torso sections are flatter at the
  sides and squarer at the front than an ellipse. Configurable alternative `n = 2` (exact
  ellipse). No measurement supports 2.3.
- **Front/back asymmetry disabled** (`front_back_asymmetry: false`). `bust_points` is a
  protrusion, not a half-depth; no input supports asymmetry.
- **Underbust level** placed at 30% of the bust-to-waist interval. `underbust` is a
  circumference with no vertical level of its own.
- **Hip section disabled by default** (`include_hip_section: false`). `hips_line` is measured
  from the crotch, so it cannot be placed in the neck-referenced frame without an assumption.

## Garment

- **Ease is a circumference**, converted to a per-section scale rather than a uniform normal
  offset. Achieved normal displacement varies around each section and is reported.
- **Armhole ease drop fraction 0.5** — underarm is dropped by this fraction; no input
  defines armhole ease as a vertical level.
- **Neckline drop** 70 mm front / 20 mm back. No neckline input exists.
- **Neckline power 1.0, armscye power 1.0** — drawn shape exponents, not derived.

## Mesh and flattening

- **Monotone cubic smoothing** of the `(a, b)` stack through measured sections; no tuning
  parameter, chosen to avoid overshoot.
- **ARAP weights**: `w_distortion=1.0`, `w_boundary=0.05`, `w_seam=0.05`,
  `w_grain=0.0`, `w_flip=0.0`; flipped triangles are rejected outright (hard gate).
- **Grain axis** along centre front/back, applied as exact rigid alignment after solve.
- **Boundary simplification** tolerance 0.6 mm, straight tolerance 0.8 mm, cubic fit
  tolerance 1.5 mm, max handle ratio 2.0.
- **Seam balance** 12 iterations / 0.5 mm tolerance.

## Reconstruction naming

The body is reconstructed as a **proxy anthropometric surface**. Scalar measurements are
not a complete scan; inferred depths, shapes, and landmark assumptions above are recorded
here and emitted into `generation-manifest.json`. They do not appear in the compatibility
pattern JSON.
