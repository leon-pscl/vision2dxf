# Reconstructed configuration

Several files in this repository were lost and rebuilt rather than recovered. Everything
below is a **value someone chose while rebuilding**, not a value carried over from the
original project. Read this before trusting a number in `transformation.yaml`, the base-block
catalog or the target fixtures.

The loss: `backend/app/config/transformation.yaml`, the whole of `assets/base-blocks/`, the
two target fixtures, and the source of sixteen Python modules (`app/base_block/`,
`app/transformation/`, `app/validation/topology.py`) and two React components
(`BaseBlockStep`, `ComparisonViewer`). None of it was ever committed.

## How much was recoverable

Three things made this a fit rather than an invention:

* **Compiled bytecode.** `__pycache__` still held `.pyc` files for all sixteen Python
  modules. Disassembling them yields every configuration key the code reads, the exact
  spelling of every catalog field, and the full vocabulary of influence-map modes,
  directions and landmark roles. **The structure of `transformation.yaml` is therefore not
  guesswork.** Only the values in it are.
* **The test suite.** It pins hard numbers and behaviour, so a rebuilt config is checked
  rather than eyeballed. 145 backend and 17 frontend tests pass.
* **The README.** Its block table, known limits and dimension definitions are specific
  enough to identify files and to reason about weights.

## Values that are pinned, not chosen

These are forced by a test or derived from a file that survived, and are as good as the
originals:

| Value | Why it is certain |
|---|---|
| `objective.w_landmark: 1.0` | asserted by `test_the_manifest_identifies_the_base_block_and_configuration` |
| `solver.method: thin_plate_spline` | asserted by `test_the_response_carries_deltas_diagnostics_and_achieved_dimensions` |
| `reference.profile: body_measurements.yaml` | the tests require the block's reference bust to be 858.0 mm, and that fixture's bust is 85.8 cm |
| the whole key schema of `transformation.yaml` | read off the bytecode; a missing or misspelled key raises |
| `base_block.catalog_directory` / `template_directories` | the paths named in the README and `.gitignore` |
| the unmapped measurement list | `semantic_map.CONDITIONAL_LANDMARKS` plus the README's "Known limits" |

## Values that were chosen

### Solver and refit limits — invented

Nothing constrains these but the suite passing. They are plausible engineering defaults,
not the original tuning.

| Key | Value | Basis |
|---|---|---|
| `objective.w_smoothness` | `1.0e-6` | started at `1.0e-3`; the solver then missed its own residual tolerance (0.645 mm against 0.5 mm), so it was lowered until the spline effectively interpolates. **Chosen to make a test pass.** |
| `solver.max_landmarks` | `64` | a torso panel resolves well under twenty; this is a guard, not a tuned limit |
| `solver.condition_number_limit` | `1.0e+12` | conventional double-precision cutoff |
| `solver.landmark_residual_tol_mm` | `0.5` | invented; interacts directly with `w_smoothness` above |
| `solver.coincident_landmark_mm` | `0.5` | invented |
| `curve_refit.samples` | `64` | invented |
| `curve_refit.max_handle_ratio` | `3.0` | invented |
| `curve_refit.max_turn_deg` | `120.0` | invented |
| `seam_balance.max_iterations` | `6` | invented |
| `seam_balance.tolerance_mm` | `1.0` | invented; half of `drafting.yaml`'s `seam_mismatch_abs_mm`, which did survive |
| `seam_balance.achieved_tolerance_mm` | `3.0` | invented |
| `seam_balance.achieved_tolerance_deg` | `1.0` | invented |

### The envelope — one entry bracketed, the rest invented

Three assertions constrain `bust_circumference` only. Against a reference bust of 858.0 mm,
the suite requires a difference of 82 mm to pass, 152 mm to warn but be confirmable, and
252 mm to be refused however it is confirmed. That forces `max_abs_mm` into `(82, 152]` and
`stop_factor × max_abs_mm` into `[152, 252)`. `100.0` and `2.0` satisfy it; so would other
pairs.

**Every other measurement's limits were set by analogy and have no empirical basis.** The
README says the envelope "declares the tested range per measurement" — no such testing was
done here. Treat the non-bust rows as placeholders.

### The influence map — reasoned, never tuned

The weights follow arithmetically from the garment-dimension definitions that survived in
`app/base_block/measurements.py`'s `DEFINITIONS` (e.g. bust girth is front panel width plus
back panel width at the underarm level, and a panel is mirrored, so a panel width is twice
a one-sided offset). They are not arbitrary. But they were **never checked against a real
pattern**, and one modelling decision in particular is mine, not the original author's:

> `bust_circumference` moves the front panel only, and `back_width_full` moves the back out
> and the front in by the same amount. That makes both dimensions land exactly and keeps
> each millimetre attributable to one difference. The cost is that if `back_width` is ever
> absent, a pure bust change widens only the front panel and the side seams must be
> balanced harder. In this schema `back_width` is always present (`ASSUMED`), so the
> degenerate case does not arise — but an even front/back split is the more conventional
> choice and the original may well have made it.

`achieved_dimensions` is the arithmetic inverse of the above and inherits the same caveat.

### The target fixtures — one value pinned, 25 invented

`fixtures/body_measurements_target_a.yaml` must have `bust: 94.0`, because
`test_deltas_endpoint_reports_differences_and_the_envelope` asserts a target of 940.0 mm.
**Every other value in `target_a`, and every value in `target_b`, was made up** — chosen to
be anatomically plausible, to stay inside the envelope above, and to run in opposite
directions as the README's comment on the two profiles requires.

## The base blocks — identified, not chosen

The three templates were found by scanning the library for the fingerprint in the README's
block table (panel count, boundary edges per panel, seam count, darts, internal contours)
and running each candidate through the recovered `derive_block`.

| Block | File | Confidence |
|---|---|---|
| `bodice-ref-c` | `dd300c91b06a7481190a63c5fcfb802c.json` | **confirmed.** Loading it reports a source seam mismatch of 4.782 mm; the README independently says block C's seams "already differ by up to 4.7 mm" |
| `bodice-ref-b` | `33578877396454b71782bdee252fe8ef.json` | **high.** Only two files match 12 edges per panel with 4 seams; the README calls B "the simplest" and this one has 90 control points against the other's 230 |
| `bodice-ref-a` | `5f8a1865c9aa84855f6122db55ad4e93.json` | **a coin flip.** `3b6ef5bdefdac651b1f82ed195fb3457.json` has an identical fingerprint and differs only in curve density (122 control points against 150). Nothing in the README or the tests separates them |

`catalog.yaml`'s `label` strings and the whole of `provenance.md` are newly written. The
provenance file's factual claims (no template was modified, which checks each block passes,
block C's carried seam defect) were all verified against the validator rather than assumed.

## The Python modules

`app/base_block/`, `app/transformation/` and `app/validation/topology.py` currently exist
**only as `.pyc` files** copied out of `__pycache__` and placed where their `.py` files
should be. Python imports them, so the application runs.

This is not a fix:

* they are unreadable and uneditable
* they are CPython 3.13 bytecode and will not load on another version. The image was
  `python:3.12-slim` and crashed on startup with `ModuleNotFoundError: No module named
  'app.base_block.loader'`; it is now **pinned to `python:3.13-slim`**, and
  `.dockerignore` carries three `!` exceptions so the bytecode is copied in at all.
  Neither the base image nor the local interpreter can move off 3.13 until there is
  source. Both carry a `ponytail:` comment saying so.
* `.gitignore` excludes `*.py[cod]`, so **they are not in version control and will not
  survive a fresh clone**

Recovering real source for these sixteen modules is the most urgent outstanding task in
this repository.

## What to do about all this

1. Have a pattern maker check one generated pattern against a real block. The suite checks
   properties (an unchanged target reproduces the block, panels and seams are preserved,
   dimensions are positive); it never checks that the geometry is *right*.
2. Replace the envelope rows with ranges you have actually tested.
3. Decide whether the front/back girth split above matches your intent.
4. Confirm which `bodice-ref-a` candidate is correct, or accept the one pinned here.
5. Get source back for the sixteen bytecode-only modules before anything else depends on
   them.
