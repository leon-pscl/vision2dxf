# Implementation Plan: Torso Proxy to Dress-Ready Surface

## Goal
Extend the current Design 3 proxy anthropometric surface pipeline from a **torso-only**
garment to a **dress-ready garment surface**, while removing inferred anatomical
assumptions and keeping the 2D pattern derived from flattening.

Scope is limited to one vertical garment continuation: torso + skirt to hem. Full-leg
garments are explicitly out of scope for this increment.

## Core rules (unchanged)
- No 2D master block, drafting formulas, retrieved patterns, or learned models.
- Every measurement conflict is reported and blocks generation; nothing is silently repaired.
- Every inferred value is either removed or becomes an explicit input or confirmed setting.
- No non-converged or flipped flattened result may be accepted.

## Phase 1 — Measurement model and level confirmation
### Changes
- `backend/app/drafting/measurements.py`
- `backend/app/schemas/api.py`
- `backend/app/config/measurements.yaml`
- frontend measurement review / semantics

### Work
1. Add required dress-relevant fields:
   - `underbust_line`
   - `hips_line`
   - `hem_line`
   - `hem` circumference
   - optional but validated `depth`/`breadth` per supported level
2. Require user confirmation for all vertical levels; no fractional auto-placement.
3. Remove `underbust_fraction_of_bust_to_waist` from the active path for dress mode.
4. Make neckline and armscye vertical definitions explicit config/user settings, not inferred
   defaults.

### Tests
- `test_measurements.py`: reject missing `hem_line`, missing `ham`, non-finite values
- `test_semantics.py`: block unresolved dress levels
- `test_reconstruction.py`: out-of-order / too-close levels still refused

### Acceptance
- Both YAML fixtures still load.
- Missing dress levels fail with structured errors.
- No silent reordering.

## Phase 2 — Cross-section solver without inferred depth
### Changes
- `backend/app/reconstruction/cross_sections.py`
- `backend/app/reconstruction/perimeter_solver.py`
- config: make depth ratios optional and non-default in dress mode

### Work
1. If a level has circumference + width + depth, solve using all three.
2. If depth is missing, require an explicit config assumption or block.
3. Keep numerical perimeter and residual per section.
4. Reject impossible width/circumference/depth combinations.

### Tests
- superellipse perimeter residual tests
- impossible section rejection
- depth present vs depth absent behavior

### Acceptance
- Every section reports target and achieved perimeter.
- No inferred depth ratio is used unless explicitly configured and recorded.

## Phase 3 — Full-body ring lofting and smoothing
### Changes
- `backend/app/reconstruction/lofting.py`
- `backend/app/reconstruction/smoothing.py`
- `backend/app/reconstruction/landmarks.py`

### Work
1. Extend ring stack from neck to hem.
2. Keep 48-column correspondence and manifold checks.
3. Preserve circumference at every confirmed level.
4. Add landmark tags for hem, side seam, center front/back.

### Tests
- ring correspondence across extended stack
- manifold and orientation checks
- zero-area face rejection
- smoothing preserves all measured levels
- monotonic no-overshoot behavior

### Acceptance
- Mesh passes through every confirmed section.
- Mesh is manifold and outward-facing.
- No degenerate faces.

## Phase 4 — Level-dependent ease and conversion documentation
### Changes
- `backend/app/garment_surface/ease.py`
- `backend/app/garment_surface/offset.py`
- reconstruction config ease section

### Work
1. Keep ease interpolated by level.
2. Remove defaults-as-assumptions from user path; require selected ease preset or explicit values.
3. Report requested and achieved circumference per level.
4. Keep ease conversion formula documented and measured.

### Tests
- ease interpolation
- circumference-to-scale conversion
- achieved-vs-requested reporting

### Acceptance
- All levels have ease residual reporting.
- No single uniform offset is applied.

## Phase 5 — Fixed dress seam network and segmentation
### Changes
- `backend/app/garment_surface/seams.py`
- `backend/app/garment_surface/segmentation.py`

### Work
1. Define predetermined seams:
   - center front
   - center back
   - left/right side seam
   - neckline
   - armholes
   - shoulder seams
   - hem
2. Segment into disk-like front/back panels that run from neckline to hem.
3. Maintain semantic labels for all seam paths.

### Tests
- seam continuity
- panel segmentation
- disk-like panel validation
- invalid patch rejection

### Acceptance
- Front and back panels are single, disk-like, connected patches.
- Seam endpoints are stable and semantically labeled.

## Phase 6 — Flattening and diagnostics
### Changes
- `backend/app/flattening/parameterization.py`
- `backend/app/flattening/distortion.py`
- `backend/app/flattening/boundary_refit.py`
- `backend/app/flattening/seam_balance.py`

### Work
1. LSCM initialization, then ARAP refinement.
2. Fix two non-collinear vertices.
3. Reject flipped triangles.
4. Report convergence, angular, area, length distortion, seam mismatch, boundary self-intersection.
5. Refit boundaries with lines and cubic Beziers within configured tolerance.

### Tests
- plane near-zero
- cylinder rectangular development
- truncated cone sector development
- symmetric torso/dress left-right consistency
- no flipped triangles
- boundary closure and self-intersection

### Acceptance
- Controlled geometry tests remain green.
- Flattened dress panels are closed and non-self-intersecting.
- All diagnostics exported.

## Phase 7 — Serialization and exports
### Changes
- `backend/app/serialization/garmageset.py`
- `backend/app/export/dxf.py`
- `backend/app/export/pdf.py`
- rendering

### Work
1. Validate reference fixture before output.
2. Map extended panels into the verified GarmageSet-like schema.
3. Export 1:1 DXF/PDF and previews.
4. Exclude mesh diagnostics and assumptions from compatibility JSON.

### Tests
- reference schema validation
- serializer
- DXF/PDF scale
- bundle contents

### Acceptance
- Schema validator accepts output.
- DXF/PDF are 1:1.
- Bundle contains manifest, diagnostics, validation, previews, exports.

## Phase 8 — Comparison harness
### Work
1. Run torso and dress fixtures through the same validators.
2. Report:
   - achieved vs requested bust/waist/shoulder widths
   - seam mismatch
   - closure
   - runtime
   - DXF scale
3. Compare with Designs 1 and 2 on shared YAML.

### Acceptance
- Same YAML can be compared across designs.
- All required diagnostics are present and parseable.

## Known limits after this increment
- Still proxy, not exact body shape.
- Skirt-only continuation first; trousers are out of scope.
- More complex seam networks, darts, shaping, and styled necklines remain future work.
- Any new inferred value must be exposed in config and recorded in `generation-manifest.json`.
