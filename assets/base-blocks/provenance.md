# Base-block provenance

Where the three catalogued blocks came from, what was checked, and what was changed.

## Source

All three templates are GarmageSet pattern files taken unmodified from the GarmageSet
library. Each is content-addressed by its own filename and pinned in `catalog.yaml` by
`sha256`, so a block that is not byte-for-byte the reviewed file is refused rather than
silently becoming a different block.

| Block | Source file | Boundary | Seams | Darts |
|---|---|---|---|---|
| `bodice-ref-a` | `5f8a1865c9aa84855f6122db55ad4e93.json` | 10 edges per panel | 6 | none |
| `bodice-ref-b` | `33578877396454b71782bdee252fe8ef.json` | 12 edges per panel | 4 | none |
| `bodice-ref-c` | `dd300c91b06a7481190a63c5fcfb802c.json` | 16 front / 10 back | 8 | two front darts |

## Modifications

**None.** No template was edited, cleaned, resampled or re-saved. The pinned bytes are
the library's bytes.

This matters because the semantic layer is *derived* when a block is loaded, never stored
beside it: panel labels in these files are garment part names in Chinese and edge labels
are mostly just numbers, so nothing a person might have tidied up carries any meaning the
pipeline relies on. Editing a template to make its semantics easier to read would change
the geometry under a pinned hash and would be refused.

The only thing declared rather than derived is the body each block is considered to fit,
and that lives in `catalog.yaml`, not in the template. It is an operator-revisable
assumption and is recorded in every generation manifest.

## Review

Each block was accepted only after passing the same geometry validator that runs on
generated patterns, plus role completeness:

* every required landmark role is present (`side_neck`, `shoulder_tip`, `underarm`,
  `waist_side` on both sides; `center_neck` and `center_hem` on the centre line)
* every semantic boundary role is present on both sides
* the panel is a whole front or back, mirrored about its own centre line, crossing that
  line at exactly two vertices
* boundaries close, do not self-intersect, and carry no zero-length or duplicate edges

A template that does not yield a complete, mirror-consistent torso block is refused with
a structured reason rather than transformed. Most of the library is refused this way: the
files are whole garment sets, not torso blocks.

## Known defects carried by a block

`bodice-ref-c`'s own paired seams differ by up to **4.8 mm** before anything is
transformed, which is past the 2 mm seam tolerance. The block is still offered, because
the difference is a property of the source pattern and not of the transformation. It is
reported on the block, inherited by every pattern made from it, and the transformation is
held to not widening it rather than to closing it. Pick another block if the seams must
match before grading.

`bodice-ref-a` carries internal waist, bust and chest reference lines as open interior
contours. They are deformed with the panel but are not modelled as darts.

None of the three blocks has a hip edge or a marked bust apex, so hip and bust-apex
measurements have no landmark to move and are reported to the operator as unmapped. See
the README's "Known limits".
