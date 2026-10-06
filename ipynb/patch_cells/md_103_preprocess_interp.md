**Interpretation.** The self-test checks **equivalence, not just the contract**. A downstream caller
does not have the original masks, so the exported bundle ships a few reference samples *with* their
inputs, which is what makes a true comparison possible. What actually breaks in integration is shape,
channel count, dtype, channel semantics, resize behaviour and pair order; the previous version's
synthetic-rectangle check caught only the first three.

Two semantic properties are asserted that a shape check would miss:

- every channel above the mask is **constant across the image** (a non-constant one means the channel
  was built from image data), and
- the height channel equals `height_cm / 200` and the weight channel equals `weight_kg / 100`.

Those last two are the normalisation constants. If someone "cleans up" a magic number in the training
code and the exported module keeps the old one, predictions stay plausible and quietly wrong — this is
the assertion that catches it. Note also that `q`-style unit confusions are impossible here: the
constants are cm-based and the test compares tensors, not derived scalars.

One structural note: the module imports `to_tensor` rather than reimplementing it, and the bundle
ships `bodym_cnn.py` alongside. One definition of preprocessing, verified from a subprocess with no
notebook globals in scope.