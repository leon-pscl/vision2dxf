**Interpretation.** V-HW and V-H are separate phases (`p08_train_vhw`, `p09_train_vh`), not one
loop. That is deliberate: if the session ends with V-HW finished and V-H half-trained, only V-H
resumes. A combined loop would restart V-HW from epoch 1 every time.

Each training call passes the **full** `EPOCHS_PROD` budget. That matters. The cosine learning-rate
schedule is built from `total_steps = epochs × steps_per_epoch`, so a session that passed a smaller
epoch count would silently change the learning rate of every step it ran. An interrupted session
stops via the time guard, not by being told to run fewer epochs, and the configured budget is
carried in `last.pt` and reused on resume.

Early stopping applies to these two variants (and to B2), with `early_stopping_patience` epochs of
tolerance and `early_stopping_min_delta_mm` as the threshold an epoch must beat the best by. It does
**not** apply to B0 or B1: B1's ridge is closed-form, and giving B0's `HistGradientBoosting`
sklearn's internal validation split would make the non-visual bar stochastic, which would make every
B0-beats-vision comparison depend on a coin flip. B0 stays deterministic and un-tuned.

The consequence is that the *number* of optimiser steps is no longer known in advance — it depends
on when validation plateaus. That is the trade the brief asked for, but it is worth stating plainly:
`TRAIN_STEPS` below is a **result**, not an input. Read the printed `epochs_completed` and
`total_steps_run` per variant before quoting any comparison against a published budget.

If a variant early-stops, `stop_reason` reads `early_stop` and `early_stopped` is `True`. If the
session guard fires first, `stop_reason` reads `session_budget` and the phase is `partial`. Both are
printed, and both are recorded in the manifest under that phase's `meta`.