### §4.7 Resume determinism, early stopping and the session time guard

**WHAT** — In smoke mode, train a tiny real network twice over identical synthetic data: once
straight through for 3 epochs, and once interrupted after 2 and resumed, the way a Kaggle session
ending at hour 9 of 12 interrupts a run. Assert the two reach the same validation MAE, then
exercise early stopping and the time guard on the same harness.

**WHY**
- [Ref] Addendum A4 requires this check with a 1e-3 mm tolerance. It is the only way to know that
  the resume machinery does not silently change the model that ships.
- [Exp] Three properties make it hold, and each is exercised rather than assumed:
  1. the learning rate is a **pure function of the global step** over the full `total_steps`
     budget, so splitting a run across sessions cannot change the schedule;
  2. shuffle order is a pure function of `(seed, epoch)`, re-seeded per epoch rather than advanced;
  3. every RNG stream — Python, NumPy, torch, CUDA — is restored from `last.pt`.
- [Exp] The interruption is produced by `max_epochs_this_call`, **not** by passing a smaller
  `epochs`. A smaller epoch budget changes `total_steps` and therefore the cosine schedule, so the
  two runs diverge from step 0 and the test measures the wrong thing. A real session ends via the
  time guard, which leaves the configured budget untouched. The cell asserts the per-epoch learning
  rates match, so this cannot regress unnoticed.
- [Exp] The `total_steps` in force is stored in `last.pt` and reused on resume. Recomputing it from
  the current epoch budget would silently alter the learning rate of every remaining step.
- [Exp] `best.pt` is only loaded on a fresh start. Loading it after a resume would overwrite the
  restored weights with the previous session's best, which is wrong once training has continued.
- [Exp] The patience counter is persisted in `last.pt`. Without that, a resumed run would reset its
  patience and never stop — so the early-stopping half of this test resumes deliberately and checks
  that it still fires.
- [Exp] An interrupted phase is marked `partial`, never `done`, and a `partial` dependency blocks
  everything downstream. The cell asserts this via the run state.
- [Exp] This runs on CPU with synthetic data. cuDNN nondeterminism can still introduce small
  differences on GPU even with deterministic flags, because some kernels have no deterministic
  implementation, so the 1e-3 bound is asserted here where that noise source is absent rather than
  claimed universally.

**OUTPUT** — `DETERMINISM`, and the run-state manifest updated with the probe phases.

**CHECK** — asserts `|straight − resumed| < 1e-3` mm; asserts the per-epoch validation history and
learning rates are identical; asserts the checkpoint directory holds exactly `last.pt` and
`best.pt`; asserts early stopping fires and still fires after a resume; asserts the time guard
marks the phase `partial` and calls back instead of raising.