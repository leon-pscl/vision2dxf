### §0.6 Resume state and the run protocol

**WHAT** — Establish the `run/` directory layout, adopt any previous run state, validate
`manifest.json` against its schema, instantiate the phase registry, print the phase-status table, and
check the working-directory budget against the Kaggle persistence caps.

**WHY**
- [Exp] Kaggle's *Files only* persistence guarantees files in `/kaggle/working` survive between
  sessions; it guarantees nothing about Python state. A committed *Save & Run All* can also start
  with an empty `/kaggle/working`. So nothing may depend on a variable surviving — every phase
  writes artefacts to disk and can be reloaded.
- [Ref] Addendum A3: resume state is looked for first in `/kaggle/working/run/`, then in
  `/kaggle/input/*/run/` (a previous version's output attached as an input dataset). Working state
  wins, and the cell prints which source it used.
- [Ref] Addendum A2: a phase is reusable only when its status is `done`, its recorded
  `config_hash` matches the current config, **and** every artefact it listed still exists. All
  three are checked independently, so a truncated working directory or a changed `smoke_test`
  cannot silently reuse stale work.
- [Exp] The manifest is validated against a schema at startup rather than trusted. One that fails
  to parse degrades to a fresh manifest instead of aborting: losing an hour of cached work beats
  losing the run.
- [Ref] Addendum A1/A7: `/kaggle/working` is capped at roughly 500 files and 20 GB. This notebook
  budgets below that and asserts it, because a run that only trips the limit after the export has
  been written cannot be fixed in the same session.
- [Ref] Addendum A0: phases 0–3 must run on CPU. That is enforced by a tripwire rather than a
  printed device string — `cpu_guard` makes any `.to('cuda')` raise inside a CPU phase.
- [Exp] The registry is duplicated from the config cell rather than imported from `config.yaml`,
  because a notebook cannot import a Python object out of its own earlier cell and YAML round-trips
  types. The assert below fails loudly if the two copies ever diverge.

**OUTPUT** — `run/manifest.json`, `run/ckpt/`, `run/results/`, `export/figs/`, `TMP`, `RUNSTATE`,
`RESUME`, `PHASES`, `DEVICE`, and the helper imports `begin` / `done` / `guard`.

**CHECK** — asserts the manifest validates; asserts every phase's dependency names exist; asserts
the working directory is under `max_files_working` files and `max_gb_working`. In `smoke_test` mode
it additionally proves the write → reload → verify cycle, verifies all three reuse conditions fail
independently, and asserts the CUDA tripwire actually fires.