### How to run this notebook on Kaggle

Kaggle persistence is assumed to be **Files only**. Nothing depends on a Python variable surviving
between sessions: every phase writes its artefacts under `/kaggle/working/run/` and records them in
`run/manifest.json`, so a session that ends early can be resumed from where it stopped. A committed
*Save & Run All* may start with an empty `/kaggle/working`; the notebook detects that and starts
fresh, or adopts a previous run state attached as an input dataset.

Phases are numbered `p00`–`p14`. The status table printed at the start of §0.5 shows which are
already done; finished phases are skipped on every subsequent run.

**Step 1 — CPU, interactive.** Accelerator: **None**. Persistence: **Files only**.
`smoke_test: true`, `run_final_eval: false`.
Run all cells. Phases 0–5 (setup, data, EDA, splits, B0, B1) need no GPU. Every assertion must
pass, including the resume self-check and the determinism test in §4.6.

**Step 2 — CPU, interactive, real settings.** `smoke_test: false`.
Run all cells again. Phases 0–5 recompute at full size. This is the last CPU-only pass.

**Step 3 — GPU.** Accelerator: **GPU T4 x2**. Persistence: **Files only**.
Run all cells. Phases 6–11 (training, the three sensitivity sweeps, conformal) run on the GPU and
write checkpoints after every epoch. Phases 0–5 are skipped.
If the session ends before training finishes, the time guard has already saved `last.pt` and marked
the phase `partial`. **Stop Session**, start a new one, and run all cells again: the training phase
resumes at the next epoch and everything else is skipped.

**Step 4 — finish or resume.** Repeat step 3 until every phase reports `done` in the status table.
Then **Stop Session**.

**Step 5 — the final record.** `run_final_eval: true`, then **Save Version → Save & Run All**.
If that run cannot complete training inside 12 h, attach the previous version's output as an input
dataset so it resumes from it (the notebook looks for `/kaggle/input/*/run/manifest.json` and says
which source it used).

**Step 6 — publish.** Upload `/kaggle/working/export/` as a **private** Kaggle Dataset for the
downstream steps. The bundle contains the licence-restricted weights, so it must not be public.

### Notes on budget

- Sessions are capped at 12 h and GPU quota at roughly 30 h per week. The three training runs plus
  three sensitivity sweeps will not fit in one session; that is what resume is for.
- `/kaggle/working` holds roughly 500 files and 20 GB. This notebook asserts a lower budget
  (`max_files_working`, `max_gb_working`) and writes regenerable caches to `/kaggle/temp`, which
  Kaggle does not keep.
- `config.yaml: force_rerun` takes a list of phase names (or variant tags) to recompute despite a
  valid cache — use it after a config change you do not want to invalidate everything for.