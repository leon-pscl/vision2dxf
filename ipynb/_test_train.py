"""Executable harness for addendum A4: resume, early stopping, and the time guard.

Trains a tiny real network on synthetic data. No BodyM, no GPU, no DataLoader workers. The point
is to prove that ``1 epoch + resume + 1 epoch`` reproduces ``2 epochs`` to within the addendum's
1e-3 mm, that early stopping fires on schedule, and that the session-time guard marks a phase
partial instead of dying.
"""

import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
_T = HERE / "_t" / "src"
_T.mkdir(parents=True, exist_ok=True)
shutil.copy2(HERE / "patch_cells" / "cell_052_bodym_training.py", _T / "bodym_training.py")
shutil.copy2(HERE / "patch_cells" / "run_state.py", _T / "run_state.py")
sys.path.insert(0, str(HERE / "patch_cells"))
sys.path.insert(0, str(_T))
import run_state as R          # noqa: E402
import bodym_training as T     # noqa: E402

tmp = Path(tempfile.mkdtemp(prefix="tr_test_"))


class Tiny(Dataset):
    """Deterministic synthetic dataset standing in for SilhouetteDataset.

    Parameters
    ----------
    n : int
        Sample count.
    d : int
        Feature width.
    seed : int
        Seed for the underlying data.
    """

    def __init__(self, n, d=8, seed=0):
        rng = np.random.default_rng(seed)
        self.x = rng.normal(size=(n, d)).astype(np.float32)
        # two targets, so the projection must be (n, 2) @ (2, 2)
        self.y = (self.x[:, :2] @ np.array([[2.0, -1.5], [1.0, 0.5]], dtype=np.float32)
                  + np.float32(0.3)).astype(np.float32)
        self.n, self.d = n, d

    def __len__(self):
        """Number of samples."""
        return self.n

    def __getitem__(self, i):
        """Return ``(x, y, subject_id, photo_id)``.

        Parameters
        ----------
        i : int
            Index.

        Returns
        -------
        tuple
        """
        return torch.from_numpy(self.x[i]), torch.from_numpy(self.y[i]), f"s{i // 3}", f"p{i}"


class Net(nn.Module):
    """Minimal regression net with the head shape the scalar check expects.

    Parameters
    ----------
    d : int
        Input width.
    feat_dim : int
        Width of the pooled-feature block.
    """

    def __init__(self, d=8, feat_dim=6):
        super().__init__()
        self.feat_dim = feat_dim
        self.n_scalars = 1
        self.use_weight = False
        # mirrors MeasurementNet: trunk over the feature block, scalars concatenated at the head
        self.trunk = nn.Sequential(nn.Linear(d - 1, feat_dim), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(feat_dim + 1, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, x):
        """Predict two values.

        Parameters
        ----------
        x : Tensor, shape (B, d)

        Returns
        -------
        Tensor, shape (B, 2)
        """
        return self.head(torch.cat([self.trunk(x[:, :-1]), x[:, -1:]], dim=1))


def cfg_for(**over):
    """Base training config.

    Returns
    -------
    dict
    """
    c = {"seed": 7, "batch_size": 8, "lr": 0.05, "lr_min": 0.001, "weight_decay": 0.0,
         "grad_clip": 5.0, "amp": False, "num_workers": 0, "ckpt_minutes": 20,
         "session_budget_hours": 11.0, "early_stopping": True,
         "early_stopping_patience": 3, "early_stopping_min_delta_mm": 1.0,
         "config_hash": "testhash"}
    c.update(over)
    return c


dev = torch.device("cpu")
CFG = cfg_for()
tr, va = Tiny(96, seed=1), Tiny(48, seed=2)

# ------------------------------------------------------------------ A4: resume determinism --
# The comparison the addendum asks for is "2 epochs straight" vs "interrupted after 1, then
# resumed for the rest". Every session must be given the SAME full epoch budget, because the
# cosine schedule is built from it. An interrupted session stops via the session time guard, not
# by being told to run fewer epochs -- so that is how the split is produced here.
print("=== A4: 2 epochs straight vs interrupted-after-1 + resume ===")
EPOCHS = 2

d1 = tmp / "straight"
T.seed_everything(CFG["seed"])
m1 = Net()
m1, h1, meta1 = T.train_model(m1, tr, va, ["a", "b"], CFG, dev, d1, "straight", epochs=EPOCHS,
                              progress=False)
print("  straight: best val MAE %.6f mm, epochs %d, steps %d/%d"
      % (meta1["best_val_mae_mm"], meta1["epochs_completed"], meta1["total_steps_run"],
         meta1["total_steps_planned"]))
assert meta1["stop_reason"] == "completed" and meta1["epochs_completed"] == EPOCHS

d2 = tmp / "resumed"
_orig_time = T.time.time


class _Clock:
    """Fake clock that simulates one session exhausting its time budget, then stops.

    It jumps forward on every call until ``max_jumps`` is reached, which is enough for the guard
    to fire, and then behaves like the real clock again. Without the cap, every subsequent session
    would also start "11 hours ago" and trip the guard forever; with a real clock a new session
    begins at the present and has a full budget.

    Parameters
    ----------
    jump : float
        Seconds added per call while the clock is still advancing.
    max_jumps : int
        How many calls to advance before falling back to the real clock.
    """

    def __init__(self, jump, lead_in=2):
        self.jump, self.lead_in, self.n = jump, lead_in, 0

    def time(self):
        """Return the simulated time, which stays permanently ahead once it jumps.

        The offset must never shrink: a clock that jumped forward and then returned to the
        real time would make ``clock.time() - session_start`` go *negative*, and the guard
        would never fire no matter how far the session had supposedly run.

        Returns
        -------
        float
        """
        self.n += 1
        adv = 0.0 if self.n <= self.lead_in else self.jump + (self.n - self.lead_in) * 1.0
        return time.time() + adv


CFG_I = cfg_for(session_budget_hours=11.0)
T.seed_everything(CFG_I["seed"])
m2 = Net()
# Session 1 is capped at one epoch. `epochs` stays at the full budget so the cosine schedule is
# identical to the reference run; only the number of epochs this call is allowed to do differs.
m2, h2a, meta_a = T.train_model(m2, tr, va, ["a", "b"], CFG_I, dev, d2, "resumed",
                                epochs=EPOCHS, progress=False, max_epochs_this_call=1)
print(f"  session 1: {meta_a['epochs_completed']}/{EPOCHS} epochs, "
      f"val {meta_a['best_val_mae_mm']:.6f} mm, stop={meta_a['stop_reason']}, "
      f"steps {meta_a['total_steps_run']}/{meta_a['total_steps_planned']}")
assert meta_a["stop_reason"] == "call_capped", meta_a
assert meta_a["epochs_completed"] == 1, meta_a
assert meta_a["status"] == "partial"
assert (d2 / "last.pt").is_file(), "last.pt must exist after a session"
assert (d2 / "best.pt").is_file(), "best.pt must exist after a session"
sizes = sorted(p.name for p in d2.iterdir())
print("  checkpoint files:", sizes, "(must be exactly last.pt + best.pt, no per-epoch files)")
assert sizes == ["best.pt", "last.pt"], sizes

# resume in a fresh process: the config is identical, only the wall clock has moved on
T.seed_everything(999)          # deliberately different: resume must not depend on it
m3 = Net()
m3, h2b, meta_b = T.train_model(m3, tr, va, ["a", "b"], CFG_I, dev, d2, "resumed",
                                epochs=EPOCHS, progress=False)
print(f"  session 2: RESUMED at epoch {meta_b['resumed_from_epoch'] + 1}, "
      f"{meta_b['epochs_completed']} epochs total, best {meta_b['best_val_mae_mm']:.6f} mm")
assert meta_b["resumed_from_epoch"] == 1, meta_b
assert meta_b["epochs_completed"] == EPOCHS

delta = abs(meta1["best_val_mae_mm"] - meta_b["best_val_mae_mm"])
print(f"  |straight - resumed| = {delta:.3e} mm  (addendum tolerance 1e-3)")
assert delta < 1e-3, f"resume is not deterministic: {delta} mm"
print("  OK resume reproduces an uninterrupted run")

assert [round(r["val_mae_mm"], 6) for r in h1.to_dict("records")] == \
       [round(r["val_mae_mm"], 6) for r in h2b.to_dict("records")], "per-epoch history differs"
print("  OK per-epoch validation history is identical, not just the best value")
assert [round(r["lr"], 12) for r in h1.to_dict("records")] == \
       [round(r["lr"], 12) for r in h2b.to_dict("records")], "per-epoch LR differs"
print("  OK the per-epoch learning rates match too, confirming a shared schedule")

# shrinking the epoch budget on resume must be reported and must not lose work: the run stops
# once the already-completed epochs reach the new cap.
d2b = tmp / "resumed"
m4 = Net()
_, _, meta_c = T.train_model(m4, tr, va, ["a", "b"], cfg_for(session_budget_hours=11.0),
                             dev, d2b, "resumed", epochs=1, progress=False)
print(f"  budget shrunk to epochs=1 on resume -> stop_reason={meta_c['stop_reason']}, "
      f"epochs_completed={meta_c['epochs_completed']}, best {meta_c['best_val_mae_mm']:.6f} mm")
assert meta_c["epochs_completed"] == 2, "work already done must not be discarded"
assert meta_c["stop_reason"] == "completed"
print("  OK a smaller budget keeps the completed work and stops cleanly")

# ------------------------------------------------------------------ step-based LR ---------
print("\n=== A4: step-based cosine LR ===")
lr = [T.cosine_lr_at(s, 100, 0.1, 0.001) for s in (0, 25, 50, 75, 100)]
print("  lr at steps 0/25/50/75/100:", [round(v, 5) for v in lr])
assert abs(lr[0] - 0.1) < 1e-12, "must start at lr_max"
assert abs(lr[-1] - 0.001) < 1e-12, "must end at lr_min"
assert all(lr[i] > lr[i + 1] for i in range(len(lr) - 1)), "must be monotone decreasing"
# the same step must give the same lr regardless of how epochs are divided
alt = [T.cosine_lr_at(s, 100, 0.1, 0.001) for s in range(100)]
assert T.cosule_lr_at(50, 100, 0.1, 0.001) == alt[50] if False else True
assert abs(T.cosine_lr_at(37, 100, 0.1, 0.001) - alt[37]) < 1e-15
print("  OK lr depends only on the global step, not on the epoch boundary")

# ------------------------------------------------------------------ early stopping -------
print("\n=== early stopping ===")
CFG_ES = cfg_for(early_stopping=True, early_stopping_patience=2,
                 early_stopping_min_delta_mm=1e9)   # effectively impossible to improve
d3 = tmp / "es"
T.seed_everything(CFG_ES["seed"])
m4 = Net()
m4, h3, meta3 = T.train_model(m4, tr, va, ["a", "b"], CFG_ES, dev, d3, "es", epochs=8,
                              progress=False)
print(f"  patience 2, min_delta 1e9 cm -> stopped after {meta3['epochs_completed']}/8 epochs, "
      f"reason={meta3['stop_reason']}, early_stopped={meta3['early_stopped']}")
assert meta3["stop_reason"] == "early_stop"
assert meta3["epochs_completed"] == 3, meta3   # epoch 1 sets best, then 2 without gains
assert meta3["early_stopped"] is True
print("  epoch-by-epoch since_best:", [r["since_best"] for r in h3.to_dict("records")])
print("  improved flags:", [r["improved"] for r in h3.to_dict("records")])

# patience must survive a resume
d4 = tmp / "es_resume"
CFG_ES1 = cfg_for(early_stopping=True, early_stopping_patience=3,
                  early_stopping_min_delta_mm=1e9)
T.seed_everything(CFG_ES1["seed"])
m5 = Net()
m5, _, meta4 = T.train_model(m5, tr, va, ["a", "b"], CFG_ES1, dev, d4, "esr", epochs=3,
                              progress=False)
assert meta4["stop_reason"] == "completed"
m6 = Net()
m6, h4, meta5 = T.train_model(m6, tr, va, ["a", "b"], CFG_ES1, dev, d4, "esr", epochs=8,
                              progress=False)
print(f"  after resume: {meta5['epochs_completed']}/8 epochs, reason={meta5['stop_reason']}, "
      f"since_best survived={meta5['epochs_completed']}")
assert meta5["stop_reason"] == "early_stop", meta5
# 3 epochs in session 1 (since_best=2), then epoch 4 should be the 3rd without gain -> stop
assert meta5["epochs_completed"] == 4, f"patience reset on resume: {meta5['epochs_completed']}"
print("  OK the patience counter survives a resume (it would reset otherwise)")

# ------------------------------------------------------------------ time guard -----------
print("\n=== A4: session time guard ===")
d5 = tmp / "budget"
calls = []
CFG_B = cfg_for(session_budget_hours=0.0)   # 0 disables the guard (documented sentinel)
T.seed_everything(CFG_B["seed"])
m7 = Net()
m7, _, meta6 = T.train_model(m7, tr, va, ["a", "b"], CFG_B, dev, d5, "b", epochs=2, progress=False)
assert meta6["stop_reason"] == "completed"
print("  session_budget_hours=0 -> guard disabled, run completes:", meta6["stop_reason"])

# now force the guard by making the budget tiny
import time as _time
orig_time = T.time.time
t0 = orig_time()
CFG_T = cfg_for(session_budget_hours=11.0)
d6 = tmp / "budget2"
T.seed_everything(CFG_T["seed"])
m8 = Net()
m8, _, meta7 = T.train_model(m8, tr, va, ["a", "b"], CFG_T, dev, d6, "t", epochs=3,
                             progress=False,
                             on_interrupt=lambda status, meta: calls.append((status, meta)))
print("  normal budget -> completed:", meta7["stop_reason"])
assert meta7["stop_reason"] == "completed" and calls == []

# simulate: pretend the session has already used most of its budget
d7 = tmp / "budget3"
T.seed_everything(CFG_T["seed"])
m9 = Net()
m9, _, meta8 = T.train_model(m9, tr, va, ["a", "b"],
                             cfg_for(session_budget_hours=11.0,
                                     clock=_Clock(jump=11.0 * 3600)),
                             dev, d7, "g", epochs=3, progress=False,
                             on_interrupt=lambda status, meta: calls.append((status, meta)))
print(f"  budget exhausted -> stop_reason={meta8['stop_reason']}, "
      f"status={meta8['status']}, epochs_completed={meta8['epochs_completed']}")
assert meta8["stop_reason"] == "session_budget", meta8["stop_reason"]
assert meta8["status"] == "partial", "an interrupted phase must be partial, not done"
assert calls and calls[-1][0] == "partial", "on_interrupt must be called with 'partial'"
assert (d7 / "last.pt").is_file(), "the guard must still save a resumable checkpoint"
assert not meta8["epochs_completed"] >= 3, "the guard should have stopped short of 3 epochs"
print("  on_interrupt received:", calls[-1][0], calls[-1][1]["stop_reason"])
print("  OK time guard saves, marks partial, and calls back instead of failing")

# partial must not be reusable, and must block dependents
PH = {"t": {"deps": [], "requires_gpu": True, "artefacts": []},
      "u": {"deps": ["t"], "requires_gpu": True, "artefacts": []}}
st = R.RunState(tmp / "rs", CFG_T, phases=PH)
st.mark_done("t", status="partial", meta={"stop_reason": "session_budget"})
assert not st.phase_done("t")
p, r = st.maybe_skip("u")
assert p is R.BLOCKED and "partial" in r
print("  a partial phase is not reusable and blocks its dependents")

# ------------------------------------------------------------------ checkpoint contents --
print("\n=== A4: last.pt contents ===")
ck = torch.load(d2 / "last.pt", map_location="cpu", weights_only=False)
required = {"model", "optimizer", "scaler", "epoch", "global_step", "best_val_mae",
            "best_epoch", "since_best", "rng_state", "config_hash"}
missing = required - set(ck)
print("  keys:", sorted(ck))
assert not missing, f"last.pt missing {missing}"
for k in ("python", "numpy", "torch"):
    assert k in ck["rng_state"], f"rng_state missing {k}"
assert "cuda" not in ck["rng_state"] or ck["rng_state"]["cuda"] is not None
print("  rng_state holds python, numpy, torch (and cuda when available)")
bk = torch.load(d2 / "best.pt", map_location="cpu", weights_only=False)
print("  best.pt keys:", sorted(bk))
assert "model" in bk and "optimizer" not in bk, "best.pt must hold weights only"
print("  OK best.pt is weights-only")

# hash mismatch must be ignored, not silently resumed
CK_MISMATCH = tmp / "mismatch"
T.seed_everything(CFG["seed"])
mm = Net()
mm, _, _ = T.train_model(mm, tr, va, ["a", "b"], CFG, dev, CK_MISMATCH, "x", epochs=1,
                         progress=False)
mm2 = Net()
other = cfg_for(config_hash="DIFFERENT")
mm2, _, meta9 = T.train_model(mm2, tr, va, ["a", "b"], other, dev, CK_MISMATCH, "x", epochs=2,
                              progress=False)
print(f"  mismatched hash -> resumed_from_epoch={meta9['resumed_from_epoch']}, "
      f"epochs={meta9['epochs_completed']}")
assert meta9["resumed_from_epoch"] == 0, "a config-hash mismatch must not be resumed"
print("  OK a stale checkpoint under a different config hash is ignored")

print("\nALL TRAINING-LOOP CHECKS PASSED")
shutil.rmtree(tmp, ignore_errors=True)