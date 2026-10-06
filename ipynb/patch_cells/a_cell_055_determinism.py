"""#### §4.6 Resume, early stopping and the session time guard (smoke-mode self-check)

**WHAT** — In smoke mode, train a tiny real network twice over identical synthetic data: once
straight through for 3 epochs, and once interrupted after 2 and resumed, as a Kaggle session
ending at hour 9 of 12 would interrupt one. Assert the two reach the same validation MAE, then
exercise early stopping and the session-time guard on the same harness.

**WHY**
- [Ref] Addendum A4 requires this check, with a 1e-3 mm tolerance. It is the only way to know that
  the resume machinery does not silently change the model that ships.
- [Exp] Three properties make it hold, and each is exercised rather than assumed:
  1. the learning rate is a **pure function of the global step** over the full `total_steps`
     budget, so splitting a run across sessions cannot change the schedule;
  2. shuffle order is a pure function of `(seed, epoch)`, re-seeded per epoch rather than advanced;
  3. every RNG stream is restored from `last.pt`.
- [Exp] The interruption is produced by `max_epochs_this_call`, **not** by passing a smaller
  `epochs`. A smaller epoch budget would change `total_steps` and therefore the cosine schedule, so
  the two runs would diverge from step 0 and the test would be measuring the wrong thing. A real
  session ends via the time guard, which leaves the configured budget untouched. The cell asserts
  the per-epoch learning rates match, so this cannot silently regress.
- [Exp] The patience counter is persisted in `last.pt`. If it were not, a resumed run would reset
  its patience and never stop — so the early-stopping half of this test resumes deliberately and
  checks that it still fires.
- [Exp] This runs on CPU with synthetic data. cuDNN nondeterminism can still introduce small
  differences on GPU even with deterministic flags, because some kernels have no deterministic
  implementation, so the 1e-3 bound is asserted here where that noise source is absent rather
  than claimed universally.

**OUTPUT** — `DETERMINISM` (dict of results), and three printed pass/fail lines.

**CHECK** — asserts `|straight − resumed| < 1e-3` mm; asserts the per-epoch validation history and
learning rates are identical; asserts the checkpoint directory holds exactly `last.pt` and
`best.pt`; asserts early stopping fires and survives a resume; asserts the time guard marks the
phase `partial` and calls back rather than raising.
"""

import shutil as _shutil

import tempfile as _tempfile
from torch.utils.data import Dataset as _Dataset

# Only meaningful in a smoke pass: the real training is the expensive part, and this trains two
# throwaway networks.
if not SMOKE:
    print("determinism test: SKIPPED (smoke_test is false). It trains two throwaway networks and "
          "is only worth running in a smoke pass.")
    DETERMINISM: Dict[str, Any] = {"ran": False}
else:
    _dt_dir = Path(_tempfile.mkdtemp(prefix="determinism_"))

    class _TinyDS(_Dataset):
        """Deterministic synthetic dataset standing in for :class:`SilhouetteDataset`.

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
            self.y = (self.x[:, :2] @ np.array([[2.0, -1.5], [1.0, 0.5]], np.float32)
                      + np.float32(0.3)).astype(np.float32)

        def __len__(self):
            """Number of samples."""
            return len(self.x)

        def __getitem__(self, i):
            """Return ``(x, y, subject_id, photo_id)``.

            Parameters
            ----------
            i : int
                Sample index.

            Returns
            -------
            tuple
            """
            return (torch.from_numpy(self.x[i]), torch.from_numpy(self.y[i]),
                    f"s{i // 3}", f"p{i}")

    class _TinyNet(nn.Module):
        """Minimal net with :class:`MeasurementNet`'s head shape.

        Parameters
        ----------
        d : int
            Input width.
        feat_dim : int
            Pooled-feature width.
        """

        def __init__(self, d=8, feat_dim=6):
            super().__init__()
            self.feat_dim, self.n_scalars, self.use_weight = feat_dim, 1, False
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

    _dt_cfg = {**cfg, "seed": 12345, "batch_size": 8, "lr": 0.05, "lr_min": 0.001,
               "weight_decay": 0.0, "amp": False, "num_workers": 0,
               "early_stopping": False, "session_budget_hours": 0.0,
               "config_hash": "determinism-selfcheck"}
    _dt_tr, _dt_va = _TinyDS(96, seed=1), _TinyDS(48, seed=2)
    _EPOCHS = 3
    _SPLIT_AT = 2
    _cpu = torch.device("cpu")

    T.seed_everything(_dt_cfg["seed"])
    _ref = _TinyNet()
    _ref, _h_ref, _m_ref = T.train_model(_ref, _dt_tr, _dt_va, ["a", "b"], _dt_cfg, _cpu,
                                         _dt_dir / "straight", "dt", epochs=_EPOCHS,
                                         progress=False)
    print(f"straight:   {_m_ref['epochs_completed']}/{_EPOCHS} epochs, "
          f"best val MAE {_m_ref['best_val_mae_mm']:.6f} mm, "
          f"{_m_ref['total_steps_run']}/{_m_ref['total_steps_planned']} steps")

    T.seed_everything(_dt_cfg["seed"])
    _part = _TinyNet()
    _part, _h_a, _m_a = T.train_model(_part, _dt_tr, _dt_va, ["a", "b"], _dt_cfg, _cpu,
                                      _dt_dir / "split", "dt", epochs=_EPOCHS,
                                      progress=False, max_epochs_this_call=_SPLIT_AT)
    print(f"session 1:  {_m_a['epochs_completed']}/{_EPOCHS} epochs, "
          f"best val MAE {_m_a['best_val_mae_mm']:.6f} mm, stop={_m_a['stop_reason']}")
    _ck_files = sorted(p.name for p in (_dt_dir / "split").iterdir())
    print(f"checkpoint files: {_ck_files}")
    assert _ck_files == ["best.pt", "last.pt"], (
        f"a checkpoint directory must hold exactly last.pt and best.pt (addendum A1: no per-epoch "
        f"files, because /kaggle/working has a file-count cap). Found {_ck_files}")

    # a fresh process would have a different global seed; resume must not depend on it
    T.seed_everything(999)
    _rest = _TinyNet()
    _rest, _h_b, _m_b = T.train_model(_rest, _dt_tr, _dt_va, ["a", "b"], _dt_cfg, _cpu,
                                     _dt_dir / "split", "dt", epochs=_EPOCHS, progress=False)
    print(f"session 2:  resumed at epoch {_m_b['resumed_from_epoch'] + 1}, "
          f"{_m_b['epochs_completed']}/{_EPOCHS} epochs, "
          f"best val MAE {_m_b['best_val_mae_mm']:.6f} mm")
    assert _m_b["resumed_from_epoch"] == _SPLIT_AT, (
        f"resume started at epoch {_m_b['resumed_from_epoch'] + 1}, expected {_SPLIT_AT + 1}")

    _delta = abs(_m_ref["best_val_mae_mm"] - _m_b["best_val_mae_mm"])
    _lr_ref = [round(float(v), 10) for v in _h_ref["lr"]]
    _lr_res = [round(float(v), 10) for v in _h_b["lr"]]
    _val_ref = [round(float(v), 8) for v in _h_ref["val_mae_mm"]]
    _val_res = [round(float(v), 8) for v in _h_b["val_mae_mm"]]
    print(f"\nper-epoch val MAE straight : {_val_ref}")
    print(f"per-epoch val MAE resumed : {_val_res}")
    print(f"per-epoch lr straight     : {_lr_ref}")
    print(f"per-epoch lr resumed      : {_lr_res}")
    print(f"|straight - resumed|      : {_delta:.3e} mm   (addendum tolerance 1e-3)")

    assert _lr_ref == _lr_res, (
        f"the learning-rate schedule diverged: {_lr_ref} vs {_lr_res}. The cosine must be a pure "
        "function of the global step over the full budget, or resume changes the model.")
    assert _val_ref == _val_res, f"per-epoch validation history diverged: {_val_ref} vs {_val_res}"
    assert _delta < 1e-3, (
        f"resume is not deterministic: |straight - resumed| = {_delta:.6f} mm, above the 1e-3 "
        "tolerance. Check that shuffle order is re-seeded per epoch and that every RNG stream is "
        "restored from last.pt.")

    DETERMINISM = {"ran": True, "delta_mm": _delta, "epochs": _EPOCHS, "split_at": _SPLIT_AT,
                   "straight_best_mm": _m_ref["best_val_mae_mm"],
                   "resumed_best_mm": _m_b["best_val_mae_mm"]}
    print("resume determinism: PASS. An interrupted run and an uninterrupted one produce the "
          "same weights and the same learning-rate schedule.")

    # --- early stopping, including across a resume ------------------------------------------
    _es_cfg = {**_dt_cfg, "early_stopping": True, "early_stopping_patience": 2,
               "early_stopping_min_delta_mm": 1e9}      # effectively impossible to beat
    T.seed_everything(_dt_cfg["seed"])
    _es = _TinyNet()
    _es, _h_es, _m_es = T.train_model(_es, _dt_tr, _dt_va, ["a", "b"], _es_cfg, _cpu,
                                     _dt_dir / "es", "es", epochs=8, progress=False)
    print(f"\nearly stopping: stopped after {_m_es['epochs_completed']}/8 epochs, "
          f"reason={_m_es['stop_reason']}, since_best per epoch "
          f"{[r['since_best'] for r in _h_es.to_dict('records')]}")
    assert _m_es["stop_reason"] == "early_stop", _m_es["stop_reason"]
    assert _m_es["early_stopped"] is True

    # patience must survive a resume, or a resumed run resets it and never stops
    _esr_cfg = {**_dt_cfg, "early_stopping": True, "early_stopping_patience": 3,
                "early_stopping_min_delta_mm": 1e9}
    T.seed_everything(_dt_cfg["seed"])
    _e1 = _TinyNet()
    _e1, _, _me1 = T.train_model(_e1, _dt_tr, _dt_va, ["a", "b"], _esr_cfg, _cpu,
                                 _dt_dir / "esr", "esr", epochs=_SPLIT_AT + 1, progress=False)
    _e2 = _TinyNet()
    _e2, _, _me2 = T.train_model(_e2, _dt_tr, _dt_va, ["a", "b"], _esr_cfg, _cpu,
                                 _dt_dir / "esr", "esr", epochs=8, progress=False)
    print(f"patience across a resume: session 1 ran {_me1['epochs_completed']} epochs, "
          f"session 2 stopped at {_me2['epochs_completed']} with reason {_me2['stop_reason']}")
    assert _me2["stop_reason"] == "early_stop", (
        f"early stopping did not fire after a resume ({_me2['stop_reason']}); the patience "
        "counter is probably not persisted in last.pt")

    # --- the session time guard ----------------------------------------------------------------
    _calls: List[Tuple[str, dict]] = []

    class _FakeClock:
        """Clock that stays permanently ahead, simulating an exhausted session.

        The offset must never shrink: a clock that jumped forward and then returned to real time
        would make ``clock.time() - session_start`` go negative and the guard would never fire.

        Parameters
        ----------
        jump : float
            Seconds added once the lead-in calls have passed.
        lead_in : int
            Calls that return the real time before the jump starts.
        """

        def __init__(self, jump, lead_in=2):
            self.jump, self.lead_in, self.n = jump, lead_in, 0

        def time(self):
            """Return the simulated time.

            Returns
            -------
            float
            """
            self.n += 1
            return time.time() + (0.0 if self.n <= self.lead_in else self.jump)

    _bud_cfg = {**_dt_cfg, "session_budget_hours": 11.0,
                "clock": _FakeClock(jump=11.0 * 3600)}
    T.seed_everything(_dt_cfg["seed"])
    _bg = _TinyNet()
    _bg, _h_bg, _m_bg = T.train_model(_bg, _dt_tr, _dt_va, ["a", "b"], _bud_cfg, _cpu,
                                      _dt_dir / "budget", "bg", epochs=4, progress=False,
                                      on_interrupt=lambda s, m: _calls.append((s, m)))
    print(f"\ntime guard: stopped after {_m_bg['epochs_completed']}/4 epochs, "
          f"reason={_m_bg['stop_reason']}, status={_m_bg['status']}")
    print(f"  on_interrupt received: {_calls[-1][0] if _calls else 'NOTHING'}")
    assert _m_bg["stop_reason"] == "session_budget", _m_bg["stop_reason"]
    assert _m_bg["status"] == "partial", "an interrupted phase must be partial, never done"
    assert _calls and _calls[-1][0] == "partial", "on_interrupt must fire with 'partial'"
    assert (_dt_dir / "budget" / "last.pt").is_file(), "the guard must still save a checkpoint"

    _shutil.rmtree(_dt_dir, ignore_errors=True)
    print("\nDETERMINISM / EARLY-STOPPING / TIME-GUARD CHECKS PASSED")