"""Training loop shared by the B2 baseline and the production variants.

Resumable by design (addendum A4). Every epoch can be interrupted without losing more than one
epoch of work, and a resumed run reproduces an uninterrupted one.

Resume determinism
------------------
Three properties make ``1 epoch + resume + 1 epoch`` numerically equivalent to ``2 epochs``:

1. **The learning rate is a pure function of the global step.** ``lr_t`` follows a cosine over
   ``total_steps = epochs * steps_per_epoch``, not over epochs. A scheduler object keyed on epochs
   cannot do this: a resumed run has a different epoch count and therefore a different schedule.
2. **Shuffle order is a pure function of the epoch.** The ``DataLoader`` generator is re-seeded
   from ``(seed, epoch)`` every epoch instead of being advanced.
3. **All RNG streams are restored from ``last.pt``**, and the phase re-seeds canonically *after*
   model construction, so the random draws consumed while building the network cannot shift the
   stream relative to a fresh process.

Augmentation already follows (2) via ``WorkerRNG`` in :mod:`bodym_perturb`.

Early stopping
--------------
Improvement is ``best_val - val > early_stopping_min_delta_mm``. After
``early_stopping_patience`` epochs without improvement, training stops. The counter is persisted in
``last.pt``: without that, a resumed run would silently reset its patience and never stop.

Time budget
-----------
Before each epoch the loop estimates whether the next one fits inside
``session_budget_hours``. If it does not, it saves, marks the phase ``partial`` and returns
normally, so the remaining phases skip with a printed reason rather than the run dying.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:                                   # atomic_save lives in run_state; keep the fallback so this
    from run_state import atomic_save  # module stays importable on its own
except ImportError:                    # pragma: no cover
    atomic_save = None


# =========================================================================================
# Learning rate and seeding
# =========================================================================================
def cosine_lr_at(step: int, total_steps: int, lr_max: float, lr_min: float) -> float:
    """Learning rate at a global step, on a cosine schedule.

    Parameters
    ----------
    step : int
        Global optimiser step, 0-based.
    total_steps : int
        Steps the whole run is budgeted for.
    lr_max : float
        Peak learning rate.
    lr_min : float
        Final learning rate.

    Returns
    -------
    float
        ``lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * step / total_steps))``.
    """
    if total_steps <= 0:
        return lr_max
    frac = min(max(step / float(total_steps), 0.0), 1.0)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * frac))


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch, and request deterministic cuDNN kernels.

    Parameters
    ----------
    seed : int
        Master seed for this training phase.
    """
    os_env = __import__("os").environ
    os_env["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception as exc:               # pragma: no cover - platform dependent
        print("cudnn determinism unavailable:", exc)


def epoch_generator(seed: int, epoch: int) -> torch.Generator:
    """A ``DataLoader`` generator whose shuffle order depends only on (seed, epoch).

    Parameters
    ----------
    seed : int
        Master seed.
    epoch : int
        1-based epoch index.

    Returns
    -------
    torch.Generator
        Seeded for this epoch alone.
    """
    g = torch.Generator()
    g.manual_seed((int(seed) * 1_000_003 + int(epoch)) % (2 ** 63 - 1))
    return g


def set_worker_seed(worker_id: int) -> None:
    """Seed a DataLoader worker deterministically.

    Parameters
    ----------
    worker_id : int
        Worker index assigned by PyTorch.
    """
    seed = (torch.initial_seed() + worker_id) % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def rng_state_snapshot() -> Dict[str, object]:
    """Capture every RNG stream that can affect training.

    Returns
    -------
    dict
        Python, NumPy, torch and (when present) all CUDA RNG states.
    """
    state: Dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    try:
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
    except Exception:                      # pragma: no cover
        pass
    return state


def rng_state_restore(state: Dict[str, object]) -> None:
    """Restore RNG streams captured by :func:`rng_state_snapshot`.

    Parameters
    ----------
    state : dict
        A snapshot. Missing keys are skipped, so an older checkpoint still loads.
    """
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except Exception:                  # pragma: no cover
            pass


# =========================================================================================
# Checkpointing
# =========================================================================================
def _save(obj, path: Path) -> Path:
    """Save atomically when :mod:`run_state` is available, otherwise directly.

    Parameters
    ----------
    obj : object
        Payload.
    path : Path
        Destination.

    Returns
    -------
    Path
        The written path.
    """
    if atomic_save is not None:
        return atomic_save(obj, path, "torch")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)
    return path


# =========================================================================================
# Inference helpers
# =========================================================================================
@torch.no_grad()
def predict_split(model: nn.Module, loader: DataLoader, device: torch.device
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference over a loader, **one row per photo pair** (fix B5).

    Parameters
    ----------
    model : nn.Module
        Trained network.
    loader : DataLoader
        Loader over a :class:`SilhouetteDataset`.
    device : torch.device
        Compute device.

    Returns
    -------
    (pred, true, subjects) : tuple of ndarray
        ``pred`` and ``true`` of shape ``(n_photo_pairs, n_targets)`` in cm, row-aligned with the
        loader's sample order, and the per-row subject ids.
    """
    model.eval()
    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []
    subs: List[str] = []
    for x, y, sid, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            p = model(x).float().cpu().numpy()
        preds.append(p)
        trues.append(np.asarray(y, dtype=float))
        subs.extend(list(sid))
    pred = np.concatenate(preds, axis=0) if preds else np.zeros((0, 0))
    true = np.concatenate(trues, axis=0) if trues else np.zeros((0, 0))
    return pred, true, np.asarray(subs, dtype=object)


def aggregate_by_subject(pred: np.ndarray, true: np.ndarray, subjects: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average per-photo-pair predictions to one row per subject.

    Used only for the clearly-labelled *secondary* subject-averaged tables.

    Parameters
    ----------
    pred, true : ndarray, shape (n_pairs, m)
        Per-pair predictions and ground truth, cm.
    subjects : ndarray, shape (n_pairs,)
        Subject id per row.

    Returns
    -------
    (pred, true, subjects) : tuple of ndarray
        Averaged predictions and truth of shape ``(n_subjects, m)``, and the ordered subject ids.
    """
    subs = np.asarray([str(v) for v in subjects])
    order = sorted(set(subs))
    out_p, out_t = [], []
    for s in order:
        idx = np.flatnonzero(subs == s)
        out_p.append(pred[idx].mean(axis=0))
        out_t.append(true[idx].mean(axis=0))
    return np.stack(out_p), np.stack(out_t), np.asarray(order, dtype=object)


# =========================================================================================
# Training
# =========================================================================================
def train_model(
    model: nn.Module,
    train_ds,
    val_ds,
    targets: Sequence[str],
    cfg: Dict,
    device: torch.device,
    ckpt_dir: Path,
    tag: str,
    epochs: int,
    worker_rng=None,
    on_interrupt=None,
    progress: bool = True,
    max_epochs_this_call: Optional[int] = None,
) -> Tuple[nn.Module, pd.DataFrame, Dict[str, object]]:
    """Train one model with L1 loss, resuming if a matching ``last.pt`` exists.

    Parameters
    ----------
    model : nn.Module
        Network to train. Weights are overwritten in place.
    train_ds, val_ds : SilhouetteDataset
        Training and validation datasets. Checkpoint selection uses **validation** data only,
        measured per photo pair.
    targets : sequence of str
        Target names, length ``model.n_outputs``.
    cfg : dict
        Configuration. Keys used: ``seed``, ``batch_size``, ``lr``, ``lr_min``, ``weight_decay``,
        ``grad_clip``, ``amp``, ``num_workers``, ``ckpt_minutes``, ``session_budget_hours``,
        ``early_stopping``, ``early_stopping_patience``, ``early_stopping_min_delta_mm``,
        ``config_hash``.
    device : torch.device
        Compute device.
    ckpt_dir : Path
        Directory for ``last.pt`` and ``best.pt``. Never accumulates per-epoch files (addendum
        A1: the working directory has a file-count cap).
    tag : str
        Variant tag, e.g. ``'vhw'``.
    epochs : int
        Total epochs for the whole run, including any already completed before a resume.
    worker_rng : WorkerRNG, optional
        Augmentation RNG handle; advanced once per epoch so streams differ across epochs.
    on_interrupt : callable, optional
        Called as ``on_interrupt(status, meta)`` when the loop stops early, so the caller can mark
        the phase ``partial`` in the manifest.
    progress : bool, default True
        Print per-epoch lines.
    max_epochs_this_call : int, optional
        Cap on epochs to run in *this* invocation, independent of the time guard. Used by the
        determinism test to interrupt at a chosen epoch while keeping the cosine schedule built
        from the full ``epochs`` budget. Leave unset in normal use: an interrupted session stops
        via the time guard, not by being told to run fewer epochs.

    Returns
    -------
    (model, history, meta) : tuple
        The model with the best validation-MAE weights loaded, the per-epoch history (including
        the epochs completed in earlier sessions), and a metadata dict with ``status``,
        ``epochs_completed``, ``epochs_configured``, ``best_epoch``, ``best_val_mae_mm``,
        ``stop_reason`` and ``resumed_from_epoch``.

    Raises
    ------
    RuntimeError
        If the first epoch's mean loss is not finite.
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path = ckpt_dir / "last.pt"
    best_path = ckpt_dir / "best.pt"

    seed = int(cfg["seed"])
    batch_size = int(cfg["batch_size"])
    lr_max = float(cfg["lr"])
    lr_min = float(cfg.get("lr_min", lr_max * 0.02))
    ckpt_minutes = float(cfg.get("ckpt_minutes", 20)) * 60.0
    budget_hours = cfg.get("session_budget_hours")
    budget_s = float(budget_hours) * 3600.0 if budget_hours else 0.0
    # A caller may inject a clock module, which is how the smoke-mode determinism test simulates
    # a session running out of time without waiting eleven hours for it.
    clock = cfg.get("clock") or time
    # Optional fixed per-epoch estimate, in seconds. The guard prefers a measurement taken in the
    # *current* session; `epoch_time_estimate_s` overrides that entirely and exists so a test can
    # supply a plausible epoch duration while the injected clock makes the budget look exhausted.
    forced_epoch_s = cfg.get("epoch_time_estimate_s")
    es_on = bool(cfg.get("early_stopping", True))
    es_patience = int(cfg.get("early_stopping_patience", 3))
    es_min_delta = float(cfg.get("early_stopping_min_delta_mm", 1.0)) / 10.0   # mm -> cm
    cfg_hash = cfg.get("config_hash")

    n_train = len(train_ds)
    steps_per_epoch = max(n_train // max(batch_size, 1), 1)
    total_steps_planned = steps_per_epoch * epochs
    total_steps = total_steps_planned
    amp_on = bool(cfg.get("amp", True) and device.type == "cuda")

    # ---- resume ----------------------------------------------------------------------------
    start_epoch = 1
    best_mae = float("inf")
    best_epoch = -1
    global_step = 0
    since_best = 0
    history: List[dict] = []
    epoch_times: List[float] = []
    resumed_from = 0
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on)
    loss_fn = nn.L1Loss()

    if last_path.is_file():
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        # A hash mismatch means the configuration changed, so the cached weights and optimiser
        # state describe a different model. `force_rerun` is the escape hatch for "same config,
        # I just want to start over".
        forced = tag in set(cfg.get("force_rerun", []) or [])
        if ck.get("config_hash") not in (None, cfg_hash) and not forced:
            print(f"  ({tag}) existing checkpoint was written with config_hash "
                  f"{ck.get('config_hash')} but this run has {cfg_hash}. Starting from scratch; "
                  f"add '{tag}' to config.yaml force_rerun to reuse it anyway.")
            ck = None
        if ck is not None:
            model.load_state_dict(ck["model"])
            # Re-seed before restoring the optimiser: `Optimizer.load_state_dict` re-creates
            # the parameter state dict and, for some optimisers, draws fresh tensors. Restoring
            # the RNG afterwards guarantees the streams match an uninterrupted run regardless
            # of what the loader consumed internally.
            opt.load_state_dict(ck["optimizer"])
            if ck.get("scaler") is not None and amp_on:
                scaler.load_state_dict(ck["scaler"])
            start_epoch = int(ck.get("epoch", 0)) + 1
            best_mae = float(ck.get("best_val_mae", float("inf")))
            best_epoch = int(ck.get("best_epoch", -1))
            global_step = int(ck.get("global_step", 0))
            since_best = int(ck.get("since_best", 0))
            history = list(ck.get("history", []))
            epoch_times = list(ck.get("epoch_times", []))
            rng_state_restore(ck.get("rng_state", {}))
            resumed_from = start_epoch - 1
            # The cosine schedule is a pure function of the global step over `total_steps`, so
            # the budget that was in force when those steps were taken must be reused verbatim.
            # Recomputing it from the *current* epochs would silently change the LR for every
            # remaining step and break resume equivalence - which is exactly the property A4
            # asks us to test.
            ck_total = ck.get("total_steps")
            if isinstance(ck_total, int) and ck_total > 0:
                if ck_total != total_steps_planned:
                    print(f"  note: config now budgets {total_steps_planned} total steps but this "
                          f"checkpoint was trained against {ck_total}. Reusing {ck_total} so the "
                          "learning-rate schedule stays continuous; the epoch budget is therefore "
                          "an upper bound on this run, not a change of schedule.")
                total_steps = ck_total
            print(f"RESUMED {tag}: resuming at epoch {start_epoch}/{epochs} "
                  f"(best val MAE {best_mae*10:.2f} mm from epoch {best_epoch}, "
                  f"since_best={since_best}, step {global_step}/{total_steps})")

    # best.pt is the weights we report, so it must agree with the checkpoint we just loaded.
    # Loading it unconditionally would silently overwrite resumed weights with the previous
    # session's best, which is only correct when nothing was resumed.
    if best_path.is_file() and resumed_from == 0:
        try:
            bk = torch.load(best_path, map_location="cpu", weights_only=False)
            if bk.get("config_hash") in (None, cfg_hash):
                model.load_state_dict(bk["model"])
        except Exception as exc:            # pragma: no cover
            print(f"  ({tag}) best.pt unreadable ({exc}); will re-select during training")

    if worker_rng is not None:
        worker_rng.set_epoch(max(start_epoch - 1, 0))

    def save_last(epoch: int) -> None:
        """Persist everything needed to resume.

        Parameters
        ----------
        epoch : int
            Epoch just completed.
        """
        _save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "optimizer": opt.state_dict(),
               "scaler": scaler.state_dict() if amp_on else None,
               "epoch": epoch,
               "global_step": global_step,
               "best_val_mae": best_mae,
               "best_epoch": best_epoch,
               "since_best": since_best,
               "epoch_times": epoch_times,
               "history": history,
               "targets": list(targets),
               "use_weight": getattr(model, "use_weight", True),
               "rng_state": rng_state_snapshot(),
               "config_hash": cfg_hash,
               "total_steps": total_steps,
               "tag": tag},
              last_path)

    stop_reason = "completed"
    session_start = clock.time()
    # Cap this call at `start_epoch - 1 + max_epochs_this_call`, but never past the full budget.
    last_epoch = epochs if max_epochs_this_call is None else min(
        epochs, start_epoch - 1 + int(max_epochs_this_call))

    for epoch in range(start_epoch, last_epoch + 1):
        # ---- time guard (A4) -------------------------------------------------------------
        # Only guard the *current* session's remaining work. A resumed run starts a fresh
        # session with a fresh budget, and `epoch_times` holds durations from the previous one,
        # so the elapsed term must not include anything from before this process began.
        if budget_s > 0 and (epoch_times or forced_epoch_s):
            mean_epoch = (float(forced_epoch_s) if forced_epoch_s
                          else float(np.mean(epoch_times[-3:])))
            spent_this_session = clock.time() - session_start
            if spent_this_session + mean_epoch * 1.15 > budget_s:
                save_last(epoch - 1)
                print(f"\n{tag}: session budget of {cfg.get('session_budget_hours')} h reached "
                      f"after epoch {epoch-1}/{epochs} (mean epoch {mean_epoch/60:.1f} min).")
                print("  Saved run/ckpt/" + tag + "/last.pt. Start a new session and run the "
                      "notebook again; this phase resumes at the next epoch and every completed "
                      "phase is skipped.")
                stop_reason = "session_budget"
                _partial = {"stop_reason": stop_reason,
                            "epochs_completed": epoch - 1,
                            "epochs_configured": epochs,
                            "epochs_remaining": epochs - (epoch - 1),
                            "best_epoch": best_epoch,
                            "best_val_mae_mm": best_mae * 10.0 if np.isfinite(best_mae) else None,
                            "next_epoch": epoch,
                            "ckpt_dir": str(ckpt_dir)}
                if on_interrupt is not None:
                    on_interrupt("partial", _partial)
                break

        # Re-seed the loader generator per epoch so shuffle order is a pure function of epoch
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda"),
                              drop_last=True, worker_init_fn=set_worker_seed,
                              generator=epoch_generator(seed, epoch),
                              persistent_workers=False)
        if worker_rng is not None:
            worker_rng.set_epoch(epoch)

        model.train()
        t0 = clock.time()
        running, n_seen = 0.0, 0
        last_ckpt = clock.time()
        for x, y, _, _ in tqdm(train_dl, desc=f"{tag} ep{epoch}", leave=False, disable=not progress):
            lr = cosine_lr_at(global_step, total_steps, lr_max, lr_min)
            for g in opt.param_groups:
                g["lr"] = lr
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_on):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            running += float(loss.item()) * len(y)
            n_seen += len(y)
            global_step += 1

            # fix B1: the scalars reach the head's first Linear directly, so after the first
            # optimiser step the weights on those input columns must be non-zero. If they are
            # all still zero the injection is dead and every later metric is silhouette-only.
            if global_step == 1 and hasattr(model, "feat_dim"):
                with torch.no_grad():
                    tail = float(model.head[0].weight[:, model.feat_dim:].abs().max())
                print(f"  [{tag}] head weights on scalar input columns after step 1: {tail:.3e}")
                assert tail > 0.0, (
                    f"{tag}: the head's scalar-input weights are still all zero after one "
                    "optimiser step - the scalar path is dead (fix B1 regression)")

            # periodic save inside the epoch, so a long epoch cannot lose the whole run (A4)
            if ckpt_minutes > 0 and (clock.time() - last_ckpt) > ckpt_minutes:
                save_last(epoch - 1)
                last_ckpt = clock.time()
                print(f"  [{tag}] checkpointed mid-epoch at step {global_step}")

        if epoch == 1 and resumed_from == 0 and not np.isfinite(running / max(n_seen, 1)):
            raise RuntimeError(f"{tag}: first-epoch loss is not finite ({running})")

        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda"))
        pred, true, _ = predict_split(model, val_dl, device)
        val_mae_cm = float(np.mean(np.abs(pred - true)))
        improved = val_mae_cm < best_mae - es_min_delta
        if improved:
            best_mae, best_epoch, since_best = val_mae_cm, epoch, 0
            _save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                   "epoch": epoch, "val_mae_mm": val_mae_cm * 10.0,
                   "config_hash": cfg_hash, "targets": list(targets)}, best_path)
        else:
            since_best += 1

        took = clock.time() - t0
        epoch_times.append(took)
        history.append({"epoch": epoch, "train_loss_cm": running / max(n_seen, 1),
                        "val_mae_cm": val_mae_cm, "val_mae_mm": val_mae_cm * 10.0,
                        "n_val_pairs": int(len(pred)), "lr": lr,
                        "improved": bool(improved), "since_best": int(since_best),
                        "seconds": round(took, 1)})
        save_last(epoch)
        last_ckpt = clock.time()

        if progress:
            print(f"  {tag} epoch {epoch:02d}/{epochs}: train {running/max(n_seen,1):.4f} cm | "
                  f"val MAE {val_mae_cm*10:.2f} mm ({len(pred)} pairs) | lr {lr:.2e} | "
                  f"{took:.0f}s" + ("  <- best" if improved else
                                     f"  (no gain x{since_best})"))

        # ---- early stopping ---------------------------------------------------------------
        # a caller-imposed cap is an interruption, not a completion
        if max_epochs_this_call is not None and epoch >= last_epoch and epoch < epochs:
            save_last(epoch)
            stop_reason = "call_capped"
            _partial = {"stop_reason": stop_reason, "epochs_completed": epoch,
                        "epochs_configured": epochs,
                        "epochs_remaining": epochs - epoch,
                        "best_epoch": best_epoch,
                        "best_val_mae_mm": best_mae * 10.0 if np.isfinite(best_mae) else None,
                        "next_epoch": epoch + 1, "ckpt_dir": str(ckpt_dir)}
            if on_interrupt is not None:
                on_interrupt("partial", _partial)
            break

        if es_on and since_best >= es_patience:
            stop_reason = "early_stop"
            print(f"{tag}: early stopping after epoch {epoch} - no val MAE improvement of at "
                  f"least {cfg.get('early_stopping_min_delta_mm')} mm for {es_patience} epochs "
                  f"(best {best_mae*10:.2f} mm at epoch {best_epoch})")
            if on_interrupt is not None:
                on_interrupt("done", {"stop_reason": stop_reason,
                                      "epochs_completed": epoch,
                                      "epochs_configured": epochs})
            break

    if best_path.is_file():
        bk = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(bk["model"])
    model.to(device)

    meta = {
        "status": "partial" if stop_reason in ("session_budget", "call_capped") else "done",
        "stop_reason": stop_reason,
        "epochs_completed": len(history),
        "epochs_configured": epochs,
        "early_stopped": bool(stop_reason == "early_stop"),
        "resumed_from_epoch": resumed_from,
        "best_epoch": best_epoch,
        "best_val_mae_mm": best_mae * 10.0 if np.isfinite(best_mae) else None,
        "total_steps_planned": total_steps,
        "total_steps_run": global_step,
        "steps_per_epoch": steps_per_epoch,
        "ckpt_dir": str(ckpt_dir),
        "n_train_samples": n_train,
    }
    print(f"{tag}: {'best epoch ' + str(best_epoch) if best_epoch > 0 else 'no epoch completed'}, "
          f"val MAE {best_mae*10:.2f} mm after {len(history)}/{epochs} epochs "
          f"(stop_reason={stop_reason}) -> {ckpt_dir}")
    return model, pd.DataFrame(history), meta