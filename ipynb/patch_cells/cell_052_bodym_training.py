"""Training loop shared by the B2 baseline and the production variants."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm.auto import tqdm          # fix A3: used in the epoch loop below


def set_worker_seed(worker_id: int) -> None:
    """Seed a DataLoader worker deterministically.

    Parameters
    ----------
    worker_id : int
        Worker index assigned by PyTorch.
    """
    import random

    seed = (torch.initial_seed() + worker_id) % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


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
) -> Tuple[nn.Module, pd.DataFrame]:
    """Train one model with L1 loss and checkpoint every epoch.

    Parameters
    ----------
    model : nn.Module
        Network to train.
    train_ds, val_ds : SilhouetteDataset
        Training and validation datasets. Their ``subject_id`` arrays define the split; the
        checkpoint selection uses **validation** data only.
    targets : sequence of str
        Target names, length ``model.n_outputs``.
    cfg : dict
        Notebook configuration (``batch_size``, ``lr``, ``weight_decay``, ``grad_clip``, ``amp``,
        ``num_workers``, ``seed``).
    device : torch.device
        Compute device.
    ckpt_dir : Path
        Directory for per-epoch checkpoints.
    tag : str
        Prefix for checkpoint filenames, e.g. ``'b2'`` or ``'vhw'``.
    epochs : int
        Number of epochs.

    Returns
    -------
    (model, history) : tuple
        The model with the best validation-MAE weights loaded, and a per-epoch history.
        Validation MAE is the **per-photo-pair** mean absolute error (fix B5): deployment
        predicts from one silhouette pair, so checkpoint selection must not reward a model
        that happens to be good on the *average* of a subject's photos.

    Raises
    ------
    RuntimeError
        If the first epoch's mean loss is not finite.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    g = torch.Generator()
    g.manual_seed(cfg["seed"])

    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                          num_workers=cfg["num_workers"], pin_memory=True, drop_last=True,
                          worker_init_fn=set_worker_seed, generator=g,
                          persistent_workers=cfg["num_workers"] > 0)
    val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=cfg["num_workers"], pin_memory=True)

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["amp"] and device.type == "cuda"))
    loss_fn = nn.L1Loss()

    history: List[dict] = []
    best_mae = float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running, n_seen = 0.0, 0
        for x, y, _, _ in tqdm(train_dl, desc=f"{tag} ep{epoch}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(cfg["amp"] and device.type == "cuda")):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            running += float(loss.item()) * len(y)
            n_seen += len(y)

        if epoch == 1 and not np.isfinite(running / max(n_seen, 1)):
            raise RuntimeError(f"{tag}: first-epoch loss is not finite ({running})")

        # fix B1: the scalars now reach the head's first Linear directly, so after the first
        # optimiser step the weights on those input columns must be non-zero. If they are all
        # still zero the injection is dead again and every later metric is silhouette-only.
        if epoch == 1:
            with torch.no_grad():
                w = model.head[0].weight
                col0 = int(model.feat_dim)                  # first scalar column
                tail = w[:, col0:].abs().max().item()
            print(f"  [{tag}] head weights on scalar input columns after step 1: {tail:.3e}")
            assert tail > 0.0, (
                f"{tag}: the head's scalar-input weights are still all zero after one optimiser "
                "step - the scalar path is dead (fix B1 regression)")
        sched.step()

        pred, true, _ = predict_split(model, val_dl, device)
        val_mae_cm = float(np.mean(np.abs(pred - true)))
        history.append({"epoch": epoch, "train_loss_cm": running / max(n_seen, 1),
                        "val_mae_cm": val_mae_cm, "val_mae_mm": val_mae_cm * 10.0,
                        "n_val_pairs": int(len(pred)),
                        "seconds": round(time.time() - t0, 1)})

        ckpt = ckpt_dir / f"{tag}_epoch{epoch:02d}.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch, "targets": list(targets),
                    "use_weight": getattr(model, "use_weight", True),
                    "img_height": getattr(train_ds, "img_height", None),
                    "img_width": getattr(train_ds, "img_width", None),
                    "val_mae_mm": val_mae_cm * 10.0}, ckpt)

        if val_mae_cm < best_mae:
            best_mae, best_epoch = val_mae_cm, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"  {tag} epoch {epoch:02d}: train {running/max(n_seen,1):.4f} cm | "
              f"val MAE {val_mae_cm*10:.2f} mm ({len(pred)} pairs) | {time.time()-t0:.0f}s"
              + ("  <- best" if epoch == best_epoch else ""))

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"{tag}: best epoch {best_epoch}, val MAE {best_mae*10:.2f} mm -> {ckpt_dir}")
    return model, pd.DataFrame(history)


@torch.no_grad()
def predict_split(model: nn.Module, loader: DataLoader, device: torch.device
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference over a loader, **one row per photo pair** (fix B5).

    The previous version averaged all of a subject's photos before comparing against that
    subject's targets. That is not what deployment does: ``infer.py`` receives exactly one
    front mask and one side mask. Averaging inflated every metric by the variance-reduction
    from pooling, and it did so unevenly - Test-A has far more photos per subject than Test-B,
    so the Test-A/Test-B gap confounded model quality with the averaging count.

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
        ``pred`` and ``true`` of shape ``(n_photo_pairs, n_targets)`` in cm, row-aligned with
        the loader's sample order, and the per-row subject ids.
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

    Used only for the clearly-labelled *secondary* subject-averaged tables. It is not the
    primary metric, because deployment predicts from a single pair (fix B5).

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
    order = sorted(set(map(str, subjects)))
    out_p, out_t = [], []
    for s in order:
        idx = np.flatnonzero(np.asarray([str(v) for v in subjects]) == s)
        out_p.append(pred[idx].mean(axis=0))
        out_t.append(true[idx].mean(axis=0))
    return np.stack(out_p), np.stack(out_t), np.asarray(order, dtype=object)