"""Phase wrapper: the pattern every phase cell follows.

Why a wrapper rather than a decorator
-------------------------------------
Each phase in this notebook needs to *define its own variables*, and those variables differ per
phase (fitted models, DataFrames, prediction arrays). A decorator cannot inject them. So the
pattern is explicit and uniform:

    _res, _why = PH("p05_b1")
    if _res is R.BLOCKED:
        b1 = b1_metrics = None
    elif _res is not None:
        <reload from _res>
    else:
        <compute>
        PH_done("p05_b1", artefacts=[...], meta={...})

Three helpers below keep the shape identical everywhere:

* :func:`begin` - the ``maybe_skip`` call plus a printed decision.
* :func:`done` - save a payload and mark the phase.
* :func:`guard` - a context manager combining the CPU tripwire with a forced device.

The ``BLOCKED`` branch is the part that matters most. When a dependency is unavailable (an
interrupted training phase, a blocked test-set lock), the dependent cell must still define its
variables, or every later cell raises ``NameError``. Assigning ``None`` makes the failure legible
at the point where the value is used, rather than as an unrelated crash three phases later.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

import run_state as R

#: Set when any phase has been blocked, so the final summary can say so loudly.
BLOCKED_PHASES: Dict[str, str] = {}

#: Every phase that was skipped because a dependency was unavailable, and why.
UNAVAILABLE: Dict[str, str] = {}


class PhaseUnavailable(RuntimeError):
    """Raised when a cell's inputs cannot exist because an upstream phase did not finish.

    A notebook cell that raises stops only that cell; the following cells still run. That is the
    behaviour we want here. The alternative — letting ``predict_index(None, ...)`` fail with an
    ``AttributeError`` three phases later — hides which decision caused the problem.

    The exception is caught by the final summary cell, which reports every unavailable phase in
    one place instead of leaving a trail of tracebacks.
    """


def begin(name: str, deps: Optional[Sequence[str]] = None) -> Tuple[Any, str]:
    """Ask the run state whether a phase should run, be skipped, or is blocked.

    Parameters
    ----------
    name : str
        Phase name from the registry.
    deps : sequence of str, optional
        Overrides the registry's dependency list.

    Returns
    -------
    (payload, reason) : tuple
        * ``(R.BLOCKED, reason)`` - a dependency is unavailable; the cell must define its
          variables as ``None`` and skip its body.
        * ``(payload, 'cached')`` - the phase is reusable; reload from ``payload``.
        * ``(None, 'run')`` - compute the phase, then call :func:`done`.
    """
    payload, reason = RUNSTATE.maybe_skip(name, deps)
    if payload is R.BLOCKED:
        BLOCKED_PHASES[name] = str(reason)
        return R.BLOCKED, str(reason)
    if reason == "cached":
        meta = payload.get("_meta", {})
        extra = ""
        if meta.get("epochs_completed") is not None:
            extra = (f" [{meta.get('epochs_completed')}/{meta.get('epochs_configured')} epochs, "
                     f"best {meta.get('best_val_mae_mm')} mm, "
                     f"stop={meta.get('stop_reason')}]")
        print(f"  loaded {name} from {RUNSTATE.results_dir(name)}"
              + (f", checkpoint {RUNSTATE.ckpt_dir(name.replace('p09_', '').replace('p08_', '')
                                                        .replace('p06_', ''))}"
                 if "train" in name else "")
              + extra)
        return payload, "cached"
    print(f"  running {name}")
    return None, "run"


def done(name: str, payload: Dict[str, Any], status: str = "done",
         extra_artefacts: Sequence[str] = (), meta: Optional[Dict[str, Any]] = None) -> List[str]:
    """Persist a phase payload and record it in the manifest.

    Parameters
    ----------
    name : str
        Phase name.
    payload : dict
        Values to write. DataFrames become parquet, ndarrays npz, everything else JSON.
    status : str, default 'done'
        One of ``run_state.STATUSES``. Use ``'partial'`` for an interrupted phase.
    extra_artefacts : sequence of str, optional
        Additional files the phase depends on, such as a checkpoint. Included in the manifest so
        their deletion also invalidates the phase.
    meta : dict, optional
        Provenance recorded alongside the status.

    Returns
    -------
    list of str
        Every artefact now listed for this phase.
    """
    arts = R.save_payload(RUNSTATE.results_dir(name), payload)
    arts = list(arts) + [str(a) for a in extra_artefacts]
    RUNSTATE.mark_done(name, artefacts=arts, meta=meta or {}, status=status)
    return arts


@contextmanager
def guard(name: str, requires_gpu: Optional[bool] = None):
    """Run a block under the right device policy for its phase.

    CPU-only phases are wrapped in :func:`run_state.cpu_guard`, which turns any attempt to move a
    tensor to CUDA into an immediate, named error rather than a silent dependence on an
    accelerator that may not be present.

    Parameters
    ----------
    name : str
        Phase name.
    requires_gpu : bool, optional
        Overrides the registry entry.

    Yields
    ------
    torch.device
        The device the phase should use.
    """
    need_gpu = (PHASES.get(name, {}).get("requires_gpu", False)
                if requires_gpu is None else requires_gpu)
    if need_gpu:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            print(f"  WARNING: {name} requires a GPU but none is available; running on CPU. "
                  "Expect this to be very slow.")
        yield dev
    else:
        with R.cpu_guard(name):
            yield torch.device("cpu")


def blocked_reason(name: str) -> str:
    """Why a phase was blocked, or an empty string.

    Parameters
    ----------
    name : str
        Phase name.

    Returns
    -------
    str
    """
    return BLOCKED_PHASES.get(name, "")


def any_blocked() -> Dict[str, str]:
    """Every blocked phase and its reason.

    Returns
    -------
    dict
    """
    return dict(BLOCKED_PHASES)


def require(*names: str) -> None:
    """Stop the current cell if any named phase did not produce usable output.

    Used at the top of every cell that consumes a trained network. A cell that raises stops only
    itself, so the rest of the notebook continues; letting ``predict_index(None, ...)`` fail
    instead would surface three phases later as an ``AttributeError`` that says nothing about the
    real cause.

    Parameters
    ----------
    *names : str
        Phase names whose outputs this cell needs.

    Raises
    ------
    PhaseUnavailable
        If any of them is unavailable. The message names the phase and the reason.
    """
    missing = [n for n in names
               if RUNSTATE.entry(n)["status"] in ("partial", "blocked", "failed", "missing")]
    if not missing:
        return
    lines = []
    for n in missing:
        st = RUNSTATE.entry(n)
        why = (st.get("meta") or {}).get("stop_reason") or st["status"]
        lines.append(f"  {n}: status={st['status']} ({why})")
        UNAVAILABLE[n] = f"{st['status']}: {why}"
    print("=" * 78)
    print("THIS CELL SKIPPED - a phase it depends on did not finish:")
    print("\n".join(lines))
    print("Fix that phase first (add its name to config.yaml force_rerun, or attach the previous")
    print("version's output as an input dataset) and run the notebook again.")
    print("=" * 78)
    raise PhaseUnavailable("; ".join(missing))