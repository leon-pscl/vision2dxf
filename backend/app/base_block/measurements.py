"""Helper to provide default ambiguous‑field confirmations used by the test suite.
The function returns a mapping ``field_name → candidate_id`` for every ambiguous field.
"""

from __future__ import annotations

from ..drafting.semantics import registry as semantics_registry


def default_confirmations() -> dict[str, str]:
    """Return a minimal set of confirmations: the first candidate for each ambiguous field.
    This mirrors the behaviour expected by ``conftest.py``.
    """
    reg = semantics_registry()
    confirmations: dict[str, str] = {}
    for name, fdef in reg.fields.items():
        if fdef.status == "ambiguous":
            # Choose the first candidate id.
            if fdef.candidates:
                confirmations[name] = fdef.candidates[0].id
    return confirmations
