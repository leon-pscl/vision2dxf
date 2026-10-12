"""Check an exported body file against the pinned GarmentCode revision (brief 3.2).

The schema is pinned, not guessed. This script compares the keys the exporter
writes with the keys the pinned revision actually reads, and reports the exact
difference. It never renames a key to make the comparison pass: if the two
disagree, that is a decision for a person to make.

    python tools/verify_garmentcode.py                       # check the schema itself
    python tools/verify_garmentcode.py data/.../body.yaml    # and load a real export
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bodymeasure.config import load_app_config, load_schema  # noqa: E402


def pinned_keys() -> set[str] | None:
    """Return the body keys the installed pygarment reads, or ``None`` if absent.

    ``None`` means the comparison could not be made, which is not the same as
    the comparison passing.
    """
    try:
        from pygarment.garmentcode.body_parameters import BodyParameters
    except ImportError:
        return None
    defaults = getattr(BodyParameters, "default_values", None)
    if callable(defaults):
        defaults = defaults()
    if isinstance(defaults, dict):
        return {key for key in defaults if not key.startswith("_")}
    return None


def compare(exported: set[str], pinned: set[str]) -> tuple[set[str], set[str]]:
    """Return the keys we export that it does not read, and the keys it wants."""
    return exported - pinned, pinned - exported


def main(argv: list[str] | None = None) -> int:
    """Report the schema difference. Returns a non-zero exit code on a mismatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("body_file", nargs="?", type=Path, help="an exported body file to load")
    arguments = parser.parse_args(argv)

    commit = load_app_config().garmentcode.commit
    exported = set(load_schema())
    print(f"pinned GarmentCode commit: {commit}")
    print(f"exported primary keys: {len(exported)}")

    if arguments.body_file:
        document = yaml.safe_load(arguments.body_file.read_text(encoding="utf-8"))
        in_file = set(document["body"])
        missing = exported - in_file
        extra = in_file - exported
        if missing or extra:
            print(f"the file does not match the schema: missing {sorted(missing)}, "
                  f"unexpected {sorted(extra)}")
            return 1
        nulls = sorted(name for name, value in document["body"].items() if value is None)
        print(f"the file matches the schema; {len(nulls)} key(s) are null: {nulls}")

    pinned = pinned_keys()
    if pinned is None:
        print(
            "pygarment is not installed at the pinned revision, so the compatibility\n"
            "check could NOT be made. Install it from the pinned commit and run this\n"
            "again before trusting an export."
        )
        return 2

    unexpected, wanted = compare(exported, pinned)
    if not unexpected and not wanted:
        print("the exported keys are exactly the keys the pinned revision reads")
        return 0
    print("SCHEMA DIFFERENCE — do not rename anything to hide it:")
    if unexpected:
        print(f"  exported but not read by GarmentCode: {sorted(unexpected)}")
    if wanted:
        print(f"  read by GarmentCode but not exported: {sorted(wanted)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
