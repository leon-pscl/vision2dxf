"""Search sample_patterns for polo/tshirt-like patterns and copy them to pattern_templates/."""

import json
import shutil
from pathlib import Path

SHIRT_LABELS = {"衣身前中", "衣身后中", "袖片"}
SKIRT_LABELS = {"裙前中", "裙后中"}
RUFFLE_LABEL = "荷叶边"

SRC = Path(__file__).resolve().parent / "sample_patterns"
DST = Path(__file__).resolve().parent / "pattern_templates"


def is_polo_or_tshirt(data: dict) -> bool:
    labels = {p["label"].strip() for p in data["panels"]}
    has_bodice = "衣身前中" in labels and "衣身后中" in labels
    has_sleeve = "袖片" in labels
    has_skirt = bool(labels & SKIRT_LABELS)
    has_ruffle = RUFFLE_LABEL in labels
    return has_bodice and has_sleeve and not has_skirt and not has_ruffle


def main() -> None:
    DST.mkdir(exist_ok=True)
    files = sorted(SRC.glob("*.json"))
    print(f"Scanning {len(files)} files...")
    copied = 0
    for f in files:
        data = json.loads(f.read_text("utf-8"))
        if is_polo_or_tshirt(data):
            shutil.copy2(f, DST / f.name)
            copied += 1
    print(f"Copied {copied} polo/tshirt patterns to {DST}")


if __name__ == "__main__":
    main()
