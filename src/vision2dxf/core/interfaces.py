from __future__ import annotations
from pathlib import Path
import json
import yaml

from .models import MeasurementProfile


def load_profiles(path: str | Path) -> list[MeasurementProfile]:
    with open(path) as f:
        raw = json.load(f)
    return [
        MeasurementProfile(
            profile_id=p["profile_id"],
            chest=p["chest"],
            waist=p["waist"],
            hip=p["hip"],
            shoulder=p["shoulder"],
            torso_length=p["torso_length"],
            armhole_depth=p["armhole_depth"],
            sleeve_length=p["sleeve_length"],
            ease=p["ease"],
        )
        for p in raw
    ]


def load_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
