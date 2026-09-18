from __future__ import annotations
import json
from pathlib import Path
from .models import PatternResult


def export_json(result: PatternResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "design_id": result.design_id,
        "points": {k: list(v) for k, v in result.points.items()},
        "segments": result.segments,
        "valid": result.valid,
        "validation_errors": result.validation_errors,
        "metadata": result.metadata,
    }
    path.write_text(json.dumps(data, indent=2))
    return path


def export_dxf(result: PatternResult, path: str | Path) -> Path | None:
    try:
        import ezdxf
    except ImportError:
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    for seg in result.segments:
        fr = result.points.get(seg["from"])
        to = result.points.get(seg["to"])
        if fr is None or to is None:
            continue
        if seg.get("type") == "curve":
            ctrl = seg.get("control_points", [])
            if len(ctrl) >= 2:
                pts = [fr] + ctrl + [to]
                fit_pts = [(p[0], p[1]) for p in pts]
                msp.add_spline(fit_points=fit_pts)
            else:
                msp.add_line(fr, to)
        else:
            msp.add_line(fr, to)

    doc.saveas(str(path))
    return path
