"""Exporters: JSON description and visual PDF rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

__all__ = ["export_json", "export_pdf"]


def export_json(data: Mapping[str, Any], path: str | Path) -> str:
    """Write the pattern dictionary to ``path`` as pretty-printed JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return str(path)


def export_pdf(
    pattern: Any,
    data: Mapping[str, Any],
    path: str | Path,
    info: Mapping[str, Any] | None = None,
) -> str:
    """Render the drafted pattern to a 1:1 PDF using matplotlib.

    The geometry already extracted in ``data`` is re-used so the drawing and
    the JSON always describe the same panels.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    point_dicts: list[Dict[str, Any]] = []
    for group in data.get("points", {}).values():
        points = group.get("points") if isinstance(group, Mapping) else None
        if points:
            point_dicts.append(dict(points))

    vertices_list = [panel["vertices"] for panel in data.get("panels", [])]

    fig, ax = pattern.draw_pattern(point_dicts, vertices_list)

    if info:
        pattern.print_info(ax, dict(info))
    pattern.add_legends(ax)
    pattern.add_scales(ax, val=5)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path))
    plt.close(fig)
    return str(path)
