from __future__ import annotations
import plotly.graph_objects as go
from .models import PatternResult


def render_preview(result: PatternResult) -> go.Figure:
    fig = go.Figure()

    for seg in result.segments:
        fr = result.points.get(seg["from"])
        to = result.points.get(seg["to"])
        if fr is None or to is None:
            continue
        xs = [fr[0], to[0]]
        ys = [fr[1], to[1]]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color="steelblue", width=2),
            showlegend=False,
        ))

    names = list(result.points.keys())
    xs = [result.points[n][0] for n in names]
    ys = [result.points[n][1] for n in names]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        marker=dict(size=6, color="crimson"),
        text=names, textposition="top center",
        showlegend=False,
    ))

    fig.update_layout(
        title=f"{result.design_id} — Pattern Preview",
        xaxis=dict(scaleanchor="y", scaleratio=1, title="x (cm)"),
        yaxis=dict(autorange="reversed", title="y (cm)"),
        width=500, height=600,
        plot_bgcolor="white",
    )
    return fig
