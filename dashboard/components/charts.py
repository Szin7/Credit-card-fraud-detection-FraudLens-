"""Shared Plotly theming so every chart on every page uses the same fonts,
colors, and spacing instead of Plotly's default look.
"""

from __future__ import annotations

import plotly.graph_objects as go

from dashboard.components.theme import COLORS, FONT_STACK


def apply_chart_theme(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_STACK, size=13, color=COLORS["text_primary"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12)),
        hoverlabel=dict(bgcolor=COLORS["surface"], font_size=12, font_family=FONT_STACK),
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor=COLORS["border"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=COLORS["border"], zeroline=False)
    return fig
