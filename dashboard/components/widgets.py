"""Small reusable HTML fragments built on the theme tokens — KPI cards, risk
badges, and empty/error states — kept as plain functions returning/rendering
HTML rather than a component class hierarchy, since there's no shared state
between them.
"""

from __future__ import annotations

import html

import streamlit as st

from dashboard.components.theme import COLORS, RISK_COLOR_MAP


def kpi_row(items: list[dict]) -> None:
    """Render a responsive row of KPI cards.

    Each item: {"label": str, "value": str, "context": str | None}.
    """
    cards = []
    for item in items:
        context_html = (
            f'<div class="fl-kpi-context">{html.escape(item["context"])}</div>'
            if item.get("context")
            else ""
        )
        cards.append(
            f"""
            <div class="fl-kpi-card">
                <div class="fl-kpi-label">{html.escape(item['label'])}</div>
                <div class="fl-kpi-value">{html.escape(str(item['value']))}</div>
                {context_html}
            </div>
            """
        )
    st.markdown(f'<div class="fl-kpi-row">{"".join(cards)}</div>', unsafe_allow_html=True)


def risk_badge_html(risk_level: str) -> str:
    text_key, bg_key = RISK_COLOR_MAP.get(risk_level, ("text_secondary", "bg"))
    color = COLORS[text_key]
    background = COLORS[bg_key]
    label = html.escape(risk_level)
    return f'<span class="fl-badge" style="color:{color};background-color:{background};">{label}</span>'


def render_empty_state(title: str, message: str) -> None:
    st.markdown(
        f"""
        <div class="fl-empty-state">
            <div class="fl-empty-state-title">{html.escape(title)}</div>
            <div>{html.escape(message)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_error_state(title: str, message: str) -> None:
    st.markdown(
        f"""
        <div class="fl-empty-state" style="border-color:{COLORS['risk_high']};border-style:solid;">
            <div class="fl-empty-state-title" style="color:{COLORS['risk_high']};">{html.escape(title)}</div>
            <div>{html.escape(message)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
