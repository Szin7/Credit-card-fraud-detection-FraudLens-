"""Design tokens and global CSS for the dashboard.

Injected once (inject_global_styles, called from app.py) rather than
styled per-widget, so every page shares one visual system instead of
drifting into inconsistent one-off styling.
"""

from __future__ import annotations

import textwrap

import streamlit as st

COLORS = {
    "bg": "#F6F7F9",
    "surface": "#FFFFFF",
    "border": "#E4E7EC",
    "text_primary": "#12141C",
    "text_secondary": "#667085",
    "text_muted": "#98A2B3",
    "accent": "#4338CA",
    "accent_soft": "#EEF2FF",
    "risk_low": "#067647",
    "risk_low_soft": "#ECFDF3",
    "risk_review": "#B54708",
    "risk_review_soft": "#FFFAEB",
    "risk_high": "#B42318",
    "risk_high_soft": "#FEF3F2",
}

FONT_STACK = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
)

RISK_COLOR_MAP = {
    "LOW": ("risk_low", "risk_low_soft"),
    "REVIEW": ("risk_review", "risk_review_soft"),
    "HIGH": ("risk_high", "risk_high_soft"),
}


def inject_global_styles() -> None:
    st.markdown(
        textwrap.dedent(
            f"""\
        <style>
        html, body, [class*="css"] {{
            font-family: {FONT_STACK};
        }}
        .stApp {{
            background-color: {COLORS['bg']};
        }}
        [data-testid="stSidebar"] {{
            background-color: {COLORS['surface']};
            border-right: 1px solid {COLORS['border']};
        }}
        [data-testid="stHeader"] {{
            background-color: transparent;
        }}
        h1, h2, h3 {{
            color: {COLORS['text_primary']};
            font-weight: 650;
            letter-spacing: -0.01em;
        }}
        p, span, label, div {{
            color: {COLORS['text_primary']};
        }}
        [data-testid="stCaptionContainer"] {{
            color: {COLORS['text_secondary']};
        }}

        /* Card containers (st.container(border=True)) */
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background-color: {COLORS['surface']};
            border: 1px solid {COLORS['border']};
            border-radius: 12px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }}

        /* Dataframes / tables */
        [data-testid="stDataFrame"] {{
            border: 1px solid {COLORS['border']};
            border-radius: 10px;
            overflow: hidden;
        }}

        /* Buttons */
        .stButton > button, .stDownloadButton > button {{
            border-radius: 8px;
            border: 1px solid {COLORS['border']};
            font-weight: 500;
        }}
        .stButton > button:hover {{
            border-color: {COLORS['accent']};
            color: {COLORS['accent']};
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab"] {{
            font-weight: 500;
        }}

        /* FraudLens custom components */
        .fl-kpi-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 8px;
        }}
        .fl-kpi-card {{
            background-color: {COLORS['surface']};
            border: 1px solid {COLORS['border']};
            border-radius: 12px;
            padding: 18px 20px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }}
        .fl-kpi-label {{
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: {COLORS['text_secondary']};
            margin-bottom: 6px;
        }}
        .fl-kpi-value {{
            font-size: 28px;
            font-weight: 700;
            color: {COLORS['text_primary']};
            line-height: 1.2;
        }}
        .fl-kpi-context {{
            font-size: 12.5px;
            color: {COLORS['text_muted']};
            margin-top: 4px;
        }}

        .fl-badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 12.5px;
            font-weight: 600;
        }}

        .fl-empty-state {{
            background-color: {COLORS['surface']};
            border: 1px dashed {COLORS['border']};
            border-radius: 12px;
            padding: 48px 24px;
            text-align: center;
            color: {COLORS['text_secondary']};
        }}
        .fl-empty-state-title {{
            font-size: 16px;
            font-weight: 600;
            color: {COLORS['text_primary']};
            margin-bottom: 6px;
        }}

        .fl-section-caption {{
            color: {COLORS['text_secondary']};
            font-size: 13px;
            margin-top: -8px;
            margin-bottom: 12px;
        }}
        </style>
        """
        ),
        unsafe_allow_html=True,
    )
