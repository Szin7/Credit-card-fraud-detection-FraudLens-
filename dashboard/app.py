"""Dashboard entry point / router.

Streamlit resolves imports relative to this script's own directory, so the
project root needs to be added to sys.path explicitly before any `src.*` or
`dashboard.*` import — the same bootstrap notebooks/01_exploration.ipynb
uses, for the same reason.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from dashboard.components.theme import inject_global_styles

st.set_page_config(
    page_title="FraudLens",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_styles()

pages = [
    st.Page("pages/overview.py", title="Overview", default=True),
    st.Page("pages/transaction_explorer.py", title="Transaction Explorer"),
    st.Page("pages/model_performance.py", title="Model Performance"),
    st.Page("pages/pipeline_health.py", title="Pipeline Health"),
]

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 4px 0 20px 0;">
            <div style="font-size: 19px; font-weight: 700; color: #12141C;">FraudLens</div>
            <div style="font-size: 12.5px; color: #667085;">Transaction Risk Intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

navigation = st.navigation(pages)
navigation.run()
