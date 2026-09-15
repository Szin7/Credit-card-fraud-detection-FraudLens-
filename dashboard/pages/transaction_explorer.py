from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.widgets import render_empty_state, render_error_state, risk_badge_html
from src.database import repository as repo

st.title("Transaction Explorer")
st.caption("Search, filter, and inspect individual scored transactions.")

PAGE_SIZE = 25

if "explorer_page" not in st.session_state:
    st.session_state.explorer_page = 0

with st.container(border=True):
    col1, col2, col3 = st.columns([1.4, 1.6, 1.6])
    with col1:
        risk_filter = st.multiselect("Risk level", ["LOW", "REVIEW", "HIGH"], default=["HIGH", "REVIEW"])
        search_id = st.text_input("Transaction ID contains", placeholder="e.g. a1b2c3")
    with col2:
        prob_range = st.slider("Fraud probability", 0.0, 1.0, (0.0, 1.0), step=0.01)
    with col3:
        min_amount = st.number_input("Min amount ($)", min_value=0.0, value=0.0, step=10.0)
        max_amount = st.number_input("Max amount ($, 0 = no limit)", min_value=0.0, value=0.0, step=10.0)

filters = dict(
    risk_levels=risk_filter or None,
    transaction_id_contains=search_id or None,
    min_probability=prob_range[0],
    max_probability=prob_range[1],
    min_amount=min_amount or None,
    max_amount=max_amount or None,
)

if st.session_state.get("_explorer_filters") != filters:
    st.session_state.explorer_page = 0
    st.session_state["_explorer_filters"] = filters

try:
    total_matches = repo.count_predictions(**filters)
except Exception as exc:  # noqa: BLE001
    render_error_state("Could not load data", f"Check that Postgres is running and reachable. ({exc})")
    st.stop()

if total_matches == 0:
    render_empty_state("No matching transactions", "Try widening the filters above.")
    st.stop()

total_pages = max(1, (total_matches - 1) // PAGE_SIZE + 1)
st.session_state.explorer_page = min(st.session_state.explorer_page, total_pages - 1)
offset = st.session_state.explorer_page * PAGE_SIZE

results = repo.search_predictions(**filters, limit=PAGE_SIZE, offset=offset)

st.write("")
nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
with nav_col1:
    if st.button("Previous", disabled=st.session_state.explorer_page == 0, use_container_width=True):
        st.session_state.explorer_page -= 1
        st.rerun()
with nav_col2:
    start = offset + 1
    end = min(offset + PAGE_SIZE, total_matches)
    st.markdown(
        f'<div style="text-align:center;color:#667085;padding-top:6px;">'
        f"{start}-{end} of {total_matches:,} matching transactions</div>",
        unsafe_allow_html=True,
    )
with nav_col3:
    if st.button("Next", disabled=st.session_state.explorer_page >= total_pages - 1, use_container_width=True):
        st.session_state.explorer_page += 1
        st.rerun()

st.write("")
with st.container(border=True):
    display_df = results.copy()
    display_df["prediction_timestamp"] = pd.to_datetime(display_df["prediction_timestamp"])

    selection = st.dataframe(
        display_df[["transaction_id", "amount", "fraud_probability", "risk_level", "is_flagged", "prediction_timestamp"]],
        column_config={
            "transaction_id": st.column_config.TextColumn("Transaction"),
            "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
            "fraud_probability": st.column_config.ProgressColumn(
                "Probability", min_value=0.0, max_value=1.0, format="%.3f"
            ),
            "risk_level": st.column_config.TextColumn("Risk"),
            "is_flagged": st.column_config.CheckboxColumn("Flagged"),
            "prediction_timestamp": st.column_config.DatetimeColumn("Predicted At", format="YYYY-MM-DD HH:mm"),
        },
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
    )

selected_rows = selection.selection.rows if selection is not None else []

st.write("")
with st.container(border=True):
    st.subheader("Transaction Detail")
    if not selected_rows:
        st.markdown(
            '<div class="fl-section-caption">Select a row above to see full prediction detail.</div>',
            unsafe_allow_html=True,
        )
    else:
        selected_id = display_df.iloc[selected_rows[0]]["transaction_id"]
        detail = repo.get_transaction_detail(selected_id)
        if detail is None:
            st.write("Transaction not found.")
        else:
            d1, d2, d3 = st.columns(3)
            with d1:
                st.markdown("**Transaction ID**")
                st.code(detail["transaction_id"], language=None)
                st.markdown("**Amount**")
                st.write(f"${detail['amount']:.2f}")
            with d2:
                st.markdown("**Fraud Probability**")
                st.write(f"{detail['fraud_probability']:.4f}")
                st.markdown("**Risk Classification**")
                st.markdown(risk_badge_html(detail["risk_level"]), unsafe_allow_html=True)
            with d3:
                st.markdown("**Model Version**")
                st.write(detail["model_version"])
                st.markdown("**Prediction Timestamp**")
                st.write(str(detail["prediction_timestamp"]))
