from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.components.charts import apply_chart_theme
from dashboard.components.theme import COLORS
from dashboard.components.widgets import kpi_row, render_empty_state, render_error_state
from src.database import repository as repo

st.title("Overview")
st.caption("Live snapshot of transaction risk across the current prediction set.")


@st.cache_data(ttl=60)
def _load_overview_data():
    return (
        repo.get_overview_stats(),
        repo.get_risk_distribution(),
        repo.get_predictions_over_time(),
        repo.get_recent_high_risk(limit=15),
    )


try:
    stats, risk_dist, trend, recent_high_risk = _load_overview_data()
except Exception as exc:  # noqa: BLE001 - surface any DB error as a page-level message
    render_error_state(
        "Could not load dashboard data",
        f"Check that Postgres is running and reachable at the configured DATABASE_URL. ({exc})",
    )
    st.stop()

if stats["total_transactions"] == 0:
    render_empty_state(
        "No predictions yet",
        "Run the pipeline to populate the dashboard: `make pipeline` (or trigger "
        "fraud_inference_pipeline in Airflow).",
    )
    st.stop()

kpi_row(
    [
        {"label": "Total Transactions", "value": f"{stats['total_transactions']:,}"},
        {"label": "Flagged", "value": f"{stats['flagged_count']:,}"},
        {"label": "Flagged Rate", "value": f"{stats['flagged_rate']:.2%}"},
        {"label": "Avg. Risk Score", "value": f"{stats['avg_probability']:.3f}"},
    ]
)

st.write("")
with st.container(border=True):
    st.subheader("Transaction Risk Over Time")
    st.markdown(
        '<div class="fl-section-caption">Transaction volume and average fraud probability, '
        "by hour since the first transaction in this batch.</div>",
        unsafe_allow_html=True,
    )

    if trend.empty:
        st.write("Not enough data to chart a trend yet.")
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=trend["hour_bucket"],
                y=trend["transaction_count"],
                name="Transactions",
                marker_color=COLORS["accent_soft"],
                marker_line_color=COLORS["border"],
                marker_line_width=1,
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=trend["hour_bucket"],
                y=trend["avg_probability"],
                name="Avg. fraud probability",
                mode="lines+markers",
                line=dict(color=COLORS["accent"], width=2.5),
                marker=dict(size=5),
            ),
            secondary_y=True,
        )
        fig.update_yaxes(title_text="Transactions", secondary_y=False)
        fig.update_yaxes(title_text="Avg. fraud probability", secondary_y=True, range=[0, 1])
        fig.update_xaxes(title_text="Hour since first transaction")
        fig = apply_chart_theme(fig, height=360)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.write("")
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader("Risk Distribution")
        risk_order = ["LOW", "REVIEW", "HIGH"]
        dist = (
            risk_dist.set_index("risk_level").reindex(risk_order).fillna(0).astype(int).reset_index()
        )
        fig = go.Figure(
            go.Bar(
                x=dist["count"],
                y=dist["risk_level"],
                orientation="h",
                marker_color=[COLORS["risk_low"], COLORS["risk_review"], COLORS["risk_high"]],
                text=dist["count"],
                textposition="outside",
            )
        )
        fig.update_xaxes(title_text="Transactions")
        fig = apply_chart_theme(fig, height=260)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with col2:
    with st.container(border=True):
        st.subheader("Transaction Volume")
        if trend.empty:
            st.write("Not enough data to chart volume yet.")
        else:
            fig = go.Figure(
                go.Scatter(
                    x=trend["hour_bucket"],
                    y=trend["transaction_count"],
                    mode="lines",
                    fill="tozeroy",
                    line=dict(color=COLORS["accent"], width=2),
                    fillcolor=COLORS["accent_soft"],
                )
            )
            fig.update_xaxes(title_text="Hour since first transaction")
            fig.update_yaxes(title_text="Transactions")
            fig = apply_chart_theme(fig, height=260)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.write("")
with st.container(border=True):
    st.subheader("Recent High-Risk Transactions")
    if recent_high_risk.empty:
        st.write("No high-risk transactions in the current prediction set.")
    else:
        display_df = recent_high_risk.copy()
        display_df["prediction_timestamp"] = pd.to_datetime(display_df["prediction_timestamp"])
        st.dataframe(
            display_df[["transaction_id", "amount", "fraud_probability", "risk_level", "prediction_timestamp"]],
            column_config={
                "transaction_id": st.column_config.TextColumn("Transaction"),
                "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
                "fraud_probability": st.column_config.ProgressColumn(
                    "Probability", min_value=0.0, max_value=1.0, format="%.3f"
                ),
                "risk_level": st.column_config.TextColumn("Risk"),
                "prediction_timestamp": st.column_config.DatetimeColumn("Predicted At", format="YYYY-MM-DD HH:mm"),
            },
            hide_index=True,
            use_container_width=True,
        )
