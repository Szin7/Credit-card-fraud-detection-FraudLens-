from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.widgets import kpi_row, render_empty_state, render_error_state
from src.database import repository as repo

st.title("Pipeline Health")
st.caption("Data engineering view: recent pipeline runs, throughput, and ingestion status.")


@st.cache_data(ttl=30)
def _load_pipeline_data():
    return repo.get_pipeline_runs(limit=50), repo.get_latest_successful_ingestion()


try:
    runs, last_ingestion = _load_pipeline_data()
except Exception as exc:  # noqa: BLE001
    render_error_state("Could not load pipeline data", f"Check that Postgres is running and reachable. ({exc})")
    st.stop()

if runs.empty:
    render_empty_state(
        "No pipeline runs yet",
        "Run the pipeline to populate this page: `make pipeline` (or trigger a DAG in Airflow).",
    )
    st.stop()

runs["started_at"] = pd.to_datetime(runs["started_at"])
runs["finished_at"] = pd.to_datetime(runs["finished_at"])
runs["duration_seconds"] = (runs["finished_at"] - runs["started_at"]).dt.total_seconds()

latest_run = runs.iloc[0]
latest_duration = latest_run["duration_seconds"]

kpi_row(
    [
        {"label": "Last Run Type", "value": latest_run["run_type"].replace("_", " ").title()},
        {"label": "Last Run Status", "value": latest_run["status"].title()},
        {
            "label": "Records Processed",
            "value": f"{int(latest_run['records_processed']):,}" if pd.notna(latest_run["records_processed"]) else "-",
        },
        {"label": "Duration", "value": f"{latest_duration:.1f}s" if pd.notna(latest_duration) else "running"},
    ]
)

st.write("")
with st.container(border=True):
    st.subheader("Last Successful Ingestion")
    if last_ingestion is None:
        st.write("No successful ingestion recorded yet.")
    else:
        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            st.markdown("**Run ID**")
            st.code(last_ingestion["run_id"], language=None)
        with ic2:
            st.markdown("**Started At**")
            st.write(str(pd.to_datetime(last_ingestion["started_at"])))
        with ic3:
            st.markdown("**Records Processed**")
            st.write(f"{last_ingestion['records_processed']:,}" if last_ingestion["records_processed"] else "-")

st.write("")
with st.container(border=True):
    st.subheader("Recent Pipeline Runs")
    display_df = runs[
        ["run_type", "status", "started_at", "duration_seconds", "records_processed", "records_rejected", "error_message"]
    ].copy()
    st.dataframe(
        display_df,
        column_config={
            "run_type": st.column_config.TextColumn("Type"),
            "status": st.column_config.TextColumn("Status"),
            "started_at": st.column_config.DatetimeColumn("Started", format="YYYY-MM-DD HH:mm:ss"),
            "duration_seconds": st.column_config.NumberColumn("Duration (s)", format="%.1f"),
            "records_processed": st.column_config.NumberColumn("Processed"),
            "records_rejected": st.column_config.NumberColumn("Rejected"),
            "error_message": st.column_config.TextColumn("Error"),
        },
        hide_index=True,
        width="stretch",
    )
