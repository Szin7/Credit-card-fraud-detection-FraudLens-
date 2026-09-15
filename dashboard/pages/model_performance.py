from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.charts import apply_chart_theme
from dashboard.components.theme import COLORS
from dashboard.components.widgets import render_empty_state, render_error_state
from src.database import repository as repo

st.title("Model Performance")
st.caption("Evaluation of the current champion model, measured on its held-out test split.")


@st.cache_data(ttl=60)
def _load_champion():
    return repo.get_champion_model()


try:
    champion = _load_champion()
except Exception as exc:  # noqa: BLE001
    render_error_state("Could not load model data", f"Check that Postgres is running and reachable. ({exc})")
    st.stop()

if champion is None:
    render_empty_state(
        "No trained model registered yet",
        "Run the training pipeline to populate this page: `make train` (or trigger "
        "fraud_training_pipeline in Airflow).",
    )
    st.stop()

curve_data = champion["curve_data"]

st.markdown(
    f"**Champion:** `{champion['model_type']}`  &nbsp;|&nbsp;  "
    f"**Trained:** {pd.to_datetime(champion['trained_at']).strftime('%Y-%m-%d %H:%M UTC')}  &nbsp;|&nbsp;  "
    f"**Decision threshold:** {champion['decision_threshold']}"
)

st.write("")
kpi_cols = st.columns(5)
kpi_items = [
    ("Precision", champion["precision"]),
    ("Recall", champion["recall"]),
    ("F1", champion["f1"]),
    ("ROC-AUC", champion["roc_auc"]),
    ("PR-AUC", champion["pr_auc"]),
]
for col, (label, value) in zip(kpi_cols, kpi_items):
    with col:
        with st.container(border=True):
            st.markdown(f'<div class="fl-kpi-label">{label}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="fl-kpi-value">{value:.4f}</div>', unsafe_allow_html=True)

st.write("")
col1, col2 = st.columns([1, 1.4])

with col1:
    with st.container(border=True):
        st.subheader("Confusion Matrix")
        st.markdown(
            '<div class="fl-section-caption">On the held-out test split, at the current decision threshold.</div>',
            unsafe_allow_html=True,
        )
        cm = curve_data["test_confusion_matrix"]
        z = [[cm["true_negatives"], cm["false_positives"]], [cm["false_negatives"], cm["true_positives"]]]
        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=["Predicted Legitimate", "Predicted Fraud"],
                y=["Actual Legitimate", "Actual Fraud"],
                colorscale=[[0, COLORS["accent_soft"]], [1, COLORS["accent"]]],
                text=z,
                texttemplate="%{text:,}",
                textfont={"size": 16},
                showscale=False,
            )
        )
        fig.update_yaxes(autorange="reversed")
        fig = apply_chart_theme(fig, height=280)
        st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})

with col2:
    with st.container(border=True):
        st.subheader("Model Comparison")
        st.markdown(
            '<div class="fl-section-caption">Every model trained in this run, ranked by validation PR-AUC.</div>',
            unsafe_allow_html=True,
        )
        comparison_df = pd.DataFrame(champion["model_comparison"]).sort_values("pr_auc", ascending=False)
        st.dataframe(
            comparison_df[["model", "precision", "recall", "f1", "roc_auc", "pr_auc"]],
            column_config={
                "model": st.column_config.TextColumn("Model"),
                "precision": st.column_config.NumberColumn("Precision", format="%.4f"),
                "recall": st.column_config.NumberColumn("Recall", format="%.4f"),
                "f1": st.column_config.NumberColumn("F1", format="%.4f"),
                "roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.4f"),
                "pr_auc": st.column_config.NumberColumn("PR-AUC", format="%.4f"),
            },
            hide_index=True,
            width="stretch",
        )

st.write("")
col3, col4 = st.columns(2)

with col3:
    with st.container(border=True):
        st.subheader("Precision-Recall Curve")
        pr = curve_data["precision_recall_curve"]
        fig = go.Figure(
            go.Scatter(
                x=pr["recall"], y=pr["precision"], mode="lines",
                line=dict(color=COLORS["accent"], width=2.5),
                fill="tozeroy", fillcolor=COLORS["accent_soft"],
            )
        )
        fig.update_xaxes(title_text="Recall", range=[0, 1])
        fig.update_yaxes(title_text="Precision", range=[0, 1.02])
        fig = apply_chart_theme(fig, height=300)
        st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})

with col4:
    with st.container(border=True):
        st.subheader("ROC Curve")
        roc = curve_data["roc_curve"]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name="Model", line=dict(color=COLORS["accent"], width=2.5))
        )
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color=COLORS["text_muted"], dash="dash"))
        )
        fig.update_xaxes(title_text="False Positive Rate", range=[0, 1])
        fig.update_yaxes(title_text="True Positive Rate", range=[0, 1.02])
        fig = apply_chart_theme(fig, height=300)
        st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})

st.write("")
with st.container(border=True):
    st.subheader("Threshold Analysis")
    st.markdown(
        '<div class="fl-section-caption">Computed on the validation split, never the test split. '
        "Move the slider to see the precision/recall/cost tradeoff at any threshold.</div>",
        unsafe_allow_html=True,
    )

    sweep = pd.DataFrame(curve_data["threshold_sweep"]).sort_values("threshold").reset_index(drop=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["precision"], name="Precision", line=dict(color=COLORS["accent"])))
    fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["recall"], name="Recall", line=dict(color=COLORS["risk_review"])))
    fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["f1"], name="F1", line=dict(color=COLORS["risk_low"])))
    fig.add_vline(x=champion["decision_threshold"], line_dash="dash", line_color=COLORS["text_muted"])
    fig.update_xaxes(title_text="Decision threshold", range=[0, 1])
    fig.update_yaxes(title_text="Score", range=[0, 1.05])
    fig = apply_chart_theme(fig, height=320)
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})

    threshold_options = sweep["threshold"].tolist()
    default_idx = min(range(len(threshold_options)), key=lambda i: abs(threshold_options[i] - champion["decision_threshold"]))
    selected_threshold = st.select_slider("Explore a threshold", options=threshold_options, value=threshold_options[default_idx])
    row = sweep[sweep["threshold"] == selected_threshold].iloc[0]
    st.markdown(
        f"At threshold **{selected_threshold:.2f}**: precision **{row['precision']:.3f}**, "
        f"recall **{row['recall']:.3f}**, F1 **{row['f1']:.3f}**, "
        f"{int(row['false_positives'])} false positives, {int(row['false_negatives'])} false negatives, "
        f"illustrative expected cost **{row['expected_cost']:.0f}** "
        "(config/settings.yaml > cost_model — not a real business figure)."
    )

    with st.expander("Full threshold table"):
        st.dataframe(
            sweep[["threshold", "precision", "recall", "f1", "false_positives", "false_negatives", "expected_cost"]],
            column_config={
                "threshold": st.column_config.NumberColumn("Threshold", format="%.2f"),
                "precision": st.column_config.NumberColumn("Precision", format="%.4f"),
                "recall": st.column_config.NumberColumn("Recall", format="%.4f"),
                "f1": st.column_config.NumberColumn("F1", format="%.4f"),
                "false_positives": st.column_config.NumberColumn("False Positives"),
                "false_negatives": st.column_config.NumberColumn("False Negatives"),
                "expected_cost": st.column_config.NumberColumn("Expected Cost", format="%.0f"),
            },
            hide_index=True,
            width="stretch",
        )

st.write("")
with st.container(border=True):
    st.subheader("Feature Importance")
    st.markdown(
        '<div class="fl-section-caption">Which model inputs most influenced predictions — V1-V28 are '
        "anonymized and carry no disclosed real-world meaning.</div>",
        unsafe_allow_html=True,
    )

    importance = curve_data.get("feature_importance", {"native": [], "shap": None})
    native_df = pd.DataFrame(importance.get("native", []))
    shap_records = importance.get("shap")

    tab_labels = ["Native Importance"] + (["SHAP"] if shap_records else [])
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        if native_df.empty:
            st.write("No importance data available for this model.")
        else:
            fig = go.Figure(
                go.Bar(x=native_df["importance"], y=native_df["feature"], orientation="h", marker_color=COLORS["accent"])
            )
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(title_text="Importance (|coefficient| or impurity-based)")
            fig = apply_chart_theme(fig, height=max(260, 24 * len(native_df)))
            st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})

    if shap_records:
        with tabs[1]:
            shap_df = pd.DataFrame(shap_records)
            fig = go.Figure(
                go.Bar(x=shap_df["importance"], y=shap_df["feature"], orientation="h", marker_color=COLORS["risk_review"])
            )
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(title_text="Mean |SHAP value|")
            fig = apply_chart_theme(fig, height=max(260, 24 * len(shap_df)))
            st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})
