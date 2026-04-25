from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

import sys

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predictive_maintenance import (  # noqa: E402
    CYCLE_COL,
    ENGINE_ID_COL,
    FAILURE_SOON_COL,
    FEATURE_WINDOWS,
    RUL_COL,
    clip_rul,
    prepare_feature_matrices,
    prepare_train_test_with_rul,
)

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}

# ── Dark-mode palette ──────────────────────────────────────────────
DARK_BG = "#0a0e17"
CARD_BG = "rgba(15, 23, 42, 0.72)"
PLOT_BG = "rgba(15, 23, 42, 0.55)"
GRID_CLR = "rgba(100, 180, 255, 0.07)"
TEXT_CLR = "#c8d6e5"
ACCENT_CYAN = "#00e5ff"
ACCENT_AMBER = "#ffab00"
ACCENT_RED = "#ff1744"
ACCENT_GREEN = "#00e676"
RISK_COLORS = {"High": "#ff1744", "Medium": "#ffab00", "Low": "#00e676"}


def _dark_layout(**overrides) -> dict:
    """Return a reusable dark Plotly layout dict."""
    base = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": PLOT_BG,
        "font": {"color": TEXT_CLR, "family": "Inter, sans-serif", "size": 13},
        "title_font": {"color": "#ffffff", "size": 16, "family": "Inter, sans-serif"},
        "legend": {"bgcolor": "rgba(0,0,0,0)", "font": {"color": TEXT_CLR}},
        "margin": {"l": 24, "r": 24, "t": 56, "b": 24},
    }
    base.update(overrides)
    return base


def _dark_axes(fig, rangeslider: bool = False) -> None:
    """Apply dark grid styling to both axes."""
    fig.update_xaxes(
        gridcolor=GRID_CLR,
        zerolinecolor=GRID_CLR,
        tickfont={"color": TEXT_CLR},
        title_font={"color": TEXT_CLR},
        rangeslider_visible=rangeslider,
    )
    fig.update_yaxes(
        gridcolor=GRID_CLR,
        zerolinecolor=GRID_CLR,
        tickfont={"color": TEXT_CLR},
        title_font={"color": TEXT_CLR},
    )


def apply_custom_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {
            --bg-primary:   #0a0e17;
            --bg-card:      rgba(15, 23, 42, 0.72);
            --bg-card-hover:rgba(20, 30, 55, 0.85);
            --border:       rgba(100, 180, 255, 0.10);
            --border-glow:  rgba(0, 229, 255, 0.25);
            --text-primary: #e2e8f0;
            --text-muted:   #94a3b8;
            --accent-cyan:  #00e5ff;
            --accent-amber: #ffab00;
            --accent-red:   #ff1744;
            --accent-green: #00e676;
        }

        html, body, .stApp {
            background: var(--bg-primary) !important;
            color: var(--text-primary);
            font-family: "Inter", -apple-system, sans-serif;
        }

        .stApp {
            background:
                radial-gradient(ellipse at 12% 8%,  rgba(0,229,255,0.06), transparent 40%),
                radial-gradient(ellipse at 85% 20%, rgba(255,171,0,0.04), transparent 35%),
                radial-gradient(ellipse at 50% 90%, rgba(0,230,118,0.03), transparent 40%),
                var(--bg-primary) !important;
        }

        .main .block-container {
            max-width: 1380px;
            padding: 1.2rem 1.6rem 1.6rem;
        }

        /* ── Typography ─────────────────────────────── */
        h1 {
            font-family: "Inter", sans-serif !important;
            font-weight: 800 !important;
            letter-spacing: -0.03em;
            background: linear-gradient(135deg, #ffffff 0%, var(--accent-cyan) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        h2, h3 {
            font-family: "Inter", sans-serif !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
            color: #ffffff !important;
        }
        h4 {
            font-family: "Inter", sans-serif !important;
            color: var(--text-primary) !important;
        }
        p, span, label, .stMarkdown, .stText, div {
            font-family: "Inter", sans-serif;
        }

        /* ── Sidebar ────────────────────────────────── */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d1321 0%, #101829 100%) !important;
            border-right: 1px solid var(--border) !important;
        }
        section[data-testid="stSidebar"] .stMarkdown h3 {
            color: var(--accent-cyan) !important;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
        }
        section[data-testid="stSidebar"] .stRadio label {
            color: var(--text-muted) !important;
            transition: color 0.2s;
        }
        section[data-testid="stSidebar"] .stRadio label:hover {
            color: var(--accent-cyan) !important;
        }

        /* ── Metric Cards ───────────────────────────── */
        [data-testid="stMetric"] {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 14px;
            padding: 0.75rem 1rem;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            box-shadow: 0 4px 24px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.04);
            transition: border-color 0.3s, box-shadow 0.3s;
        }
        [data-testid="stMetric"]:hover {
            border-color: var(--border-glow) !important;
            box-shadow: 0 4px 32px rgba(0,229,255,0.08), inset 0 1px 0 rgba(255,255,255,0.06);
        }
        [data-testid="stMetric"] label {
            color: var(--text-muted) !important;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-family: "JetBrains Mono", monospace !important;
            font-weight: 600;
        }

        /* ── DataFrame ──────────────────────────────── */
        [data-testid="stDataFrame"] {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px;
            overflow: hidden;
        }

        /* ── Tabs ───────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {
            background: transparent;
            gap: 0;
            border-bottom: 1px solid var(--border);
        }
        .stTabs [data-baseweb="tab"] {
            color: var(--text-muted) !important;
            font-weight: 500;
            border-bottom: 2px solid transparent;
            transition: color 0.2s, border-color 0.2s;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: var(--accent-cyan) !important;
        }
        .stTabs [aria-selected="true"] {
            color: var(--accent-cyan) !important;
            border-bottom-color: var(--accent-cyan) !important;
        }

        /* ── Container borders ──────────────────────── */
        [data-testid="stVerticalBlock"] > div[data-testid="stExpander"],
        div[data-testid="stContainer"] {
            border-color: var(--border) !important;
        }

        /* ── Buttons ────────────────────────────────── */
        .stDownloadButton button {
            background: linear-gradient(135deg, rgba(0,229,255,0.12), rgba(0,229,255,0.04)) !important;
            border: 1px solid rgba(0,229,255,0.3) !important;
            color: var(--accent-cyan) !important;
            font-weight: 600;
            border-radius: 10px;
            transition: all 0.25s;
        }
        .stDownloadButton button:hover {
            background: linear-gradient(135deg, rgba(0,229,255,0.22), rgba(0,229,255,0.08)) !important;
            box-shadow: 0 0 20px rgba(0,229,255,0.12);
        }

        /* ── Slider / select ────────────────────────── */
        .stSlider label, .stMultiSelect label, .stSelectbox label {
            color: var(--text-muted) !important;
        }

        /* ── Banner ─────────────────────────────────── */
        .pm-banner {
            border: 1px solid var(--border);
            border-left: 4px solid var(--accent-cyan);
            border-radius: 12px;
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            padding: 0.75rem 1rem;
            margin-bottom: 1rem;
            color: var(--text-muted);
            font-size: 0.94rem;
            line-height: 1.5;
        }

        .pm-caption {
            color: var(--text-muted);
            font-size: 0.92rem;
            margin-top: -0.3rem;
            letter-spacing: 0.02em;
        }

        /* ── Scrollbar ──────────────────────────────── */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(100,180,255,0.15); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(100,180,255,0.25); }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data() -> dict[str, object]:
    models_dir = ROOT_DIR / "models"

    metadata = json.loads((models_dir / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((models_dir / "metrics.json").read_text(encoding="utf-8"))

    train_df, test_df = prepare_train_test_with_rul(ROOT_DIR / "data", subset=metadata["subset"])
    train_df = clip_rul(train_df, cap=metadata["rul_cap"])
    test_df = clip_rul(test_df, cap=metadata["rul_cap"])

    feature_pack = prepare_feature_matrices(
        train_df=train_df,
        test_df=test_df,
        dropped_sensors=metadata["dropped_sensors"],
        windows=tuple(metadata.get("feature_windows", FEATURE_WINDOWS)),
        include_raw_sensors=bool(metadata.get("include_raw_sensors", False)),
    )

    return {
        "metadata": metadata,
        "metrics": metrics,
        "train_df": train_df,
        "test_df": test_df,
        "feature_pack": feature_pack,
    }


@st.cache_resource
def load_models() -> tuple[object, object]:
    models_dir = ROOT_DIR / "models"
    rul_model = joblib.load(models_dir / "rul_model.pkl")
    classifier = joblib.load(models_dir / "classifier.pkl")
    return rul_model, classifier


@st.cache_resource
def build_explainer(_model) -> shap.TreeExplainer:
    return shap.TreeExplainer(_model)


def risk_level(rul_value: float) -> str:
    if rul_value < 30:
        return "High"
    if rul_value <= 70:
        return "Medium"
    return "Low"


def build_latest_fleet_predictions(
    test_frame: pd.DataFrame,
    feature_cols: list[str],
    rul_model,
    classifier,
    rul_cap: int,
    failure_threshold: int,
) -> pd.DataFrame:
    latest_idx = test_frame.groupby(ENGINE_ID_COL)[CYCLE_COL].idxmax()
    latest = test_frame.loc[latest_idx].copy().sort_values(ENGINE_ID_COL).reset_index(drop=True)

    latest["predicted_rul"] = np.clip(rul_model.predict(latest[feature_cols]), 0, rul_cap)
    latest["failure_prob_30"] = classifier.predict_proba(latest[feature_cols])[:, 1]
    latest[FAILURE_SOON_COL] = (latest["predicted_rul"] <= failure_threshold).astype(int)
    latest["risk_level"] = latest["predicted_rul"].apply(risk_level)

    return latest


def render_banner(text: str) -> None:
    st.markdown(f"<div class='pm-banner'>{text}</div>", unsafe_allow_html=True)


def render_shap_waterfall(explainer, feature_row: pd.Series, feature_cols: list[str]) -> None:
    shap_values = explainer.shap_values(feature_row[feature_cols].to_frame().T)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = float(base_value.flatten()[0])

    explanation = shap.Explanation(
        values=np.asarray(shap_values)[0],
        base_values=base_value,
        data=feature_row[feature_cols].values,
        feature_names=feature_cols,
    )

    with plt.rc_context({
        "figure.facecolor": DARK_BG,
        "axes.facecolor": DARK_BG,
        "text.color": TEXT_CLR,
        "axes.labelcolor": TEXT_CLR,
        "xtick.color": TEXT_CLR,
        "ytick.color": TEXT_CLR,
    }):
        plt.figure(figsize=(9, 5.5))
        shap.plots.waterfall(explanation, max_display=12, show=False)
        fig = plt.gcf()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)


def page_fleet(latest_fleet: pd.DataFrame) -> None:
    st.subheader("Fleet Overview")
    render_banner(
        "Filter the live fleet by cycle and risk to surface engines requiring immediate intervention. "
        "Bubble size reflects the probability of failure within 30 cycles."
    )

    high_risk_count = int((latest_fleet["predicted_rul"] < 30).sum())
    medium_risk_count = int(
        ((latest_fleet["predicted_rul"] >= 30) & (latest_fleet["predicted_rul"] <= 70)).sum()
    )
    low_risk_count = int((latest_fleet["predicted_rul"] > 70).sum())
    immediate_share = high_risk_count / max(1, len(latest_fleet))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Engines Requiring Immediate Attention", high_risk_count)
    col2.metric("Medium Risk Engines", medium_risk_count)
    col3.metric("Low Risk Engines", low_risk_count)
    col4.metric("Immediate Attention Share", f"{immediate_share:.1%}")

    with st.container(border=True):
        filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 2])
        with filter_col1:
            selected_levels = st.multiselect(
                "Risk Levels",
                options=["High", "Medium", "Low"],
                default=["High", "Medium", "Low"],
            )
        with filter_col2:
            max_cycle = int(latest_fleet[CYCLE_COL].max())
            selected_cycle_range = st.slider(
                "Cycle Range",
                min_value=0,
                max_value=max_cycle,
                value=(0, max_cycle),
            )
        with filter_col3:
            min_failure_prob = st.slider(
                "Minimum Failure Probability (30 cycles)",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
            )

    filtered_fleet = latest_fleet[
        latest_fleet["risk_level"].isin(selected_levels)
        & latest_fleet[CYCLE_COL].between(selected_cycle_range[0], selected_cycle_range[1])
        & (latest_fleet["failure_prob_30"] >= min_failure_prob)
    ].copy()

    if filtered_fleet.empty:
        st.warning("No engines match the selected filters.")
        return

    scatter_fig = px.scatter(
        filtered_fleet,
        x=CYCLE_COL,
        y="predicted_rul",
        color="risk_level",
        size="failure_prob_30",
        size_max=28,
        custom_data=[ENGINE_ID_COL, "failure_prob_30"],
        hover_data={
            ENGINE_ID_COL: True,
            CYCLE_COL: True,
            "predicted_rul": ":.1f",
            "failure_prob_30": ":.1%",
        },
        color_discrete_map=RISK_COLORS,
        title="Fleet Risk Map — Cycle vs Predicted RUL",
        labels={
            CYCLE_COL: "Current Cycle",
            "predicted_rul": "Predicted RUL",
            "risk_level": "Risk Level",
            "failure_prob_30": "Failure Probability (30 cycles)",
            ENGINE_ID_COL: "Engine ID",
        },
        template="plotly_dark",
    )
    scatter_fig.update_traces(
        marker={"line": {"width": 0.8, "color": "rgba(0,0,0,0.4)"}, "opacity": 0.92},
        hovertemplate=(
            "<b>Engine %{customdata[0]}</b><br>"
            "Cycle %{x}<br>"
            "Predicted RUL %{y:.1f}<br>"
            "Failure Prob %{customdata[1]:.1%}<extra></extra>"
        ),
    )
    scatter_fig.update_layout(**_dark_layout(
        height=480,
        legend_title_text="Risk Level",
        legend_title_font={"color": TEXT_CLR},
    ))
    _dark_axes(scatter_fig, rangeslider=True)

    risk_counts = (
        filtered_fleet["risk_level"]
        .value_counts()
        .reindex(["High", "Medium", "Low"], fill_value=0)
        .reset_index()
    )
    risk_counts.columns = ["risk_level", "count"]

    risk_fig = px.bar(
        risk_counts,
        x="count",
        y="risk_level",
        orientation="h",
        color="risk_level",
        color_discrete_map=RISK_COLORS,
        title="Risk Distribution",
        text="count",
        template="plotly_dark",
    )
    risk_fig.update_layout(**_dark_layout(
        showlegend=False,
        height=480,
        xaxis_title="Engine Count",
        yaxis_title="",
    ))
    risk_fig.update_traces(
        textposition="outside", cliponaxis=False,
        textfont={"color": TEXT_CLR},
    )
    _dark_axes(risk_fig)

    chart_col1, chart_col2 = st.columns([2, 1])
    with chart_col1:
        st.plotly_chart(scatter_fig, use_container_width=True, config=PLOTLY_CONFIG)
    with chart_col2:
        st.plotly_chart(risk_fig, use_container_width=True, config=PLOTLY_CONFIG)

    fleet_table = filtered_fleet[
        [ENGINE_ID_COL, CYCLE_COL, "predicted_rul", "risk_level", "failure_prob_30"]
    ].rename(
        columns={
            ENGINE_ID_COL: "Engine ID",
            CYCLE_COL: "Current Cycle",
            "predicted_rul": "Predicted RUL",
            "risk_level": "Risk Level",
            "failure_prob_30": "Failure Risk (%)",
        }
    )
    fleet_table["Failure Risk (%)"] = (fleet_table["Failure Risk (%)"] * 100).round(1)
    fleet_table = fleet_table.sort_values(by=["Predicted RUL", "Failure Risk (%)"])

    st.markdown("#### Engine Priority Table")
    st.dataframe(
        fleet_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Engine ID": st.column_config.NumberColumn(format="%d"),
            "Current Cycle": st.column_config.NumberColumn(format="%d"),
            "Predicted RUL": st.column_config.NumberColumn(format="%.1f cycles"),
            "Failure Risk (%)": st.column_config.ProgressColumn(
                "Failure Risk (%)",
                min_value=0.0,
                max_value=100.0,
                format="%.1f%%",
            ),
        },
    )

    csv_payload = fleet_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Filtered Fleet Snapshot",
        data=csv_payload,
        file_name="fleet_snapshot.csv",
        mime="text/csv",
    )


def page_engine(
    latest_fleet: pd.DataFrame,
    test_df: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_cols: list[str],
    dashboard_sensors: list[str],
    classifier,
    rul_model,
    explainer,
    failure_threshold: int,
    rul_cap: int,
) -> None:
    st.subheader("Single Engine Deep Dive")
    render_banner(
        "Inspect one engine in detail: compare sensor behavior and RUL trajectory, then review "
        "SHAP reasoning behind the latest model decision."
    )

    engine_ids = latest_fleet[ENGINE_ID_COL].sort_values().tolist()
    available_sensors = [column for column in test_df.columns if column.startswith("sensor_")]
    default_sensors = [sensor for sensor in dashboard_sensors if sensor in available_sensors][:3]
    if not default_sensors:
        default_sensors = available_sensors[:3]

    control_col1, control_col2 = st.columns([1, 2])
    with control_col1:
        selected_engine = st.selectbox("Engine ID", engine_ids)
    with control_col2:
        selected_sensors = st.multiselect(
            "Sensors to Visualize",
            options=available_sensors,
            default=default_sensors,
            max_selections=5,
        )
    if not selected_sensors:
        selected_sensors = default_sensors[:1]

    engine_history = test_df[test_df[ENGINE_ID_COL] == selected_engine].sort_values(CYCLE_COL)
    engine_feature_history = test_frame[test_frame[ENGINE_ID_COL] == selected_engine].sort_values(CYCLE_COL)
    latest_row = latest_fleet[latest_fleet[ENGINE_ID_COL] == selected_engine].iloc[0]
    latest_feature_row = engine_feature_history.iloc[-1]

    failure_probability = float(classifier.predict_proba(latest_feature_row[feature_cols].to_frame().T)[0, 1])
    failure_label = "HIGH" if failure_probability >= 0.5 else "LOW"

    rul_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(latest_row["predicted_rul"]),
            number={"suffix": " cycles", "font": {"color": "#ffffff", "family": "JetBrains Mono"}},
            title={"text": "Predicted RUL", "font": {"color": TEXT_CLR}},
            gauge={
                "axis": {"range": [0, float(rul_cap)], "tickcolor": TEXT_CLR, "tickfont": {"color": TEXT_CLR}},
                "bar": {"color": ACCENT_CYAN},
                "bgcolor": "rgba(15,23,42,0.4)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "rgba(255, 23, 68, 0.18)"},
                    {"range": [30, 70], "color": "rgba(255, 171, 0, 0.15)"},
                    {"range": [70, float(rul_cap)], "color": "rgba(0, 230, 118, 0.12)"},
                ],
            },
        )
    )
    rul_gauge.update_layout(**_dark_layout(height=300))

    failure_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=failure_probability,
            number={"valueformat": ".1%", "font": {"color": "#ffffff", "family": "JetBrains Mono"}},
            title={"text": f"Failure Risk — {failure_threshold} Cycles ({failure_label})", "font": {"color": TEXT_CLR}},
            gauge={
                "axis": {"range": [0, 1], "tickcolor": TEXT_CLR, "tickfont": {"color": TEXT_CLR}},
                "bar": {"color": ACCENT_RED},
                "bgcolor": "rgba(15,23,42,0.4)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 0.5], "color": "rgba(0, 230, 118, 0.12)"},
                    {"range": [0.5, 1], "color": "rgba(255, 23, 68, 0.15)"},
                ],
                "threshold": {
                    "line": {"color": ACCENT_AMBER, "width": 3},
                    "thickness": 0.75,
                    "value": 0.5,
                },
            },
        )
    )
    failure_gauge.update_layout(**_dark_layout(height=300))

    gauge_col1, gauge_col2 = st.columns(2)
    with gauge_col1:
        st.plotly_chart(rul_gauge, use_container_width=True, config=PLOTLY_CONFIG)
    with gauge_col2:
        st.plotly_chart(failure_gauge, use_container_width=True, config=PLOTLY_CONFIG)

    tab_sensors, tab_rul, tab_explain = st.tabs(["⚡ Sensor Behavior", "📉 RUL Tracking", "🔍 Explainability"])

    with tab_sensors:
        chart_df = engine_history[[CYCLE_COL, *selected_sensors]].melt(
            id_vars=[CYCLE_COL], var_name="sensor", value_name="value"
        )
        sensor_fig = px.line(
            chart_df,
            x=CYCLE_COL,
            y="value",
            color="sensor",
            title=f"Sensor Trends — Engine {selected_engine}",
            template="plotly_dark",
        )
        sensor_fig.update_traces(line={"width": 2})
        sensor_fig.update_layout(**_dark_layout(height=460))
        _dark_axes(sensor_fig, rangeslider=True)
        st.plotly_chart(sensor_fig, use_container_width=True, config=PLOTLY_CONFIG)

    with tab_rul:
        predicted_history = np.clip(
            rul_model.predict(engine_feature_history[feature_cols]),
            0,
            rul_cap,
        )
        rul_curve = pd.DataFrame(
            {
                CYCLE_COL: engine_history[CYCLE_COL].values,
                "Actual RUL": engine_history[RUL_COL].values,
                "Predicted RUL": predicted_history,
            }
        )
        rul_melt = rul_curve.melt(id_vars=[CYCLE_COL], var_name="series", value_name="rul")

        rul_fig = px.line(
            rul_melt,
            x=CYCLE_COL,
            y="rul",
            color="series",
            template="plotly_dark",
            title=f"Actual vs Predicted RUL — Engine {selected_engine}",
            color_discrete_map={"Actual RUL": ACCENT_CYAN, "Predicted RUL": ACCENT_RED},
        )
        rul_fig.update_traces(line={"width": 2.5})
        rul_fig.add_hline(
            y=failure_threshold,
            line_dash="dash",
            line_color=ACCENT_AMBER,
            annotation_text=f"Failure Threshold ({failure_threshold})",
            annotation_position="top left",
            annotation_font={"color": ACCENT_AMBER},
        )
        rul_fig.update_layout(**_dark_layout(height=450))
        _dark_axes(rul_fig)
        rul_fig.update_yaxes(title="RUL")
        st.plotly_chart(rul_fig, use_container_width=True, config=PLOTLY_CONFIG)

    with tab_explain:
        latest_shap = explainer.shap_values(latest_feature_row[feature_cols].to_frame().T)
        if isinstance(latest_shap, list):
            latest_shap = latest_shap[0]
        latest_shap_values = np.asarray(latest_shap)[0]

        top_idx = np.argsort(np.abs(latest_shap_values))[::-1][:10]
        top_shap_df = pd.DataFrame(
            {
                "feature": [feature_cols[idx] for idx in top_idx],
                "shap_value": [latest_shap_values[idx] for idx in top_idx],
            }
        )
        top_shap_df["direction"] = np.where(
            top_shap_df["shap_value"] >= 0,
            "Increases Predicted RUL",
            "Decreases Predicted RUL",
        )
        top_shap_df = top_shap_df.sort_values("shap_value")

        shap_bar = px.bar(
            top_shap_df,
            x="shap_value",
            y="feature",
            color="direction",
            orientation="h",
            template="plotly_dark",
            title="Top SHAP Contributions — Latest Cycle",
            color_discrete_map={
                "Increases Predicted RUL": ACCENT_GREEN,
                "Decreases Predicted RUL": ACCENT_RED,
            },
        )
        shap_bar.update_layout(**_dark_layout(
            height=450,
            legend_title_text="Contribution",
        ))
        shap_bar.update_xaxes(title="SHAP value impact")
        _dark_axes(shap_bar)
        st.plotly_chart(shap_bar, use_container_width=True, config=PLOTLY_CONFIG)

        st.markdown("### SHAP Waterfall (Latest Reading)")
        render_shap_waterfall(explainer, latest_feature_row, feature_cols)


def page_performance(metrics: dict, plots_dir: Path, rul_model, feature_cols: list[str]) -> None:
    st.subheader("Model Performance")
    render_banner(
        "Compare regression and classification quality side-by-side, then inspect model-derived feature "
        "importance against SHAP explainability output."
    )

    reg = metrics["regression"]["xgboost"]
    cls = metrics["classification"]["xgboost"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Regression RMSE", f"{reg['rmse']:.3f}")
    col2.metric("Regression MAE", f"{reg['mae']:.3f}")
    col3.metric("Regression R²", f"{reg['r2']:.3f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Precision", f"{cls['precision']:.3f}")
    col5.metric("Recall", f"{cls['recall']:.3f}")
    col6.metric("F1", f"{cls['f1']:.3f}")

    tab_reg, tab_cls, tab_exp = st.tabs(["📊 Regression", "🎯 Classification", "🔍 Explainability"])

    with tab_reg:
        baseline = metrics["regression"]["baseline_linear_regression"]
        rmse_gain = (baseline["rmse"] - reg["rmse"]) / baseline["rmse"]
        mae_gain = (baseline["mae"] - reg["mae"]) / baseline["mae"]

        gain_col1, gain_col2 = st.columns(2)
        gain_col1.metric("RMSE Improvement vs Baseline", f"{rmse_gain:.1%}")
        gain_col2.metric("MAE Improvement vs Baseline", f"{mae_gain:.1%}")

        reg_compare_df = pd.DataFrame(
            {
                "metric": ["RMSE", "MAE", "RMSE", "MAE"],
                "model": ["LinearRegression", "LinearRegression", "XGBoost", "XGBoost"],
                "value": [baseline["rmse"], baseline["mae"], reg["rmse"], reg["mae"]],
            }
        )
        reg_compare_fig = px.bar(
            reg_compare_df,
            x="metric",
            y="value",
            color="model",
            barmode="group",
            title="Regression Error Comparison",
            template="plotly_dark",
            text="value",
            labels={"value": "Error (Lower is Better)", "metric": "Metric"},
            color_discrete_map={"LinearRegression": "#475569", "XGBoost": ACCENT_CYAN},
        )
        reg_compare_fig.update_traces(
            texttemplate="%{text:.2f}", textposition="outside",
            textfont={"color": TEXT_CLR},
        )
        reg_compare_fig.update_layout(**_dark_layout(height=450))
        _dark_axes(reg_compare_fig)
        st.plotly_chart(reg_compare_fig, use_container_width=True, config=PLOTLY_CONFIG)

    with tab_cls:
        cm = np.asarray(metrics["classification"]["confusion_matrix"])
        cm_fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Predicted: No Failure", "Predicted: Failure"],
                y=["Actual: No Failure", "Actual: Failure"],
                text=cm,
                texttemplate="%{text}",
                textfont={"color": "#ffffff", "size": 16},
                colorscale=[[0, "#0f172a"], [0.5, "#164e63"], [1, ACCENT_CYAN]],
                showscale=False,
            )
        )
        cm_fig.update_layout(**_dark_layout(
            title="Interactive Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="Actual Label",
            height=440,
        ))
        _dark_axes(cm_fig)

        cls_profile = pd.DataFrame(
            {
                "metric": ["Precision", "Recall", "F1", "ROC-AUC"],
                "value": [cls["precision"], cls["recall"], cls["f1"], cls["roc_auc"]],
            }
        )
        radar_fig = px.line_polar(
            cls_profile,
            r="value",
            theta="metric",
            line_close=True,
            range_r=[0, 1],
            template="plotly_dark",
            title="Classification Metric Profile",
        )
        radar_fig.update_traces(
            fill="toself",
            line_color=ACCENT_CYAN,
            fillcolor="rgba(0, 229, 255, 0.12)",
        )
        radar_fig.update_layout(**_dark_layout(height=440))

        cls_col1, cls_col2 = st.columns(2)
        with cls_col1:
            st.plotly_chart(cm_fig, use_container_width=True, config=PLOTLY_CONFIG)
        with cls_col2:
            st.plotly_chart(radar_fig, use_container_width=True, config=PLOTLY_CONFIG)

    with tab_exp:
        if hasattr(rul_model, "feature_importances_"):
            importance_df = pd.DataFrame(
                {
                    "feature": feature_cols,
                    "importance": rul_model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)
            top_importance_df = importance_df.head(10).sort_values("importance", ascending=True)
            importance_fig = px.bar(
                top_importance_df,
                x="importance",
                y="feature",
                orientation="h",
                title="Top 10 Model Feature Importances",
                template="plotly_dark",
                labels={"importance": "Importance", "feature": "Feature"},
                color_discrete_sequence=[ACCENT_CYAN],
            )
            importance_fig.update_layout(**_dark_layout(height=440))
            _dark_axes(importance_fig)
            st.plotly_chart(importance_fig, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.info("Model does not expose feature importances for interactive plotting.")

        shap_path = plots_dir / "shap_summary_bar.png"
        st.markdown("#### SHAP Global Importance")
        st.image(str(shap_path), use_container_width=True)


def main() -> None:
    apply_custom_theme()
    st.title("Jet Engine Predictive Maintenance Dashboard")
    st.markdown(
        "<p class='pm-caption'>NASA CMAPSS FD001 | XGBoost RUL and Failure-Risk Forecasting with SHAP Explainability</p>",
        unsafe_allow_html=True,
    )

    data = load_data()
    rul_model, classifier = load_models()
    explainer = build_explainer(rul_model)

    metadata = data["metadata"]
    metrics = data["metrics"]
    test_df = data["test_df"]
    test_frame = data["feature_pack"]["test_frame"]
    feature_cols = metadata["feature_columns"]

    latest_fleet = build_latest_fleet_predictions(
        test_frame=test_frame,
        feature_cols=feature_cols,
        rul_model=rul_model,
        classifier=classifier,
        rul_cap=metadata["rul_cap"],
        failure_threshold=metadata["failure_threshold"],
    )

    with st.sidebar:
        st.markdown("### Navigation")
        st.markdown(
            "<span style='color:#94a3b8;font-size:0.88rem'>Inspect fleet health, engine diagnostics, or model quality.</span>",
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown("### Fleet Snapshot")
        st.metric("🔴 High Risk", int((latest_fleet["risk_level"] == "High").sum()))
        st.metric("🟡 Medium Risk", int((latest_fleet["risk_level"] == "Medium").sum()))
        st.metric("🟢 Low Risk", int((latest_fleet["risk_level"] == "Low").sum()))

    pages = {
        "🛰️  Fleet Overview": lambda: page_fleet(latest_fleet),
        "🔬  Engine Deep Dive": lambda: page_engine(
            latest_fleet=latest_fleet,
            test_df=test_df,
            test_frame=test_frame,
            feature_cols=feature_cols,
            dashboard_sensors=metadata.get("top_dashboard_sensors", ["sensor_11", "sensor_4", "sensor_12"]),
            classifier=classifier,
            rul_model=rul_model,
            explainer=explainer,
            failure_threshold=metadata["failure_threshold"],
            rul_cap=metadata["rul_cap"],
        ),
        "📈  Model Performance": lambda: page_performance(
            metrics=metrics,
            plots_dir=ROOT_DIR / "plots",
            rul_model=rul_model,
            feature_cols=feature_cols,
        ),
    }

    selected_page = st.sidebar.radio("Pages", list(pages.keys()), label_visibility="collapsed")
    pages[selected_page]()


if __name__ == "__main__":
    main()
