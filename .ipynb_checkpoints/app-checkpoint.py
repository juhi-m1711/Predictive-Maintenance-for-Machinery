"""
app.py
------
Predictive Maintenance for Machinery — Control Room
A Streamlit web app for live failure detection & diagnosis,
built on top of the trained models from the MLDS project notebook.

Run with: streamlit run app.py
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# PAGE CONFIG (must be first Streamlit call)
# ============================================================
st.set_page_config(
    page_title="Predictive Maintenance | Control Room",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# LOAD ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    with open("model_artifacts.pkl", "rb") as f:
        return pickle.load(f)

try:
    A = load_artifacts()
except FileNotFoundError:
    st.error(
        "**model_artifacts.pkl not found.** Run `python train_models.py` first "
        "(with `maintenance_dataset.csv` in the same folder) to generate the trained models, "
        "then restart this app."
    )
    st.stop()

FEATURE_COLS = A["feature_cols"]
TYPE_MAP = A["type_mapping"]
BINARY_SCALER = A["binary_scaler"]
BINARY_MODELS = A["binary_models"]
BEST_BINARY_NAME = A["best_binary_model_name"]
BEST_BINARY_MODEL = BINARY_MODELS[BEST_BINARY_NAME]
MULTI_SCALER = A["multi_scaler"]
MULTI_MODEL = A["multi_model"]
MULTI_CLASS_MAP = A["multi_class_mapping"]
FAILURE_FULL_NAMES = A["failure_full_names"]
DATA_RANGES = A["data_ranges"]
BINARY_RESULTS = A["binary_results"]

# ============================================================
# CONTROL ROOM THEME (dark, industrial, instrumentation-style)
# ============================================================
CSS = """
<style>
:root {
    --bg-deep:      #0a0e14;
    --bg-panel:     #11161f;
    --bg-panel-2:   #161c28;
    --border-dim:   #232b3a;
    --amber:        #ffb454;
    --amber-dim:    #8a6328;
    --cyan:         #5ad7e0;
    --green-ok:     #4ade80;
    --red-alarm:    #f87171;
    --text-main:    #e6ebf2;
    --text-dim:     #7c8696;
    --mono: 'JetBrains Mono', 'Courier New', monospace;
}

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

.stApp {
    background: radial-gradient(circle at 20% 0%, #0d1320 0%, var(--bg-deep) 55%);
    color: var(--text-main);
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: var(--bg-panel);
    border-right: 1px solid var(--border-dim);
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--amber);
    font-family: var(--mono);
    letter-spacing: 0.08em;
    font-size: 0.85rem;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border-dim);
    padding-bottom: 8px;
    margin-top: 4px;
}

/* ---------- Headings ---------- */
h1 {
    font-weight: 800 !important;
    letter-spacing: -0.02em;
    color: var(--text-main) !important;
}
h2, h3 { color: var(--text-main) !important; }

/* ---------- Top status strip ---------- */
.status-strip {
    display: flex;
    gap: 14px;
    align-items: center;
    padding: 10px 18px;
    background: var(--bg-panel);
    border: 1px solid var(--border-dim);
    border-radius: 8px;
    font-family: var(--mono);
    font-size: 0.78rem;
    color: var(--text-dim);
    margin-bottom: 18px;
}
.status-dot {
    width: 9px; height: 9px; border-radius: 50%;
    background: var(--green-ok);
    box-shadow: 0 0 8px var(--green-ok);
    display: inline-block;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.45; }
}

/* ---------- Panels / cards ---------- */
.panel {
    background: var(--bg-panel);
    border: 1px solid var(--border-dim);
    border-radius: 10px;
    padding: 20px 22px;
    margin-bottom: 16px;
}
.panel-label {
    font-family: var(--mono);
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 6px;
}

/* ---------- Verdict banners ---------- */
.verdict-ok {
    background: linear-gradient(135deg, rgba(74,222,128,0.12), rgba(74,222,128,0.03));
    border: 1px solid rgba(74,222,128,0.4);
    border-radius: 10px;
    padding: 28px 26px;
    text-align: center;
}
.verdict-alarm {
    background: linear-gradient(135deg, rgba(248,113,113,0.15), rgba(248,113,113,0.04));
    border: 1px solid rgba(248,113,113,0.5);
    border-radius: 10px;
    padding: 28px 26px;
    text-align: center;
    animation: alarm-glow 1.6s infinite;
}
@keyframes alarm-glow {
    0%, 100% { box-shadow: 0 0 0px rgba(248,113,113,0); }
    50% { box-shadow: 0 0 28px rgba(248,113,113,0.25); }
}
.verdict-title {
    font-family: var(--mono);
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    margin-bottom: 4px;
}
.verdict-ok .verdict-title { color: var(--green-ok); }
.verdict-alarm .verdict-title { color: var(--red-alarm); }
.verdict-sub { color: var(--text-dim); font-size: 0.88rem; }

/* ---------- Metric tiles ---------- */
[data-testid="stMetric"] {
    background: var(--bg-panel-2);
    border: 1px solid var(--border-dim);
    border-radius: 8px;
    padding: 14px 16px 10px 16px;
}
[data-testid="stMetricLabel"] {
    font-family: var(--mono);
    font-size: 0.7rem !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-dim) !important;
}
[data-testid="stMetricValue"] {
    color: var(--cyan) !important;
    font-weight: 700 !important;
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: 1px solid var(--border-dim);
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--mono);
    font-size: 0.82rem;
    letter-spacing: 0.03em;
    color: var(--text-dim);
    background: transparent;
    border-radius: 6px 6px 0 0;
}
.stTabs [aria-selected="true"] {
    color: var(--amber) !important;
    border-bottom: 2px solid var(--amber) !important;
}

/* ---------- Buttons ---------- */
.stButton button {
    background: linear-gradient(135deg, #ffb454, #e8943a);
    color: #1a1206;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.4rem;
    letter-spacing: 0.02em;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(255,180,84,0.25);
}

/* ---------- Sliders ---------- */
[data-testid="stSlider"] [role="slider"] {
    background: var(--amber) !important;
}

/* ---------- Dataframe ---------- */
[data-testid="stDataFrame"] { border: 1px solid var(--border-dim); border-radius: 8px; }

/* ---------- Divider ---------- */
hr { border-color: var(--border-dim) !important; }

/* ---------- Caption / small text ---------- */
.small-dim { color: var(--text-dim); font-size: 0.82rem; }
.mono { font-family: var(--mono); }

/* ---------- Gauge label override ---------- */
.gauge-caption {
    text-align: center;
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--text-dim);
    margin-top: -10px;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ============================================================
# HELPERS
# ============================================================
def predict_failure_probability(machine_type, air_temp, process_temp, rpm, torque, tool_wear):
    type_encoded = TYPE_MAP[machine_type]
    new_data = pd.DataFrame({
        "Type_Encoded": [type_encoded],
        "Air temperature [K]": [air_temp],
        "Process temperature [K]": [process_temp],
        "Rotational speed [rpm]": [rpm],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear],
    })[FEATURE_COLS]
    scaled = BINARY_SCALER.transform(new_data)
    pred = BEST_BINARY_MODEL.predict(scaled)[0]
    proba = BEST_BINARY_MODEL.predict_proba(scaled)[0][1]
    return int(pred), float(proba), new_data

def predict_failure_type(machine_type, air_temp, process_temp, rpm, torque, tool_wear):
    type_encoded = TYPE_MAP[machine_type]
    new_data = pd.DataFrame({
        "Type_Encoded": [type_encoded],
        "Air temperature [K]": [air_temp],
        "Process temperature [K]": [process_temp],
        "Rotational speed [rpm]": [rpm],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear],
    })[FEATURE_COLS]
    scaled = MULTI_SCALER.transform(new_data)
    pred_idx = MULTI_MODEL.predict(scaled)[0]
    proba = MULTI_MODEL.predict_proba(scaled)[0]
    code = MULTI_CLASS_MAP[pred_idx]
    class_probs = {MULTI_CLASS_MAP[i]: float(p) for i, p in enumerate(proba)}
    return code, class_probs

def gauge(value, label, vmin, vmax, suffix="", danger_zone=None):
    """Render a compact instrumentation-style gauge."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": suffix, "font": {"color": "#e6ebf2", "size": 26, "family": "JetBrains Mono"}},
        gauge={
            "axis": {"range": [vmin, vmax], "tickcolor": "#7c8696", "tickfont": {"color": "#7c8696", "size": 9}},
            "bar": {"color": "#5ad7e0", "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": (
                [{"range": danger_zone, "color": "rgba(248,113,113,0.18)"}] if danger_zone else []
            ),
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        height=170,
        margin=dict(l=18, r=18, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# ============================================================
# SIDEBAR — SENSOR CONTROL PANEL
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Sensor Input Panel")
    st.markdown(
        "<p class='small-dim'>Set live readings for the machine you want to assess.</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    machine_type = st.selectbox(
        "Product Quality Type",
        options=list(TYPE_MAP.keys()),
        index=list(TYPE_MAP.keys()).index("M") if "M" in TYPE_MAP else 0,
        help="L = Low, M = Medium, H = High grade product line.",
    )

    air_lo, air_hi, air_mean = DATA_RANGES["Air temperature [K]"]
    air_temp = st.slider("Air Temperature (K)", float(air_lo - 2), float(air_hi + 2), float(round(air_mean, 1)), 0.1)

    proc_lo, proc_hi, proc_mean = DATA_RANGES["Process temperature [K]"]
    process_temp = st.slider("Process Temperature (K)", float(proc_lo - 2), float(proc_hi + 2), float(round(proc_mean, 1)), 0.1)

    rpm_lo, rpm_hi, rpm_mean = DATA_RANGES["Rotational speed [rpm]"]
    rpm = st.slider("Rotational Speed (rpm)", float(rpm_lo - 100), float(rpm_hi + 200), float(round(rpm_mean)), 1.0)

    torque_lo, torque_hi, torque_mean = DATA_RANGES["Torque [Nm]"]
    torque = st.slider("Torque (Nm)", float(max(0, torque_lo - 5)), float(torque_hi + 5), float(round(torque_mean, 1)), 0.1)

    wear_lo, wear_hi, wear_mean = DATA_RANGES["Tool wear [min]"]
    tool_wear = st.slider("Tool Wear (min)", float(wear_lo), float(wear_hi + 20), float(round(wear_mean)), 1.0)

    st.markdown("---")
    run = st.button("▶  Run Diagnostic", use_container_width=True)

    st.markdown("---")
    st.markdown(
        f"<p class='small-dim mono'>MODEL: {BEST_BINARY_NAME}<br>"
        f"DATASET: AI4I 2020 ({A['n_rows']:,} records)<br>"
        f"BASE FAILURE RATE: {A['overall_failure_rate']*100:.2f}%</p>",
        unsafe_allow_html=True
    )

# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="status-strip">
        <span class="status-dot"></span>
        <span>SYSTEM ONLINE</span>
        <span style="color:#232b3a">|</span>
        <span>PREDICTIVE MAINTENANCE ENGINE v1.0</span>
        <span style="color:#232b3a">|</span>
        <span>MODEL: RANDOM FOREST CLASSIFIER</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("# ⚙️ Predictive Maintenance Control Room")
st.markdown(
    "<p class='small-dim'>Live machine health diagnostics powered by supervised machine learning, "
    "trained on the AI4I 2020 Predictive Maintenance Dataset.</p>",
    unsafe_allow_html=True
)
st.markdown("")

# ============================================================
# TABS
# ============================================================
tab_diag, tab_sensors, tab_model, tab_about = st.tabs(
    ["🩺  DIAGNOSTIC", "📡  SENSOR READOUT", "📊  MODEL PERFORMANCE", "ℹ️  ABOUT"]
)

# ------------------------------------------------------------
# TAB 1 — DIAGNOSTIC (the main predict experience)
# ------------------------------------------------------------
with tab_diag:
    if not run:
        st.markdown(
            """
            <div class="panel" style="text-align:center; padding: 50px 20px;">
                <p style="font-size:1.1rem; color:#7c8696; margin-bottom:6px;">No diagnostic run yet</p>
                <p class="small-dim">Set sensor readings in the left panel and click <b style="color:#ffb454;">Run Diagnostic</b> to assess this machine.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        pred, proba, input_df = predict_failure_probability(
            machine_type, air_temp, process_temp, rpm, torque, tool_wear
        )

        col_verdict, col_gauge = st.columns([1.4, 1])

        with col_verdict:
            if pred == 1:
                st.markdown(
                    f"""
                    <div class="verdict-alarm">
                        <div class="verdict-title">⚠ FAILURE PREDICTED</div>
                        <div class="verdict-sub">Model confidence: {proba*100:.1f}% probability of failure</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="verdict-ok">
                        <div class="verdict-title">✓ MACHINE HEALTHY</div>
                        <div class="verdict-sub">Model confidence: {(1-proba)*100:.1f}% probability of normal operation</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with col_gauge:
            fig = gauge(proba * 100, "Failure Risk", 0, 100, suffix="%", danger_zone=[70, 100])
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown("<p class='gauge-caption'>FAILURE PROBABILITY</p>", unsafe_allow_html=True)

        st.markdown("")

        # ---- If failure predicted, run stage 2 diagnosis ----
        if pred == 1:
            st.markdown("### 🔧 Failure Type Diagnosis")
            st.markdown(
                "<p class='small-dim'>Stage 2 model — identifying the most likely root cause to guide maintenance action.</p>",
                unsafe_allow_html=True
            )

            code, class_probs = predict_failure_type(
                machine_type, air_temp, process_temp, rpm, torque, tool_wear
            )
            full_name = FAILURE_FULL_NAMES.get(code, code)

            col_diag1, col_diag2 = st.columns([1, 1.3])

            with col_diag1:
                st.markdown(
                    f"""
                    <div class="panel">
                        <div class="panel-label">Most Likely Cause</div>
                        <div style="font-size:1.4rem; font-weight:700; color:#ffb454; font-family:'JetBrains Mono';">{code}</div>
                        <div style="color:#e6ebf2; margin-top:2px;">{full_name}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col_diag2:
                probs_df = pd.DataFrame({
                    "Failure Type": [f"{k} — {FAILURE_FULL_NAMES.get(k,k)}" for k in class_probs.keys()],
                    "Probability": list(class_probs.values())
                }).sort_values("Probability", ascending=True)

                fig_bar = px.bar(
                    probs_df, x="Probability", y="Failure Type", orientation="h",
                    color="Probability", color_continuous_scale=["#161c28", "#ffb454"],
                )
                fig_bar.update_layout(
                    height=200, margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e6ebf2", showlegend=False, coloraxis_showscale=False,
                    xaxis=dict(range=[0,1], gridcolor="#232b3a"),
                )
                st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

            st.info(
                "ℹ️ **RNF (Random Failure)** is, by the dataset's own design, not tied to any sensor pattern — "
                "so this model will rarely (and shouldn't be expected to) predict it. This reflects a genuine "
                "real-world limit: some failures truly are unpredictable from sensor data alone.",
                icon="ℹ️"
            )

        # ---- Input summary ----
        with st.expander("📋 View input readings sent to the model"):
            st.dataframe(input_df.rename(columns={"Type_Encoded": "Type (encoded)"}), use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# TAB 2 — SENSOR READOUT (gauges for context / feel of a control room)
# ------------------------------------------------------------
with tab_sensors:
    st.markdown("### 📡 Live Sensor Readout")
    st.markdown("<p class='small-dim'>Current input values plotted against the operating range observed in the training data.</p>", unsafe_allow_html=True)
    st.markdown("")

    g1, g2, g3 = st.columns(3)
    with g1:
        lo, hi, _ = DATA_RANGES["Air temperature [K]"]
        st.plotly_chart(gauge(air_temp, "Air Temp", lo-2, hi+2, suffix="K"), use_container_width=True, config={"displayModeBar": False})
        st.markdown("<p class='gauge-caption'>AIR TEMPERATURE</p>", unsafe_allow_html=True)
    with g2:
        lo, hi, _ = DATA_RANGES["Process temperature [K]"]
        st.plotly_chart(gauge(process_temp, "Process Temp", lo-2, hi+2, suffix="K"), use_container_width=True, config={"displayModeBar": False})
        st.markdown("<p class='gauge-caption'>PROCESS TEMPERATURE</p>", unsafe_allow_html=True)
    with g3:
        lo, hi, _ = DATA_RANGES["Rotational speed [rpm]"]
        st.plotly_chart(gauge(rpm, "RPM", lo-100, hi+200, suffix=""), use_container_width=True, config={"displayModeBar": False})
        st.markdown("<p class='gauge-caption'>ROTATIONAL SPEED</p>", unsafe_allow_html=True)

    g4, g5, g6 = st.columns(3)
    with g4:
        lo, hi, _ = DATA_RANGES["Torque [Nm]"]
        st.plotly_chart(gauge(torque, "Torque", max(0,lo-5), hi+5, suffix="Nm"), use_container_width=True, config={"displayModeBar": False})
        st.markdown("<p class='gauge-caption'>TORQUE</p>", unsafe_allow_html=True)
    with g5:
        lo, hi, _ = DATA_RANGES["Tool wear [min]"]
        st.plotly_chart(gauge(tool_wear, "Tool Wear", lo, hi+20, suffix="m", danger_zone=[200, hi+20]), use_container_width=True, config={"displayModeBar": False})
        st.markdown("<p class='gauge-caption'>TOOL WEAR</p>", unsafe_allow_html=True)
    with g6:
        type_full = {"L": "Low grade", "M": "Medium grade", "H": "High grade"}.get(machine_type, "")
        st.markdown(
            f"""
            <div class="panel" style="height:170px; display:flex; flex-direction:column; justify-content:center;">
                <div class="panel-label">Product Type</div>
                <div style="font-size:2rem; font-weight:800; color:#5ad7e0; font-family:'JetBrains Mono';">{machine_type}</div>
                <div class="small-dim">{type_full}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<p class='gauge-caption'>QUALITY TYPE</p>", unsafe_allow_html=True)

# ------------------------------------------------------------
# TAB 3 — MODEL PERFORMANCE
# ------------------------------------------------------------
with tab_model:
    st.markdown("### 📊 Model Performance Summary")
    st.markdown(
        "<p class='small-dim'>How the candidate models performed on the held-out test set "
        "(never seen during training). Random Forest was selected as the production model "
        "for its strongest recall/F1 balance — critical in maintenance, where missing a real "
        "failure is far costlier than a false alarm.</p>",
        unsafe_allow_html=True
    )
    st.markdown("")

    rows = []
    for name, res in BINARY_RESULTS.items():
        m = res["metrics"]
        rows.append({
            "Model": name,
            "Accuracy": m["Accuracy"],
            "Precision": m["Precision"],
            "Recall": m["Recall"],
            "F1 Score": m["F1 Score"],
            "ROC-AUC": m["ROC-AUC"],
        })
    comp_df = pd.DataFrame(rows).set_index("Model")

    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.dataframe(comp_df.style.format("{:.3f}").highlight_max(axis=0, color="#1c3a2e"), use_container_width=True)
        st.markdown(
            f"<p class='small-dim'>✓ Production model: <b style='color:#ffb454;'>{BEST_BINARY_NAME}</b></p>",
            unsafe_allow_html=True
        )

    with c2:
        plot_df = comp_df.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig_cmp = px.bar(
            plot_df, x="Metric", y="Score", color="Model", barmode="group",
            color_discrete_sequence=["#5ad7e0", "#ffb454", "#4ade80"]
        )
        fig_cmp.update_layout(
            height=320, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e6ebf2", legend=dict(orientation="h", y=1.15),
            yaxis=dict(gridcolor="#232b3a"), xaxis=dict(gridcolor="#232b3a"),
        )
        st.plotly_chart(fig_cmp, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")
    st.markdown("### 🎯 Feature Importance — What Drives a Failure Prediction")
    imp = A["binary_feature_importance"].sort_values(ascending=True)
    fig_imp = px.bar(
        x=imp.values, y=imp.index, orientation="h",
        color=imp.values, color_continuous_scale=["#161c28", "#5ad7e0"]
    )
    fig_imp.update_layout(
        height=280, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e6ebf2", showlegend=False, coloraxis_showscale=False,
        xaxis=dict(title="Importance", gridcolor="#232b3a"), yaxis=dict(title=""),
    )
    st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")
    st.markdown("### 🧩 Failure Type Classification Report")
    st.markdown("<p class='small-dim'>Performance of the Stage 2 multiclass model on machines that actually failed.</p>", unsafe_allow_html=True)
    multi_rep_df = pd.DataFrame(A["multi_report"]).T
    keep_idx = [i for i in multi_rep_df.index if i in list(FAILURE_FULL_NAMES.keys()) + ["accuracy", "macro avg", "weighted avg"]]
    multi_rep_df = multi_rep_df.loc[keep_idx]
    st.dataframe(multi_rep_df.style.format("{:.3f}"), use_container_width=True)

# ------------------------------------------------------------
# TAB 4 — ABOUT
# ------------------------------------------------------------
with tab_about:
    st.markdown("### ℹ️ About This Project")
    st.markdown(
        """
        <div class="panel">
        <p><b>Predictive Maintenance for Machinery</b> is an MLDS project that uses supervised machine
        learning to anticipate equipment failure before it happens, rather than relying on reactive or
        fixed-schedule maintenance.</p>

        <p><b>Pipeline:</b></p>
        <ol>
            <li><b>Stage 1 — Failure Detection (Binary):</b> Logistic Regression, Decision Tree, and
            Random Forest were trained and compared. The training data was rebalanced with SMOTE to
            counter the dataset's severe class imbalance (~3.4% failure rate), while evaluation was
            kept on the original, untouched test set for an honest read of real-world performance.</li>
            <li><b>Stage 2 — Failure Diagnosis (Multiclass):</b> For machines flagged as likely to fail,
            a second Random Forest model identifies the most probable failure type — Tool Wear, Heat
            Dissipation, Power, Overstrain, or Random Failure.</li>
        </ol>

        <p><b>Dataset:</b> AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository
        (Matzka, 2020) — 10,000 synthetic but realistic records of milling machine operation.</p>

        <p><b>Why Recall matters here:</b> In a maintenance context, missing a real failure (false
        negative) is far costlier than a false alarm (false positive) — a missed failure can mean
        unplanned downtime or a safety incident. So the production model was chosen for its recall/F1
        balance, not raw accuracy.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🛠 How to Run This App Locally")
    st.code(
        "pip install streamlit pandas numpy scikit-learn imbalanced-learn plotly\n"
        "python train_models.py    # one-time: trains models, saves model_artifacts.pkl\n"
        "streamlit run app.py",
        language="bash"
    )

    st.markdown(
        "<p class='small-dim'>Built as part of an MLDS coursework submission. "
        "Models are trained once via <code>train_models.py</code> and loaded by this app "
        "for instant predictions — no retraining happens on each run.</p>",
        unsafe_allow_html=True
    )