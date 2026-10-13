"""
dashboard.py
-------------
Interactive Streamlit Web Dashboard for Adaptive AI-Driven CPU Scheduling.

Sections:
    ① Sidebar Controls       – Sliders for simulation parameters.
    ② Run Button             – Triggers ML classification + simulation.
    ③ AI Insight Cards       – ML label, confidence, selected policy.
    ④ Gantt Chart            – Plotly timeline of process execution blocks.
    ⑤ Process Metrics Table  – Per-process completion / waiting / turnaround times.
    ⑥ Comparison Bar Charts  – AWT and ATAT across all five schedulers.

Run with:
    streamlit run dashboard.py
"""

import sys
import os

# Ensure the project root is on PYTHONPATH so all modules resolve.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from core.simulation_engine import SimulationEngine, generate_processes
from scheduler.adaptive_scheduler import AdaptiveScheduler
from metrics.calculator import (
    processes_to_dataframe,
    comparison_dataframe,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Adaptive AI CPU Scheduler",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark glassmorphism theme
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #0a0e1a;
    color: #e2e8f0;
}

/* ── Gradient hero header ── */
.hero-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 0.2rem;
}

.hero-sub {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 400;
    margin-bottom: 2rem;
}

/* ── Metric cards ── */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(102,126,234,0.25);
}
.metric-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 1.9rem;
    font-weight: 700;
    line-height: 1;
}
.metric-sub {
    font-size: 0.82rem;
    color: #94a3b8;
    margin-top: 0.3rem;
}

/* Colour variants */
.card-purple .metric-value { color: #a78bfa; }
.card-cyan   .metric-value { color: #67e8f9; }
.card-green  .metric-value { color: #6ee7b7; }
.card-orange .metric-value { color: #fbbf24; }
.card-pink   .metric-value { color: #f9a8d4; }

/* ── Section headings ── */
.section-heading {
    font-size: 1.25rem;
    font-weight: 700;
    color: #e2e8f0;
    border-left: 4px solid #667eea;
    padding-left: 0.75rem;
    margin: 1.8rem 0 1rem 0;
}

/* ── Justification box ── */
.justification-box {
    background: rgba(102,126,234,0.12);
    border: 1px solid rgba(102,126,234,0.30);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    font-size: 0.92rem;
    color: #c7d2fe;
    line-height: 1.7;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(15,20,40,0.98);
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* ── Tables ── */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 2rem;
    font-size: 1rem;
    font-weight: 600;
    width: 100%;
    transition: opacity 0.2s ease, transform 0.1s ease;
}
.stButton > button:hover {
    opacity: 0.90;
    transform: translateY(-1px);
}

/* ── Confidence bar ── */
.conf-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 99px;
    height: 8px;
    margin-top: 8px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 8px;
    border-radius: 99px;
    background: linear-gradient(90deg, #667eea, #f093fb);
    transition: width 0.8s ease;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.07); margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Colour palette for Gantt chart (one colour per process, cycling)
# ---------------------------------------------------------------------------

PROCESS_COLORS = [
    "#667eea", "#f093fb", "#4ade80", "#fbbf24", "#67e8f9",
    "#f87171", "#a78bfa", "#34d399", "#fb923c", "#38bdf8",
    "#e879f9", "#86efac", "#fde68a", "#7dd3fc", "#fca5a5",
]


def _get_color(pid: int) -> str:
    return PROCESS_COLORS[(pid - 1) % len(PROCESS_COLORS)]


# ---------------------------------------------------------------------------
# Gantt chart builder
# ---------------------------------------------------------------------------

def build_gantt(timeline, processes, algorithm_name: str) -> go.Figure:
    """
    Build an interactive Plotly Gantt chart from execution timeline intervals.

    Args:
        timeline       : List of (pid, start, end) tuples.
        processes      : List of Process objects (for metadata).
        algorithm_name : Title annotation.

    Returns:
        Plotly Figure.
    """
    if not timeline:
        fig = go.Figure()
        fig.update_layout(title="No data", template="plotly_dark")
        return fig

    pid_map = {p.pid: p for p in processes}
    rows = []
    for pid, start, end in timeline:
        proc = pid_map.get(pid)
        io_f = proc.io_frequency if proc else 0.0
        rows.append(dict(
            Task=f"P{pid}",
            Start=start,
            Finish=end,
            PID=pid,
            Duration=end - start,
            IOFreq=round(io_f, 3),
        ))

    df = pd.DataFrame(rows)

    fig = go.Figure()

    for _, row in df.iterrows():
        pid = row["PID"]
        color = _get_color(pid)
        fig.add_trace(go.Bar(
            x=[row["Duration"]],
            y=[f"P{pid}"],
            base=[row["Start"]],
            orientation="h",
            marker=dict(
                color=color,
                line=dict(color="rgba(0,0,0,0.4)", width=1),
                opacity=0.88,
            ),
            text=f"  P{pid} [{row['Start']}→{row['Finish']}]",
            textposition="inside",
            insidetextanchor="start",
            hovertemplate=(
                f"<b>Process P{pid}</b><br>"
                f"Start: {row['Start']}<br>"
                f"End: {row['Finish']}<br>"
                f"Duration: {row['Duration']} ticks<br>"
                f"I/O Freq: {row['IOFreq']}"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    # Add a single invisible legend entry per PID for colour reference.
    seen_pids = set()
    for _, row in df.drop_duplicates("PID").iterrows():
        pid = row["PID"]
        if pid not in seen_pids:
            fig.add_trace(go.Bar(
                x=[0], y=[f"P{pid}"],
                orientation="h",
                marker=dict(color=_get_color(pid)),
                name=f"P{pid}",
                showlegend=True,
                visible="legendonly",
            ))
            seen_pids.add(pid)

    unique_procs = sorted(df["PID"].unique())
    y_labels = [f"P{pid}" for pid in unique_procs]

    fig.update_layout(
        title=dict(
            text=f"<b>Gantt Chart — {algorithm_name}</b>",
            font=dict(size=16, color="#e2e8f0"),
            x=0.0,
        ),
        xaxis=dict(
            title="Time (ticks)",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.07)",
            color="#94a3b8",
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            title="Process",
            categoryorder="array",
            categoryarray=y_labels,
            color="#94a3b8",
            tickfont=dict(size=12),
        ),
        barmode="stack",
        paper_bgcolor="rgba(10,14,26,0)",
        plot_bgcolor="rgba(10,14,26,0.4)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
        height=max(280, 60 + len(unique_procs) * 42),
        margin=dict(l=60, r=30, t=70, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Comparison bar chart builder
# ---------------------------------------------------------------------------

def build_comparison_chart(all_results: dict) -> go.Figure:
    """
    Side-by-side bar chart comparing AWT and ATAT across all schedulers.
    """
    labels, awt_vals, atat_vals = [], [], []

    for key, res in all_results.items():
        if "error" in res:
            continue
        algo_short = {
            "FCFS": "FCFS",
            "SJF": "SJF",
            "RR": "Round Robin",
            "Priority": "Priority",
            "Adaptive": "🧠 Adaptive AI",
        }.get(key, key)
        labels.append(algo_short)
        awt_vals.append(res.get("avg_waiting_time", 0.0))
        atat_vals.append(res.get("avg_turnaround_time", 0.0))

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Avg Waiting Time",
        x=labels,
        y=awt_vals,
        marker=dict(
            color=["#667eea"] * (len(labels) - 1) + ["#f093fb"],
            line=dict(width=0),
            opacity=0.88,
        ),
        text=[f"{v:.2f}" for v in awt_vals],
        textposition="outside",
        textfont=dict(size=11, color="#e2e8f0"),
    ))

    fig.add_trace(go.Bar(
        name="Avg Turnaround Time",
        x=labels,
        y=atat_vals,
        marker=dict(
            color=["#4ade80"] * (len(labels) - 1) + ["#fbbf24"],
            line=dict(width=0),
            opacity=0.88,
        ),
        text=[f"{v:.2f}" for v in atat_vals],
        textposition="outside",
        textfont=dict(size=11, color="#e2e8f0"),
    ))

    fig.update_layout(
        barmode="group",
        title=dict(
            text="<b>Scheduler Comparison — AWT vs ATAT</b>",
            font=dict(size=16, color="#e2e8f0"),
            x=0.0,
        ),
        xaxis=dict(color="#94a3b8", tickfont=dict(size=12)),
        yaxis=dict(
            title="Time (ticks)",
            color="#94a3b8",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.07)",
        ),
        paper_bgcolor="rgba(10,14,26,0)",
        plot_bgcolor="rgba(10,14,26,0.4)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12),
        ),
        height=420,
        margin=dict(l=60, r=30, t=70, b=60),
    )
    return fig


# ---------------------------------------------------------------------------
# Metric card HTML builder
# ---------------------------------------------------------------------------

def metric_card(label: str, value: str, sub: str = "", css_class: str = "card-purple") -> str:
    return f"""
    <div class="metric-card {css_class}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
    </div>
    """


def confidence_bar(conf: float) -> str:
    pct = int(conf * 100)
    return f"""
    <div class="conf-bar-bg">
        <div class="conf-bar-fill" style="width:{pct}%;"></div>
    </div>
    """


# ---------------------------------------------------------------------------
# ── SIDEBAR ──
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Simulation Controls")
    st.markdown("---")

    process_count = st.slider(
        "Process Count",
        min_value=3, max_value=30, value=10, step=1,
        help="Number of processes to generate for this simulation batch.",
    )
    max_arrival = st.slider(
        "Max Arrival Time",
        min_value=0, max_value=50, value=20, step=1,
        help="Upper bound (ticks) for random process arrival times.",
    )
    burst_min = st.slider(
        "Min Burst Time",
        min_value=1, max_value=20, value=2, step=1,
    )
    burst_max = st.slider(
        "Max Burst Time",
        min_value=2, max_value=80, value=20, step=1,
    )
    io_probability = st.slider(
        "I/O Probability",
        min_value=0.0, max_value=1.0, value=0.4, step=0.05,
        help="Fraction of processes that are I/O-heavy.",
    )
    rr_quantum = st.slider(
        "Round Robin Quantum",
        min_value=1, max_value=20, value=4, step=1,
        help="Base time quantum for Round Robin (and adaptive RR).",
    )
    random_seed = st.number_input(
        "Random Seed (0 = random)",
        min_value=0, max_value=9999, value=42, step=1,
    )

    st.markdown("---")
    st.markdown("### 🤖 ML Model")

    retrain_btn = st.button("🔄 Retrain ML Model")
    if retrain_btn:
        with st.spinner("Generating dataset & training model…"):
            try:
                from ml.data_generator import generate_dataset
                from ml.train_model import train
                from ml.predictor import reload_model

                data_path = os.path.join(_ROOT, "data", "workload_dataset.csv")
                model_path = os.path.join(_ROOT, "ml", "model.joblib")
                generate_dataset(save=True, output_path=data_path)
                train(data_path=data_path, model_path=model_path)
                reload_model()
                st.success("✅ Model retrained successfully!")
            except Exception as exc:
                st.error(f"Training failed: {exc}")

    st.markdown("---")
    st.caption("Adaptive AI CPU Scheduler v1.0")
    st.caption("Powered by scikit-learn + Streamlit")


# ---------------------------------------------------------------------------
# ── HERO HEADER ──
# ---------------------------------------------------------------------------

st.markdown('<div class="hero-header">🧠 Adaptive AI-Driven CPU Scheduler</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Machine Learning–powered workload classification with real-time '
    'scheduling policy adaptation. Combines scikit-learn Random Forest classification '
    'with FCFS, SJF, Round Robin, and Priority algorithms.</div>',
    unsafe_allow_html=True,
)

st.markdown("---")

# ---------------------------------------------------------------------------
# ── RUN BUTTON ──
# ---------------------------------------------------------------------------

run_col, info_col = st.columns([1, 3])
with run_col:
    run_btn = st.button("▶ Run Simulation & AI Adaptation", use_container_width=True)

with info_col:
    st.info(
        f"Ready to simulate **{process_count} processes** | "
        f"Burst range [{burst_min}–{burst_max}] | "
        f"I/O probability {io_probability:.0%} | "
        f"RR Quantum {rr_quantum}"
    )

# ---------------------------------------------------------------------------
# ── SIMULATION LOGIC ──
# ---------------------------------------------------------------------------

if run_btn:
    seed_val = int(random_seed) if random_seed > 0 else None

    with st.spinner("🔄 Generating processes & running AI scheduling pipeline…"):
        try:
            # Generate process batch.
            processes = generate_processes(
                count=process_count,
                max_arrival_time=max_arrival,
                burst_time_range=(burst_min, burst_max),
                io_probability=io_probability,
                seed=seed_val,
            )

            # Run adaptive (AI) scheduler.
            adaptive = AdaptiveScheduler(base_quantum=rr_quantum)
            adaptive_result = adaptive.run(processes, override_quantum=None)

            # Run all classic schedulers for comparison.
            engine = SimulationEngine(processes)
            all_classic = engine.run_all(time_quantum=rr_quantum)

            # Merge all results for comparison chart.
            all_results = {**all_classic, "Adaptive": adaptive_result}

        except Exception as exc:
            st.error(f"❌ Simulation failed: {exc}")
            st.exception(exc)
            st.stop()

    # ── Store in session for persistence across re-renders ──
    st.session_state["adaptive_result"] = adaptive_result
    st.session_state["all_results"] = all_results
    st.session_state["processes"] = processes


# Retrieve from session state (so results persist after widget interactions).
if "adaptive_result" not in st.session_state:
    st.markdown(
        """
        <div style="text-align:center; padding: 4rem; color:#475569;">
            <div style="font-size:4rem;">🚀</div>
            <div style="font-size:1.2rem; font-weight:600; margin-top:1rem;">
                Configure parameters in the sidebar and click <b>Run Simulation</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

adaptive_result = st.session_state["adaptive_result"]
all_results = st.session_state["all_results"]
processes = st.session_state["processes"]

prediction = adaptive_result["prediction"]
label = prediction["label"]
confidence = prediction["confidence"]
selected_algo = adaptive_result["selected_algorithm"]
justification = adaptive_result["justification"]
time_quantum_used = adaptive_result["time_quantum"]

# ---------------------------------------------------------------------------
# ── SECTION 1: AI INSIGHT CARDS ──
# ---------------------------------------------------------------------------

st.markdown('<div class="section-heading">🤖 AI Classification & Policy Decision</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)

# Emoji per class
class_emoji = {"CPU-bound": "⚡", "I/O-bound": "💾", "Mixed": "🔀"}.get(label, "🧠")

with c1:
    st.markdown(
        metric_card("Workload Class", f"{class_emoji} {label}", sub="ML model prediction", css_class="card-purple"),
        unsafe_allow_html=True,
    )
with c2:
    conf_pct = f"{confidence * 100:.1f}%"
    st.markdown(
        metric_card("Confidence", conf_pct, sub=confidence_bar(confidence), css_class="card-cyan"),
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        metric_card("Selected Policy", selected_algo.split("(")[0].strip(), sub=f"Q={time_quantum_used} ticks", css_class="card-green"),
        unsafe_allow_html=True,
    )
with c4:
    awt = adaptive_result.get("avg_waiting_time", 0.0)
    st.markdown(
        metric_card("Avg Waiting Time", f"{awt:.2f}", sub="ticks (adaptive)", css_class="card-orange"),
        unsafe_allow_html=True,
    )
with c5:
    atat = adaptive_result.get("avg_turnaround_time", 0.0)
    st.markdown(
        metric_card("Avg Turnaround", f"{atat:.2f}", sub="ticks (adaptive)", css_class="card-pink"),
        unsafe_allow_html=True,
    )

# Justification box
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    f'<div class="justification-box">💡 <b>Decision Rationale:</b> {justification}</div>',
    unsafe_allow_html=True,
)

# Probability breakdown
st.markdown("<br>", unsafe_allow_html=True)
prob_cols = st.columns(3)
probs = prediction.get("probabilities", {})
for idx, (cls_name, prob_val) in enumerate(probs.items()):
    with prob_cols[idx]:
        st.metric(
            label=cls_name,
            value=f"{prob_val * 100:.1f}%",
            delta=None,
        )
        st.progress(float(prob_val))

st.markdown("---")

# ---------------------------------------------------------------------------
# ── SECTION 2: GANTT CHART ──
# ---------------------------------------------------------------------------

st.markdown('<div class="section-heading">📊 Execution Gantt Chart</div>', unsafe_allow_html=True)

gantt_tab1, gantt_tab2 = st.tabs(["🧠 Adaptive AI Scheduler", "🔁 Compare All Algorithms"])

with gantt_tab1:
    gantt_fig = build_gantt(
        timeline=adaptive_result["timeline"],
        processes=adaptive_result["processes"],
        algorithm_name=adaptive_result["selected_algorithm"],
    )
    st.plotly_chart(gantt_fig, use_container_width=True, key="gantt_adaptive")

with gantt_tab2:
    def _gantt_label(k):
        if k == "Adaptive":
            return f"🧠 AI Adaptive → {all_results[k].get('algorithm', k)}"
        return all_results[k].get("algorithm", k)

    algo_select = st.selectbox(
        "Select Algorithm to View Gantt",
        options=list(all_results.keys()),
        format_func=_gantt_label,
    )
    sel_result = all_results[algo_select]
    if "error" in sel_result:
        st.error(f"Error: {sel_result['error']}")
    else:
        fig2 = build_gantt(
            timeline=sel_result["timeline"],
            processes=sel_result["processes"],
            algorithm_name=_gantt_label(algo_select),
        )
        st.plotly_chart(fig2, use_container_width=True, key=f"gantt_{algo_select}")

st.markdown("---")

# ---------------------------------------------------------------------------
# ── SECTION 3: PROCESS METRICS TABLE ──
# ---------------------------------------------------------------------------

st.markdown('<div class="section-heading">📋 Process-Level Metrics</div>', unsafe_allow_html=True)

table_tab1, table_tab2 = st.tabs(["🧠 Adaptive AI Scheduler", "📊 All Classic Schedulers"])

def _highlight_col(col):
    """Colour-scale a column from green (low) to red (high) using pure CSS — no matplotlib needed."""
    if col.max() == col.min():
        return [""] * len(col)
    norm = (col - col.min()) / (col.max() - col.min())   # 0 = best (green), 1 = worst (red)
    styles = []
    for v in norm:
        r = int(220 * v + 40 * (1 - v))
        g = int(40 * v + 180 * (1 - v))
        b = 80
        styles.append(f"background-color: rgba({r},{g},{b},0.30); color: #e2e8f0;")
    return styles

with table_tab1:
    from metrics.calculator import processes_to_dataframe
    proc_df = processes_to_dataframe(adaptive_result["processes"])
    st.dataframe(
        proc_df.style
            .format(precision=2)
            .apply(_highlight_col, subset=["Waiting Time", "Turnaround Time"]),
        use_container_width=True,
        height=min(600, 80 + len(proc_df) * 38),
    )

with table_tab2:
    for algo_key, res in all_results.items():
        if "error" in res:
            continue
        with st.expander(f"**{res['algorithm']}** — AWT: {res.get('avg_waiting_time',0):.2f} | ATAT: {res.get('avg_turnaround_time',0):.2f}"):
            df_res = processes_to_dataframe(res["processes"])
            st.dataframe(df_res.style.format(precision=2), use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# ── SECTION 4: COMPARISON CHARTS ──
# ---------------------------------------------------------------------------

st.markdown('<div class="section-heading">📈 Algorithm Comparison</div>', unsafe_allow_html=True)

# Bar chart — AWT vs ATAT
comp_fig = build_comparison_chart(all_results)
st.plotly_chart(comp_fig, use_container_width=True)

# Numeric comparison table
from metrics.calculator import comparison_dataframe
comp_df = comparison_dataframe(all_results)

# Move index into a plain column to avoid non-unique index issues with Styler
selected_algo = adaptive_result["selected_algorithm"]
comp_df = comp_df.reset_index()   # 'Algorithm' becomes a regular column
comp_df["Algorithm"] = comp_df["Algorithm"].apply(
    lambda name: f"🧠 {name}" if name == selected_algo else name
)

# Highlight minimum AWT and ATAT (numeric columns only — no emoji in index)
def _highlight_min(s):
    is_min = s == s.min()
    return [
        "background-color: rgba(110,231,183,0.20); color: #6ee7b7; font-weight:700"
        if v else ""
        for v in is_min
    ]

st.dataframe(
    comp_df.style
        .format(precision=3, subset=["Avg Waiting Time", "Avg Turnaround Time",
                                      "Avg Response Time", "Throughput"])
        .apply(_highlight_min, subset=["Avg Waiting Time", "Avg Turnaround Time"]),
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")

# ---------------------------------------------------------------------------
# ── SECTION 5: FEATURE ENGINEERING INSIGHT ──
# ---------------------------------------------------------------------------

st.markdown('<div class="section-heading">🔬 Extracted Feature Vector</div>', unsafe_allow_html=True)

feat_dict = prediction.get("features", {})
feat_names = list(feat_dict.keys())
feat_vals = list(feat_dict.values())

feat_fig = go.Figure(go.Bar(
    x=feat_names,
    y=feat_vals,
    marker=dict(
        color=["#667eea", "#f093fb", "#4ade80", "#fbbf24", "#67e8f9"],
        opacity=0.85,
        line=dict(width=0),
    ),
    text=[f"{v:.3f}" for v in feat_vals],
    textposition="outside",
    textfont=dict(size=12, color="#e2e8f0"),
))

feat_fig.update_layout(
    title=dict(
        text="<b>Feature Vector fed to ML Classifier</b>",
        font=dict(size=14, color="#e2e8f0"),
    ),
    xaxis=dict(color="#94a3b8", tickfont=dict(size=12)),
    yaxis=dict(color="#94a3b8", showgrid=True, gridcolor="rgba(255,255,255,0.07)"),
    paper_bgcolor="rgba(10,14,26,0)",
    plot_bgcolor="rgba(10,14,26,0.4)",
    font=dict(family="Inter, sans-serif", color="#e2e8f0"),
    height=320,
    margin=dict(l=40, r=30, t=60, b=60),
)
st.plotly_chart(feat_fig, use_container_width=True)

# ---------------------------------------------------------------------------
# ── FOOTER ──
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#475569; font-size:0.85rem; padding:1rem 0;">
        🧠 Adaptive AI-Driven CPU Scheduler &nbsp;·&nbsp;
        Built with Streamlit, scikit-learn, and Plotly &nbsp;·&nbsp;
        Random Forest Classifier with 1,500+ training samples
    </div>
    """,
    unsafe_allow_html=True,
)
