"""
dashboard.py
-------------
Interactive Streamlit Dashboard — Adaptive AI-Driven CPU Scheduler.

Key Features
────────────
• Dual-Core SMP Parallel Dispatching (Core 0 Compute Engine + Core 1 Interactive Engine).
• Dynamic Priority Aging & Anti-Starvation Safe Fallback Watchdog (Deterministic Guard).
• True Runtime Adaptation via TrueAdaptiveScheduler (policy switches mid-sim).
• Parallel Dual-Core Gantt Chart & Policy-coloured Process Gantt.
• Adaptation Timeline Log showing every policy-transition and starvation-rescue event.
• Workload Input Modes:
    1. Random Generator   – sliders
    2. Manual Table Input – st.data_editor (add/edit/delete rows)
    3. Classic OS Presets – pre-built scenario dropdown

Run:
    streamlit run dashboard.py
"""

import sys
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from core.process import Process
from core.simulation_engine import SimulationEngine, generate_processes
from scheduler.adaptive_scheduler import (
    TrueAdaptiveScheduler,
    DualCoreAdaptiveScheduler,
    POLICY_COLORS,
    POLICY_SRTF,
    POLICY_RR,
    POLICY_PRIORITY,
    POLICY_STARVATION,
)
from metrics.calculator import (
    processes_to_dataframe,
    comparison_dataframe,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Adaptive AI CPU Scheduler",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — dark glassmorphism
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #0a0e1a;
    color: #e2e8f0;
}
.hero-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.6rem; font-weight: 800; line-height: 1.2; margin-bottom: 0.2rem;
}
.hero-sub { color: #94a3b8; font-size: 1.05rem; font-weight: 400; margin-bottom: 1.8rem; }

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px; padding: 1.2rem 1.4rem;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 12px 40px rgba(102,126,234,0.25); }
.metric-label { font-size: 0.76rem; font-weight: 600; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem; }
.metric-value { font-size: 1.8rem; font-weight: 700; line-height: 1; }
.metric-sub   { font-size: 0.80rem; color: #94a3b8; margin-top: 0.3rem; }
.card-purple .metric-value { color: #a78bfa; }
.card-cyan   .metric-value { color: #67e8f9; }
.card-green  .metric-value { color: #6ee7b7; }
.card-orange .metric-value { color: #fbbf24; }
.card-pink   .metric-value { color: #f9a8d4; }
.card-red    .metric-value { color: #f87171; }

/* Section heading */
.section-heading {
    font-size: 1.25rem; font-weight: 700; color: #e2e8f0;
    border-left: 4px solid #667eea; padding-left: 0.75rem;
    margin: 1.8rem 0 1rem 0;
}

/* Justification box */
.justification-box {
    background: rgba(102,126,234,0.12); border: 1px solid rgba(102,126,234,0.30);
    border-radius: 12px; padding: 1rem 1.4rem; font-size: 0.92rem;
    color: #c7d2fe; line-height: 1.7;
}

/* Policy badges */
.badge {
    display: inline-block; border-radius: 6px; padding: 2px 10px;
    font-size: 0.78rem; font-weight: 600; margin: 2px;
}
.badge-srtf       { background: rgba(102,126,234,0.30); color: #a5b4fc; }
.badge-rr         { background: rgba(240,147,251,0.30); color: #f5d0fe; }
.badge-priority   { background: rgba(251,191,36,0.30);  color: #fde68a; }
.badge-starvation { background: rgba(239,68,68,0.30);   color: #fca5a5; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(15,20,40,0.98);
    border-right: 1px solid rgba(255,255,255,0.06);
}
/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white; border: none; border-radius: 12px;
    padding: 0.7rem 2rem; font-size: 1rem; font-weight: 600;
    width: 100%; transition: opacity 0.2s ease, transform 0.1s ease;
}
.stButton > button:hover { opacity: 0.90; transform: translateY(-1px); }

hr { border-color: rgba(255,255,255,0.07); margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants: Classic OS Presets
# ─────────────────────────────────────────────────────────────────────────────

PRESETS: dict = {
    "Convoy Effect Demo": [
        {"PID": 1, "Arrival Time": 0, "Burst Time": 30, "Priority": 5, "I/O Frequency": 0.00},
        {"PID": 2, "Arrival Time": 1, "Burst Time": 2,  "Priority": 5, "I/O Frequency": 0.10},
        {"PID": 3, "Arrival Time": 2, "Burst Time": 2,  "Priority": 5, "I/O Frequency": 0.10},
        {"PID": 4, "Arrival Time": 3, "Burst Time": 2,  "Priority": 5, "I/O Frequency": 0.10},
        {"PID": 5, "Arrival Time": 4, "Burst Time": 2,  "Priority": 5, "I/O Frequency": 0.10},
    ],
    "Starvation Risk Scenario": [
        {"PID": 1, "Arrival Time": 0, "Burst Time": 25, "Priority": 5, "I/O Frequency": 0.05},
        {"PID": 2, "Arrival Time": 2, "Burst Time": 4,  "Priority": 1, "I/O Frequency": 0.80},
        {"PID": 3, "Arrival Time": 4, "Burst Time": 5,  "Priority": 1, "I/O Frequency": 0.70},
        {"PID": 4, "Arrival Time": 6, "Burst Time": 4,  "Priority": 1, "I/O Frequency": 0.85},
        {"PID": 5, "Arrival Time": 8, "Burst Time": 3,  "Priority": 1, "I/O Frequency": 0.90},
    ],
    "Mixed Batch & Interactive": [
        {"PID": 1, "Arrival Time": 0, "Burst Time": 20, "Priority": 3, "I/O Frequency": 0.05},
        {"PID": 2, "Arrival Time": 1, "Burst Time": 3,  "Priority": 4, "I/O Frequency": 0.80},
        {"PID": 3, "Arrival Time": 2, "Burst Time": 15, "Priority": 3, "I/O Frequency": 0.05},
        {"PID": 4, "Arrival Time": 3, "Burst Time": 2,  "Priority": 4, "I/O Frequency": 0.90},
        {"PID": 5, "Arrival Time": 4, "Burst Time": 12, "Priority": 3, "I/O Frequency": 0.10},
        {"PID": 6, "Arrival Time": 5, "Burst Time": 4,  "Priority": 4, "I/O Frequency": 0.75},
    ],
}

PRESET_DESCRIPTIONS = {
    "Convoy Effect Demo": (
        "P1 (burst=30) monopolises the CPU; P2–P5 wait behind it. FCFS performs poorly. "
        "The adaptive scheduler detects high burst variance and triggers SRTF or Dual-Core load balancing."
    ),
    "Starvation Risk Scenario": (
        "P1 has low priority while a stream of high-priority processes (P2–P5) arrives. "
        "Static priority would starve P1 indefinitely. The Anti-Starvation Watchdog triggers deterministic "
        "safe fallback preemption once P1's wait crosses the starvation threshold."
    ),
    "Mixed Batch & Interactive": (
        "Alternating heavy compute tasks and interactive I/O tasks. "
        "In Dual-Core mode, Core 0 runs compute jobs while Core 1 runs interactive jobs simultaneously."
    ),
}

def _default_manual_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"PID": 1, "Arrival Time": 0,  "Burst Time": 12, "Priority": 4, "I/O Frequency": 0.10},
        {"PID": 2, "Arrival Time": 2,  "Burst Time": 4,  "Priority": 1, "I/O Frequency": 0.70},
        {"PID": 3, "Arrival Time": 3,  "Burst Time": 8,  "Priority": 3, "I/O Frequency": 0.20},
        {"PID": 4, "Arrival Time": 5,  "Burst Time": 3,  "Priority": 2, "I/O Frequency": 0.80},
        {"PID": 5, "Arrival Time": 7,  "Burst Time": 10, "Priority": 5, "I/O Frequency": 0.05},
    ])

def _df_to_processes(df: pd.DataFrame) -> list:
    processes = []
    seen_pids = set()
    for _, row in df.iterrows():
        try:
            pid  = int(row["PID"])
            arr  = max(0, int(row["Arrival Time"]))
            burst= max(1, int(row["Burst Time"]))
            prio = max(1, min(10, int(row["Priority"])))
            io   = max(0.0, min(1.0, float(row["I/O Frequency"])))
            if pid in seen_pids:
                continue
            seen_pids.add(pid)
            processes.append(Process(pid=pid, arrival_time=arr, burst_time=burst,
                                     priority=prio, io_frequency=io))
        except Exception:
            pass
    return sorted(processes, key=lambda p: (p.arrival_time, p.pid))

# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────

PROCESS_COLORS = [
    "#667eea","#f093fb","#4ade80","#fbbf24","#67e8f9",
    "#f87171","#a78bfa","#34d399","#fb923c","#38bdf8",
    "#e879f9","#86efac","#fde68a","#7dd3fc","#fca5a5",
]

def _pid_color(pid: int) -> str:
    return PROCESS_COLORS[(pid - 1) % len(PROCESS_COLORS)]

# ─────────────────────────────────────────────────────────────────────────────
# Dual-Core Parallel Gantt Chart (Core 0 & Core 1 lanes)
# ─────────────────────────────────────────────────────────────────────────────

def build_dual_core_gantt(dual_result: dict, color_by_policy: bool = False) -> go.Figure:
    """
    Renders parallel CPU core execution lanes:
    Row 1: Core 0 (Compute Engine)
    Row 2: Core 1 (Interactive Engine)
    """
    timeline = dual_result.get("timeline", [])
    if not timeline:
        fig = go.Figure()
        fig.update_layout(title="No execution data", template="plotly_dark")
        return fig

    rows = []
    for entry in timeline:
        core_id, pid, start, end, pol = entry
        core_name = f"Core {core_id} ({'Compute' if core_id == 0 else 'Interactive'})"
        rows.append(dict(core=core_name, pid=pid, start=start, end=end, duration=end-start, policy=pol))

    df = pd.DataFrame(rows)
    fig = go.Figure()

    seen_legend = set()
    for _, r in df.iterrows():
        pid, pol = r["pid"], r["policy"]
        color = POLICY_COLORS.get(pol, _pid_color(pid)) if color_by_policy else _pid_color(pid)
        legend_key = pol if color_by_policy else f"P{pid}"
        show_leg = legend_key not in seen_legend
        if show_leg:
            seen_legend.add(legend_key)

        fig.add_trace(go.Bar(
            x=[r["duration"]],
            y=[r["core"]],
            base=[r["start"]],
            orientation="h",
            marker=dict(color=color, opacity=0.90, line=dict(color="rgba(0,0,0,0.3)", width=1)),
            text=f"  P{pid} ({pol[:4]})" if r["duration"] >= 2 else f"P{pid}",
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate=(
                f"<b>{r['core']}</b><br>"
                f"Process: P{pid}<br>"
                f"Policy: {pol}<br>"
                f"Window: [{r['start']} → {r['end']}] ({r['duration']} ticks)"
                "<extra></extra>"
            ),
            name=legend_key,
            showlegend=show_leg,
        ))

    fig.update_layout(
        title=dict(text="<b>⚡ Dual-Core Parallel Execution Timeline (SMP Dispatching)</b>",
                   font=dict(size=16, color="#e2e8f0"), x=0.0),
        xaxis=dict(title="Time (ticks)", showgrid=True, gridcolor="rgba(255,255,255,0.07)", color="#94a3b8"),
        yaxis=dict(title="Execution Lane", color="#94a3b8"),
        barmode="stack",
        paper_bgcolor="rgba(10,14,26,0)",
        plot_bgcolor="rgba(10,14,26,0.4)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
        height=320,
        margin=dict(l=80, r=30, t=70, b=50),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Single-Core / Process-Level Gantt Chart
# ─────────────────────────────────────────────────────────────────────────────

def build_gantt(
    timeline,
    processes,
    algorithm_name: str,
    color_by_policy: bool = False,
    adaptation_log: list = None,
) -> go.Figure:
    if not timeline:
        fig = go.Figure()
        fig.update_layout(title="No execution data", template="plotly_dark")
        return fig

    pid_map = {p.pid: p for p in processes}
    # Check if entry is 4-tuple or 5-tuple
    is_dual = len(timeline[0]) == 5
    has_policy = len(timeline[0]) >= 4

    rows = []
    for entry in timeline:
        if is_dual:
            core_id, pid, start, end, pol = entry
        elif has_policy:
            pid, start, end, pol = entry
        else:
            pid, start, end = entry
            pol = algorithm_name
        proc = pid_map.get(pid)
        rows.append(dict(pid=pid, start=start, end=end, duration=end-start,
                         policy=pol, io_freq=round(proc.io_frequency, 3) if proc else 0))

    df = pd.DataFrame(rows)
    fig = go.Figure()

    seen_legend = set()
    for _, row in df.iterrows():
        pid, pol = row["pid"], row["policy"]
        color = POLICY_COLORS.get(pol, _pid_color(pid)) if color_by_policy else _pid_color(pid)
        legend_key = pol if color_by_policy else f"P{pid}"
        show_leg = legend_key not in seen_legend
        if show_leg:
            seen_legend.add(legend_key)

        fig.add_trace(go.Bar(
            x=[row["duration"]],
            y=[f"P{pid}"],
            base=[row["start"]],
            orientation="h",
            marker=dict(color=color, opacity=0.88, line=dict(color="rgba(0,0,0,0.3)", width=1)),
            text=f"  P{pid} [{row['start']}→{row['end']}]" if row["duration"] > 1 else "",
            textposition="inside",
            insidetextanchor="start",
            hovertemplate=(
                f"<b>P{pid}</b><br>Policy: {pol}<br>"
                f"Start: {row['start']} | End: {row['end']}<br>"
                f"Duration: {row['duration']} ticks<br>I/O Freq: {row['io_freq']}"
                "<extra></extra>"
            ),
            name=legend_key,
            showlegend=show_leg,
        ))

    unique_pids = sorted(df["pid"].unique())
    y_labels = [f"P{p}" for p in unique_pids]

    fig.update_layout(
        title=dict(text=f"<b>Process Gantt Chart — {algorithm_name}</b>",
                   font=dict(size=16, color="#e2e8f0"), x=0.0),
        xaxis=dict(title="Time (ticks)", showgrid=True, gridcolor="rgba(255,255,255,0.07)", color="#94a3b8"),
        yaxis=dict(title="Process", categoryorder="array", categoryarray=y_labels, color="#94a3b8"),
        barmode="stack",
        paper_bgcolor="rgba(10,14,26,0)",
        plot_bgcolor="rgba(10,14,26,0.4)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
        height=max(300, 70 + len(unique_pids) * 44),
        margin=dict(l=60, r=30, t=70, b=50),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_chart(all_results: dict) -> go.Figure:
    short_names = {
        "FCFS":          "FCFS",
        "SJF":           "SJF (NP)",
        "SRTF":          "SRTF",
        "RR":            "Round Robin",
        "Priority":      "Priority (Static)",
        "PriorityAging": "Priority (Aging)",
        "TrueAdaptive":  "🧠 Adaptive Single-Core",
        "DualCore":      "⚡ Adaptive Dual-Core",
    }
    labels, awt_vals, atat_vals = [], [], []
    for key, res in all_results.items():
        if "error" in res:
            continue
        labels.append(short_names.get(key, res.get("algorithm", key)))
        awt_vals.append(res.get("avg_waiting_time", 0.0))
        atat_vals.append(res.get("avg_turnaround_time", 0.0))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Avg Waiting Time", x=labels, y=awt_vals,
        marker=dict(color="#667eea", opacity=0.88),
        text=[f"{v:.2f}" for v in awt_vals], textposition="outside",
        textfont=dict(size=11, color="#e2e8f0")
    ))
    fig.add_trace(go.Bar(
        name="Avg Turnaround Time", x=labels, y=atat_vals,
        marker=dict(color="#4ade80", opacity=0.88),
        text=[f"{v:.2f}" for v in atat_vals], textposition="outside",
        textfont=dict(size=11, color="#e2e8f0")
    ))

    fig.update_layout(
        barmode="group",
        title=dict(text="<b>Performance Benchmarks — Waiting Time & Turnaround Time</b>",
                   font=dict(size=16, color="#e2e8f0"), x=0.0),
        xaxis=dict(color="#94a3b8", tickfont=dict(size=11)),
        yaxis=dict(title="Time (ticks)", color="#94a3b8", showgrid=True, gridcolor="rgba(255,255,255,0.07)"),
        paper_bgcolor="rgba(10,14,26,0)",
        plot_bgcolor="rgba(10,14,26,0.4)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12)),
        height=420, margin=dict(l=60, r=30, t=70, b=60),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Adaptation timeline log renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_adaptation_log(adaptation_log: list):
    if not adaptation_log:
        st.info("No policy transitions recorded.")
        return

    BADGE = {
        POLICY_SRTF:       '<span class="badge badge-srtf">SRTF</span>',
        POLICY_RR:         '<span class="badge badge-rr">Round Robin</span>',
        POLICY_PRIORITY:   '<span class="badge badge-priority">Priority</span>',
        POLICY_STARVATION: '<span class="badge badge-starvation">⚠️ Starvation Guard</span>',
    }

    rows_html = ""
    for ev in adaptation_log:
        badge = BADGE.get(ev.get("policy"), ev.get("policy", "—"))
        pids  = ", ".join(f"P{p}" for p in ev.get("queue_pids", []))
        core_info = f"<span style='color:#67e8f9;font-weight:600;'>{ev.get('core')}</span>" if "core" in ev else "CPU"
        bg    = "rgba(239,68,68,0.12)" if ev.get("policy") == POLICY_STARVATION else "rgba(102,126,234,0.06)"

        rows_html += f"""
        <tr style="border-bottom:1px solid rgba(255,255,255,0.06);background:{bg};">
            <td style="padding:8px 12px;color:#94a3b8;font-variant-numeric:tabular-nums;">t = {ev['time']}</td>
            <td style="padding:8px 12px;">{core_info}</td>
            <td style="padding:8px 12px;">{badge}</td>
            <td style="padding:8px 12px;color:#94a3b8;">{ev.get('queue_size',0)} ready</td>
            <td style="padding:8px 12px;color:#64748b;font-size:0.82rem;">{pids}</td>
            <td style="padding:8px 12px;color:#c7d2fe;font-size:0.88rem;">{ev['reason']}</td>
        </tr>"""

    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse;font-family:'Inter',sans-serif;">
        <thead>
            <tr style="border-bottom:2px solid rgba(255,255,255,0.12);">
                <th style="padding:8px 12px;text-align:left;color:#64748b;font-size:0.78rem;text-transform:uppercase;">Time</th>
                <th style="padding:8px 12px;text-align:left;color:#64748b;font-size:0.78rem;text-transform:uppercase;">Core</th>
                <th style="padding:8px 12px;text-align:left;color:#64748b;font-size:0.78rem;text-transform:uppercase;">Policy</th>
                <th style="padding:8px 12px;text-align:left;color:#64748b;font-size:0.78rem;text-transform:uppercase;">Queue</th>
                <th style="padding:8px 12px;text-align:left;color:#64748b;font-size:0.78rem;text-transform:uppercase;">Active PIDs</th>
                <th style="padding:8px 12px;text-align:left;color:#64748b;font-size:0.78rem;text-transform:uppercase;">Telemetry & Decision Reason</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Metric card helpers
# ─────────────────────────────────────────────────────────────────────────────

def metric_card(label, value, sub="", css_class="card-purple"):
    return f"""
    <div class="metric-card {css_class}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
    </div>"""

def _highlight_col(col):
    if col.max() == col.min():
        return [""] * len(col)
    norm = (col - col.min()) / (col.max() - col.min())
    return [f"background-color:rgba({int(220*v+40*(1-v))},{int(40*v+180*(1-v))},80,0.30);color:#e2e8f0;" for v in norm]

def _highlight_min(s):
    is_min = s == s.min()
    return ["background-color:rgba(110,231,183,0.20);color:#6ee7b7;font-weight:700" if v else "" for v in is_min]

# ─────────────────────────────────────────────────────────────────────────────
# ── SIDEBAR ──
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Simulation Controls")
    st.markdown("---")

    input_mode = st.radio(
        "Workload Generation Mode",
        ["🎲 Random Generator", "✏️ Manual Table Input", "📋 Classic OS Presets"],
        help="Choose how to create the process batch for this simulation.",
    )

    st.markdown("---")

    if input_mode == "🎲 Random Generator":
        process_count = st.slider("Process Count",    min_value=3,  max_value=30, value=10)
        max_arrival   = st.slider("Max Arrival Time", min_value=0,  max_value=50, value=20)
        burst_min     = st.slider("Min Burst Time",   min_value=1,  max_value=20, value=2)
        burst_max     = st.slider("Max Burst Time",   min_value=2,  max_value=80, value=20)
        io_probability= st.slider("I/O Probability",  min_value=0.0,max_value=1.0,value=0.4, step=0.05)
        random_seed   = st.number_input("Random Seed (0=random)", min_value=0, max_value=9999, value=42)

    elif input_mode == "✏️ Manual Table Input":
        st.markdown("**Edit the process table below.**")
        if "manual_df" not in st.session_state:
            st.session_state["manual_df"] = _default_manual_df()
        edited_df = st.data_editor(
            st.session_state["manual_df"],
            key="proc_table_editor",
            num_rows="dynamic",
            use_container_width=True,
        )
        st.session_state["manual_df"] = edited_df

    else:
        preset_name = st.selectbox("Select Scenario", options=list(PRESETS.keys()))
        st.markdown(f'<div class="justification-box">📖 {PRESET_DESCRIPTIONS[preset_name]}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🖥️ CPU Architecture & Safeguards")

    cpu_architecture = st.radio(
        "CPU Architecture Mode",
        ["⚡ Dual Core (SMP)", "🖥️ Single Core"],
        help="Simulate dual-core symmetric multiprocessing vs single-core execution."
    )

    enable_aging = st.checkbox(
        "Enable Priority Aging",
        value=True,
        help="Dynamically increments priority of waiting threads to eliminate starvation."
    )

    starvation_thresh = st.slider(
        "Anti-Starvation Threshold (ticks)",
        min_value=5, max_value=35, value=15,
        help="Maximum continuous wait before Watchdog forces deterministic fallback preemption."
    )

    aging_speed = st.slider(
        "Aging Interval (ticks for +1 priority)",
        min_value=1, max_value=8, value=4,
    )

    rr_quantum = st.slider(
        "Round Robin Base Quantum",
        min_value=1, max_value=20, value=4,
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
                data_path  = os.path.join(_ROOT, "data", "workload_dataset.csv")
                model_path = os.path.join(_ROOT, "ml", "model.joblib")
                generate_dataset(save=True, output_path=data_path)
                train(data_path=data_path, model_path=model_path)
                reload_model()
                st.success("Model retrained successfully!")
            except Exception as exc:
                st.error(f"Training failed: {exc}")

    st.markdown("---")
    st.caption("Adaptive AI CPU Scheduler v2.1")
    st.caption("Dual-Core SMP & Starvation Guard Edition")

# ─────────────────────────────────────────────────────────────────────────────
# ── HERO HEADER ──
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="hero-header">🧠 Adaptive AI-Driven CPU Scheduling</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Dynamic Workload Adaptation & Real-Time Telemetry in Multi-Core Operating Systems — '
    'featuring SMP Dual-Core Parallel Dispatching & an Anti-Starvation Watchdog.</div>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# ── RUN BUTTON ──
# ─────────────────────────────────────────────────────────────────────────────

run_col, info_col = st.columns([1, 3])
with run_col:
    run_btn = st.button("▶ Run Simulation & Adaptive Scheduling", use_container_width=True)
with info_col:
    arch_badge = "⚡ Dual Core (SMP)" if cpu_architecture.startswith("⚡") else "🖥️ Single Core"
    aging_badge = "🛡️ Anti-Starvation Active" if enable_aging else "Static Priorities"
    st.info(f"Target Architecture: **{arch_badge}** | Mode: **{aging_badge}** (Threshold={starvation_thresh}t) | Base Q={rr_quantum}")

# ─────────────────────────────────────────────────────────────────────────────
# ── SIMULATION ──
# ─────────────────────────────────────────────────────────────────────────────

if run_btn:
    with st.spinner(f"Simulating under {cpu_architecture} with Adaptive Telemetry..."):
        try:
            if input_mode == "🎲 Random Generator":
                seed = int(random_seed) if random_seed > 0 else None
                processes = generate_processes(
                    count=process_count,
                    max_arrival_time=max_arrival,
                    burst_time_range=(burst_min, burst_max),
                    io_probability=io_probability,
                    seed=seed,
                )
            elif input_mode == "✏️ Manual Table Input":
                processes = _df_to_processes(st.session_state["manual_df"])
                if not processes:
                    st.error("No valid processes in the table.")
                    st.stop()
            else:
                rows = PRESETS[preset_name]
                processes = [
                    Process(
                        pid=r["PID"],
                        arrival_time=r["Arrival Time"],
                        burst_time=r["Burst Time"],
                        priority=r["Priority"],
                        io_frequency=r["I/O Frequency"],
                    )
                    for r in rows
                ]

            is_dual = "Dual Core" in cpu_architecture

            if is_dual:
                dual_sched = DualCoreAdaptiveScheduler(
                    base_quantum=rr_quantum,
                    starvation_threshold=starvation_thresh,
                    enable_aging=enable_aging,
                    aging_interval=aging_speed,
                )
                adaptive_result = dual_sched.run(processes, base_quantum=rr_quantum)
            else:
                single_sched = TrueAdaptiveScheduler(
                    base_quantum=rr_quantum,
                    starvation_threshold=starvation_thresh,
                    enable_aging=enable_aging,
                    aging_interval=aging_speed,
                )
                adaptive_result = single_sched.run(processes, base_quantum=rr_quantum)

            # Classic single-core schedulers for baseline comparison
            engine = SimulationEngine(processes)
            all_classic = engine.run_all(time_quantum=rr_quantum, aging_interval=aging_speed)

            # Store in results
            all_results = {**all_classic}
            if is_dual:
                all_results["DualCore"] = adaptive_result
            else:
                all_results["TrueAdaptive"] = adaptive_result

        except Exception as exc:
            st.error(f"Simulation failed: {exc}")
            st.exception(exc)
            st.stop()

    st.session_state["adaptive_result"] = adaptive_result
    st.session_state["all_results"]     = all_results
    st.session_state["processes"]       = processes
    st.session_state["is_dual"]         = is_dual

if "adaptive_result" not in st.session_state:
    st.markdown("""
    <div style="text-align:center;padding:4rem;color:#475569;">
        <div style="font-size:4rem;">⚡</div>
        <div style="font-size:1.2rem;font-weight:600;margin-top:1rem;">
            Click <b>Run Simulation</b> to observe Adaptive Scheduling & Multi-Core Execution
        </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

adaptive_result = st.session_state["adaptive_result"]
all_results     = st.session_state["all_results"]
processes       = st.session_state["processes"]
is_dual         = st.session_state.get("is_dual", True)
adaptation_log  = adaptive_result.get("adaptation_log", [])

# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION 1: AI INSIGHT & TELEMETRY CARDS ──
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-heading">🤖 AI Telemetry & Multi-Core System Telemetry</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)

label = adaptive_result.get("workload_label", "Mixed")
class_emoji = {"CPU-bound":"⚡","I/O-bound":"💾","Mixed":"🔀"}.get(label, "🧠")
starve_rescues = adaptive_result.get("starvation_rescues", 0)

with c1:
    st.markdown(metric_card("Workload Class", f"{class_emoji} {label}", "ML Classification", "card-purple"), unsafe_allow_html=True)
with c2:
    arch_label = "Dual-Core SMP" if is_dual else "Single-Core"
    st.markdown(metric_card("Architecture", arch_label, "Execution Mode", "card-cyan"), unsafe_allow_html=True)
with c3:
    st.markdown(metric_card("Starvation Rescues", str(starve_rescues), "Watchdog Invocations", "card-red"), unsafe_allow_html=True)
with c4:
    awt = adaptive_result.get("avg_waiting_time", 0.0)
    st.markdown(metric_card("Avg Waiting Time", f"{awt:.2f}", "ticks (adaptive)", "card-orange"), unsafe_allow_html=True)
with c5:
    atat = adaptive_result.get("avg_turnaround_time", 0.0)
    st.markdown(metric_card("Avg Turnaround", f"{atat:.2f}", "ticks (adaptive)", "card-green"), unsafe_allow_html=True)

if is_dual:
    u0 = adaptive_result.get("utilization_core0", 0.0)
    u1 = adaptive_result.get("utilization_core1", 0.0)
    st.markdown(
        f'<div class="justification-box" style="margin-top:12px;">'
        f'⚡ <b>Multi-Core Load Telemetry:</b> Core 0 Utilization: <b>{u0:.1f}%</b> | '
        f'Core 1 Utilization: <b>{u1:.1f}%</b> | '
        f'Overall SMP Efficiency: <b>{adaptive_result.get("overall_utilization", 0):.1f}%</b>. '
        f'{adaptive_result.get("justification", "")}</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="justification-box" style="margin-top:12px;">'
        f'💡 <b>Single-Core Telemetry:</b> {adaptive_result.get("justification", "")}</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION 2: GANTT CHART ──
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-heading">📊 Execution Gantt Chart</div>', unsafe_allow_html=True)

if is_dual:
    gantt_tab1, gantt_tab2, gantt_tab3 = st.tabs([
        "⚡ Dual-Core Parallel Lanes (Core 0 & Core 1)",
        "🧠 Process Timeline (by Policy)",
        "🔁 Compare All Baseline Schedulers",
    ])

    with gantt_tab1:
        dual_fig = build_dual_core_gantt(adaptive_result, color_by_policy=True)
        st.plotly_chart(dual_fig, use_container_width=True, key="gantt_dual")
        st.caption("Visualizes Core 0 (Compute Engine) and Core 1 (Interactive Engine) running in parallel.")

    with gantt_tab2:
        proc_fig = build_gantt(
            timeline=adaptive_result["timeline"],
            processes=adaptive_result["processes"],
            algorithm_name="Dual-Core Adaptive Scheduler",
            color_by_policy=True,
            adaptation_log=adaptation_log,
        )
        st.plotly_chart(proc_fig, use_container_width=True, key="gantt_proc_dual")

    with gantt_tab3:
        sel_algo = st.selectbox("Select Baseline Algorithm", options=list(all_results.keys()), key="sel_baseline_dual")
        r = all_results[sel_algo]
        if "error" not in r:
            f = build_gantt(r["timeline"], r["processes"], sel_algo)
            st.plotly_chart(f, use_container_width=True, key=f"gantt_{sel_algo}")

else:
    gantt_tab1, gantt_tab2 = st.tabs(["🧠 True Adaptive AI (by Policy)", "🔁 Compare All Baseline Schedulers"])

    with gantt_tab1:
        adaptive_fig = build_gantt(
            timeline=adaptive_result["timeline"],
            processes=adaptive_result["processes"],
            algorithm_name="True Adaptive AI Scheduler",
            color_by_policy=True,
            adaptation_log=adaptation_log,
        )
        st.plotly_chart(adaptive_fig, use_container_width=True, key="gantt_single_adaptive")

    with gantt_tab2:
        sel_algo = st.selectbox("Select Baseline Algorithm", options=list(all_results.keys()), key="sel_baseline_single")
        r = all_results[sel_algo]
        if "error" not in r:
            f = build_gantt(r["timeline"], r["processes"], sel_algo)
            st.plotly_chart(f, use_container_width=True, key=f"gantt_{sel_algo}")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION 3: ADAPTATION & STARVATION WATCHDOG LOG ──
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-heading">⚡ Adaptation & Anti-Starvation Watchdog Log</div>', unsafe_allow_html=True)
st.caption("Logs real-time queue classification, dynamic policy transitions, work-stealing, and deterministic safe fallbacks.")

render_adaptation_log(adaptation_log)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION 4: PROCESS METRICS TABLE ──
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-heading">📋 Process-Level Metrics & Fairness</div>', unsafe_allow_html=True)

table_tab1, table_tab2 = st.tabs(["🧠 Adaptive AI Schedulers", "📊 All Baseline Schedulers"])

with table_tab1:
    proc_df = processes_to_dataframe(adaptive_result["processes"])
    st.dataframe(
        proc_df.style.format(precision=2).apply(_highlight_col, subset=["Waiting Time", "Turnaround Time"]),
        use_container_width=True,
        hide_index=True,
    )

with table_tab2:
    for algo_key, res in all_results.items():
        if "error" in res or algo_key in ("TrueAdaptive", "DualCore"):
            continue
        label_str = res["algorithm"]
        with st.expander(f"**{label_str}** — AWT: {res.get('avg_waiting_time',0):.2f} | ATAT: {res.get('avg_turnaround_time',0):.2f}"):
            df_res = processes_to_dataframe(res["processes"])
            st.dataframe(df_res.style.format(precision=2), use_container_width=True, hide_index=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION 5: COMPARISON BENCHMARKS ──
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-heading">📈 Comprehensive Performance Benchmarking</div>', unsafe_allow_html=True)

comp_fig = build_comparison_chart(all_results)
st.plotly_chart(comp_fig, use_container_width=True, key="comp_chart")

comp_df = comparison_dataframe(all_results).reset_index()
st.dataframe(
    comp_df.style.format(precision=3, subset=["Avg Waiting Time", "Avg Turnaround Time", "Avg Response Time", "Throughput"])
           .apply(_highlight_min, subset=["Avg Waiting Time", "Avg Turnaround Time"]),
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION 6: ML FEATURE TELEMETRY ──
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-heading">🔬 Telemetry & Predictive Feature Vector</div>', unsafe_allow_html=True)

try:
    from ml.predictor import extract_features, FEATURE_COLS
    feat_vec = extract_features(processes)
    feat_fig = go.Figure(go.Bar(
        x=FEATURE_COLS, y=feat_vec.tolist(),
        marker=dict(color=["#667eea","#f093fb","#4ade80","#fbbf24","#67e8f9"], opacity=0.85),
        text=[f"{v:.3f}" for v in feat_vec],
        textposition="outside",
        textfont=dict(size=12, color="#e2e8f0"),
    ))
    feat_fig.update_layout(
        title=dict(text="<b>Live Workload Telemetry Vector</b>", font=dict(size=14, color="#e2e8f0")),
        xaxis=dict(color="#94a3b8"),
        yaxis=dict(color="#94a3b8", showgrid=True, gridcolor="rgba(255,255,255,0.07)"),
        paper_bgcolor="rgba(10,14,26,0)",
        plot_bgcolor="rgba(10,14,26,0.4)",
        height=320, margin=dict(l=40, r=30, t=60, b=60),
    )
    st.plotly_chart(feat_fig, use_container_width=True, key="feature_chart")
except Exception:
    pass

st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#475569;font-size:0.85rem;padding:1rem 0;">
    🧠 Adaptive AI-Driven CPU Scheduling v2.1 &nbsp;·&nbsp;
    Dual-Core SMP & Anti-Starvation Guard &nbsp;·&nbsp;
    Streamlit + scikit-learn + Plotly
</div>""", unsafe_allow_html=True)
