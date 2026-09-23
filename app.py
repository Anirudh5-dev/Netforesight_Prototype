"""NetForeSight — AI-Based Network Attack Forecasting (offline MVP).

Runs fully offline: ``streamlit run app.py``. All data, inference
(deterministic demo engine) and visualizations execute locally.
"""
from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from modules import ATTACK_STAGES, SEVERITY_COLORS
from modules import visualizations as V
from modules.data_loader import (
    DataLoadError, describe_dataset, load_demo_dataset, load_pcap,
    load_uploaded_csv,
)
from modules.explainability import (
    feature_contributions, narrative, shap_style_values,
)
from modules.feature_extractor import (
    extract_features, normalize_features, summarize_flows,
)
from modules.forecast_engine import (
    DEFAULT_THRESHOLD, PIPELINE_STAGES, SCENARIO_PHASES, DemoWorldModel,
    TorchLSTMWorldModel, build_world_model, scenario_phase,
)
from modules.windowing import (
    NetworkWindow, generate_windows, state_matrix, windows_to_dataframe,
)

st.set_page_config(page_title="NetForeSight — Network Attack Forecasting",
                   page_icon="🛡️", layout="wide")

# --------------------------------------------------------------------------- #
# Styling                                                                      #
# --------------------------------------------------------------------------- #

CSS = """
<style>
.stApp { background: radial-gradient(1200px 600px at 80% -10%, #14264d 0%,
  #0a0f1e 55%, #070b16 100%); }
section[data-testid="stSidebar"] { background: #0b1226;
  border-right: 1px solid rgba(148,163,184,0.15); }
.brand { font-size: 22px; font-weight: 800; letter-spacing: 2px; color: #fff; }
.brand span { color: #22d3ee; }
.tagline { color: #94a3b8; font-size: 12px; margin: 2px 0 12px; }
.kpi-card { background: rgba(17,26,48,0.75); border: 1px solid
  rgba(148,163,184,0.18); border-radius: 12px; padding: 14px 16px;
  backdrop-filter: blur(6px); }
.kpi-label { color: #94a3b8; font-size: 12px; letter-spacing: 1px;
  text-transform: uppercase; }
.kpi-value { font-size: 30px; font-weight: 800; margin: 2px 0; }
.kpi-sub { color: #94a3b8; font-size: 12px; }
.panel { background: rgba(17,26,48,0.75); border: 1px solid
  rgba(148,163,184,0.18); border-radius: 12px; padding: 16px 18px;
  margin-bottom: 14px; }
.panel h4 { margin-top: 0; }
.badge { display: inline-block; border: 1px solid #22d3ee; color: #22d3ee;
  border-radius: 999px; padding: 2px 10px; font-size: 11px; font-weight: 700;
  letter-spacing: 1px; margin-right: 6px; }
.badge-dim { border-color: #475569; color: #94a3b8; }
.alert-card { background: linear-gradient(135deg, rgba(127,29,29,0.35),
  rgba(17,26,48,0.9)); border: 1px solid rgba(239,68,68,0.55);
  border-radius: 12px; padding: 18px 20px; margin: 6px 0 16px; }
.warn-card { background: linear-gradient(135deg, rgba(124,45,18,0.30),
  rgba(17,26,48,0.9)); border: 1px solid rgba(251,146,60,0.5);
  border-radius: 12px; padding: 14px 18px; margin: 6px 0 16px; }
.tl-wrap { display: flex; align-items: stretch; flex-wrap: wrap; }
.tl-card { background: rgba(17,26,48,0.85); border-radius: 10px;
  padding: 12px 16px; min-width: 150px; flex: 1; text-align: center; }
.tl-stage { font-weight: 700; color: #e6edf7; margin-top: 4px; }
.tl-arrow { color: #475569; font-size: 18px; text-align: center;
  padding: 2px 6px; align-self: center; }
.info-chip { background: rgba(17,26,48,0.85); border: 1px solid
  rgba(148,163,184,0.25); border-radius: 8px; padding: 8px 14px;
  color: #cbd5e1; font-size: 14px; }
.win-card { background: rgba(17,26,48,0.75); border: 1px solid
  rgba(148,163,184,0.18); border-radius: 12px; padding: 14px 16px;
  height: 100%; }
.win-card h5 { margin: 0 0 2px; color: #fff; }
.step-card { background: rgba(17,26,48,0.85); border: 1px solid
  rgba(34,211,238,0.35); border-radius: 10px; padding: 12px 14px;
  text-align: center; }
.check-line { color: #86efac; font-family: monospace; font-size: 13px;
  margin: 1px 0; }
.status-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%;
  background: #22c55e; margin-right: 6px; box-shadow: 0 0 8px #22c55e; }
.status-dot.amber { background: #f59e0b; box-shadow: 0 0 8px #f59e0b; }
.report-box { background: #0d1528; border: 1px solid rgba(148,163,184,0.25);
  border-radius: 10px; padding: 18px 22px; font-family: monospace;
  font-size: 13px; color: #dbe7f5; white-space: pre-wrap; }
div[data-testid="stMetric"] { background: rgba(17,26,48,0.75);
  border: 1px solid rgba(148,163,184,0.18); border-radius: 12px; padding: 10px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Session state                                                                #
# --------------------------------------------------------------------------- #

_DEFAULTS = {
    "page": "Dashboard",
    "flows": None,
    "windows": [],
    "visible_ids": [],
    "forecast": None,
    "contributions": {},
    "summary": {},
    "dataset_info": {},
    "is_demo": True,
    "horizon": 5,
    "threshold": DEFAULT_THRESHOLD,
    "n_windows": 5,
    "demo_phase": 7,
    "analysis_steps": [],
    "alert_dismissed": False,
    "show_investigation": False,
    "analyst_notes": "",
    "investigations": [],
    "inv_counter": 1,
    "selected_flow": None,
    "node_selected": "10.0.2.15",
    "revealed": 0,
    "scenario_log": [],
    "explain_step": "T+2",
    "backend": "demo",
}
for _k, _v in _DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


# --------------------------------------------------------------------------- #
# Analysis pipeline                                                            #
# --------------------------------------------------------------------------- #

PIPELINE_LABELS = ["File Loaded", "Feature Extraction", "Normalization",
                   "Time Window Generation", "State Representation",
                   "Forecasting", "Explainability", "Analysis Complete"]


def run_analysis(df: pd.DataFrame, name: str, ftype: str, size: str,
                 scenario_final: float | None = None,
                 show_progress: bool = True) -> None:
    """Execute the offline pipeline and store every artifact in session."""
    steps: list[str] = []

    def _mark(label: str):
        steps.append(label)
        if show_progress:
            status.update(label=f"⚙ {label} …")

    status = st.status("⚙ Analyzing network traffic …",
                       expanded=show_progress) if show_progress else None
    try:
        _mark(PIPELINE_LABELS[0])
        feats = extract_features(df)
        _mark(PIPELINE_LABELS[1])
        feats, _ = normalize_features(feats)
        _mark(PIPELINE_LABELS[2])
        windows = generate_windows(feats, st.session_state.n_windows)
        _mark(PIPELINE_LABELS[3])
        _ = state_matrix(windows)
        _mark(PIPELINE_LABELS[4])
        model = build_world_model(st.session_state.backend)
        forecast = model.predict(windows, horizon=st.session_state.horizon,
                                 threshold=st.session_state.threshold,
                                 scenario_final=scenario_final)
        _mark(PIPELINE_LABELS[5])
        contribs = feature_contributions(forecast.get("signals"))
        _mark(PIPELINE_LABELS[6])
        _mark(PIPELINE_LABELS[7])
    finally:
        if status is not None:
            status.update(label="✅ Analysis complete", state="complete",
                          expanded=False)

    st.session_state.flows = feats
    st.session_state.windows = windows
    st.session_state.forecast = forecast
    st.session_state.contributions = contribs
    st.session_state.summary = summarize_flows(feats)
    st.session_state.dataset_info = describe_dataset(feats, name, ftype, size)
    st.session_state.analysis_steps = steps
    st.session_state.alert_dismissed = False
    st.session_state.show_investigation = False
    st.session_state.revealed = 0
    # Reset per-dataset UI selections.
    crit = feats[feats["risk"].isin(["HIGH", "CRITICAL"])]
    st.session_state.selected_flow = (int(crit.index[0])
                                      if len(crit) else int(feats.index[0]))


def apply_demo_phase(phase: int) -> None:
    """Recompute the visible demo view for a scenario phase."""
    st.session_state.demo_phase = int(phase)
    st.session_state.is_demo = True
    phase_spec = scenario_phase(phase)
    windows: list[NetworkWindow] = st.session_state.windows
    if not windows:  # fresh session: load demo first
        run_analysis(load_demo_dataset(), "demo_network.csv", "CSV",
                     "—", show_progress=False)
        windows = st.session_state.windows
    k = min(int(phase), 4, len(windows))
    visible = windows[:k]
    st.session_state.visible_ids = [w.id for w in visible]
    model = build_world_model(st.session_state.backend)
    forecast = model.predict(visible, horizon=st.session_state.horizon,
                             threshold=st.session_state.threshold,
                             scenario_final=phase_spec.final_risk)
    st.session_state.forecast = forecast
    st.session_state.contributions = feature_contributions(
        forecast.get("signals"))
    feats: pd.DataFrame = st.session_state.flows
    # Flow KPIs always cover the full observed session; the scenario phase
    # only moves the "current time" (visible windows) and the forecast.
    st.session_state.summary = summarize_flows(feats)
    st.session_state.alert_dismissed = False
    st.session_state.revealed = 0


def ensure_data() -> None:
    if st.session_state.flows is None:
        run_analysis(load_demo_dataset(), "demo_network.csv", "CSV", "—",
                     show_progress=False)
        st.session_state.is_demo = True
        apply_demo_phase(st.session_state.demo_phase)


ensure_data()

# --------------------------------------------------------------------------- #
# Sidebar                                                                      #
# --------------------------------------------------------------------------- #

st.sidebar.markdown('<div class="brand">NET<span>FORESIGHT</span></div>',
                    unsafe_allow_html=True)
st.sidebar.markdown('<div class="tagline">Network Attack Forecasting · '
                    'Offline MVP</div>', unsafe_allow_html=True)

NAV = ["🏠 Dashboard", "📡 Traffic Monitor", "🔮 Attack Forecast",
       "🧠 Explainability", "🌐 Network Graph", "📊 Analytics",
       "📄 Reports", "⚙ Settings"]
choice = st.sidebar.radio("Navigate", NAV,
                          index=[n.split(" ", 1)[1] for n in NAV].index(
                              st.session_state.page)
                          if st.session_state.page in
                          [n.split(" ", 1)[1] for n in NAV] else 0,
                          label_visibility="collapsed")
st.session_state.page = choice.split(" ", 1)[1]

st.sidebar.markdown("---")
st.sidebar.markdown("**SYSTEM STATUS**")
st.sidebar.markdown('<span class="status-dot"></span>OFFLINE / DEMO MODE',
                    unsafe_allow_html=True)
fc = st.session_state.forecast or {}
st.sidebar.markdown(f"Model: `{(fc.get('model', '—')).split(' (')[0]}`")
st.sidebar.markdown(f"Dataset: `{st.session_state.dataset_info.get('file_name', '—')}`")
st.sidebar.markdown("---")
st.sidebar.caption("SIMULATED / OFFLINE DATA — demo inference only.")

page = st.session_state.page

# --------------------------------------------------------------------------- #
# Shared render helpers                                                        #
# --------------------------------------------------------------------------- #

def header_block() -> None:
    st.markdown("# NETFORESIGHT")
    st.markdown("### AI-Powered Network Attack Forecasting")
    st.markdown("**“Predict the next stage. Investigate before compromise.”**")
    st.markdown(
        '<span class="badge">● DEMO MODE</span>'
        '<span class="badge">SIMULATED / OFFLINE DATA</span>'
        '<span class="badge badge-dim">OFFLINE MVP</span>'
        '<span class="badge badge-dim">PROTOTYPE WORLD MODEL</span>',
        unsafe_allow_html=True)
    st.markdown("")


def status_strip() -> None:
    info = st.session_state.dataset_info
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="kpi-card"><div class="kpi-label">Data Source</div>'
                f'<div style="font-weight:700">'
                f'{"Demo Network Traffic" if st.session_state.is_demo else info.get("file_name", "—")}'
                f'</div></div>', unsafe_allow_html=True)
    c2.markdown('<div class="kpi-card"><div class="kpi-label">Analysis Status</div>'
                '<div style="font-weight:700;color:#22c55e">● Complete</div></div>',
                unsafe_allow_html=True)
    c3.markdown('<div class="kpi-card"><div class="kpi-label">Model Status</div>'
                '<div style="font-weight:700;color:#22d3ee">Forecast Engine Ready</div></div>',
                unsafe_allow_html=True)
    c4.markdown('<div class="kpi-card"><div class="kpi-label">Environment</div>'
                '<div style="font-weight:700">Offline MVP</div></div>',
                unsafe_allow_html=True)


def kpi_row() -> None:
    f = st.session_state.forecast
    s = st.session_state.summary
    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.markdown(V.kpi_card("Current Network Risk", f"{f['current_risk']:.0f}%",
                             f"Stage: {f['current_stage']}", "#22d3ee"),
                  unsafe_allow_html=True)
    r1c2.markdown(V.kpi_card("Forecast Risk", f"{f['risk']:.0f}%",
                             "Demo Forecast · primary horizon", "#fb923c"),
                  unsafe_allow_html=True)
    r1c3.markdown(V.kpi_card("Predicted Stage", f["predicted_stage"],
                             "MITRE ATT&CK mapping", "#fb923c"),
                  unsafe_allow_html=True)
    r2c1, r2c2, r2c3 = st.columns(3)
    r2c1.markdown(V.kpi_card("Forecast Confidence", f"{f['confidence']:.0f}%",
                             "Demo Forecast", "#22d3ee"),
                  unsafe_allow_html=True)
    r2c2.markdown(V.kpi_card("Active Flows", f"{s['active_flows']:,}",
                             "Observed in scope", "#e6edf7"),
                  unsafe_allow_html=True)
    r2c3.markdown(V.kpi_card("Suspicious Flows", f"{s['suspicious_flows']:,}",
                             f"{s['suspicious_pct']:.2f}% of scope", "#ef4444"),
                  unsafe_allow_html=True)


def forecast_warning_block() -> None:
    f = st.session_state.forecast
    if f.get("crosses_threshold"):
        st.markdown(
            '<div class="warn-card"><h4 style="margin:0 0 4px">⚠ Forecast Warning</h4>'
            '“The predicted network trajectory crosses the configured risk '
            'threshold within the next few time windows.”</div>',
            unsafe_allow_html=True)


def alert_card() -> None:
    f = st.session_state.forecast
    if not f.get("crosses_threshold") or st.session_state.alert_dismissed:
        return
    st.markdown(
        f"""<div class="alert-card"><h3 style="margin:0 0 6px">
        ⚠ NETWORK ATTACK FORECAST</h3>
        <div style="color:#cbd5e1">Increasing attacker progression probability
        detected.</div>
        <div style="display:flex;gap:26px;margin-top:10px;flex-wrap:wrap">
        <div>Predicted Stage:<br><b style="color:#fb923c;font-size:18px">
        {f['predicted_stage']}</b></div>
        <div>Forecast Risk:<br><b style="color:#ef4444;font-size:18px">
        {f['risk']:.0f}%</b></div>
        <div>Confidence:<br><b style="color:#22d3ee;font-size:18px">
        {f['confidence']:.0f}%</b></div>
        <div>Estimated Warning Lead:<br><b>Before predicted stage transition</b></div>
        </div></div>""", unsafe_allow_html=True)
    b1, b2, b3, b4 = st.columns(4)
    if b1.button("🔍 Investigate", width="stretch"):
        st.session_state.show_investigation = True
    if b2.button("🧠 View Explanation", width="stretch"):
        st.session_state.page = "Explainability"
        st.rerun()
    if b3.button("📝 Add Analyst Note", width="stretch"):
        st.session_state.show_investigation = True
    if b4.button("Dismiss", width="stretch"):
        st.session_state.alert_dismissed = True
        st.rerun()


def investigation_panel() -> None:
    if not st.session_state.show_investigation:
        return
    f = st.session_state.forecast
    feats: pd.DataFrame = st.session_state.flows
    flagged = feats[feats["risk"].isin(["HIGH", "CRITICAL"])].sort_values(
        "risk_score", ascending=False).head(5)
    top_src = (flagged["src_ip"].value_counts().index[0]
               if len(flagged) else "—")
    top_dst = flagged["dst_ip"].unique().tolist()[:4]
    top_signals = sorted(st.session_state.contributions.items(),
                         key=lambda kv: kv[1], reverse=True)[:3]
    inv_id = f"INV-20260923-{st.session_state.inv_counter:03d}"
    st.markdown(f"""<div class="panel"><h4>🔍 Analyst Investigation — {inv_id}</h4>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px 22px;
    color:#cbd5e1;font-size:14px">
    <div>Forecast Time: <b>{datetime.now().strftime('%H:%M:%S')}</b></div>
    <div>Affected Source: <b style="color:#ef4444">{top_src}</b></div>
    <div>Affected Destinations: <b>{', '.join(top_dst)}</b></div>
    <div>Predicted Stage: <b style="color:#fb923c">{f['predicted_stage']}</b></div>
    <div>Risk: <b>{f['risk']:.0f}%</b> · Confidence: <b>{f['confidence']:.0f}%</b></div>
    <div>Top Signals: <b>{', '.join(s for s, _ in top_signals)}</b></div>
    <div>Flagged Flows: <b>{len(flagged)} shown / """
                f"""{st.session_state.summary['suspicious_flows']:,} total</b></div>
    <div>Related Windows: <b>{', '.join(f'W{i}' for i in st.session_state.visible_ids[-2:])}</b></div>
    </div></div>""", unsafe_allow_html=True)
    st.dataframe(flagged[["timestamp", "src_ip", "dst_ip", "dst_port",
                           "protocol", "tcp_flags", "risk"]],
                 width="stretch", hide_index=True)
    st.session_state.analyst_notes = st.text_area(
        "Analyst Notes", value=st.session_state.analyst_notes, height=110,
        placeholder="Record hypothesis, next steps, escalation decision …")
    c1, c2 = st.columns([1, 4])
    if c1.button("💾 Save Investigation", type="primary"):
        st.session_state.investigations.append({
            "id": inv_id, "time": datetime.now().strftime("%H:%M:%S"),
            "stage": f["predicted_stage"], "risk": f["risk"],
            "notes": st.session_state.analyst_notes})
        st.session_state.inv_counter += 1
        st.success(f"Investigation {inv_id} saved to this session.")
    if st.session_state.investigations:
        with st.expander(f"Saved investigations "
                         f"({len(st.session_state.investigations)})"):
            for inv in st.session_state.investigations:
                st.markdown(f"**{inv['id']}** · {inv['time']} · "
                            f"{inv['stage']} · {inv['risk']:.0f}% — "
                            f"{inv['notes'] or '_no notes_'}")

# --------------------------------------------------------------------------- #
# Pages                                                                        #
# --------------------------------------------------------------------------- #

def page_dashboard() -> None:
    header_block()
    status_strip()
    st.markdown("")
    kpi_row()
    st.markdown("")

    f = st.session_state.forecast
    st.plotly_chart(V.risk_forecast_chart(f), width="stretch")
    st.caption("Demo Forecast — simulated trajectory. Observed (solid) vs "
               "forecast (dashed) with ±6% confidence band.")
    forecast_warning_block()
    alert_card()
    investigation_panel()

    st.markdown("### Attack Progression Timeline")
    st.markdown(V.timeline_html(f["timeline"], f["predicted_stage"],
                                f["confidence"]), unsafe_allow_html=True)
    st.caption("Observed = confirmed from current data · Suspected = possible "
               "stage · Predicted = world-model forecast · Future = not observed.")
    st.markdown("")

    # ---- Analyze Network Traffic ---- #
    st.markdown("## Analyze Network Traffic")
    tab_pcap, tab_csv, tab_demo = st.tabs(
        ["Upload PCAP", "Upload CSV", "Load Demo Dataset"])
    with tab_pcap:
        up = st.file_uploader("Choose a .pcap / .pcapng file", type=["pcap", "pcapng"])
        if up is not None and st.button("▶ Process PCAP", type="primary"):
            try:
                df = load_pcap(up)
                run_analysis(df, up.name, "PCAP", f"{len(up.getvalue())/1024:.1f} KB")
                st.session_state.is_demo = False
                st.session_state.visible_ids = [w.id for w in st.session_state.windows]
                st.success(f"Processed {len(df):,} flows from {up.name}.")
                st.rerun()
            except DataLoadError as exc:
                st.warning(f"⚠ Unable to process the selected file. {exc}")
    with tab_csv:
        up = st.file_uploader("Choose a .csv file", type=["csv"])
        if up is not None and st.button("▶ Process CSV", type="primary"):
            try:
                df = load_uploaded_csv(up)
                run_analysis(df, up.name, "CSV", f"{up.size/1024:.1f} KB")
                st.session_state.is_demo = False
                st.session_state.visible_ids = [w.id for w in st.session_state.windows]
                st.success(f"Processed {len(df):,} records from {up.name}.")
                st.rerun()
            except DataLoadError as exc:
                st.warning(f"⚠ Unable to process the selected file. {exc}")
    with tab_demo:
        st.markdown("Realistic simulated enterprise traffic with a progressive "
                    "internal attack (12,481 flows, 5 time windows).")
        if st.button("📥 Load Demo Dataset", type="primary"):
            run_analysis(load_demo_dataset(), "demo_network.csv", "CSV", "—")
            st.session_state.is_demo = True
            apply_demo_phase(7)
            st.success("Demo dataset loaded.")
            st.rerun()

    info = st.session_state.dataset_info
    if info:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("File Name", info.get("file_name", "—"))
        c2.metric("File Type", info.get("file_type", "—"))
        c3.metric("File Size", info.get("file_size", "—"))
        c4.metric("Records / Packets", f"{info.get('records', 0):,}")
        c5.metric("Analysis Status", info.get("status", "—"))
    if st.session_state.analysis_steps:
        st.markdown("**Processing workflow**")
        st.markdown("\n".join(f'<div class="check-line">✓ {s}</div>'
                              for s in st.session_state.analysis_steps),
                    unsafe_allow_html=True)
    st.markdown("")

    # ---- Network State Windows ---- #
    st.markdown("## Network State Windows")
    st.caption("Ordered network states — the sequence the world model learns "
               "from. Demo mode shows the four observed windows plus the "
               "predicted next state.")
    wins = [w for w in st.session_state.windows
            if w.id in st.session_state.visible_ids]
    if st.session_state.is_demo:
        wins = [w for w in wins if w.id <= 4]
    cols = st.columns(5 if len(wins) >= 4 else max(len(wins), 1))
    for col, w in zip(cols, wins):
        col.markdown(
            f"""<div class="win-card"><h5>Window {w.id:02d}</h5>
            <div style="color:#22d3ee;font-size:13px;margin-bottom:8px">{w.label}</div>
            <div style="font-size:13px;color:#cbd5e1">
            Flows: <b>{w.flows:,}</b><br>Unique Sources: <b>{w.unique_src}</b>
            <br>Unique Destinations: <b>{w.unique_dst}</b><br>
            SYN Count: <b>{w.syn_count:,}</b><br>Average IAT: <b>{w.avg_iat}s</b>
            <br>Risk: <b>{w.risk:.0f}%</b></div></div>""",
            unsafe_allow_html=True)
    if st.session_state.is_demo and len(cols) > len(wins):
        pass
    if st.session_state.is_demo:
        (cols[len(wins)] if len(wins) < len(cols) else st).markdown(
            f"""<div class="win-card" style="border-style:dashed;
            border-color:rgba(251,146,60,0.6)"><h5>Window 05</h5>
            <div style="color:#fb923c;font-size:13px;margin-bottom:8px">
            Predicted High-Risk State</div>
            <div style="font-size:13px;color:#cbd5e1">
            Forecast Risk: <b>{f['forecast_risk'][0]:.0f}%</b><br>
            Predicted Stage: <b>{f['predicted_stage']}</b><br>
            <span class="badge" style="margin-top:6px">DEMO FORECAST</span></div></div>""",
            unsafe_allow_html=True)
    chart_wins = [w for w in st.session_state.windows
                  if w.id in st.session_state.visible_ids]
    if st.session_state.is_demo:
        chart_wins = [w for w in chart_wins if w.id <= 4]
    if chart_wins:  # append the predicted next state
        last = chart_wins[-1]
        chart_wins = chart_wins + [NetworkWindow(
            id=last.id + 1, label="Predicted High-Risk State",
            start=last.end, end=last.end, flows=0, unique_src=0,
            unique_dst=0, syn_count=0, avg_iat=0.0, retrans=0,
            risk=float(f["forecast_risk"][0]), predicted=True)]
    if chart_wins:
        st.plotly_chart(V.window_risk_chart(chart_wins),
                        width="stretch")
    st.markdown("")

    # ---- Demo scenario ---- #
    st.markdown("## Demo Scenario: Progressive Internal Attack")
    st.markdown(" → ".join(f"`{s}`" for s in PIPELINE_STAGES))
    st.caption("Observe → Window → Learn → Forecast → Explain → "
               "Prioritize → Review")
    if st.button("▶ Run Demo Scenario", type="primary"):
        if st.session_state.flows is None or not st.session_state.is_demo:
            run_analysis(load_demo_dataset(), "demo_network.csv", "CSV", "—",
                         show_progress=False)
            st.session_state.is_demo = True
        log: list[str] = []
        with st.status("▶ Running demo scenario …", expanded=True) as stt:
            for ph in SCENARIO_PHASES:
                apply_demo_phase(ph.phase)
                msg = (f"Phase {ph.phase}/7 — {ph.title} "
                       f"(risk {st.session_state.forecast['current_risk']:.0f}%"
                       f" → forecast {ph.final_risk:.0f}%)")
                log.append("✓ " + msg)
                stt.update(label=f"▶ {msg}")
                time.sleep(0.45)
            stt.update(label="✅ Demo scenario complete — threshold crossed, "
                             "lateral movement forecast.",
                       state="complete", expanded=False)
        st.session_state.scenario_log = log
        st.rerun()
    if st.session_state.scenario_log:
        with st.expander("Scenario run log", expanded=False):
            for line in st.session_state.scenario_log:
                st.markdown(f'<div class="check-line">{line}</div>',
                            unsafe_allow_html=True)
    st.caption("Prototype limits: simulated inference, no live monitoring, "
               "forecasts are not validated detections.")


def _filtered_flows() -> pd.DataFrame:
    df: pd.DataFrame = st.session_state.flows
    f1, f2, f3 = st.columns(3)
    risks = f1.multiselect("Risk", ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                           default=["LOW", "MEDIUM", "HIGH", "CRITICAL"])
    protos = f2.multiselect("Protocol", sorted(df["protocol"].unique()),
                            default=sorted(df["protocol"].unique()))
    sort_col = f3.selectbox("Sort by",
                            ["timestamp", "risk_score", "bytes", "packets",
                             "duration", "dst_port"])
    f4, f5, f6 = st.columns(3)
    src_q = f4.text_input("Source IP contains", "")
    dst_q = f5.text_input("Destination IP contains", "")
    asc = f6.checkbox("Ascending order", value=False)
    span_min = int((df["timestamp"].max() - df["timestamp"].min()).total_seconds() // 60)
    t0, t1 = st.slider("Time range (minutes from start)", 0, max(span_min, 1),
                       (0, max(span_min, 1)))
    base = df["timestamp"].min()
    out = df[df["risk"].isin(risks) & df["protocol"].isin(protos)]
    if src_q:
        out = out[out["src_ip"].str.contains(src_q)]
    if dst_q:
        out = out[out["dst_ip"].str.contains(dst_q)]
    out = out[(out["timestamp"] >= base + pd.Timedelta(minutes=t0)) &
              (out["timestamp"] <= base + pd.Timedelta(minutes=t1))]
    out = out.sort_values(sort_col, ascending=asc)
    return out


def page_traffic() -> None:
    st.markdown("# Traffic Monitor")
    st.markdown('<span class="badge">● DEMO MODE</span>'
                '<span class="badge badge-dim">SIMULATED / OFFLINE DATA</span>',
                unsafe_allow_html=True)
    out = _filtered_flows()
    st.caption(f"{len(out):,} flows match the current filters "
               f"(showing up to 1,000).")
    show = out.head(1000)
    table = pd.DataFrame({
        "Timestamp": show["timestamp"].dt.strftime("%H:%M:%S"),
        "Source IP": show["src_ip"], "Destination IP": show["dst_ip"],
        "Source Port": show["src_port"], "Destination Port": show["dst_port"],
        "Protocol": show["protocol"], "TCP Flags": show["tcp_flags"],
        "Packets": show["packets"], "Bytes": show["bytes_str"],
        "Flow Duration": show["duration_str"], "Risk": show["risk"]})
    st.dataframe(table, width="stretch", hide_index=True,
                 height=380)

    st.markdown("### Suspicious flow investigation")
    susp = st.session_state.flows[
        st.session_state.flows["risk"].isin(["HIGH", "CRITICAL"])]
    if susp.empty:
        st.info("No HIGH/CRITICAL flows in the current scope.")
        return
    labels = {int(i): (f"{r.src_ip} → {r.dst_ip}:{r.dst_port} "
                       f"[{r.protocol}/{r.tcp_flags}] · {r.risk}")
              for i, r in susp.iterrows()}
    sel = st.selectbox("Select a suspicious flow", list(labels.keys()),
                       index=(list(labels.keys()).index(
                           st.session_state.selected_flow)
                           if st.session_state.selected_flow in labels else 0),
                       format_func=lambda i: labels[i])
    st.session_state.selected_flow = sel
    r = st.session_state.flows.loc[sel]
    st.markdown(f"""<div class="panel"><h4>Flow Details {V.severity_badge(r['risk'])}</h4>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px 22px;
    color:#cbd5e1;font-size:14px">
    <div>Source: <b>{r['src_ip']}:{r['src_port']}</b></div>
    <div>Destination: <b>{r['dst_ip']}:{r['dst_port']}</b></div>
    <div>Protocol / Flags: <b>{r['protocol']} / {r['tcp_flags']}</b></div>
    <div>Packets: <b>{r['packets']:,}</b></div>
    <div>Bytes: <b>{r['bytes']:,} ({r['bytes_str']})</b></div>
    <div>Flow Duration: <b>{r['duration_str']}</b></div>
    <div>Mean IAT: <b>{r['iat_mean']}s</b></div>
    <div>TTL: <b>{r['ttl']}</b></div>
    <div>Retransmissions: <b>{r['retransmissions']}</b></div>
    </div>
    <div style="margin-top:10px;color:#cbd5e1;font-size:14px">
    Risk contribution: <b style="color:#fb923c">{r['risk_contrib']:.1f}/100</b></div>
    <div style="background:#0d1528;border-radius:6px;height:10px;margin-top:6px">
    <div style="width:{r['risk_contrib']:.0f}%;height:10px;border-radius:6px;
    background:linear-gradient(90deg,#22d3ee,#fb923c,#ef4444)"></div></div>
    </div>""", unsafe_allow_html=True)


def page_forecast() -> None:
    st.markdown("# Attack Forecast")
    st.markdown('<span class="badge">DEMO FORECAST</span> '
                '<span class="badge badge-dim">PROTOTYPE WORLD MODEL</span>',
                unsafe_allow_html=True)
    f = st.session_state.forecast
    st.markdown("### Current State")
    c1, c2 = st.columns(2)
    c1.markdown(V.kpi_card("Current Risk", f"{f['current_risk']:.0f}%",
                           f"Current Stage: {f['current_stage']}", "#22d3ee"),
                unsafe_allow_html=True)
    c2.markdown(V.kpi_card("Forecast Engine", f["model"].split(" (")[0],
                           "Offline · deterministic demo inference", "#22d3ee"),
                unsafe_allow_html=True)

    st.markdown("### Forecast Horizon")
    horizon = st.segmented_control("Steps", [1, 2, 3, 4, 5],
                                   default=st.session_state.horizon)
    if horizon != st.session_state.horizon:
        st.session_state.horizon = horizon
        phase_final = (scenario_phase(st.session_state.demo_phase).final_risk
                       if st.session_state.is_demo else None)
        vis = [w for w in st.session_state.windows
               if w.id in st.session_state.visible_ids] or st.session_state.windows
        model = build_world_model(st.session_state.backend)
        st.session_state.forecast = model.predict(
            vis, horizon=horizon, threshold=st.session_state.threshold,
            scenario_final=phase_final)
        st.session_state.revealed = 0
        st.rerun()
    f = st.session_state.forecast

    if st.button("🔮 Simulate Future", type="primary"):
        with st.status("🔮 Simulating future network states …",
                       expanded=True) as stt:
            chain = ["Current Network State"] + [
                f"Future State {i + 1} — {s['stage']} "
                f"({s['risk']:.0f}%, conf {s['confidence']:.0f}%)"
                for i, s in enumerate(f["future_stages"])]
            stt.write("Current Network State ↓")
            for link in chain[1:]:
                time.sleep(0.4)
                stt.write(link + " ↓")
            st.session_state.revealed = len(f["future_stages"])
            stt.update(label="✅ Future simulation complete",
                       state="complete", expanded=False)
        st.rerun()

    # Default: the full K-step forecast is visible immediately (showcase
    # ready); "Simulate Future" progressively reveals the same chain with
    # an animated status log.
    steps = f["future_stages"]
    cols = st.columns(len(steps))
    for col, s in zip(cols, steps):
        col.markdown(
            f"""<div class="step-card"><div style="color:#94a3b8;font-size:12px">
            {s['step']}</div><div style="font-size:26px;font-weight:800;
            color:#fb923c">{s['risk']:.0f}%</div>
            <div style="font-weight:700;font-size:13px">{s['stage']}</div>
            <div style="color:#94a3b8;font-size:12px">Confidence:
            {s['confidence']:.0f}%</div></div>""",
            unsafe_allow_html=True)
    st.plotly_chart(V.kstep_forecast_chart(f["future_stages"],
                                           f["current_risk"]),
                    width="stretch")
    forecast_warning_block()
    alert_card()


def page_explain() -> None:
    st.markdown("# Why is NetForeSight Forecasting an Attack?")
    st.markdown('<span class="badge">DEMO EXPLANATION</span> '
                '<span class="badge badge-dim">SIMULATED ATTRIBUTION</span>',
                unsafe_allow_html=True)
    f = st.session_state.forecast
    step = st.selectbox("Forecast window",
                        ["T+1", "T+2", "T+3", "T+4", "T+5"][:f["horizon"]],
                        index=min(1, f["horizon"] - 1))
    st.session_state.explain_step = step
    s = next(x for x in f["future_stages"] if x["step"] == step)
    base_signals = dict(f.get("signals") or {})
    scale = 0.85 + 0.15 * (s["risk"] / max(f["current_risk"], 1))
    adj = {k: (v * scale if k in ("syn_rate", "dst_spread", "retrans") else v)
           for k, v in base_signals.items()}
    contribs = feature_contributions(adj if base_signals else None)
    st.plotly_chart(V.contribution_bar_chart(contribs),
                    width="stretch")
    st.markdown(f"> {narrative(contribs, 'current window', s['stage'])}")
    st.markdown("### SHAP-style feature contributions "
                f"({step} → {s['stage']})")
    rows = shap_style_values(contribs)
    html = ['<div class="panel">'
            '<div style="display:grid;grid-template-columns:1fr 120px 1fr;'
            'gap:4px 12px;font-size:14px">'
            '<div style="color:#94a3b8"><b>Feature</b></div>'
            '<div style="color:#94a3b8"><b>Contribution</b></div>'
            '<div style="color:#94a3b8"><b>Effect</b></div>']
    for row in rows:
        color = "#f87171" if row["value"] >= 0 else "#22c55e"
        bar = "█" * max(int(abs(row["value"]) * 30), 1)
        html.append(
            f"<div>{row['feature']}</div>"
            f"<div style='color:{color};font-family:monospace'>"
            f"{row['value']:+.2f}</div>"
            f"<div style='color:{color};font-family:monospace'>{bar}</div>")
    html.append("</div><div style='margin-top:8px'>"
                '<span class="badge">DEMO EXPLANATION</span></div></div>')
    st.markdown("".join(html), unsafe_allow_html=True)
    st.caption("Positive values push the forecast toward attack progression; "
               "negative values would push toward benign. Values are "
               "simulated attention-style attribution, not fitted SHAP values.")


def _node_stats(ip: str) -> dict:
    df: pd.DataFrame = st.session_state.flows
    if "/" in ip:  # aggregate node
        prefix = ip.split("/")[0].rsplit(".", 1)[0] + "."
        sub = df[df["src_ip"].str.startswith(prefix) |
                 df["dst_ip"].str.startswith(prefix)]
    else:
        sub = df[(df["src_ip"] == ip) | (df["dst_ip"] == ip)]
    susp = sub[sub["risk"].isin(["HIGH", "CRITICAL"])]
    return {
        "connections": len(sub),
        "suspicious": len(susp),
        "ports": sub["dst_port"].value_counts().head(4).index.tolist(),
        "risk": round(float(sub["risk_score"].max() * 100), 1) if len(sub) else 0.0,
    }


def page_graph() -> None:
    st.markdown("# Network Graph")
    st.markdown('<span class="badge">● DEMO MODE</span> '
                '<span class="badge badge-dim">SIMULATED TOPOLOGY</span>',
                unsafe_allow_html=True)
    f = st.session_state.forecast
    st.plotly_chart(V.network_graph_figure(st.session_state.node_selected),
                    width="stretch")
    st.caption("Suspicious communication paths are highlighted in red.")
    node = st.selectbox("Select a node to inspect",
                        [n["ip"] for n in V.GRAPH_NODES],
                        index=[n["ip"] for n in V.GRAPH_NODES].index(
                            st.session_state.node_selected),
                        format_func=lambda ip: next(
                            f"{n['ip']} — {n['role']}" for n in V.GRAPH_NODES
                            if n["ip"] == ip))
    st.session_state.node_selected = node
    meta = next(n for n in V.GRAPH_NODES if n["ip"] == node)
    stats = _node_stats(node)
    behav = ("Expected to participate in forecasted "
             f"{f['predicted_stage'].lower()}" if node in
             ("10.0.2.15", "10.0.4.21", "10.0.4.22", "10.0.4.27")
             else "No malicious behaviour forecast")
    st.markdown(f"""<div class="panel"><h4>Node — {node}</h4>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px 22px;
    color:#cbd5e1;font-size:14px">
    <div>IP: <b>{node}</b></div><div>Role: <b>{meta['role']}</b></div>
    <div>Connections: <b>{stats['connections']:,}</b></div>
    <div>Suspicious Connections: <b style="color:#ef4444">
    {stats['suspicious']:,}</b></div>
    <div>Targeted Ports: <b>{', '.join(map(str, stats['ports'])) or '—'}</b></div>
    <div>Current Risk: <b>{stats['risk']:.0f}%</b></div>
    </div><div style="margin-top:8px;color:#cbd5e1;font-size:14px">
    Predicted Behavior: <b style="color:#fb923c">{behav}</b></div></div>""",
                unsafe_allow_html=True)


def page_analytics() -> None:
    st.markdown("# Analytics")
    st.markdown('<span class="badge">● DEMO MODE</span> '
                '<span class="badge badge-dim">SIMULATED / OFFLINE DATA</span>',
                unsafe_allow_html=True)
    df: pd.DataFrame = st.session_state.flows
    r1c1, r1c2 = st.columns(2)
    r1c1.plotly_chart(V.stage_distribution_chart(df), width="stretch")
    r1c2.plotly_chart(V.risk_histogram(df), width="stretch")
    r2c1, r2c2 = st.columns(2)
    r2c1.plotly_chart(V.suspicious_timeseries(df), width="stretch")
    r2c2.plotly_chart(V.protocol_donut(df), width="stretch")
    r3c1, r3c2 = st.columns(2)
    r3c1.plotly_chart(V.top_ports_chart(df), width="stretch")
    r3c2.plotly_chart(V.flow_volume_chart(df), width="stretch")


def _report_text() -> str:
    f = st.session_state.forecast
    info = st.session_state.dataset_info
    contribs = st.session_state.contributions
    feats: pd.DataFrame = st.session_state.flows
    flagged = feats[feats["risk"].isin(["HIGH", "CRITICAL"])].sort_values(
        "risk_score", ascending=False).head(10)
    lines = [
        "NETFORESIGHT", "Network Attack Forecast Report",
        "─" * 40, "",
        f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset: {info.get('file_name', '—')}",
        "Environment: Offline MVP (Demo Forecast)",
        f"Model Status: {f.get('model', '—')}", "",
        f"Current Risk: {f['current_risk']:.0f}%",
        f"Forecast Risk: {f['risk']:.0f}%",
        f"Predicted Stage: {f['predicted_stage']}",
        f"Confidence: {f['confidence']:.0f}%", "",
        "Top Contributing Signals:"]
    for i, (k, v) in enumerate(sorted(contribs.items(),
                                      key=lambda kv: kv[1], reverse=True), 1):
        lines.append(f"{i}. {k} — {v:.1f}%")
    lines += ["", "Flagged Flows (top 10):"]
    for _, r in flagged.iterrows():
        lines.append(f"- {r['timestamp']} {r['src_ip']} → {r['dst_ip']}:"
                     f"{r['dst_port']} {r['protocol']}/{r['tcp_flags']} "
                     f"[{r['risk']}]")
    lines += ["", "Attack Progression:"]
    for t in f["timeline"]:
        mark = {"Observed": "✓", "Suspected": "◐",
                "Predicted": "⚠", "Future": "○"}[t["state"]]
        lines.append(f"{mark} {t['stage']} — {t['state']}")
    lines += ["", f"Analyst Notes: {st.session_state.analyst_notes or '—'}",
              "", "Note: simulated demo inference — not a validated detection."]
    return "\n".join(lines)


def page_reports() -> None:
    st.markdown("# Security Forecast Report")
    st.markdown('<span class="badge">DEMO FORECAST</span> '
                '<span class="badge badge-dim">OFFLINE MVP</span>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="report-box">{_report_text()}</div>',
                unsafe_allow_html=True)
    st.markdown("")
    c1, c2 = st.columns(2)
    c1.download_button("📄 Export Report", _report_text(),
                       file_name="netforesight_report.txt", mime="text/plain",
                       width="stretch")
    csv = st.session_state.flows[[
        "timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
        "tcp_flags", "packets", "bytes", "duration", "iat_mean", "ttl",
        "retransmissions", "risk_score", "risk"]].to_csv(index=False)
    c2.download_button("⬇ Download CSV", csv,
                       file_name="netforesight_flows.csv", mime="text/csv",
                       width="stretch")


def page_settings() -> None:
    st.markdown("# Settings")
    st.markdown("### Forecast configuration")
    thr = st.slider("Alert Threshold (%)", 50, 90,
                    int(st.session_state.threshold * 100))
    if thr != int(st.session_state.threshold * 100):
        st.session_state.threshold = thr / 100.0
        vis = [w for w in st.session_state.windows
               if w.id in st.session_state.visible_ids] or st.session_state.windows
        phase_final = (scenario_phase(st.session_state.demo_phase).final_risk
                       if st.session_state.is_demo else None)
        model = build_world_model(st.session_state.backend)
        st.session_state.forecast = model.predict(
            vis, horizon=st.session_state.horizon,
            threshold=st.session_state.threshold, scenario_final=phase_final)
        st.rerun()
    n_win = st.selectbox("Time windows", [3, 4, 5, 6],
                         index=[3, 4, 5, 6].index(st.session_state.n_windows))
    if n_win != st.session_state.n_windows:
        st.session_state.n_windows = n_win
        info = st.session_state.dataset_info
        run_analysis(st.session_state.flows, info.get("file_name", "data"),
                     info.get("file_type", "CSV"), info.get("file_size", "—"),
                     scenario_final=(scenario_phase(
                         st.session_state.demo_phase).final_risk
                         if st.session_state.is_demo else None),
                     show_progress=False)
        if st.session_state.is_demo:
            apply_demo_phase(st.session_state.demo_phase)
        else:
            st.session_state.visible_ids = [w.id for w in
                                            st.session_state.windows]
        st.rerun()
    backend = st.radio("Forecast backend", ["demo", "torch"],
                       index=["demo", "torch"].index(st.session_state.backend),
                       format_func=lambda b: "DemoWorldModel (Prototype)"
                       if b == "demo" else
                       "TorchLSTMWorldModel (Untrained Placeholder)")
    if backend != st.session_state.backend:
        st.session_state.backend = backend
        st.rerun()
    tw = TorchLSTMWorldModel()
    st.info(f"**DemoWorldModel** — deterministic demo inference (active "
            f"backend: `{st.session_state.backend}`).\n\n"
            f"**TorchLSTMWorldModel** — reference LSTM→K-states→stage-logits "
            f"architecture; weights: `untrained-placeholder`; "
            f"torch installed: `{tw.torch_available}`. Swap the trained "
            f"module in later without UI changes.")
    st.markdown("### Session")
    st.write(f"Saved investigations: {len(st.session_state.investigations)}")
    if st.button("🧹 Reset Session"):
        for k in list(_DEFAULTS):
            st.session_state[k] = _DEFAULTS[k] if not isinstance(
                _DEFAULTS[k], (list, dict)) else type(_DEFAULTS[k])()
        st.rerun()
    st.markdown("### Environment")
    st.code("Data Source: local files only\nInference: local (no cloud APIs)\n"
            "Connectivity required: none at runtime", language="text")


PAGES = {"Dashboard": page_dashboard, "Traffic Monitor": page_traffic,
         "Attack Forecast": page_forecast, "Explainability": page_explain,
         "Network Graph": page_graph, "Analytics": page_analytics,
         "Reports": page_reports, "Settings": page_settings}
PAGES[page]()
st.markdown("---")
st.caption("NetForeSight · Offline MVP · Simulated demo inference — "
           "forecasts are illustrative, not validated detections. "
           "The analyst remains in control: no automated blocking.")
