"""Plotly chart builders and HTML card helpers (all local, offline)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import SEVERITY_COLORS

BG = "#0a0f1e"
CARD = "#111a30"
GRID = "rgba(148,163,184,0.15)"
TEXT = "#e6edf7"
MUTED = "#94a3b8"
CYAN = "#22d3ee"
ORANGE = "#fb923c"
RED = "#ef4444"

SEVERITY_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def _base_layout(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT, "family": "Inter, Segoe UI, sans-serif"},
        margin={"l": 50, "r": 20, "t": 40, "b": 40},
        legend={"orientation": "h", "y": 1.08, "x": 0,
                "font": {"size": 11, "color": MUTED}},
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, tickfont={"color": MUTED})
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont={"color": MUTED})
    return fig


# --------------------------------------------------------------------------- #
# Forecast                                                                     #
# --------------------------------------------------------------------------- #

def risk_forecast_chart(forecast: dict) -> go.Figure:
    obs = forecast["observed_risk"]
    fut = forecast["forecast_risk"]
    thr = forecast["threshold"] * 100.0
    n_obs = len(obs)
    y_fut = [obs[-1]] + fut
    x_all = [f"T+{i}" for i in range(n_obs + len(y_fut) - 1)]
    x_obs = x_all[:n_obs]
    x_fut = x_all[n_obs - 1: n_obs - 1 + len(y_fut)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_obs, y=obs, mode="lines+markers",
                             name="Observed Risk", line={"color": CYAN, "width": 3},
                             marker={"size": 8, "color": CYAN},
                             hovertemplate="%{x}<br>Observed: %{y:.1f}%<extra></extra>"))
    band = 6.0
    fig.add_trace(go.Scatter(
        x=x_fut + x_fut[::-1],
        y=[v + band for v in y_fut] + [v - band for v in y_fut][::-1],
        fill="toself", fillcolor="rgba(251,146,60,0.15)",
        line={"width": 0}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x_fut, y=y_fut, mode="lines+markers",
                             name="Forecast Risk",
                             line={"color": ORANGE, "width": 3, "dash": "dash"},
                             marker={"size": 9, "color": ORANGE,
                                     "symbol": "diamond"},
                             hovertemplate="%{x}<br>Forecast: %{y:.1f}%<extra></extra>"))
    fig.add_hline(y=thr, line_dash="dot", line_color=RED, line_width=2,
                  annotation_text=f"Alert Threshold = {thr:.0f}%",
                  annotation_position="top right",
                  annotation_font_color=RED)
    now_x = x_obs[-1]
    fig.add_shape(type="line", x0=now_x, x1=now_x, xref="x",
                  y0=0, y1=1, yref="paper",
                  line={"dash": "dash", "color": MUTED, "width": 1})
    fig.add_annotation(x=now_x, y=97, xref="x", yref="y", text="NOW",
                       showarrow=False, font={"color": MUTED, "size": 10})
    if forecast.get("crosses_threshold") and forecast.get("crossing_index") is not None:
        cx = x_fut[1 + forecast["crossing_index"]] \
            if len(x_fut) > 1 + forecast["crossing_index"] else x_fut[-1]
        cy = fut[forecast["crossing_index"]]
        fig.add_trace(go.Scatter(x=[cx], y=[cy], mode="markers+text",
                                 name="Threshold crossing",
                                 marker={"size": 14, "color": RED,
                                         "symbol": "x"},
                                 text=["crossing"], textposition="top center",
                                 textfont={"color": RED, "size": 11},
                                 hovertemplate="Threshold crossed at %{x}: %{y:.1f}%<extra></extra>"))
    fig.update_yaxes(title="Risk (%)", range=[0, 100])
    fig.update_xaxes(title="Time window")
    fig.update_layout(title={"text": "Future Attack Risk Forecast", "x": 0,
                             "font": {"size": 16, "color": TEXT}})
    return _base_layout(fig, height=420)


def kstep_forecast_chart(future_stages: list[dict], current_risk: float) -> go.Figure:
    x = ["Now"] + [s["step"] for s in future_stages]
    y = [current_risk] + [s["risk"] for s in future_stages]
    colors = [CYAN] + [ORANGE] * len(future_stages)
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="lines+markers",
        line={"color": ORANGE, "width": 3},
        marker={"size": 10, "color": colors},
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
    fig.update_yaxes(title="Risk (%)", range=[0, 100])
    fig.update_xaxes(title="Forecast step")
    fig.update_layout(title={"text": "K-Step Risk Trajectory", "x": 0,
                             "font": {"size": 15, "color": TEXT}},
                      showlegend=False)
    return _base_layout(fig, height=320)


def contribution_bar_chart(contributions: dict[str, float]) -> go.Figure:
    items = sorted(contributions.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    colors = ["#38bdf8", "#818cf8", "#22d3ee", "#fbbf24", "#fb923c", "#f87171"]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker={"color": colors[-len(labels):]},
        text=[f"{v:.0f}%" for v in values], textposition="outside",
        textfont={"color": TEXT},
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    fig.update_xaxes(title="Contribution (%)", range=[0, max(values) * 1.35])
    fig.update_layout(title={"text": "Top Contributing Traffic Signals", "x": 0,
                             "font": {"size": 15, "color": TEXT}},
                      showlegend=False)
    return _base_layout(fig, height=360)


# --------------------------------------------------------------------------- #
# Analytics                                                                    #
# --------------------------------------------------------------------------- #

def stage_distribution_chart(flows: pd.DataFrame) -> go.Figure:
    from . import stage_for_risk
    stages = flows["risk_score"].apply(lambda s: stage_for_risk(s * 100))
    order = ["Reconnaissance", "Initial Access", "Lateral Movement",
             "Command & Control", "Exfiltration"]
    counts = stages.value_counts()
    vals = [int(counts.get(s, 0)) for s in order]
    fig = go.Figure(go.Bar(
        x=order, y=vals, marker={"color": ["#38bdf8", "#22d3ee", "#fb923c", "#f87171", "#a855f7"]},
        text=vals, textposition="outside", textfont={"color": TEXT},
        hovertemplate="%{x}: %{y} flows<extra></extra>"))
    fig.update_xaxes(tickangle=-18)
    fig.update_layout(title={"text": "Attack Stage Distribution", "x": 0,
                             "font": {"size": 14, "color": TEXT}},
                      showlegend=False)
    return _base_layout(fig, height=340)


def risk_histogram(flows: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=flows["risk_score"] * 100, nbinsx=30,
        marker={"color": CYAN, "line": {"color": BG, "width": 0.5}},
        hovertemplate="Risk %{x:.0f}%: %{y} flows<extra></extra>"))
    fig.update_xaxes(title="Flow risk score (%)")
    fig.update_yaxes(title="Flows")
    fig.update_layout(title={"text": "Risk Distribution", "x": 0,
                             "font": {"size": 14, "color": TEXT}},
                      showlegend=False)
    return _base_layout(fig, height=340)


def suspicious_timeseries(flows: pd.DataFrame) -> go.Figure:
    df = flows.copy()
    df["bucket"] = df["timestamp"].dt.floor("30s")
    total = df.groupby("bucket").size()
    susp = df[df["risk"].isin(["HIGH", "CRITICAL"])].groupby("bucket").size()
    idx = total.index
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=total.values, mode="lines",
                             name="All flows", line={"color": CYAN, "width": 2},
                             hovertemplate="%{x|%H:%M:%S}<br>%{y} flows<extra></extra>"))
    fig.add_trace(go.Scatter(x=susp.index, y=susp.values, mode="lines+markers",
                             name="Suspicious", line={"color": RED, "width": 2},
                             marker={"size": 5},
                             hovertemplate="%{x|%H:%M:%S}<br>%{y} suspicious<extra></extra>"))
    fig.update_xaxes(title="Time")
    fig.update_yaxes(title="Flows / 30 s")
    fig.update_layout(title={"text": "Suspicious Traffic Over Time", "x": 0,
                             "font": {"size": 14, "color": TEXT}})
    return _base_layout(fig, height=340)


def protocol_donut(flows: pd.DataFrame) -> go.Figure:
    counts = flows["protocol"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(), values=counts.values.tolist(), hole=0.55,
        marker={"colors": [CYAN, ORANGE, "#a855f7", "#22c55e", MUTED]},
        textinfo="label+percent", textfont={"color": TEXT},
        hovertemplate="%{label}: %{value} flows<extra></extra>"))
    fig.update_layout(title={"text": "Protocol Distribution", "x": 0,
                             "font": {"size": 14, "color": TEXT}},
                      showlegend=False)
    return _base_layout(fig, height=340)


def top_ports_chart(flows: pd.DataFrame, top_n: int = 8) -> go.Figure:
    counts = flows["dst_port"].value_counts().head(top_n).sort_values()
    fig = go.Figure(go.Bar(
        x=counts.values, y=[f"Port {p}" for p in counts.index], orientation="h",
        marker={"color": ORANGE},
        text=counts.values, textposition="outside", textfont={"color": TEXT},
        hovertemplate="%{y}: %{x} flows<extra></extra>"))
    fig.update_xaxes(title="Flows")
    fig.update_layout(title={"text": "Top Targeted Ports", "x": 0,
                             "font": {"size": 14, "color": TEXT}},
                      showlegend=False)
    return _base_layout(fig, height=340)


def flow_volume_chart(flows: pd.DataFrame) -> go.Figure:
    df = flows.copy()
    df["bucket"] = df["timestamp"].dt.floor("30s")
    vol = df.groupby("bucket").size()
    fig = go.Figure(go.Scatter(x=vol.index, y=vol.values, mode="lines",
                               fill="tozeroy", fillcolor="rgba(34,211,238,0.15)",
                               line={"color": CYAN, "width": 2},
                               hovertemplate="%{x|%H:%M:%S}<br>%{y} flows<extra></extra>"))
    fig.update_xaxes(title="Time")
    fig.update_yaxes(title="Flows / 30 s")
    fig.update_layout(title={"text": "Flow Volume", "x": 0,
                             "font": {"size": 14, "color": TEXT}},
                      showlegend=False)
    return _base_layout(fig, height=340)


def window_risk_chart(windows) -> go.Figure:
    labels = [f"W{w.id}" for w in windows]
    risks = [w.risk for w in windows]
    colors = [SEVERITY_COLORS[
        "CRITICAL" if r >= 80 else "HIGH" if r >= 60 else "MEDIUM" if r >= 35 else "LOW"]
        for r in risks]
    fig = go.Figure(go.Bar(x=labels, y=risks, marker={"color": colors},
                           text=[f"{r:.0f}%" for r in risks],
                           textposition="outside", textfont={"color": TEXT},
                           hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
    fig.update_yaxes(title="Window risk (%)", range=[0, 100])
    fig.update_layout(title={"text": "Risk by Network State Window", "x": 0,
                             "font": {"size": 14, "color": TEXT}},
                      showlegend=False)
    return _base_layout(fig, height=300)


# --------------------------------------------------------------------------- #
# Network graph                                                                #
# --------------------------------------------------------------------------- #

GRAPH_NODES = [
    {"ip": "203.0.113.7", "role": "External Client", "x": 0.0, "y": 2.0},
    {"ip": "10.0.1.1", "role": "Gateway", "x": 1.0, "y": 2.0},
    {"ip": "10.0.2.15", "role": "Compromised Host (suspect)", "x": 2.0, "y": 2.6},
    {"ip": "10.0.2.0/24", "role": "Workstations", "x": 2.0, "y": 1.4},
    {"ip": "10.0.4.21", "role": "Server A (targeted)", "x": 3.2, "y": 2.8},
    {"ip": "10.0.4.22", "role": "Server B (targeted)", "x": 3.2, "y": 2.0},
    {"ip": "10.0.4.27", "role": "Server C (targeted)", "x": 3.2, "y": 1.2},
]

GRAPH_EDGES = [
    ("203.0.113.7", "10.0.1.1", False),
    ("10.0.1.1", "10.0.2.15", False),
    ("10.0.1.1", "10.0.2.0/24", False),
    ("10.0.2.15", "10.0.4.21", True),
    ("10.0.2.15", "10.0.4.22", True),
    ("10.0.2.15", "10.0.4.27", True),
    ("10.0.2.0/24", "10.0.4.21", False),
    ("10.0.2.0/24", "10.0.4.22", False),
]


def network_graph_figure(selected_ip: str | None = None) -> go.Figure:
    pos = {n["ip"]: (n["x"], n["y"]) for n in GRAPH_NODES}
    fig = go.Figure()
    for a, b, susp in GRAPH_EDGES:
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line={"color": RED if susp else "rgba(148,163,184,0.5)",
                  "width": 3 if susp else 1.5,
                  "dash": "solid" if susp else "dot"},
            showlegend=False, hoverinfo="skip"))
    for n in GRAPH_NODES:
        susp = n["ip"] == "10.0.2.15" or "(targeted)" in n["role"]
        sel = n["ip"] == selected_ip
        fig.add_trace(go.Scatter(
            x=[n["x"]], y=[n["y"]], mode="markers+text",
            marker={"size": 30 if sel else 22,
                    "color": RED if n["ip"] == "10.0.2.15" else
                    (ORANGE if susp else CYAN),
                    "line": {"color": "#fff" if sel else BG, "width": 3 if sel else 2}},
            text=[n["role"]], textposition="bottom center",
            textfont={"color": TEXT, "size": 10},
            name=n["ip"],
            hovertemplate=f"{n['ip']}<br>{n['role']}<extra></extra>"))
    fig.update_xaxes(visible=False, range=[-0.4, 3.7])
    fig.update_yaxes(visible=False, range=[0.6, 3.3])
    fig.update_layout(title={"text": "Network Graph — suspicious paths highlighted",
                             "x": 0, "font": {"size": 14, "color": TEXT}},
                      showlegend=False)
    return _base_layout(fig, height=420)


# --------------------------------------------------------------------------- #
# HTML helpers                                                                 #
# --------------------------------------------------------------------------- #

def kpi_card(label: str, value: str, sub: str = "", accent: str = CYAN) -> str:
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value" style="color:{accent}">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>"""


def severity_badge(sev: str) -> str:
    color = SEVERITY_COLORS.get(sev, MUTED)
    return (f'<span class="badge" style="border-color:{color};color:{color}">'
            f'{sev}</span>')


def timeline_html(timeline: list[dict], predicted_stage: str,
                  confidence: float) -> str:
    style = {"Observed": ("✓", "#22c55e", "solid"),
             "Suspected": ("◐", "#eab308", "solid"),
             "Predicted": ("⚠", "#fb923c", "dashed"),
             "Future": ("○", "#64748b", "dotted")}
    cards = []
    for i, item in enumerate(timeline):
        mark, color, border = style[item["state"]]
        tag = (f'<span class="badge" style="border-color:{color};color:{color}">'
               f'{item["state"].upper()}</span>')
        arrow = '<div class="tl-arrow">↓</div>' if i < len(timeline) - 1 else ""
        cards.append(f"""
        <div class="tl-card" style="border:1px {border} {color}55">
          <div style="font-size:20px;color:{color}">{mark}</div>
          <div class="tl-stage">{item["stage"]}</div>
          <div style="margin-top:6px">{tag}</div>
        </div>{arrow}""")
    return f"""
    <div class="tl-wrap">{''.join(cards)}</div>
    <div style="display:flex;gap:12px;margin-top:14px;flex-wrap:wrap">
      <div class="info-chip">Predicted Next Stage: <b style="color:{ORANGE}">
        {predicted_stage.upper()}</b></div>
      <div class="info-chip">Confidence: <b style="color:{CYAN}">
        {confidence:.0f}%</b></div>
    </div>"""
