"""Time-windowed network state representation.

Traffic is grouped into ordered, chronological windows. Each window is
summarized as a *network state* (volumes, diversity, timing behaviour and a
risk estimate) instead of classifying isolated flows. The ordered states are
the input sequence for the world model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class NetworkWindow:
    id: int
    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    flows: int
    unique_src: int
    unique_dst: int
    syn_count: int
    avg_iat: float
    retrans: int
    risk: float  # 0-100
    top_ports: list = field(default_factory=list)
    state_vector: np.ndarray = field(default_factory=lambda: np.zeros(8))
    predicted: bool = False


def _window_label(risk: float, syn_rate: float) -> str:
    if risk < 25:
        return "Normal Activity"
    if risk < 42:
        return "Increased Reconnaissance"
    if risk < 58:
        return "Suspicious Port Activity"
    if risk < 72:
        return "Possible Internal Movement"
    return "High-Risk Activity"


def _window_risk(sub: pd.DataFrame) -> float:
    """Heuristic window risk (0-100) from aggregate traffic signals.

    Weighted toward attack-progression indicators (service-directed SYN,
    suspicious-flow share) while background signals (IAT compression,
    destination spread) set the floor for quiet windows.
    """
    syn = sub["tcp_flags"].str.contains("SYN", na=False)
    svc_syn = float((sub["dst_port"].isin(
        [22, 135, 139, 445, 3389, 5985]) & syn).mean())
    syn_rate = float(syn.mean())
    n = max(len(sub), 1)
    dst_spread = min(sub["dst_ip"].nunique() / n * 6.0, 1.0)
    iat_sig = float(np.clip(1.0 - sub["iat_mean"].mean(), 0.0, 1.0))
    susp = float(sub["risk"].isin(["HIGH", "CRITICAL"]).mean())
    risk = 0.7 + 100.0 * (0.50 * svc_syn + 0.13 * syn_rate + 0.02 * dst_spread
                          + 10.0 * susp + 0.26 * iat_sig)
    return round(float(np.clip(risk, 2, 99)), 1)


def generate_windows(flows: pd.DataFrame, n_windows: int = 5) -> list[NetworkWindow]:
    """Split flows into ``n_windows`` equal-duration chronological windows."""
    if flows.empty:
        return []
    t0, t1 = flows["timestamp"].min(), flows["timestamp"].max()
    span = max((t1 - t0).total_seconds(), 1.0)
    edges = [t0 + pd.Timedelta(seconds=span * i / n_windows)
             for i in range(n_windows + 1)]
    edges[-1] = t1  # include the final record despite float rounding
    windows: list[NetworkWindow] = []
    for i in range(n_windows):
        lo, hi = edges[i], edges[i + 1]
        sub = flows[(flows["timestamp"] >= lo)
                    & ((flows["timestamp"] < hi) | (hi == edges[-1]) & (flows["timestamp"] <= hi))]
        if sub.empty:
            continue
        risk = _window_risk(sub)
        syn = int(sub["tcp_flags"].str.contains("SYN", na=False).sum())
        syn_rate = syn / max(len(sub), 1)
        top_ports = sub["dst_port"].value_counts().head(3).index.tolist()
        vec = np.array([
            len(sub) / 5000.0,
            sub["src_ip"].nunique() / 80.0,
            sub["dst_ip"].nunique() / 250.0,
            syn_rate,
            float(np.clip(1.0 - sub["iat_mean"].mean(), 0, 1)),
            float(np.clip(sub["retransmissions"].mean() / 3.0, 0, 1)),
            float(sub["risk_score"].mean()),
            risk / 100.0,
        ], dtype=float).round(4)
        windows.append(NetworkWindow(
            id=i + 1,
            label=_window_label(risk, syn_rate),
            start=sub["timestamp"].min(), end=sub["timestamp"].max(),
            flows=len(sub),
            unique_src=int(sub["src_ip"].nunique()),
            unique_dst=int(sub["dst_ip"].nunique()),
            syn_count=syn,
            avg_iat=round(float(sub["iat_mean"].mean()), 3),
            retrans=int(sub["retransmissions"].sum()),
            risk=risk,
            top_ports=top_ports,
            state_vector=vec,
        ))
    for k, w in enumerate(windows, start=1):
        w.id = k
    return windows


def windows_to_dataframe(windows: list[NetworkWindow]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Window": f"Window {w.id:02d}",
        "Label": w.label,
        "Flows": w.flows,
        "Unique Sources": w.unique_src,
        "Unique Destinations": w.unique_dst,
        "SYN Count": w.syn_count,
        "Avg IAT (s)": w.avg_iat,
        "Risk %": w.risk,
    } for w in windows])


def state_matrix(windows: list[NetworkWindow]) -> np.ndarray:
    """Stacked state vectors: the sequence consumed by the world model."""
    if not windows:
        return np.zeros((0, 8))
    return np.stack([w.state_vector for w in windows])
