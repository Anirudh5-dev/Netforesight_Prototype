"""Flow-level feature extraction and normalization (offline, local only)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _fmt_bytes(n: int) -> str:
    if n >= 1_048_576:
        return f"{n / 1_048_576:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def _fmt_dur(s: float) -> str:
    return f"{s:.2f}s"


def extract_features(flows: pd.DataFrame) -> pd.DataFrame:
    """Enrich a standardized flow table with derived model features."""
    df = flows.copy()
    df["is_syn"] = df["tcp_flags"].str.contains("SYN", na=False).astype(int)
    df["is_rst"] = df["tcp_flags"].str.contains("RST", na=False).astype(int)
    df["is_internal_dst"] = df["dst_ip"].astype(str).str.startswith(
        ("10.", "192.168.", "172.")).astype(int)
    df["is_svc_port"] = df["dst_port"].isin(
        [22, 135, 139, 445, 3389, 5985]).astype(int)
    df["bytes_per_pkt"] = (df["bytes"] / df["packets"].clip(lower=1)).round(1)
    df["pkts_per_sec"] = (df["packets"] / df["duration"].clip(lower=0.01)).round(1)
    # Per-flow risk contribution used by the investigation panel (0-100).
    df["risk_contrib"] = (
        40 * df["risk_score"]
        + 20 * df["is_syn"] * df["is_svc_port"]
        + 10 * df["is_svc_port"]
        + 10 * (df["retransmissions"].clip(upper=10) / 10)
        + 10 * (1 - (df["iat_mean"].clip(upper=1.0)))
        + 10 * df["is_rst"]
    ).clip(0, 100).round(1)
    df["bytes_str"] = df["bytes"].apply(_fmt_bytes)
    df["duration_str"] = df["duration"].apply(_fmt_dur)
    return df


def normalize_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Min-max normalization of numeric model inputs to [0, 1]."""
    numeric = ["packets", "bytes", "duration", "iat_mean", "ttl",
               "retransmissions", "risk_score", "bytes_per_pkt", "pkts_per_sec"]
    stats: dict = {}
    for col in numeric:
        lo, hi = float(df[col].min()), float(df[col].max())
        span = (hi - lo) or 1.0
        df[f"{col}_n"] = ((df[col] - lo) / span).round(4)
        stats[col] = {"min": lo, "max": hi}
    return df, stats


def summarize_flows(df: pd.DataFrame) -> dict:
    suspicious = df[df["risk"].isin(["HIGH", "CRITICAL"])]
    return {
        "active_flows": int(len(df)),
        "suspicious_flows": int(len(suspicious)),
        "suspicious_pct": round(100 * len(suspicious) / max(len(df), 1), 2),
        "unique_src": int(df["src_ip"].nunique()),
        "unique_dst": int(df["dst_ip"].nunique()),
    }
