"""Explainability: demo attribution over traffic signals (offline, local).

The prototype uses an attention-style attribution layer: per-signal raw
scores are modulated by the observed window statistics and normalized to
percentages, with a SHAP-style signed contribution table for the panel.
All outputs are labelled as demo/simulated attribution in the UI.
"""
from __future__ import annotations

import numpy as np

from . import MODEL_FEATURES

# Base profile (fractions) — calibrated so the reference demo state yields
# 31 / 24 / 18 / 12 / 9 / 6.
_BASE_PROFILE = {
    "SYN Activity": 0.31,
    "Destination Port Pattern": 0.22,
    "Inter-arrival Time": 0.18,
    "Retransmissions": 0.12,
    "TTL Variation": 0.10,
    "Flow Duration": 0.07,
}


def _signal_strengths(signals: dict | None) -> dict[str, float]:
    if not signals:
        return dict(_BASE_PROFILE)
    syn = float(np.clip(signals.get("syn_rate", 0.2) * 3.0, 0.15, 1.0))
    dst = float(np.clip(signals.get("dst_spread", 0.3) * 2.5, 0.15, 1.0))
    iat = float(np.clip(1.0 - signals.get("iat", 0.5), 0.10, 1.0))
    retx = float(np.clip(signals.get("retrans", 0.1) * 4.0, 0.10, 1.0))
    ttl = 0.55
    dur = 0.45
    return {
        "SYN Activity": syn,
        "Destination Port Pattern": dst,
        "Inter-arrival Time": iat,
        "Retransmissions": retx,
        "TTL Variation": ttl,
        "Flow Duration": dur,
    }


def feature_contributions(signals: dict | None = None) -> dict[str, float]:
    """Percentage contribution per traffic signal (sums to 100)."""
    strengths = _signal_strengths(signals)
    raw = {k: _BASE_PROFILE[k] * (0.55 + 0.45 * strengths[k])
           for k in MODEL_FEATURES}
    total = sum(raw.values()) or 1.0
    pct = {k: round(100.0 * v / total, 1) for k, v in raw.items()}
    # Fix rounding drift on the largest contributor.
    drift = round(100.0 - sum(pct.values()), 1)
    top = max(pct, key=pct.get)
    pct[top] = round(pct[top] + drift, 1)
    return pct


def shap_style_values(contributions: dict[str, float]) -> list[dict]:
    """Signed SHAP-style contributions (+/- risk push), demo attribution."""
    order = ["SYN Activity", "Destination Port Pattern", "Inter-arrival Time",
             "Retransmissions", "TTL Variation", "Flow Duration"]
    short = {"SYN Activity": "TCP SYN Count",
             "Destination Port Pattern": "Destination Port",
             "Inter-arrival Time": "Mean IAT",
             "Retransmissions": "Retransmissions",
             "TTL Variation": "TTL Variance",
             "Flow Duration": "Flow Duration"}
    return [{"feature": short[k],
             "value": round(contributions.get(k, 0.0) / 100.0, 2)}
            for k in order]


def narrative(contributions: dict[str, float], window_label: str,
              predicted_stage: str) -> str:
    top = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)[:3]
    parts = ", ".join(f"{k.lower()} ({v:.0f}% contribution)"
                      for k, v in top)
    return (
        f"The observed traffic trajectory ({window_label.lower()}) contains "
        f"{parts}, which together push the world model toward forecasting "
        f"'{predicted_stage}'. Rising SYN volume against internal service "
        f"ports, combined with compressed inter-arrival behaviour, is "
        f"consistent with early-stage attack progression rather than "
        f"isolated benign bursts."
    )
