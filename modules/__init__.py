"""NetForeSight shared constants and small utilities."""
from __future__ import annotations

ATTACK_STAGES = [
    "Reconnaissance",
    "Initial Access",
    "Lateral Movement",
    "Command & Control",
    "Exfiltration",
]

# Upper-exclusive risk boundaries mapping risk (0-100) -> stage.
STAGE_BOUNDARIES = [
    (45.0, "Reconnaissance"),
    (70.0, "Initial Access"),
    (85.0, "Lateral Movement"),
    (95.0, "Command & Control"),
    (101.0, "Exfiltration"),
]

SEVERITY_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
SEVERITY_COLORS = {
    "LOW": "#22c55e",
    "MEDIUM": "#eab308",
    "HIGH": "#f97316",
    "CRITICAL": "#ef4444",
}

MODEL_FEATURES = [
    "SYN Activity",
    "Destination Port Pattern",
    "Inter-arrival Time",
    "Retransmissions",
    "TTL Variation",
    "Flow Duration",
]


def stage_for_risk(risk_pct: float) -> str:
    """Map a 0-100 risk value to a MITRE-ATT&CK-style attack stage."""
    for boundary, stage in STAGE_BOUNDARIES:
        if risk_pct < boundary:
            return stage
    return ATTACK_STAGES[-1]


def severity_for_score(score: float) -> str:
    """Map a 0-1 risk score to a severity label."""
    if score >= 0.80:
        return "CRITICAL"
    if score >= 0.60:
        return "HIGH"
    if score >= 0.35:
        return "MEDIUM"
    return "LOW"
