"""Forecast engine: world-model abstraction + deterministic demo engine.

Conceptual architecture::

    Previous N Network Windows (state vectors)
                    |
                LSTM Model            <-  TorchLSTMWorldModel (placeholder)
                    |
            Next K Network States
                    |
           Stage Probabilities (Reconnaissance ... Exfiltration)

The 40% prototype ships with :class:`DemoWorldModel`, a deterministic,
heuristic stand-in that produces realistic demo trajectories. The Streamlit
UI only depends on the :class:`WorldModel` interface, so a trained PyTorch
LSTM can replace the demo engine later without touching the UI.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from . import ATTACK_STAGES, stage_for_risk
from .windowing import NetworkWindow

DEFAULT_THRESHOLD = 0.70
DEFAULT_HORIZON = 5

# Step deltas of the reference demo trajectory (current 68 -> 79,84,87,90,92).
_REFERENCE_DELTAS = [11.0, 16.0, 19.0, 22.0, 24.0]
_REFERENCE_CONFS = [87.0, 91.0, 88.0, 83.0, 78.0]


# --------------------------------------------------------------------------- #
# Demo scenario progression                                                   #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ScenarioPhase:
    phase: int
    title: str
    final_risk: float  # forecast risk at the end of the visible horizon
    note: str


SCENARIO_PHASES: list[ScenarioPhase] = [
    ScenarioPhase(1, "Normal network behavior", 28.0,
                  "Baseline traffic. No attacker progression detected."),
    ScenarioPhase(2, "Reconnaissance activity increases", 44.0,
                  "Scanning behaviour emerging at the perimeter."),
    ScenarioPhase(3, "SYN activity increases", 58.0,
                  "Rising SYN rate and shorter inter-arrival times."),
    ScenarioPhase(4, "Multiple internal ports are probed", 66.0,
                  "Service ports (445/3389/22) probed from an internal host."),
    ScenarioPhase(5, "Connections to multiple internal systems increase", 69.0,
                  "One source fans out to several internal destinations."),
    ScenarioPhase(6, "Model forecasts lateral movement", 79.0,
                  "World model predicts Lateral Movement within 2 windows."),
    ScenarioPhase(7, "Future risk crosses the alert threshold", 84.0,
                  "Forecast trajectory crosses the 70% alert threshold."),
]

PIPELINE_STAGES = ["Observe", "Window", "Learn", "Forecast",
                   "Explain", "Prioritize", "Review"]


def scenario_phase(phase: int) -> ScenarioPhase:
    phase = int(np.clip(phase, 1, len(SCENARIO_PHASES)))
    return SCENARIO_PHASES[phase - 1]


# --------------------------------------------------------------------------- #
# World model abstraction                                                     #
# --------------------------------------------------------------------------- #

class WorldModel(ABC):
    """Interface every forecast backend must implement."""

    name = "WorldModel"

    @abstractmethod
    def predict(self, network_windows: list[NetworkWindow], horizon: int = 5,
                threshold: float = DEFAULT_THRESHOLD,
                scenario_final: float | None = None) -> dict:
        """Return a forecast dict (see ``forecast_engine`` docs)."""


def _shape_trajectory(current: float, final: float, horizon: int) -> list[float]:
    """Deterministic K-step risk trajectory from current toward final.

    Anchored so the primary forecast point (T+2, or T+1 for horizon 1)
    lands exactly on ``final``; later steps extend the reference curve.
    """
    anchor = _REFERENCE_DELTAS[1] if horizon >= 2 else _REFERENCE_DELTAS[0]
    scale = (final - current) / anchor if anchor else 0.0
    out = []
    for i in range(horizon):
        ref = _REFERENCE_DELTAS[i] if i < len(_REFERENCE_DELTAS) else 24.0 + 2 * (i - 4)
        out.append(round(float(np.clip(current + ref * scale, 2.0, 97.0)), 1))
    return out


def _shape_confidences(horizon: int, progress: float) -> list[float]:
    """Confidence profile peaking near the primary forecast point."""
    out = []
    for i in range(horizon):
        ref = _REFERENCE_CONFS[i] if i < len(_REFERENCE_CONFS) else 76.0
        out.append(round(ref * progress + 84.0 * (1.0 - progress), 1))
    return out


def _build_timeline(current_stage: str, predicted_stage: str) -> list[dict]:
    cur = ATTACK_STAGES.index(current_stage)
    pred = ATTACK_STAGES.index(predicted_stage)
    timeline = []
    for i, stage in enumerate(ATTACK_STAGES):
        if i <= cur:
            state = "Observed"
        elif i == pred and pred > cur:
            state = "Predicted"
        elif i == cur + 1:
            state = "Suspected"
        else:
            state = "Future"
        timeline.append({"stage": stage, "state": state})
    return timeline


def build_forecast(network_windows: list[NetworkWindow], horizon: int = 5,
                   threshold: float = DEFAULT_THRESHOLD,
                   scenario_final: float | None = None,
                   model_name: str = "DemoWorldModel") -> dict:
    """Shared deterministic forecast builder used by every backend."""
    horizon = int(np.clip(horizon, 1, 5))
    risks = [w.risk for w in network_windows] or [20.0]
    observed = risks[-4:]
    while len(observed) < 4:  # left-pad short histories
        observed = [observed[0]] + observed
    observed = [round(float(r), 1) for r in observed]
    current = observed[-1]

    if scenario_final is None:
        # Data-driven default: extrapolate recent momentum.
        momentum = (observed[-1] - observed[0]) / max(len(observed) - 1, 1)
        scenario_final = float(np.clip(current + momentum * 2.2, 5, 95))

    cards = _shape_trajectory(current, float(scenario_final), horizon)
    progress = float(np.clip((scenario_final - 20.0) / 64.0, 0.0, 1.0))
    confs = _shape_confidences(horizon, progress)

    future_stages = [{
        "step": f"T+{i + 1}",
        "risk": cards[i],
        "stage": stage_for_risk(cards[i]),
        "confidence": confs[i],
    } for i in range(horizon)]

    # Headline forecast = card closest to the primary horizon point (T+2).
    head_idx = min(1, horizon - 1)
    head = future_stages[head_idx]
    current_stage = stage_for_risk(current)
    predicted_stage = stage_for_risk(head["risk"])

    forecast_risk = cards[:2] if horizon >= 2 else cards[:1]
    crossing = next((i for i, v in enumerate(forecast_risk)
                     if v >= threshold * 100.0), None)

    signals = {}
    if network_windows:
        last = network_windows[-1]
        n = max(last.flows, 1)
        signals = {
            "syn_rate": round(last.syn_count / n, 4),
            "dst_spread": round(min(last.unique_dst / max(n, 1) * 6.0, 1.0), 4),
            "iat": last.avg_iat,
            "retrans": round(last.retrans / n, 4),
            "risk": last.risk,
        }

    return {
        "model": model_name,
        "demo": True,
        "current_risk": current,
        "current_stage": current_stage,
        "observed_risk": observed,
        "forecast_risk": forecast_risk,
        "risk": head["risk"],
        "confidence": head["confidence"],
        "predicted_stage": predicted_stage,
        "future_stages": future_stages,
        "threshold": float(threshold),
        "crossing_index": crossing,
        "crosses_threshold": crossing is not None,
        "timeline": _build_timeline(current_stage, predicted_stage),
        "signals": signals,
        "horizon": horizon,
    }


class DemoWorldModel(WorldModel):
    """Deterministic prototype engine (documented demo inference)."""

    name = "DemoWorldModel (Prototype)"

    def predict(self, network_windows: list[NetworkWindow], horizon: int = 5,
                threshold: float = DEFAULT_THRESHOLD,
                scenario_final: float | None = None) -> dict:
        return build_forecast(network_windows, horizon, threshold,
                              scenario_final, model_name=self.name)


class TorchLSTMWorldModel(WorldModel):
    """Reference PyTorch world-model architecture (untrained placeholder).

    The ``nn.Module`` below is the integration target for the production
    model. Until trained weights exist, :meth:`predict` reuses the same
    deterministic shaping as the demo engine and is explicitly flagged as
    a placeholder, so the UI contract stays identical.
    """

    name = "TorchLSTMWorldModel (Untrained Placeholder)"

    def __init__(self, input_dim: int = 8, hidden_dim: int = 64,
                 num_layers: int = 2, horizon: int = 5):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.horizon = horizon
        self.module = self._build_module()

    def _build_module(self):
        try:
            import torch
            import torch.nn as nn

            class _LSTMWorldModel(nn.Module):
                """Prev-N window states -> next-K states + stage logits."""

                def __init__(self, input_dim, hidden_dim, num_layers, horizon):
                    super().__init__()
                    self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                                        batch_first=True)
                    self.state_head = nn.Linear(hidden_dim, input_dim * horizon)
                    self.stage_head = nn.Linear(hidden_dim, 5 * horizon)
                    self.horizon = horizon

                def forward(self, x):
                    _, (h_n, _) = self.lstm(x)
                    h = h_n[-1]
                    states = self.state_head(h).view(-1, self.horizon, x.size(-1))
                    stages = self.stage_head(h).view(-1, self.horizon, 5)
                    return states, stages

            return _LSTMWorldModel(self.input_dim, self.hidden_dim,
                                   self.num_layers, self.horizon)
        except ImportError:
            return None  # torch optional at runtime; demo path still works

    @property
    def torch_available(self) -> bool:
        return self.module is not None

    def forward_states(self, state_matrix: np.ndarray):
        """Run the (untrained) torch module over a state sequence."""
        if self.module is None:
            raise RuntimeError("PyTorch is not installed in this environment.")
        import torch
        with torch.no_grad():
            x = torch.tensor(state_matrix, dtype=torch.float32).unsqueeze(0)
            return self.module(x)

    def predict(self, network_windows: list[NetworkWindow], horizon: int = 5,
                threshold: float = DEFAULT_THRESHOLD,
                scenario_final: float | None = None) -> dict:
        out = build_forecast(network_windows, horizon, threshold,
                             scenario_final, model_name=self.name)
        out["weights"] = "untrained-placeholder"
        out["torch_available"] = self.torch_available
        return out


def build_world_model(kind: str = "demo") -> WorldModel:
    if kind == "torch":
        return TorchLSTMWorldModel()
    return DemoWorldModel()
