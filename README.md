# NETFORESIGHT — AI-Based Network Attack Forecasting (Offline MVP)

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-app-red)
![Offline](https://img.shields.io/badge/offline-no%20cloud%20APIs-green)
![Prototype](https://img.shields.io/badge/status-40%25%20prototype-orange)

> **“Predict the next stage. Investigate before compromise.”**

NetForeSight is a **40% functional showcase prototype** of a SOC analyst
dashboard that forecasts **attack progression** (not just benign/malicious
classification) from network traffic. Traffic is grouped into ordered
**time windows**, each window becomes a **network state**, and a
**world model** forecasts K-steps-ahead risk, MITRE ATT&CK-style stages and
explainable traffic signals.

The production LSTM world model is **not** part of this prototype: a
deterministic **demo inference engine** produces realistic trajectories so
the full analyst workflow is functional and believable. Everything runs
**locally and offline** — no cloud APIs, no external databases.

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the printed local URL (default http://localhost:8501).

`torch` and `scapy` are optional at runtime: without them the app still
works fully via the demo engine + CSV/demo data. PCAP upload requires
`scapy`.

## Workflow demonstrated

```text
PCAP / CSV Upload → Feature Extraction → Normalization → Time Windows →
Network State Representation → World Model Forecast → K-Step Prediction →
Attack Stage Mapping → Explainability → SOC Analyst Investigation
```

Pages: **Dashboard** (risk forecast, progression timeline, alert,
investigation, upload, state windows, demo scenario), **Traffic Monitor**,
**Attack Forecast** (K-step simulation), **Explainability** (signal
attribution + SHAP-style panel), **Network Graph**, **Analytics**,
**Reports** (export + CSV download), **Settings** (threshold, windows,
backend, reset).

Try **▶ Run Demo Scenario** on the dashboard: a 7-phase progressive
internal attack (normal → reconnaissance → SYN surge → port probing →
fan-out → forecasted lateral movement → threshold crossing).

## Project structure

```text
app.py                    Streamlit application (all pages, session state)
requirements.txt
README.md
data/
    demo_network.csv      Deterministic demo traffic (12,481 flows)
modules/
    __init__.py           Stages, severities, shared constants
    data_loader.py        Demo dataset, CSV upload, PCAP parsing (scapy)
    feature_extractor.py  Flow features, normalization, session summary
    windowing.py          Time-window generation, network state vectors
    forecast_engine.py    WorldModel ABC, DemoWorldModel, TorchLSTMWorldModel
    explainability.py     Attention-style attribution, SHAP-style table
    visualizations.py     Plotly charts, network graph, HTML cards
```

## Model contract

`WorldModel.predict(network_windows, horizon, threshold, scenario_final)`
returns:

```python
{
    "risk": 0.84, "confidence": 0.91,
    "predicted_stage": "Lateral Movement",
    "future_stages": [...], "feature_contributions": {...}, ...
}
```

Replace `DemoWorldModel` with the trained `TorchLSTMWorldModel` (reference
LSTM → next-K-states → stage-logits architecture already sketched in
`modules/forecast_engine.py`) without changing the UI.

## Prototype limits

Simulated/demo inference only · no live-network monitoring · forecasts are
not validated detections · the analyst stays in control (no automated
blocking). Demo provenance is labelled throughout the UI.
