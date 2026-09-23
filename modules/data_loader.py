"""Offline data loading: demo dataset, CSV uploads and optional PCAP parsing.

Supported CSV layouts (auto-detected):

* NetForeSight flow CSV (``timestamp, src_ip, dst_ip, ...``)
* Generic traffic CSVs via forgiving column aliases
* CICIDS2017 flow CSVs (e.g. ``Thursday-WorkingHours-...-WebAttacks``):
  ``Source IP`` / ``Destination IP`` / ``Timestamp`` / ``Destination Port``,
  ``Flow Duration`` + ``Flow IAT Mean`` in microseconds, ``* Flag Count``
  columns, packet/byte totals and a ``Label`` column mapped to risk scores.

Everything runs locally. PCAP support needs ``scapy`` (listed in
requirements.txt); the app works without it by using CSV / demo data.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import severity_for_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEMO_CSV = DATA_DIR / "demo_network.csv"

FLOW_COLUMNS = [
    "timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
    "tcp_flags", "packets", "bytes", "duration", "iat_mean", "ttl",
    "retransmissions", "risk_score", "risk",
]

# Forgiving aliases accepted from user-supplied CSV files.
COLUMN_ALIASES = {
    "timestamp": {"timestamp", "time", "datetime", "ts", "start_time"},
    "src_ip": {"src_ip", "source_ip", "src", "source", "srcip"},
    "dst_ip": {"dst_ip", "dest_ip", "destination_ip", "dst", "destination", "dstip"},
    "src_port": {"src_port", "source_port", "sport", "srcport"},
    "dst_port": {"dst_port", "dest_port", "destination_port", "dport", "dstport"},
    "protocol": {"protocol", "proto"},
    "tcp_flags": {"tcp_flags", "flags", "tcpflags"},
    "packets": {"packets", "packet_count", "pkts"},
    "bytes": {"bytes", "byte_count", "octets"},
    "duration": {"duration", "flow_duration", "dur"},
    "iat_mean": {"iat_mean", "iat", "mean_iat", "inter_arrival"},
    "ttl": {"ttl", "ip_ttl"},
    "retransmissions": {"retransmissions", "retrans", "retx"},
    "risk_score": {"risk_score", "score"},
    "risk": {"risk", "severity", "label"},
}


class DataLoadError(Exception):
    """User-facing data loading failure with a clear remediation message."""


# --------------------------------------------------------------------------- #
# Demo dataset                                                                #
# --------------------------------------------------------------------------- #

def _window_spec() -> list[dict]:
    """Deterministic per-window traffic mix for the demo scenario."""
    return [
        # window, rows, normal-host pools, attacker flows, scanner flows
        {"rows": 1204, "n_src": 42, "n_dst": 61, "syn_p": 0.050,
         "iat_base": 0.55, "attacker": 6, "scanner": 40, "med_p": 0.02},
        {"rows": 1893, "n_src": 48, "n_dst": 102, "syn_p": 0.130,
         "iat_base": 0.34, "attacker": 45, "scanner": 150, "med_p": 0.03},
        {"rows": 2221, "n_src": 50, "n_dst": 147, "syn_p": 0.190,
         "iat_base": 0.30, "attacker": 150, "scanner": 260, "med_p": 0.04},
        {"rows": 2784, "n_src": 55, "n_dst": 188, "syn_p": 0.100,
         "iat_base": 0.26, "attacker": 320, "scanner": 300, "med_p": 0.05},
        {"rows": 4379, "n_src": 64, "n_dst": 214, "syn_p": 0.080,
         "iat_base": 0.27, "attacker": 420, "scanner": 320, "med_p": 0.05},
    ]


def generate_demo_dataset(seed: int = 42) -> pd.DataFrame:
    """Build the deterministic demo flow table (used if the CSV is missing)."""
    rng = np.random.default_rng(seed)
    attacker_ip = "10.0.2.15"
    targets = ["10.0.4.21", "10.0.4.22", "10.0.4.27"]
    svc_ports = [445, 3389, 22, 135, 139, 5985]
    web_ports = [80, 443]
    base = pd.Timestamp("2026-09-23 14:15:00")
    frames: list[pd.DataFrame] = []

    for w, spec in enumerate(_window_spec()):
        n = spec["rows"]
        seg_start = base + pd.Timedelta(seconds=120 * w)
        # Strictly interior timestamps so equal-duration splits recover
        # the exact per-window row counts.
        ts = seg_start + pd.to_timedelta(
            np.sort(rng.uniform(0.5, 119.5, n)), unit="s")

        src_pool = [f"10.0.2.{i}" for i in range(2, 2 + spec["n_src"])]
        dst_pool = [f"10.0.4.{i}" for i in range(2, 2 + spec["n_dst"])]
        src = rng.choice(src_pool, n)
        dst = rng.choice(dst_pool, n)

        proto = rng.choice(["TCP", "TCP", "TCP", "TCP", "UDP", "ICMP"], n,
                           p=[0.62, 0.10, 0.08, 0.06, 0.10, 0.04])
        dport = np.where(
            proto == "UDP",
            rng.choice([53, 53, 53, 123, 161], n),
            np.where(rng.random(n) < 0.72,
                     rng.choice(web_ports + [25, 110, 993, 3306, 8080], n),
                     rng.choice(svc_ports, n)))
        sport = rng.integers(1024, 65535, n)
        syn = rng.random(n) < spec["syn_p"]
        flags = np.where(syn, "SYN",
                         rng.choice(["ACK", "ACK", "PSH-ACK", "FIN-ACK"], n))
        pkts = np.clip(rng.poisson(9, n) + 1, 1, 400).astype(int)
        byt = (pkts * rng.uniform(60, 900, n)).astype(int)
        dur = np.round(rng.exponential(0.6, n) + 0.02, 3)
        iat = np.round(np.clip(
            rng.exponential(spec["iat_base"], n), 0.005, 5.0), 3)
        ttl = rng.choice([64, 64, 64, 128, 128, 255], n)
        retx = np.clip(rng.poisson(0.4, n), 0, 25).astype(int)
        score = np.clip(rng.beta(2.2, 9.0, n) * 0.55, 0.0, 0.6)

        df = pd.DataFrame({
            "timestamp": ts, "src_ip": src, "dst_ip": dst,
            "src_port": sport, "dst_port": dport, "protocol": proto,
            "tcp_flags": flags, "packets": pkts, "bytes": byt,
            "duration": dur, "iat_mean": iat, "ttl": ttl,
            "retransmissions": retx, "risk_score": np.round(score, 3),
        })

        # --- attacker lateral-movement style flows ------------------------ #
        na = spec["attacker"]
        if na:
            idx = rng.choice(n, na, replace=False)
            df.loc[idx, "src_ip"] = attacker_ip
            df.loc[idx, "dst_ip"] = rng.choice(targets, na)
            df.loc[idx, "dst_port"] = rng.choice(svc_ports, na)
            df.loc[idx, "protocol"] = "TCP"
            df.loc[idx, "tcp_flags"] = rng.choice(
                ["SYN", "SYN", "SYN", "ACK", "RST"], na)
            df.loc[idx, "packets"] = np.clip(rng.poisson(14, na) + 2, 2, 200)
            df.loc[idx, "duration"] = np.round(rng.exponential(0.35, na) + 0.02, 3)
            df.loc[idx, "iat_mean"] = np.round(
                np.clip(rng.exponential(0.16, na), 0.005, 2.0), 3)
            df.loc[idx, "retransmissions"] = np.clip(rng.poisson(1.6, na), 0, 25)
            df.loc[idx, "risk_score"] = np.round(
                rng.uniform(0.62, 0.97, na), 3)

        # --- reconnaissance scanner flows --------------------------------- #
        ns = spec["scanner"]
        if ns:
            idx = rng.choice(n, ns, replace=False)
            scanners = rng.choice(
                ["10.0.2.15", "10.0.2.9", "10.0.2.31", "203.0.113.7"], ns)
            df.loc[idx, "src_ip"] = scanners
            df.loc[idx, "dst_ip"] = rng.choice(dst_pool, ns)
            df.loc[idx, "dst_port"] = rng.integers(1, 1024, ns)
            df.loc[idx, "protocol"] = "TCP"
            df.loc[idx, "tcp_flags"] = "SYN"
            df.loc[idx, "packets"] = rng.integers(1, 6, ns)
            df.loc[idx, "duration"] = np.round(rng.uniform(0.01, 0.2, ns), 3)
            df.loc[idx, "iat_mean"] = np.round(
                np.clip(rng.exponential(0.09, ns), 0.005, 1.0), 3)
            df.loc[idx, "risk_score"] = np.round(
                rng.uniform(0.38, 0.72, ns), 3)

        # --- background suspicious-looking but benign flows ---------------- #
        med = rng.random(n) < spec["med_p"]
        df.loc[med, "risk_score"] = np.round(rng.uniform(0.36, 0.55, med.sum()), 3)
        frames.append(df)

    flows = pd.concat(frames, ignore_index=True)
    flows = flows.sort_values("timestamp").reset_index(drop=True)
    flows["risk"] = flows["risk_score"].apply(severity_for_score)

    # Calibrate: exactly 327 suspicious (HIGH + CRITICAL) flows.
    flows = _calibrate_suspicious(flows, target=327, seed=seed)
    return flows[FLOW_COLUMNS]


def _calibrate_suspicious(flows: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    """Force the suspicious (HIGH + CRITICAL) count to exactly ``target``."""
    flows = flows.copy()
    susp_idx = flows[flows["risk"].isin(["HIGH", "CRITICAL"])].index
    current = len(susp_idx)
    by_score = flows.loc[susp_idx].sort_values("risk_score")
    if current > target:  # demote the lowest-scoring suspicious rows
        drop = by_score.index[:current - target]
        flows.loc[drop, "risk_score"] = 0.55
        flows.loc[drop, "risk"] = "MEDIUM"
    elif current < target:  # promote the highest-scoring benign rows
        benign = flows[~flows.index.isin(susp_idx)].sort_values(
            "risk_score", ascending=False)
        need = min(target - current, len(benign))
        flows.loc[benign.index[:need], "risk_score"] = 0.62
        flows.loc[benign.index[:need], "risk"] = "HIGH"
    assert int(flows["risk"].isin(["HIGH", "CRITICAL"]).sum()) == target
    return flows


def ensure_demo_csv() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DEMO_CSV.exists():
        generate_demo_dataset().to_csv(DEMO_CSV, index=False)
    return DEMO_CSV


def load_demo_dataset() -> tuple[pd.DataFrame, list[str]]:
    ensure_demo_csv()
    df = pd.read_csv(DEMO_CSV, parse_dates=["timestamp"])
    if df.empty:
        raise DataLoadError("Demo dataset is empty. Re-generate it or upload a CSV.")
    return _standardize(df)


# --------------------------------------------------------------------------- #
# Uploaded files                                                              #
# --------------------------------------------------------------------------- #

def _norm_col(name: object) -> str:
    """Lowercase alphanumeric fingerprint: 'Source IP' -> 'sourceip'."""
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def _cic_label_score(label: object) -> float:
    """Deterministic risk score from a CICIDS2017 attack label."""
    lab = str(label).strip().lower()
    if lab == "benign":
        return 0.10
    if any(k in lab for k in ("portscan", "patator")):
        return 0.68
    if any(k in lab for k in ("heartbleed", "infiltration", "bot", "dos",
                              "ddos", "hulk", "goldeneye", "slowloris",
                              "slowhttptest", "sql", "xss", "brute force",
                              "web attack")):
        return 0.85
    return 0.75  # any other non-benign label


def _apply_cicids_mapping(out: pd.DataFrame, raw: pd.DataFrame,
                          norm_map: dict[str, object]) -> None:
    """Fill NetForeSight fields from CICIDS2017 composite columns (in place).

    Runs only when the CIC fingerprint (fwd/bwd packet totals) is present.
    Durations and IATs are converted from microseconds to seconds, TCP
    flags are composed from ``* Flag Count`` columns and risk scores come
    from the ``Label`` column unless explicit scores already exist.
    """
    def _get(*names):
        for n in names:
            if n in norm_map:
                return pd.to_numeric(raw[norm_map[n]], errors="coerce")
        return None

    fwd_pkts = _get("totalfwdpackets")
    bwd_pkts = _get("totalbackwardpackets")
    if fwd_pkts is None or bwd_pkts is None:
        return  # not a CICIDS2017 layout
    out["packets"] = (fwd_pkts.fillna(0) + bwd_pkts.fillna(0)
                      ).astype(int).clip(lower=1).values
    fwd_b = _get("totallengthoffwdpackets")
    bwd_b = _get("totallengthofbwdpackets")
    if fwd_b is not None and bwd_b is not None:
        out["bytes"] = (fwd_b.fillna(0) + bwd_b.fillna(0)
                        ).astype(int).clip(lower=1).values
    dur = _get("flowduration")
    if dur is not None:
        out["duration"] = (dur.fillna(1000) / 1e6).clip(lower=0.01).values
    iat = _get("flowiatmean")
    if iat is not None:
        out["iat_mean"] = (iat.fillna(500000) / 1e6).clip(lower=0.001).values

    flag_order = [("synflagcount", "SYN"), ("ackflagcount", "ACK"),
                  ("finflagcount", "FIN"), ("rstflagcount", "RST"),
                  ("pshflagcount", "PSH"), ("urgflagcount", "URG")]
    flag_cols = [(raw[norm_map[n]], tag) for n, tag in flag_order
                 if n in norm_map]
    if flag_cols:
        mat = np.column_stack([
            pd.to_numeric(c, errors="coerce").fillna(0).values > 0
            for c, _ in flag_cols])
        tags = [tag for _, tag in flag_cols]
        out["tcp_flags"] = [
            "-".join(t for t, on in zip(tags, row) if on) or "ACK"
            for row in mat]

    if "risk_score" not in out and "label" in norm_map:
        out["risk_score"] = raw[norm_map["label"]].apply(
            _cic_label_score).round(3).values


def _standardize(df: pd.DataFrame, dayfirst: bool = False,
                 cic: bool = False) -> tuple[pd.DataFrame, list[str]]:
    # Normalized alias lookup so 'Source IP', 'source_ip' and 'SRCIP' match.
    norm_map = {_norm_col(c): c for c in df.columns}
    alias_map: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            alias_map.setdefault(_norm_col(alias), canonical)
    mapped: dict[str, pd.Series] = {}
    for nrm, orig in norm_map.items():
        if nrm in alias_map:
            mapped[alias_map[nrm]] = df[orig]
    missing = [c for c in ("timestamp", "src_ip", "dst_ip") if c not in mapped]
    notes: list[str] = []
    if not mapped:
        raise DataLoadError(
            "Required traffic features were not detected "
            "(missing: timestamp, src_ip, dst_ip). "
            "Load the Demo Dataset or upload a supported traffic CSV.")
    out = pd.DataFrame(mapped)
    n_rows = len(out)
    if "timestamp" in missing:
        # No clock in file: rebuild a monotonic clock from capture order so
        # windowing/forecasting still work. Clearly labelled as synthetic.
        base = (pd.Timestamp("2017-07-06 08:00:00") if cic
                else pd.Timestamp("2026-09-23 14:15:00"))
        span = 4 * 3600 if cic else 3600
        out["timestamp"] = base + pd.to_timedelta(
            np.linspace(0, span, n_rows), unit="s")
        notes.append("Timestamps reconstructed from file row order "
                     "(source file has no timestamp column).")
    if "src_ip" in missing or "dst_ip" in missing:
        # No endpoints in file: deterministic synthetic IDs from the
        # documentation (TEST-NET) ranges so they are never mistaken
        # for real infrastructure.
        lab = df[norm_map["label"]] if "label" in norm_map else None
        if lab is not None:
            attack = (lab.astype(str).str.strip().str.lower()
                      != "benign").values
        else:
            attack = np.zeros(n_rows, dtype=bool)
        pool = (np.arange(n_rows) * 2654435761) % 59 + 2
        out["src_ip"] = np.where(
            attack, "203.0.113.7", "192.0.2." + pool.astype(str))
        out["dst_ip"] = "198.51.100.50"
        notes.append("Endpoint IPs are synthetic documentation-range IDs "
                     "(source file has no IP columns): attacks from "
                     "203.0.113.7, benign hosts from 192.0.2.0/24, "
                     "server 198.51.100.50.")
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce",
                                      dayfirst=dayfirst)
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if out.empty:
        raise DataLoadError("No valid timestamped records found in the file.")
    for col in ("src_ip", "dst_ip"):
        out[col] = out[col].astype(str).str.strip()
    # CICIDS2017 composite fields (microsecond units, flag counts, labels).
    _apply_cicids_mapping(out, df, norm_map)
    def _col(name: str, default):
        if name in out.columns:
            return out[name]
        return pd.Series(default, index=out.index)

    out["src_port"] = pd.to_numeric(_col("src_port", 0), errors="coerce").fillna(0).astype(int)
    out["dst_port"] = pd.to_numeric(_col("dst_port", 0), errors="coerce").fillna(0).astype(int)
    if "protocol" in out.columns:
        out["protocol"] = out["protocol"].fillna("TCP").astype(str).str.upper().str.strip()
    else:  # CICIDS2017 has no protocol column: infer from service ports.
        out["protocol"] = np.where(out["dst_port"].isin([53, 123, 161, 1900, 5353]),
                                   "UDP", "TCP")
    out["tcp_flags"] = _col("tcp_flags", "ACK").fillna("ACK").astype(str).str.upper().str.strip()
    out["packets"] = pd.to_numeric(_col("packets", 1), errors="coerce").fillna(1).astype(int).clip(lower=1)
    out["bytes"] = pd.to_numeric(_col("bytes", 64), errors="coerce").fillna(64).astype(int).clip(lower=1)
    out["duration"] = pd.to_numeric(_col("duration", 0.1), errors="coerce").fillna(0.1).clip(lower=0.01)
    out["iat_mean"] = pd.to_numeric(_col("iat_mean", 0.5), errors="coerce").fillna(0.5).clip(lower=0.001)
    out["ttl"] = pd.to_numeric(_col("ttl", 64), errors="coerce").fillna(64).astype(int).clip(1, 255)
    out["retransmissions"] = pd.to_numeric(
        _col("retransmissions", 0), errors="coerce").fillna(0).astype(int).clip(lower=0)
    if "risk_score" not in out:
        syn = out["tcp_flags"].str.contains("SYN", na=False).astype(float)
        svc = out["dst_port"].isin([22, 135, 139, 445, 3389, 5985]).astype(float)
        out["risk_score"] = (0.12 + 0.30 * syn + 0.25 * svc).clip(0, 0.95).round(3)
    out["risk_score"] = pd.to_numeric(out["risk_score"], errors="coerce").fillna(0.1).clip(0, 1)
    out["risk"] = out["risk_score"].apply(severity_for_score)
    return out[FLOW_COLUMNS], notes


def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        raise DataLoadError(f"Unable to read the CSV file: {exc}")
    if df.empty:
        raise DataLoadError("The uploaded CSV is empty.")
    cols = {_norm_col(c) for c in df.columns}
    cic = {"totalfwdpackets", "totalbackwardpackets"} <= cols
    # CICIDS2017 timestamps use D/M/YYYY ordering.
    return _standardize(df, dayfirst=cic, cic=cic)


def load_pcap(uploaded_file) -> pd.DataFrame:
    try:
        from scapy.all import PcapReader, IP, TCP, UDP  # type: ignore
    except ImportError as exc:
        raise DataLoadError(
            "PCAP parsing needs the optional 'scapy' package, which is not "
            "installed in this environment. Load the Demo Dataset or upload "
            "a supported traffic CSV.") from exc
    try:
        raw = uploaded_file.getvalue()
    except Exception as exc:
        raise DataLoadError(f"Unable to read the uploaded file: {exc}")
    flows: dict[tuple, dict] = {}
    count = 0
    try:
        reader = PcapReader(io.BytesIO(raw))
        for pkt in reader:
            count += 1
            if count > 200_000:
                break
            if not pkt.haslayer(IP):
                continue
            ip = pkt[IP]
            if pkt.haslayer(TCP):
                proto, sport, dport = "TCP", pkt[TCP].sport, pkt[TCP].dport
                flags = pkt[TCP].sprintf("%TCP.flags%") or "ACK"
                seq = int(pkt[TCP].seq)
            elif pkt.haslayer(UDP):
                proto, sport, dport = "UDP", pkt[UDP].sport, pkt[UDP].dport
                flags, seq = "", -1
            else:
                continue
            key = (ip.src, ip.dst, sport, dport, proto)
            ts = float(pkt.time)
            cell = flows.get(key)
            if cell is None:
                flows[key] = {"ts": [ts], "bytes": [len(pkt)], "ttl": [int(ip.ttl)],
                              "flags": [flags], "seqs": [seq]}
            else:
                cell["ts"].append(ts)
                cell["bytes"].append(len(pkt))
                cell["ttl"].append(int(ip.ttl))
                cell["flags"].append(flags)
                cell["seqs"].append(seq)
        reader.close()
    except Exception as exc:
        raise DataLoadError(
            "Unsupported or corrupted PCAP file. Load the Demo Dataset or "
            f"upload a supported traffic CSV. ({exc})")
    if not flows:
        raise DataLoadError("No IP traffic found in the PCAP file.")
    rows = []
    for (src, dst, sport, dport, proto), cell in flows.items():
        ts = np.array(cell["ts"])
        iats = np.diff(np.sort(ts))
        flag_str = max(set(cell["flags"]), key=cell["flags"].count) if cell["flags"] else "ACK"
        syn = 1.0 if "S" in flag_str else 0.0
        svc = 1.0 if dport in (22, 135, 139, 445, 3389, 5985) else 0.0
        n = len(ts)
        seen: set[int] = set()
        retx = 0
        for s in cell["seqs"]:
            if s in seen and s >= 0:
                retx += 1
            seen.add(s)
        rows.append({
            "timestamp": pd.to_datetime(ts.min(), unit="s"),
            "src_ip": src, "dst_ip": dst, "src_port": int(sport),
            "dst_port": int(dport), "protocol": proto,
            "tcp_flags": flag_str or "ACK", "packets": n,
            "bytes": int(sum(cell["bytes"])),
            "duration": round(float(ts.max() - ts.min()) + 0.01, 3),
            "iat_mean": round(float(iats.mean()) if len(iats) else 0.5, 3),
            "ttl": int(np.median(cell["ttl"])),
            "retransmissions": int(retx),
            "risk_score": round(float(np.clip(0.12 + 0.30 * syn + 0.25 * svc, 0, 0.95)), 3),
        })
    out = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    out["risk"] = out["risk_score"].apply(severity_for_score)
    return out[FLOW_COLUMNS], []


def describe_dataset(df: pd.DataFrame, name: str, ftype: str, size: str,
                     notes: list[str] | None = None) -> dict:
    return {
        "file_name": name,
        "file_type": ftype,
        "file_size": size,
        "records": len(df),
        "time_start": df["timestamp"].min(),
        "time_end": df["timestamp"].max(),
        "status": "Complete",
        "notes": notes or [],
    }
