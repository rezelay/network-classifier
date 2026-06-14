import csv
import logging
import os
import signal
import sys
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from flask import Flask, jsonify, request

sns.set_theme(style="whitegrid")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

app = Flask(__name__)
# Suppress Flask's per-request access log — we print our own
log_werkzeug = logging.getLogger("werkzeug")
log_werkzeug.setLevel(logging.ERROR)

# ── In-memory session log ─────────────────────────────────────────────────────
SESSION: list[dict] = []

# ── Load model artifacts once at startup ──────────────────────────────────────
mlp    = joblib.load("model/mlp.joblib")
scaler = joblib.load("model/scaler.joblib")
TRAIN_COLUMNS = list(scaler.feature_names_in_)

# ── Lookup tables ─────────────────────────────────────────────────────────────
PROTOCOL_MAP = {1: "icmp", 6: "tcp", 17: "udp"}

PORT_TO_SERVICE = {
    20:   "ftp_data",  21:   "ftp",       22:   "ssh",
    23:   "telnet",    25:   "smtp",       37:   "time",
    42:   "name",      43:   "whois",      53:   "domain",
    57:   "mtp",       66:   "sql_net",    69:   "tftp_u",
    70:   "gopher",    77:   "rje",        79:   "finger",
    80:   "http",      87:   "link",       95:   "supdup",
    110:  "pop_3",     111:  "sunrpc",     113:  "auth",
    117:  "uucp_path", 119:  "nntp",       123:  "ntp_u",
    137:  "netbios_ns", 138: "netbios_dgm", 139: "netbios_ssn",
    143:  "imap4",     161:  "snmp",       179:  "bgp",
    194:  "IRC",       210:  "Z39_50",     389:  "ldap",
    443:  "http_443",  445:  "netbios_ssn", 512: "exec",
    513:  "login",     514:  "shell",      515:  "printer",
    530:  "courier",   540:  "uucp",       543:  "klogin",
    544:  "kshell",    993:  "imap4",      995:  "pop_3",
    2784: "http_2784", 5631: "vmnet",      6000: "X11",
    8001: "http_8001",
}

# Ports typically used by scan tools or unknown services → "private"
PRIVATE_PORT_THRESHOLD = 1024


def derive_service(dst_port: int, protocol: int) -> str:
    if protocol == 1:           # ICMP has no ports
        return "ecr_i"
    if dst_port in PORT_TO_SERVICE:
        return PORT_TO_SERVICE[dst_port]
    if dst_port < PRIVATE_PORT_THRESHOLD:
        return "other"
    return "private"


def derive_flag(flow: dict, protocol: int) -> str:
    if protocol != 6:           # UDP/ICMP don't have TCP flags
        return "SF"
    syn = flow.get("syn_flag_cnt", 0)
    fin = flow.get("fin_flag_cnt", 0)
    rst = flow.get("rst_flag_cnt", 0)
    ack = flow.get("ack_flag_cnt", 0)

    if syn > 0 and fin > 0 and rst == 0:
        return "SF"             # normal close
    if syn > 0 and ack == 0 and fin == 0 and rst == 0:
        return "S0"             # connection attempt, no reply
    if syn > 0 and ack > 0 and fin == 0 and rst == 0:
        return "S1"             # established, not yet closed
    if rst > 0 and syn > 0 and ack == 0:
        return "REJ"            # connection rejected
    if rst > 0 and syn == 0 and fin == 0:
        return "RSTO"           # reset by originator
    if rst > 0 and fin > 0:
        return "RSTR"           # reset by responder
    if fin > 0 and syn == 0:
        return "SH"             # SYN+FIN (unusual)
    return "OTH"


def map_to_nslkdd(flow: dict) -> dict:
    protocol  = flow.get("protocol", 6)
    dst_port  = int(flow.get("dst_port", 0))
    fwd_pkts  = max(int(flow.get("tot_fwd_pkts", 1)), 1)

    syn_rate = min(flow.get("syn_flag_cnt", 0) / fwd_pkts, 1.0)
    rst_rate = min(flow.get("rst_flag_cnt", 0) / fwd_pkts, 1.0)

    # flow_duration is in microseconds in CICFlowMeter; convert to seconds
    duration_sec = int(flow.get("flow_duration", 0) / 1_000_000)

    land = int(
        flow.get("src_ip") == flow.get("dst_ip")
        and flow.get("src_port") == dst_port
    )

    return {
        # ── Directly mappable ────────────────────────────────────────────────
        "duration":              duration_sec,
        "protocol_type":         PROTOCOL_MAP.get(protocol, "tcp"),
        "service":               derive_service(dst_port, protocol),
        "flag":                  derive_flag(flow, protocol),
        "src_bytes":             flow.get("totlen_fwd_pkts", 0),
        "dst_bytes":             flow.get("totlen_bwd_pkts", 0),
        "land":                  land,
        "urgent":                flow.get("urg_flag_cnt", 0),
        # ── Approximate from available stats ─────────────────────────────────
        "serror_rate":           syn_rate,
        "srv_serror_rate":       syn_rate,
        "rerror_rate":           rst_rate,
        "srv_rerror_rate":       rst_rate,
        # ── Not extractable from modern traffic — zeroed ─────────────────────
        "wrong_fragment":        0,
        "hot":                   0,
        "num_failed_logins":     0,
        "logged_in":             0,
        "num_compromised":       0,
        "root_shell":            0,
        "su_attempted":          0,
        "num_root":              0,
        "num_file_creations":    0,
        "num_shells":            0,
        "num_access_files":      0,
        "num_outbound_cmds":     0,
        "is_host_login":         0,
        "is_guest_login":        0,
        # ── Window-based stats not available per-flow — sensible defaults ─────
        "count":                 1,
        "srv_count":             1,
        "same_srv_rate":         1.0,
        "diff_srv_rate":         0.0,
        "srv_diff_host_rate":    0.0,
        "dst_host_count":        1,
        "dst_host_srv_count":    1,
        "dst_host_same_srv_rate":     1.0,
        "dst_host_diff_srv_rate":     0.0,
        "dst_host_same_src_port_rate": 1.0,
        "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate":       syn_rate,
        "dst_host_srv_serror_rate":   syn_rate,
        "dst_host_rerror_rate":       rst_rate,
        "dst_host_srv_rerror_rate":   rst_rate,
    }


def preprocess(features: dict) -> np.ndarray:
    df = pd.DataFrame([features])
    df = pd.get_dummies(df, columns=["protocol_type", "service", "flag"], dtype=float)
    df = df.reindex(columns=TRAIN_COLUMNS, fill_value=0.0)
    return scaler.transform(df)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    flow = request.get_json(silent=True)
    if not flow:
        return jsonify({"error": "invalid or missing JSON body"}), 400

    features    = map_to_nslkdd(flow)
    X           = preprocess(features)
    prediction  = int(mlp.predict(X)[0])
    probability = float(mlp.predict_proba(X)[0][1])
    label       = "ATTACK" if prediction == 1 else "normal"

    log.info(
        "%-8s  prob=%.2f  %s:%s → %s:%s  proto=%-4s  svc=%-12s  flag=%s",
        label,
        probability,
        flow.get("src_ip", "?"),
        flow.get("src_port", "?"),
        flow.get("dst_ip", "?"),
        flow.get("dst_port", "?"),
        features["protocol_type"],
        features["service"],
        features["flag"],
    )

    SESSION.append({
        "timestamp":          datetime.now().isoformat(timespec="seconds"),
        "prediction":         label,
        "attack_probability": round(probability, 4),
        "src_ip":             flow.get("src_ip"),
        "dst_ip":             flow.get("dst_ip"),
        "src_port":           flow.get("src_port"),
        "dst_port":           flow.get("dst_port"),
        "protocol":           features["protocol_type"],
        "service":            features["service"],
        "flag":               features["flag"],
    })

    return jsonify({
        "prediction":         label.lower(),
        "attack_probability": round(probability, 4),
        "src_ip":             flow.get("src_ip"),
        "dst_ip":             flow.get("dst_ip"),
        "src_port":           flow.get("src_port"),
        "dst_port":           flow.get("dst_port"),
        "protocol":           PROTOCOL_MAP.get(flow.get("protocol", 6), "unknown"),
        "service":            features["service"],
        "flag":               features["flag"],
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "mlp", "features": len(TRAIN_COLUMNS)})


# ── Session report ────────────────────────────────────────────────────────────
def generate_session_report():
    if not SESSION:
        log.info("No flows recorded — nothing to save.")
        return

    out = "live_outputs"
    os.makedirs(out, exist_ok=True)
    df = pd.DataFrame(SESSION)

    # 1. Flow log CSV
    df.to_csv(f"{out}/flows.csv", index=False)

    # 2. Summary text
    total   = len(df)
    attacks = (df["prediction"] == "ATTACK").sum()
    normal  = total - attacks
    with open(f"{out}/summary.txt", "w") as f:
        f.write(f"Session summary — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'─' * 40}\n")
        f.write(f"Total flows   : {total}\n")
        f.write(f"Normal        : {normal}  ({normal/total*100:.1f}%)\n")
        f.write(f"Attack        : {attacks}  ({attacks/total*100:.1f}%)\n")

    # 3. Prediction distribution bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["prediction"].value_counts()
    ax.bar(counts.index, counts.values, color=["steelblue" if l == "normal" else "tomato" for l in counts.index], width=0.4)
    for i, v in enumerate(counts.values):
        ax.text(i, v + total * 0.01, f"{v:,}", ha="center")
    ax.set_title("Prediction Distribution")
    ax.set_ylabel("Flows")
    plt.tight_layout()
    plt.savefig(f"{out}/prediction_distribution.png", dpi=150)
    plt.close()

    # 4. Attack probability histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["attack_probability"], bins=20, color="steelblue", edgecolor="white")
    ax.axvline(0.5, color="tomato", linestyle="--", lw=1.5, label="Decision threshold")
    ax.set_xlabel("Attack Probability")
    ax.set_ylabel("Flows")
    ax.set_title("Attack Probability Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{out}/probability_histogram.png", dpi=150)
    plt.close()

    # 5. Timeline
    df["ts"] = pd.to_datetime(df["timestamp"])
    df["label_num"] = (df["prediction"] == "ATTACK").astype(int)
    fig, ax = plt.subplots(figsize=(10, 3))
    colors = df["prediction"].map({"normal": "steelblue", "ATTACK": "tomato"})
    ax.scatter(df["ts"], df["attack_probability"], c=colors, s=20, alpha=0.7)
    ax.axhline(0.5, color="gray", linestyle="--", lw=1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Attack Probability")
    ax.set_title("Classification Timeline")
    plt.tight_layout()
    plt.savefig(f"{out}/timeline.png", dpi=150)
    plt.close()

    log.info("Session report saved to %s/ (%d flows)", out, total)


def _handle_sigint(sig, frame):
    log.info("Interrupted — generating session report…")
    generate_session_report()
    sys.exit(0)


signal.signal(signal.SIGINT, _handle_sigint)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)