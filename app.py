"""
app.py  —  Network Intrusion Detection System (Gradio)
Run:  python app.py
"""

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ── load model artifacts ──────────────────────────────────────────────────────
rf            = joblib.load("model/rf_model.pkl")
scaler        = joblib.load("model/scaler.pkl")
encoders      = joblib.load("model/encoders.pkl")
feature_names = joblib.load("model/feature_names.pkl")
medians       = joblib.load("model/medians.pkl")
unique_vals   = joblib.load("model/unique_vals.pkl")
report        = joblib.load("model/report.pkl")
conf_matrix   = joblib.load("model/conf_matrix.pkl")
feat_imp      = joblib.load("model/feat_importance.pkl")
attack_dist   = joblib.load("model/attack_dist.pkl")
dataset_info  = joblib.load("model/dataset_info.pkl")

EXAMPLES = [
    ["udp",  "-",    "INT", 0.001, 2, 0, 200,  0,   64,  0,   100.0],
    ["tcp",  "-",    "FIN", 0.0,   1, 0, 40,   0,   62,  0,   5000.0],
    ["tcp",  "-",    "REQ", 0.0,   1, 0, 40,   0,   62,  252, 3000.0],
    ["tcp",  "http", "FIN", 0.2,   8, 6, 1500, 900, 62,  252, 70.0],
    ["tcp",  "-",    "CON", 0.0,   3, 3, 120,  120, 62,  252, 1000.0],
]

# ── helper ────────────────────────────────────────────────────────────────────
def build_row(proto, service, state, dur, spkts, dpkts, sbytes, dbytes, sttl, dttl, rate):
    row = medians.copy()
    row["dur"]    = dur
    row["spkts"]  = spkts
    row["dpkts"]  = dpkts
    row["sbytes"] = sbytes
    row["dbytes"] = dbytes
    row["sttl"]   = sttl
    row["dttl"]   = dttl
    row["rate"]   = rate
    for col, val in [("proto", proto), ("service", service), ("state", state)]:
        le    = encoders[col]
        known = list(le.classes_)
        row[col] = int(le.transform([val])[0]) if val in known else -1
    row["total_bytes"]   = row["sbytes"] + row["dbytes"]
    row["total_pkts"]    = row["spkts"]  + row["dpkts"]
    row["bytes_per_pkt"] = row["total_bytes"] / (row["total_pkts"] + 1)
    row["ttl_diff"]      = abs(row["sttl"] - row["dttl"])
    return pd.DataFrame([row])[feature_names]

def predict(proto, service, state, dur, spkts, dpkts, sbytes, dbytes, sttl, dttl, rate):
    input_df      = build_row(proto, service, state, float(dur), int(spkts), int(dpkts),
                              int(sbytes), int(dbytes), int(sttl), int(dttl), float(rate))
    input_scaled  = scaler.transform(input_df)
    prediction    = rf.predict(input_scaled)[0]
    probabilities = rf.predict_proba(input_scaled)[0]
    prob_normal   = probabilities[0] * 100
    prob_attack   = probabilities[1] * 100
    label      = "✅  NORMAL TRAFFIC" if prediction == 0 else "🚨  ATTACK DETECTED!"
    confidence = f"Normal: {prob_normal:.1f}%     Attack: {prob_attack:.1f}%"
    return label, confidence

# ── dashboard figures (computed once at startup) ──────────────────────────────
def _attack_dist_fig():
    fig, ax = plt.subplots(figsize=(7, 4))
    cats   = list(attack_dist.keys())
    counts = list(attack_dist.values())
    colors = plt.cm.viridis_r(np.linspace(0.2, 0.9, len(cats)))
    bars   = ax.barh(cats, counts, color=colors)
    ax.set_xlabel("Number of Records")
    ax.invert_yaxis()
    for bar, val in zip(bars, counts):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=8)
    plt.tight_layout()
    return fig

def _pie_fig():
    fig, ax = plt.subplots(figsize=(5, 4))
    n, a = dataset_info["n_normal"], dataset_info["n_attacks"]
    ax.pie([n, a], labels=[f"Normal\n{n:,}", f"Attack\n{a:,}"],
           colors=["steelblue", "tomato"], autopct="%1.1f%%", startangle=90,
           wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax.set_title("Training Set")
    plt.tight_layout()
    return fig

def _perf_fig():
    fig, ax = plt.subplots(figsize=(7, 3))
    metrics  = ["Precision", "Recall", "F1-Score"]
    normal_v = [report["Normal"]["precision"], report["Normal"]["recall"], report["Normal"]["f1-score"]]
    attack_v = [report["Attack"]["precision"], report["Attack"]["recall"], report["Attack"]["f1-score"]]
    x, w = np.arange(len(metrics)), 0.3
    ax.bar(x - w / 2, normal_v, w, label="Normal", color="steelblue")
    ax.bar(x + w / 2, attack_v, w, label="Attack",  color="tomato")
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 by Class"); ax.legend()
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    return fig

def _feat_imp_fig():
    fi_features   = feat_imp["feature"][::-1]
    fi_importance = feat_imp["importance"][::-1]
    fig, ax = plt.subplots(figsize=(6, 6))
    palette = plt.cm.viridis(np.linspace(0.2, 0.85, len(fi_features)))
    ax.barh(fi_features, fi_importance, color=palette)
    ax.set_xlabel("Importance Score")
    ax.set_title("Random Forest Feature Importance")
    plt.tight_layout()
    return fig

def _conf_matrix_fig():
    cm_arr = np.array(conf_matrix)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Predicted Normal", "Predicted Attack"],
                yticklabels=["Actual Normal",    "Actual Attack"])
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig

# ── build UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="NIDS — Network Intrusion Detection") as demo:
    gr.Markdown("# 🔒 Network Intrusion Detection System")
    gr.Markdown("UNSW-NB15 Dataset  ·  Random Forest Classifier  ·  90%+ Accuracy")

    with gr.Tabs():

        # ── Tab 1: Dashboard ──────────────────────────────────────────────────
        with gr.Tab("🏠  Dashboard"):
            with gr.Row():
                gr.Textbox(value=f"{dataset_info['train_rows']:,}",              label="Training Rows",    interactive=False)
                gr.Textbox(value=f"{dataset_info['test_rows']:,}",               label="Testing Rows",     interactive=False)
                gr.Textbox(value=str(dataset_info["n_features"]),                label="Features Used",    interactive=False)
                gr.Textbox(value=f"{report['accuracy']*100:.1f}%",               label="Model Accuracy",   interactive=False)
                gr.Textbox(value=f"{report['Attack']['precision']*100:.1f}%",    label="Attack Precision", interactive=False)
            with gr.Row():
                gr.Plot(value=_attack_dist_fig, label="Attack Category Distribution")
                gr.Plot(value=_pie_fig,         label="Normal vs Attack Split")
            gr.Plot(value=_perf_fig, label="Precision / Recall / F1 by Class")

        # ── Tab 2: Live Prediction ────────────────────────────────────────────
        with gr.Tab("🔍  Live Prediction"):
            with gr.Row():
                with gr.Column():
                    proto   = gr.Dropdown(choices=unique_vals["proto"],    label="Protocol",          value="udp")
                    service = gr.Dropdown(choices=unique_vals["service"],  label="Service",           value="-")
                    state   = gr.Dropdown(choices=unique_vals["state"],    label="Connection State",  value="INT")
                with gr.Column():
                    spkts  = gr.Number(label="Source Packets",       value=2,     precision=0)
                    dpkts  = gr.Number(label="Destination Packets",  value=0,     precision=0)
                    sbytes = gr.Number(label="Source Bytes",         value=200,   precision=0)
                    dbytes = gr.Number(label="Destination Bytes",    value=0,     precision=0)
                with gr.Column():
                    dur  = gr.Number(label="Duration (seconds)",  value=0.001)
                    rate = gr.Number(label="Rate (packets/sec)",  value=100.0)
                    sttl = gr.Slider(0, 255, label="Source TTL",       value=64,  step=1)
                    dttl = gr.Slider(0, 255, label="Destination TTL",  value=0,   step=1)

            predict_btn   = gr.Button("🔍  Predict", variant="primary")
            result_label  = gr.Textbox(label="Result",      interactive=False)
            result_conf   = gr.Textbox(label="Confidence",  interactive=False)

            predict_btn.click(
                fn=predict,
                inputs=[proto, service, state, dur, spkts, dpkts, sbytes, dbytes, sttl, dttl, rate],
                outputs=[result_label, result_conf],
            )

            gr.Examples(
                examples=EXAMPLES,
                inputs=[proto, service, state, dur, spkts, dpkts, sbytes, dbytes, sttl, dttl, rate],
                label="Quick Examples  (Normal · DoS · Reconnaissance · Exploits · Generic)",
            )

        # ── Tab 3: Model Insights ─────────────────────────────────────────────
        with gr.Tab("📊  Model Insights"):
            with gr.Row():
                gr.Plot(value=_feat_imp_fig,    label="Top 15 Feature Importance")
                gr.Plot(value=_conf_matrix_fig, label="Confusion Matrix")

            tn = conf_matrix[0][0]; fp = conf_matrix[0][1]
            fn = conf_matrix[1][0]; tp = conf_matrix[1][1]
            gr.Markdown(f"""
| Box | Meaning | Count |
|-----|---------|-------|
| True Normal   | Normal traffic correctly identified | {tn:,} |
| True Attack   | Attacks correctly caught            | {tp:,} |
| False Alarm   | Normal flagged as attack            | {fp:,} |
| Missed Attack | Attack not detected                 | {fn:,} |
            """)

            report_rows = [
                ["Normal",       report["Normal"]["precision"],       report["Normal"]["recall"],       report["Normal"]["f1-score"],       int(report["Normal"]["support"])],
                ["Attack",       report["Attack"]["precision"],       report["Attack"]["recall"],       report["Attack"]["f1-score"],       int(report["Attack"]["support"])],
                ["Macro Avg",    report["macro avg"]["precision"],    report["macro avg"]["recall"],    report["macro avg"]["f1-score"],    int(report["macro avg"]["support"])],
                ["Weighted Avg", report["weighted avg"]["precision"], report["weighted avg"]["recall"], report["weighted avg"]["f1-score"], int(report["weighted avg"]["support"])],
            ]
            gr.Dataframe(
                value=report_rows,
                headers=["Class", "Precision", "Recall", "F1-Score", "Support"],
                label="Full Classification Report",
            )

        # ── Tab 4: About ──────────────────────────────────────────────────────
        with gr.Tab("ℹ️  About"):
            gr.Markdown("""
## About This Project

A **Network Intrusion Detection System (NIDS)** that uses Machine Learning to classify
network traffic as either **normal** or an **attack** in real time.

---
### Pipeline

| Step | What was done |
|------|---------------|
| **EDA**                 | 8 charts — class balance, attack types, protocols, bytes, TTL, correlation |
| **Preprocessing**       | Dropped irrelevant columns, Label Encoding for text, StandardScaler |
| **Feature Engineering** | `total_bytes`, `total_pkts`, `bytes_per_pkt`, `ttl_diff` |
| **Model Training**      | Random Forest — 100 decision trees |

---
### Dataset: UNSW-NB15
Created by the **Cyber Range Lab of UNSW Canberra**.
Contains real normal activity and nine categories of synthetic attack traffic.

---
### Attack Types

| Attack | Description |
|--------|-------------|
| **Generic**        | Attacks that work against any cipher suite |
| **Exploits**       | Known software vulnerabilities |
| **Fuzzers**        | Random data to crash or find bugs |
| **DoS**            | Denial of Service — flood the target |
| **Reconnaissance** | Scanning to gather info before attacking |
| **Backdoor**       | Secret channel to access a system |
| **Shellcode**      | Small code that launches a shell |
| **Worms**          | Self-replicating malware |
            """)

demo.launch()
