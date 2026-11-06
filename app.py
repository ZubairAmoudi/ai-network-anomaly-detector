"""
AI Network Traffic Anomaly Detector - Streamlit SOC Dashboard.
Provides real-time NetFlow traffic visualization, live threat telemetry,
interactive Plotly analytics, and ML model explainability.
"""

import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_generator import generate_network_traffic, save_dataset
from model import FEATURE_COLUMNS, load_model_bundle, predict_traffic, train_and_save_models


st.set_page_config(
    page_title="SentinelNet AI | Network Traffic Anomaly Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom SOC Dark Styling
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }
    .threat-banner {
        padding: 0.85rem 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .severity-low {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        color: #10b981;
    }
    .severity-elevated {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid #f59e0b;
        color: #f59e0b;
    }
    .severity-high {
        background: rgba(249, 115, 22, 0.18);
        border: 1px solid #f97316;
        color: #f97316;
    }
    .severity-critical {
        background: rgba(239, 68, 68, 0.22);
        border: 1px solid #ef4444;
        color: #ef4444;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading AI Anomaly Detection Engine...")
def get_model_bundle():
    return load_model_bundle()


def compute_overall_threat_level(df_pred: pd.DataFrame) -> tuple[str, str, float]:
    """Compute overall SOC threat severity level, CSS class, and composite risk index."""
    if df_pred.empty:
        return "LOW (NOMINAL)", "severity-low", 0.0

    anomaly_rate = float((df_pred["Predicted_Label"] == "Anomaly").mean())
    crit_count = int((df_pred["Threat_Severity"] == "CRITICAL").sum())
    avg_top_score = float(df_pred["Anomaly_Score"].nlargest(min(25, len(df_pred))).mean())
    risk_index = min(100.0, round((anomaly_rate * 120.0) + (avg_top_score * 45.0), 1))

    if anomaly_rate >= 0.30 or crit_count >= 25 or risk_index >= 75:
        return "CRITICAL (ACTIVE DDoS / INTRUSION)", "severity-critical", risk_index
    if anomaly_rate >= 0.18 or crit_count >= 10 or risk_index >= 50:
        return "HIGH (ELEVATED ANOMALY ACTIVITY)", "severity-high", risk_index
    if anomaly_rate >= 0.08 or risk_index >= 25:
        return "MODERATE (SUSPICIOUS PROBING)", "severity-elevated", risk_index
    return "LOW (BASELINE NOMINAL)", "severity-low", risk_index


def init_session_state(bundle: dict):
    """Initialize live traffic buffer in Streamlit session state."""
    if "traffic_buffer" not in st.session_state:
        data_file = os.path.join("data", "network_traffic.csv")
        if os.path.exists(data_file):
            base_df = pd.read_csv(data_file).tail(600).reset_index(drop=True)
        else:
            base_df = generate_network_traffic(n_samples=600, anomaly_ratio=0.18, seed=42)
        st.session_state.traffic_buffer = predict_traffic(
            base_df, bundle, engine="Hybrid Ensemble (RF + IsoForest)"
        )
    if "stream_counter" not in st.session_state:
        st.session_state.stream_counter = 1


def append_live_batch(bundle: dict, n_flows: int, anomaly_ratio: float, engine: str):
    """Simulate a new batch of live NetFlow packets arriving on the wire."""
    st.session_state.stream_counter += 1
    new_raw = generate_network_traffic(
        n_samples=n_flows,
        anomaly_ratio=anomaly_ratio,
        seed=int(time.time() * 1000) % 100000 + st.session_state.stream_counter,
        start_time=datetime.now(),
    )
    new_pred = predict_traffic(new_raw, bundle, engine=engine)
    combined = pd.concat([st.session_state.traffic_buffer, new_pred], ignore_index=True)
    # Keep rolling window of up to 2,500 flows for snappy rendering
    st.session_state.traffic_buffer = combined.tail(2500).reset_index(drop=True)


def main():
    bundle = get_model_bundle()
    init_session_state(bundle)

    # Sidebar Controls
    with st.sidebar:
        st.title("🛡️ SentinelNet AI")
        st.caption("Enterprise NetFlow Anomaly & DDoS Detection System")
        st.divider()

        st.subheader("⚙️ Detection Engine")
        selected_engine = st.selectbox(
            "ML Inference Mode",
            options=[
                "Hybrid Ensemble (RF + IsoForest)",
                "Random Forest (Supervised)",
                "Isolation Forest (Unsupervised)",
            ],
            index=0,
            help="Choose between Supervised Random Forest, Unsupervised Isolation Forest, or the Hybrid Ensemble.",
        )

        st.divider()
        st.subheader("📡 Traffic Stream Simulator")
        batch_size = st.slider("Batch Size (Flows per Tick)", min_value=15, max_value=250, value=60, step=15)
        sim_anomaly_rate = st.slider(
            "Simulated Threat Intensity (%)",
            min_value=0,
            max_value=80,
            value=18,
            step=2,
            help="Percentage of anomalous/attack flows injected in new batches.",
        )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("➕ Capture Batch", width="stretch", type="primary"):
                append_live_batch(bundle, batch_size, sim_anomaly_rate / 100.0, selected_engine)
        with col_btn2:
            if st.button("🚨 DDoS Surge", width="stretch"):
                append_live_batch(bundle, 140, 0.72, selected_engine)

        live_auto = st.toggle("⚡ Auto-Stream Live Feed", value=False)
        refresh_interval = st.select_slider(
            "Stream Refresh Rate (sec)", options=[1.0, 2.0, 3.0, 5.0], value=2.0
        )

        st.divider()
        st.subheader("🔍 Feed Filters")
        protocols_filter = st.multiselect(
            "Filter Protocols", options=["TCP", "UDP", "ICMP"], default=["TCP", "UDP", "ICMP"]
        )
        status_filter = st.radio(
            "Traffic Classification", options=["All Traffic", "Anomalies Only", "Normal Only"], index=0
        )
        feed_window = st.slider("Analysis Window (Recent Flows)", 100, 2000, 500, step=100)

        st.divider()
        if st.button("🔄 Reset & Retrain Models", width="stretch"):
            with st.spinner("Regenerating dataset and retraining ML models..."):
                save_dataset("data/network_traffic.csv", n_samples=6000)
                train_and_save_models()
                st.cache_resource.clear()
                del st.session_state["traffic_buffer"]
                st.rerun()

    # Re-score current buffer with selected ML engine
    df_all = predict_traffic(st.session_state.traffic_buffer.tail(feed_window), bundle, engine=selected_engine)
    if protocols_filter:
        df_view = df_all[df_all["Protocol"].isin(protocols_filter)].copy()
    else:
        df_view = df_all.copy()

    if status_filter == "Anomalies Only":
        df_filtered = df_view[df_view["Predicted_Label"] == "Anomaly"].copy()
    elif status_filter == "Normal Only":
        df_filtered = df_view[df_view["Predicted_Label"] == "Normal"].copy()
    else:
        df_filtered = df_view.copy()

    # Header & Threat Status Banner
    st.markdown("## 🛡️ AI Network Traffic Anomaly Detector")
    threat_label, threat_css, risk_index = compute_overall_threat_level(df_view)
    st.markdown(
        f"""
        <div class="threat-banner {threat_css}">
            <span>📡 <strong>LIVE SOC TELEMETRY STATUS:</strong> {threat_label}</span>
            <span>Engine: <strong>{selected_engine}</strong> &nbsp;|&nbsp; Threat Risk Index: <strong>{risk_index:.1f} / 100</strong></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top Live Metrics Row
    total_flows = len(df_view)
    total_packets = int(df_view["Packet_Count"].sum()) if not df_view.empty else 0
    anomaly_flows = int((df_view["Predicted_Label"] == "Anomaly").sum())
    normal_flows = total_flows - anomaly_flows
    anomaly_pct = (anomaly_flows / total_flows * 100.0) if total_flows > 0 else 0.0
    normal_pct = 100.0 - anomaly_pct
    unique_malicious_ips = (
        int(df_view.loc[df_view["Predicted_Label"] == "Anomaly", "IP_Source"].nunique())
        if anomaly_flows > 0
        else 0
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Total Packets Analyzed",
        f"{total_packets:,}",
        delta=f"{total_flows:,} NetFlow records",
    )
    m2.metric(
        "Normal vs Anomaly Ratio",
        f"{normal_pct:.1f}% / {anomaly_pct:.1f}%",
        delta=f"{anomaly_flows:,} Anomalies ({normal_flows:,} Normal)",
        delta_color="inverse",
    )
    m3.metric(
        "Threat Severity Level",
        threat_label.split(" ")[0],
        delta=f"Risk Score: {risk_index:.1f}/100",
        delta_color="inverse" if risk_index >= 35 else "normal",
    )
    m4.metric(
        "Flagged Source IPs",
        f"{unique_malicious_ips}",
        delta=f"Avg Score: {df_view['Anomaly_Score'].mean():.3f}" if not df_view.empty else "0.000",
        delta_color="off",
    )

    # Main Navigation Tabs
    tab_dashboard, tab_feed, tab_model, tab_inspector = st.tabs(
        [
            "📊 Real-Time Analytics & Plotly Charts",
            "🖧 Live NetFlow Traffic Feed",
            "🧠 ML Model Performance & Explainability",
            "🔬 Interactive Packet Probe",
        ]
    )

    with tab_dashboard:
        if df_view.empty:
            st.warning("No NetFlow traffic matches the current filter selection.")
        else:
            # Chart 1: Anomaly Occurrences Over Time
            st.subheader("📈 Anomaly Occurrences & Traffic Volume Over Time")
            df_time = df_view.copy()
            df_time["Flow_Sequence"] = np.arange(1, len(df_time) + 1)
            df_time["Time_Bucket"] = (df_time["Flow_Sequence"] - 1) // max(1, len(df_time) // 25)

            bucket_agg = (
                df_time.groupby(["Time_Bucket", "Predicted_Label"], as_index=False)
                .agg(
                    Flow_Count=("Port", "count"),
                    Total_Packets=("Packet_Count", "sum"),
                    Last_Timestamp=("Timestamp", "last"),
                )
            )

            fig_timeline = px.area(
                bucket_agg,
                x="Last_Timestamp",
                y="Flow_Count",
                color="Predicted_Label",
                color_discrete_map={"Normal": "#10b981", "Anomaly": "#ef4444"},
                markers=True,
                title="Real-Time NetFlow Stream: Normal vs. Anomaly Occurrences Over Time",
                labels={
                    "Last_Timestamp": "Capture Timestamp",
                    "Flow_Count": "NetFlow Count per Window",
                    "Predicted_Label": "Classification",
                },
                template="plotly_dark",
            )
            fig_timeline.update_layout(height=360, margin=dict(l=20, r=20, t=45, b=20), hovermode="x unified")
            st.plotly_chart(fig_timeline, width="stretch")

            # Row 2: Packet Size Distributions & Attack Vector Breakdown
            c_left, c_right = st.columns(2)

            with c_left:
                st.subheader("📦 Packet Size Distribution (Normal vs. Anomaly)")
                fig_pkt = px.histogram(
                    df_view,
                    x="Packet_Size",
                    color="Predicted_Label",
                    marginal="box",
                    nbins=45,
                    barmode="overlay",
                    opacity=0.78,
                    color_discrete_map={"Normal": "#10b981", "Anomaly": "#ef4444"},
                    title="Packet Payload Size (Bytes) Distribution & Quartiles",
                    labels={"Packet_Size": "Packet Size (Bytes)", "Predicted_Label": "Traffic Class"},
                    template="plotly_dark",
                )
                fig_pkt.update_layout(height=380, margin=dict(l=20, r=20, t=45, b=20))
                st.plotly_chart(fig_pkt, width="stretch")

            with c_right:
                st.subheader("🎯 Detected Threat Signatures & Attack Types")
                atk_counts = (
                    df_view["Predicted_Attack"]
                    .value_counts()
                    .rename_axis("Attack_Category")
                    .reset_index(name="Count")
                )
                fig_atk = px.bar(
                    atk_counts,
                    x="Count",
                    y="Attack_Category",
                    orientation="h",
                    color="Attack_Category",
                    text="Count",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    title="Flow Classification by Signature / Attack Vector",
                    template="plotly_dark",
                )
                fig_atk.update_layout(
                    height=380,
                    showlegend=False,
                    margin=dict(l=20, r=20, t=45, b=20),
                    yaxis={"categoryorder": "total ascending"},
                )
                st.plotly_chart(fig_atk, width="stretch")

            # Row 3: Flow Duration vs Packet Size Scatter & Protocol Breakdown
            c_bot1, c_bot2 = st.columns([3, 2])
            with c_bot1:
                fig_scatter = px.scatter(
                    df_view.tail(350),
                    x="Flow_Duration",
                    y="Packet_Size",
                    size="Packet_Count",
                    color="Anomaly_Score",
                    symbol="Protocol",
                    hover_data=["IP_Source", "IP_Dest", "Port", "Predicted_Attack", "Threat_Severity"],
                    color_continuous_scale="Turbo",
                    log_x=True,
                    title="Multi-Dimensional Anomaly Cluster (Flow Duration vs. Packet Size vs. Volume)",
                    labels={
                        "Flow_Duration": "Flow Duration (ms, log scale)",
                        "Packet_Size": "Packet Size (Bytes)",
                        "Anomaly_Score": "Anomaly Score",
                    },
                    template="plotly_dark",
                )
                fig_scatter.update_layout(height=370, margin=dict(l=20, r=20, t=45, b=20))
                st.plotly_chart(fig_scatter, width="stretch")

            with c_bot2:
                proto_grp = (
                    df_view.groupby(["Protocol", "Predicted_Label"], as_index=False)
                    .size()
                    .rename(columns={"size": "Flows"})
                )
                fig_proto = px.sunburst(
                    proto_grp,
                    path=["Protocol", "Predicted_Label"],
                    values="Flows",
                    color="Predicted_Label",
                    color_discrete_map={"(?)": "#334155", "Normal": "#10b981", "Anomaly": "#ef4444"},
                    title="Protocol & Anomaly Hierarchy",
                    template="plotly_dark",
                )
                fig_proto.update_layout(height=370, margin=dict(l=20, r=20, t=45, b=20))
                st.plotly_chart(fig_proto, width="stretch")

    with tab_feed:
        st.subheader("🖧 Real-Time NetFlow Packet Feed Visualizer")
        st.caption(
            f"Showing {len(df_filtered):,} flows (sorted newest first). "
            "Anomalous flows are highlighted with risk scores and threat severity."
        )

        display_cols = [
            "Timestamp",
            "IP_Source",
            "IP_Dest",
            "Port",
            "Protocol",
            "TCP_Flags",
            "Packet_Size",
            "Flow_Duration",
            "Packet_Count",
            "Anomaly_Score",
            "Threat_Severity",
            "Predicted_Attack",
            "Predicted_Label",
        ]
        feed_df = df_filtered[display_cols].iloc[::-1].head(250).reset_index(drop=True)

        def style_row(row):
            if row["Predicted_Label"] == "Anomaly":
                return ["background-color: rgba(239, 68, 68, 0.18); color: #fca5a5;"] * len(row)
            return ["background-color: rgba(16, 185, 129, 0.05);"] * len(row)

        st.dataframe(
            feed_df.style.apply(style_row, axis=1).format(
                {
                    "Packet_Size": "{:.1f} B",
                    "Flow_Duration": "{:.1f} ms",
                    "Anomaly_Score": "{:.3f}",
                }
            ),
            width="stretch",
            height=520,
        )

        csv_bytes = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export Filtered NetFlow Logs (CSV)",
            data=csv_bytes,
            file_name="sentinelnet_netflow_logs.csv",
            mime="text/csv",
        )

    with tab_model:
        st.subheader("🧠 Machine Learning Model Evaluation & Feature Explainability")
        metrics = bundle["metrics"]

        col_rf, col_iso, col_hyb = st.columns(3)
        with col_rf:
            st.markdown("#### 🌲 Supervised Random Forest")
            rf_m = metrics["random_forest"]
            st.metric("Test Accuracy", f"{rf_m['accuracy']*100:.2f}%")
            st.metric("F1-Score / ROC-AUC", f"{rf_m['f1_score']*100:.2f}%  |  {rf_m['roc_auc']:.4f}")
            st.caption(f"Precision: {rf_m['precision']*100:.2f}% | Recall: {rf_m['recall']*100:.2f}%")

        with col_iso:
            st.markdown("#### 🔍 Unsupervised Isolation Forest")
            iso_m = metrics["isolation_forest"]
            st.metric("Test Accuracy", f"{iso_m['accuracy']*100:.2f}%")
            st.metric("F1-Score / ROC-AUC", f"{iso_m['f1_score']*100:.2f}%  |  {iso_m['roc_auc']:.4f}")
            st.caption(f"Precision: {iso_m['precision']*100:.2f}% | Recall: {iso_m['recall']*100:.2f}%")

        with col_hyb:
            st.markdown("#### ⚡ Hybrid Ensemble (Default)")
            hyb_m = metrics["hybrid_ensemble"]
            st.metric("Test Accuracy", f"{hyb_m['accuracy']*100:.2f}%")
            st.metric("F1-Score / ROC-AUC", f"{hyb_m['f1_score']*100:.2f}%  |  {hyb_m['roc_auc']:.4f}")
            st.caption(f"Precision: {hyb_m['precision']*100:.2f}% | Recall: {hyb_m['recall']*100:.2f}%")

        st.divider()
        c_imp, c_cm = st.columns(2)
        with c_imp:
            imp_df = (
                pd.DataFrame(
                    {
                        "Feature": list(metrics["feature_importances"].keys()),
                        "Importance": list(metrics["feature_importances"].values()),
                    }
                )
                .sort_values("Importance", ascending=True)
            )
            fig_imp = px.bar(
                imp_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Tealgrn",
                title="Random Forest Feature Importance Ranking",
                template="plotly_dark",
            )
            fig_imp.update_layout(height=360, margin=dict(l=20, r=20, t=45, b=20))
            st.plotly_chart(fig_imp, width="stretch")

        with c_cm:
            cm = np.array(metrics["random_forest"]["confusion_matrix"])
            fig_cm = go.Figure(
                data=go.Heatmap(
                    z=cm,
                    x=["Predicted Normal", "Predicted Anomaly"],
                    y=["Actual Normal", "Actual Anomaly"],
                    text=cm,
                    texttemplate="%{text}",
                    colorscale="Blues",
                )
            )
            fig_cm.update_layout(
                title="Confusion Matrix (Test Set - 1,500 Flows)",
                template="plotly_dark",
                height=360,
                margin=dict(l=20, r=20, t=45, b=20),
            )
            st.plotly_chart(fig_cm, width="stretch")

    with tab_inspector:
        st.subheader("🔬 Custom NetFlow Packet Inspector & Simulator")
        st.write("Test how the AI model scores an individual custom NetFlow record in real time.")

        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            p_src = st.text_input("Source IP", value="185.220.101.44")
            p_dst = st.text_input("Destination IP", value="10.0.100.10")
            p_port = st.number_input("Destination Port", min_value=0, max_value=65535, value=80)
        with ic2:
            p_proto = st.selectbox("Protocol", options=["TCP", "UDP", "ICMP"], index=0)
            p_flag = st.selectbox("TCP Flags", options=["SYN", "ACK", "PSH-ACK", "SYN-ACK", "RST", "NONE"], index=0)
            p_size = st.number_input("Packet Size (Bytes)", min_value=20.0, max_value=9000.0, value=60.0, step=10.0)
        with ic3:
            p_dur = st.number_input("Flow Duration (ms)", min_value=0.1, max_value=200000.0, value=8.5, step=5.0)
            p_cnt = st.number_input("Packet Count", min_value=1, max_value=10000, value=950, step=10)

        sample_df = pd.DataFrame(
            [
                {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "IP_Source": p_src,
                    "IP_Dest": p_dst,
                    "Port": int(p_port),
                    "Protocol": p_proto,
                    "TCP_Flags": p_flag,
                    "Packet_Size": float(p_size),
                    "Flow_Duration": float(p_dur),
                    "Packet_Count": int(p_cnt),
                }
            ]
        )
        sample_res = predict_traffic(sample_df, bundle, engine=selected_engine).iloc[0]

        r1, r2, r3 = st.columns(3)
        r1.metric("Prediction Verdict", sample_res["Predicted_Label"])
        r2.metric("Anomaly Confidence Score", f"{sample_res['Anomaly_Score']*100:.1f}%")
        r3.metric("Signature Classification", f"{sample_res['Predicted_Attack']} ({sample_res['Threat_Severity']})")

    # Handle Auto-Stream Live Feed toggle
    if live_auto:
        time.sleep(float(refresh_interval))
        append_live_batch(bundle, batch_size, sim_anomaly_rate / 100.0, selected_engine)
        st.rerun()


if __name__ == "__main__":
    main()

