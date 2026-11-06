"""
AI Network Traffic Anomaly Detection Model Pipeline.
Trains and evaluates both:
1. Random Forest Classifier (Supervised high-precision anomaly & threat classifier)
2. Isolation Forest (Unsupervised zero-day anomaly detector)

Saves trained artifacts and evaluation metrics for real-time inference in Streamlit.
"""

import json
import os
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_generator import generate_network_traffic, save_dataset


PROTOCOL_MAP = {"TCP": 0, "UDP": 1, "ICMP": 2}
FLAG_MAP = {"NONE": 0, "ACK": 1, "PSH-ACK": 2, "SYN-ACK": 3, "SYN": 4, "RST": 5, "FIN": 6}

FEATURE_COLUMNS = [
    "Port",
    "Protocol_Code",
    "TCP_Flag_Code",
    "Packet_Size",
    "Flow_Duration",
    "Packet_Count",
    "Packets_Per_Sec",
    "Bytes_Per_Sec",
    "Is_Well_Known_Port",
    "Is_External_Source",
]


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw NetFlow DataFrame into numeric ML feature matrix."""
    feat = pd.DataFrame(index=df.index)
    feat["Port"] = df["Port"].astype(float)
    feat["Protocol_Code"] = df["Protocol"].map(PROTOCOL_MAP).fillna(0).astype(float)

    if "TCP_Flags" in df.columns:
        feat["TCP_Flag_Code"] = df["TCP_Flags"].map(FLAG_MAP).fillna(0).astype(float)
    else:
        feat["TCP_Flag_Code"] = 0.0

    feat["Packet_Size"] = df["Packet_Size"].astype(float)
    feat["Flow_Duration"] = df["Flow_Duration"].astype(float)

    if "Packet_Count" in df.columns:
        pkt_count = df["Packet_Count"].astype(float)
    else:
        pkt_count = np.maximum(1.0, np.round(feat["Flow_Duration"] / 50.0))
    feat["Packet_Count"] = pkt_count

    duration_sec = np.maximum(feat["Flow_Duration"] / 1000.0, 0.001)
    if "Packets_Per_Sec" in df.columns:
        feat["Packets_Per_Sec"] = df["Packets_Per_Sec"].astype(float)
    else:
        feat["Packets_Per_Sec"] = pkt_count / duration_sec

    if "Bytes_Per_Sec" in df.columns:
        feat["Bytes_Per_Sec"] = df["Bytes_Per_Sec"].astype(float)
    else:
        feat["Bytes_Per_Sec"] = (feat["Packet_Size"] * pkt_count) / duration_sec

    feat["Is_Well_Known_Port"] = feat["Port"].isin([22, 53, 80, 123, 443, 993, 3306, 5432, 8080]).astype(float)
    feat["Is_External_Source"] = (
        ~df["IP_Source"].astype(str).str.startswith(("10.", "192.168.", "172.16."))
    ).astype(float)

    return feat[FEATURE_COLUMNS]


def train_and_save_models(
    data_path: str = "data/network_traffic.csv",
    model_dir: str = "models",
) -> dict[str, Any]:
    """Train Random Forest and Isolation Forest models, evaluate, and save artifacts."""
    os.makedirs(model_dir, exist_ok=True)

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        df = save_dataset(output_path=data_path, n_samples=6000)

    X = extract_features(df)
    y_binary = (df["Label"] == "Anomaly").to_numpy(dtype=int)
    y_attack = (
        df["Attack_Type"].to_numpy(dtype=str)
        if "Attack_Type" in df.columns
        else df["Label"].to_numpy(dtype=str)
    )

    X_train, X_test, y_train, y_test, atk_train, atk_test = train_test_split(
        X, y_binary, y_attack, test_size=0.25, random_state=42, stratify=y_binary
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Supervised Random Forest Classifier (Binary Anomaly + Multi-class Attack Type)
    rf_binary = RandomForestClassifier(
        n_estimators=120,
        max_depth=14,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    rf_binary.fit(X_train_scaled, y_train)

    rf_multiclass = RandomForestClassifier(
        n_estimators=120,
        max_depth=14,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    rf_multiclass.fit(X_train_scaled, atk_train)

    # 2. Unsupervised Isolation Forest
    contamination_rate = float(np.clip(np.mean(y_train), 0.05, 0.45))
    iso_forest = IsolationForest(
        n_estimators=150,
        contamination=contamination_rate,
        max_samples="auto",
        random_state=42,
        n_jobs=1,
    )
    iso_forest.fit(X_train_scaled)

    # Evaluate Random Forest
    rf_preds = rf_binary.predict(X_test_scaled)
    rf_probs = rf_binary.predict_proba(X_test_scaled)[:, 1]

    # Evaluate Isolation Forest (-1 = Anomaly -> 1, 1 = Normal -> 0)
    iso_raw_preds = iso_forest.predict(X_test_scaled)
    iso_preds = (iso_raw_preds == -1).astype(int)
    iso_scores = -iso_forest.score_samples(X_test_scaled)

    # Ensemble / Hybrid Score (70% RF probability + 30% normalized Isolation Forest score)
    iso_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-8)
    hybrid_probs = 0.75 * rf_probs + 0.25 * iso_norm
    hybrid_preds = (hybrid_probs >= 0.5).astype(int)

    metrics = {
        "dataset_size": int(len(df)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "anomaly_ratio": round(float(np.mean(y_binary)), 4),
        "feature_names": FEATURE_COLUMNS,
        "feature_importances": {
            col: round(float(imp), 4)
            for col, imp in zip(FEATURE_COLUMNS, rf_binary.feature_importances_)
        },
        "random_forest": {
            "accuracy": round(float(accuracy_score(y_test, rf_preds)), 4),
            "precision": round(float(precision_score(y_test, rf_preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, rf_preds, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, rf_preds, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, rf_probs)), 4),
            "confusion_matrix": confusion_matrix(y_test, rf_preds).tolist(),
        },
        "isolation_forest": {
            "accuracy": round(float(accuracy_score(y_test, iso_preds)), 4),
            "precision": round(float(precision_score(y_test, iso_preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, iso_preds, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, iso_preds, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, iso_scores)), 4),
            "confusion_matrix": confusion_matrix(y_test, iso_preds).tolist(),
        },
        "hybrid_ensemble": {
            "accuracy": round(float(accuracy_score(y_test, hybrid_preds)), 4),
            "precision": round(float(precision_score(y_test, hybrid_preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, hybrid_preds, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, hybrid_preds, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, hybrid_probs)), 4),
        },
    }

    bundle = {
        "scaler": scaler,
        "rf_binary": rf_binary,
        "rf_multiclass": rf_multiclass,
        "isolation_forest": iso_forest,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
        "iso_score_min": float(iso_scores.min()),
        "iso_score_max": float(iso_scores.max()),
    }

    model_path = os.path.join(model_dir, "anomaly_detector.joblib")
    metrics_path = os.path.join(model_dir, "metrics.json")

    joblib.dump(bundle, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return bundle


def load_model_bundle(model_dir: str = "models") -> dict[str, Any]:
    """Load saved ML model bundle, or train automatically if not found."""
    model_path = os.path.join(model_dir, "anomaly_detector.joblib")
    if not os.path.exists(model_path):
        return train_and_save_models(model_dir=model_dir)
    return joblib.load(model_path)


def predict_traffic(
    df: pd.DataFrame,
    bundle: dict[str, Any],
    engine: str = "Hybrid Ensemble (RF + IsoForest)",
) -> pd.DataFrame:
    """
    Run anomaly detection inference on a NetFlow DataFrame.
    Appends:
        - Predicted_Label ('Normal' / 'Anomaly')
        - Predicted_Attack ('Normal', 'DDoS_SYN_Flood', etc.)
        - Anomaly_Score (0.0 to 1.0 confidence / risk score)
        - Threat_Severity ('LOW', 'MODERATE', 'HIGH', 'CRITICAL')
    """
    result = df.copy()
    X = extract_features(result)
    X_scaled = bundle["scaler"].transform(X)

    rf_probs = bundle["rf_binary"].predict_proba(X_scaled)[:, 1]
    rf_attacks = bundle["rf_multiclass"].predict(X_scaled)

    iso_raw = -bundle["isolation_forest"].score_samples(X_scaled)
    iso_min = bundle.get("iso_score_min", 0.35)
    iso_max = bundle.get("iso_score_max", 0.75)
    iso_norm = np.clip((iso_raw - iso_min) / (iso_max - iso_min + 1e-8), 0.0, 1.0)
    iso_preds = bundle["isolation_forest"].predict(X_scaled) == -1

    if "Isolation Forest" in engine and "Hybrid" not in engine:
        scores = iso_norm
        is_anomaly = iso_preds
    elif "Random Forest" in engine:
        scores = rf_probs
        is_anomaly = rf_probs >= 0.5
    else:  # Hybrid Ensemble
        scores = 0.75 * rf_probs + 0.25 * iso_norm
        is_anomaly = (scores >= 0.5) | (rf_probs >= 0.65)

    result["Anomaly_Score"] = np.round(scores, 4)
    result["Predicted_Label"] = np.where(is_anomaly, "Anomaly", "Normal")
    result["Predicted_Attack"] = np.where(
        is_anomaly,
        np.where(rf_attacks == "Normal", "Zero_Day_Anomaly", rf_attacks),
        "Normal",
    )

    conditions = [
        result["Anomaly_Score"] >= 0.85,
        result["Anomaly_Score"] >= 0.60,
        result["Anomaly_Score"] >= 0.35,
    ]
    choices = ["CRITICAL", "HIGH", "MODERATE"]
    result["Threat_Severity"] = np.select(conditions, choices, default="LOW")

    return result


if __name__ == "__main__":
    print("Training AI Network Traffic Anomaly Detection Models...")
    artifact = train_and_save_models()
    m = artifact["metrics"]
    print("\n--- Model Evaluation Summary (Test Set) ---")
    print(f"Total Dataset Flows: {m['dataset_size']:,} (Anomaly Ratio: {m['anomaly_ratio']*100:.1f}%)")
    print(
        f"[Random Forest]    Accuracy: {m['random_forest']['accuracy']*100:.2f}% | "
        f"Precision: {m['random_forest']['precision']*100:.2f}% | "
        f"Recall: {m['random_forest']['recall']*100:.2f}% | "
        f"F1: {m['random_forest']['f1_score']*100:.2f}% | "
        f"ROC-AUC: {m['random_forest']['roc_auc']:.4f}"
    )
    print(
        f"[Isolation Forest] Accuracy: {m['isolation_forest']['accuracy']*100:.2f}% | "
        f"Precision: {m['isolation_forest']['precision']*100:.2f}% | "
        f"Recall: {m['isolation_forest']['recall']*100:.2f}% | "
        f"F1: {m['isolation_forest']['f1_score']*100:.2f}% | "
        f"ROC-AUC: {m['isolation_forest']['roc_auc']:.4f}"
    )
    print(
        f"[Hybrid Ensemble]  Accuracy: {m['hybrid_ensemble']['accuracy']*100:.2f}% | "
        f"F1: {m['hybrid_ensemble']['f1_score']*100:.2f}% | "
        f"ROC-AUC: {m['hybrid_ensemble']['roc_auc']:.4f}"
    )
    print("\nSaved trained model bundle to: models/anomaly_detector.joblib")

