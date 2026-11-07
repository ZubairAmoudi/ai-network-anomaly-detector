import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

def evaluate_anomaly_detector():
    print("=== Starting Anomaly Detection Model Evaluation ===")
    
    # Define data paths and trained model
    data_path = "data/processed/network_traffic_cleaned.csv"
    model_path = "models/isolation_forest_model.pkl"
    
    # Check if files exist
    if not os.path.exists(data_path):
        print(f"[-] Error: Data file not found at path: {data_path} (run model.py to generate it)")
        return
    if not os.path.exists(model_path):
        print(f"[-] Warning: Model not found at {model_path}, searching in models directory...")
        model_files = [f for f in os.listdir("models") if f.endswith('.pkl') or f.endswith('.joblib')]
        if model_files:
            model_path = os.path.join("models", model_files[0])
            print(f"[*] Model found: {model_path}")
        else:
            print("[-] Critical Error: No saved trained model found!")
            return

    # 2. Load data and model
    df = pd.read_csv(data_path)
    # Score only the held-out test rows: rows used for training would inflate the metrics.
    if "Label" in df.columns:
        from model import holdout_indices
        total_rows = len(df)
        _, test_idx = holdout_indices(df)
        df = df.iloc[test_idx].reset_index(drop=True)
        print(f"[*] Evaluating on held-out test split: {len(df)} of {total_rows} rows.")
    else:
        print("[!] Warning: 'Label' column missing; cannot reproduce test split, evaluating on all rows.")
    loaded_data = joblib.load(model_path)
    scaler = None

    # Check if saved file is a dictionary containing the model
    if isinstance(loaded_data, dict):
        scaler = loaded_data.get("scaler", None)
        model = None
        for key in ("model", "classifier", "rf_binary", "isolation_forest", "estimator"):
            if key in loaded_data and hasattr(loaded_data[key], "predict"):
                model = loaded_data[key]
                break
        if model is None:
            for val in loaded_data.values():
                if hasattr(val, "predict"):
                    model = val
                    break
        if model is None:
            print("[-] Critical Error: Saved dictionary does not contain a valid prediction model object!")
            return
        print("[*] Model successfully extracted from the saved dictionary.")
    else:
        model = loaded_data

    print("[+] Data and model loaded successfully.")

    # 3. Prepare target variables (y_true) and features (X) safely to avoid UnboundLocalError
    y_true = None
    label_col = next((col for col in ("Label", "label", "Is_Anomaly") if col in df.columns), None)

    if label_col is not None:
        raw_labels = df[label_col]
        if pd.api.types.is_numeric_dtype(raw_labels):
            y_true = np.where(raw_labels == -1, 1, raw_labels.to_numpy(dtype=int))
        else:
            y_true = np.where(
                raw_labels.astype(str).str.strip().str.lower().isin(["anomaly", "attack", "1", "true"]),
                1,
                0,
            )
    elif "Attack_Type" in df.columns:
        print("[!] Notice: 'Label' column not found, deriving true labels from 'Attack_Type' column.")
        y_true = np.where(df["Attack_Type"].astype(str).str.lower() != "normal", 1, 0)
    else:
        print("[-] Warning: 'Label' column not found in data file! Proceeding with predictions without ground truth metrics.")

    expected_features = getattr(model, "n_features_in_", None)
    X = df.drop(columns=["Label", "label", "Is_Anomaly", "Attack_Type"], errors="ignore")

    if expected_features is not None and X.select_dtypes(include=[np.number]).shape[1] != expected_features:
        try:
            from model import extract_features
            X = extract_features(df)
        except Exception:
            pass

    # Select numeric columns only for the model
    X = X.select_dtypes(include=[np.number])

    if X.empty:
        print("[-] Error: No valid numeric feature columns found for prediction in (X)!")
        return

    # Apply StandardScaler if attached to model and feature count matches
    if scaler is not None and getattr(scaler, "n_features_in_", None) == X.shape[1]:
        X = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)

    # 4. Predict using the trained model
    raw_preds = model.predict(X.to_numpy() if not hasattr(model, "feature_names_in_") and isinstance(X, pd.DataFrame) else X)
    unique_preds = set(np.unique(raw_preds))

    # Standardize predictions to match 0 (normal) and 1 (anomaly/attack)
    if unique_preds.issubset({0, 1}):
        y_pred = np.asarray(raw_preds, dtype=int)
    elif unique_preds.issubset({-1, 1}):
        y_pred = np.where(np.asarray(raw_preds) == -1, 1, 0)
    else:
        y_pred = np.where(
            pd.Series(raw_preds).astype(str).str.lower().isin(["anomaly", "attack", "1", "-1"]),
            1,
            0,
        )

    if y_true is None:
        print(f"[+] Prediction completed for {len(y_pred)} records. Detected anomalies: {int(np.sum(y_pred))}")
        return

    # 5. Calculate base performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)

    # 6. Print clear reports and results
    print("\n" + "="*40)
    print("       AI Model Evaluation Results")
    print("="*40)
    print(f"[*] Overall Accuracy              : {accuracy * 100:.2f}%")
    print(f"[*] Precision                     : {precision * 100:.2f}%")
    print(f"[*] Recall                        : {recall * 100:.2f}%")
    print("\n[+] Confusion Matrix:")
    print(conf_matrix)
    print("\n[+] Classification Report:")
    print(classification_report(y_true, y_pred))
    print("="*40)

if __name__ == "__main__":
    evaluate_anomaly_detector()