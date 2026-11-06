# 🛡️ AI Network Traffic Anomaly Detector

An enterprise-grade, real-time network traffic monitoring and cyber threat detection system powered by Machine Learning and interactive Streamlit visualization.

---

## 🚀 Key Features

* **Advanced ML Inference Engine:** Utilizes a robust Hybrid Ensemble model (Isolation Forest & Random Forest) to accurately classify network flows.
* **Real-Time Stream Simulation:** Simulates live NetFlow traffic streams, adjusting threat intensities, batch sizes, and automated alert frequencies dynamically.
* **Interactive Dashboard:** Built with Streamlit and Plotly for high-performance visual analytics, tracking traffic volume, anomaly ratios, and live threat risk indexes.
* **Rigorous Evaluation Suite:** Comprehensive evaluation scripts (`evaluate.py`) calculating key performance metrics (Accuracy, Precision, Recall, and Confusion Matrix).

---

## 📂 Project Architecture

```text
Project/
│
├── data/                    # Dataset storage (e.g., network_traffic.csv)
├── models/                  # Trained ML models and scalers (.pkl / .joblib)
├── app.py                   # Main Streamlit interactive dashboard
├── model.py                 # Core machine learning training and inference logic
├── evaluate.py              # Model performance evaluation and confusion matrix script
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
