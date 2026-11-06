# AI Network Traffic Anomaly Detector - Environment Setup Script
$ErrorActionPreference = "Stop"

Write-Host "[1/4] Creating Python virtual environment (.venv)..." -ForegroundColor Cyan
if (-Not (Test-Path ".venv")) {
    python -m venv .venv
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor Yellow
}

Write-Host "[2/4] Upgrading pip and installing dependencies..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "[3/4] Generating synthetic NetFlow dataset..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe data_generator.py

Write-Host "[4/4] Training and evaluating Anomaly Detection ML models..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe model.py

Write-Host "`nSetup complete! Run the dashboard with:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\streamlit.exe run app.py" -ForegroundColor White

