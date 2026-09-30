# MPLADS AI Monitor — v6.4 Refined Package

## What is included
- backend/database.py     — Postgres / SQLite (MS SQL removed)
- backend/main.py         — 6.4-DeepAnalytics FastAPI
- backend/final_risk.py   — FINAL_RISK_VERSION 4.1
- backend/__init__.py
- backend/risk_engine.py, data_loader.py, feature_engineering.py, load_sql.py
- dashboard/app.py        — 6.4-Dash-DeepAnalytics
- ml/anomaly_model.py     — MODEL_VERSION 4.1
- requirements.txt, run_all.py, diagnose_api.py, CHANGELOG_V6_4.txt

## Quick start
1. Place this folder next to your existing `data/` and `models/` (or copy those folders in).
2. python -m pip install -r requirements.txt --upgrade
3. python run_all.py
4. Open http://127.0.0.1:8050  (Dash) and http://127.0.0.1:8000/docs (API)

Local demos need no database config (SQLite auto).  
For cloud: set DATABASE_URL to a free Postgres connection string.

Data files (final_risk_data.csv etc.) are NOT inside this zip — use the ones from your existing project.
