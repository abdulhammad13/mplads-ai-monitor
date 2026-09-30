# MPLADS AI MONITOR

### Explainable Anomaly & Risk Monitoring for MPLADS Implementation

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Dash](https://img.shields.io/badge/UI-Dash-119DFF?logo=plotly&logoColor=white)](https://dash.plotly.com/)
[![Plotly](https://img.shields.io/badge/Analytics-Plotly-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![SQLite](https://img.shields.io/badge/Database-SQLite-003B57?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![ML](https://img.shields.io/badge/ML-Isolation%20Forest-6F42C1)](https://scikit-learn.org/)

> **SIH 2026 · Problem Statement SIH26102 · Team Fresh Minds · Team ID 120613**

MPLADS AI Monitor is an analytical monitoring system designed to surface unusual implementation patterns, execution risks, financial signals, and evidence-rich work profiles from MPLADS data.

The system is built around a simple principle:

> **Evidence-rich alerts over opaque scores.**

It combines a FastAPI analytical backend with an interactive Dash/Plotly dashboard, explainable anomaly/risk signals, state-level geography views, work-level investigation, and an AI Copilot that answers questions using the active analytical context.

---

## Why this project matters

Large public-program datasets can contain thousands of works spread across states, implementing agencies, financial stages, and completion states. Reviewing every record manually is difficult.

MPLADS AI Monitor helps an analyst move from:

**Scope → Explain → Compare → Investigate**

without replacing human review.

The platform is intended to help reviewers identify records or patterns that deserve closer examination; an anomaly or risk signal is **not, by itself, proof of fraud or wrongdoing**.

---

## Core capabilities

| Capability | What it provides |
|---|---|
| **Scope** | Filtered population, work counts, completion, expenditure and utilization |
| **Risk monitoring** | Explainable anomaly/risk signals from analytical features |
| **State comparison** | Cross-state metrics using the current analytical denominator |
| **Geography** | India state-level risk heatmap synchronized with filters |
| **Work investigation** | Work profile, timing, financial and evidence signals |
| **AI Copilot** | Natural-language questions grounded in current dashboard context |
| **Evidence** | Signal details and supporting context for manual review |

---

## System architecture

```text
                         ┌───────────────────────────┐
                         │        Dash / Plotly      │
                         │  Filters · Charts · Maps  │
                         │  Work Investigation       │
                         │  AI Copilot UI            │
                         └─────────────┬─────────────┘
                                       │ HTTP
                                       ▼
                         ┌───────────────────────────┐
                         │          FastAPI          │
                         │ Analytics · Chat · Scope  │
                         │ Evidence · Work Profiles │
                         └─────────────┬─────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
        ┌───────────┐            ┌────────────┐          ┌─────────────┐
        │  SQLite   │            │ ML / Risk  │          │ Data / CSV  │
        │ Analytical│            │ Isolation  │          │ Processing  │
        │   Store   │            │  Forest    │          │ & Features  │
        └───────────┘            └────────────┘          └─────────────┘
```

---

## Technology stack

- **Frontend:** Dash, Plotly, custom CSS/HTML
- **Backend:** FastAPI, Uvicorn
- **Data:** Python, Polars, Pandas, SQLite, SQLAlchemy
- **Machine learning:** scikit-learn Isolation Forest
- **Visualization:** Plotly
- **Supporting utilities:** Requests, python-dotenv, joblib

The application is intentionally separated so the dashboard presents analytical results rather than recomputing the core ML/business logic inside the UI.

---

## Project structure

```text
mplads_v64_refined/
├── backend/                 # FastAPI endpoints and backend services
├── dashboard/
│   ├── assets/              # logos, team portraits, GeoJSON and UI assets
│   ├── app.py               # main Dash frontend
│   └── ...
├── data/                    # analytical/data inputs
├── docs/                    # project documentation
├── ml/                      # anomaly / ML pipeline
├── models/                  # trained model artifacts
├── .env.example             # environment variable template
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Quick start — Windows PowerShell

### 1. Clone the repository

```powershell
git clone https://github.com/abdulhammad13/mplads-ai-monitor.git
cd mplads-ai-monitor
```

### 2. Create and activate the virtual environment

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

```powershell
Copy-Item .env.example .env
```

Review `.env` before running the application.

### 5. Start the backend

```powershell
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

### 6. Start the dashboard

```powershell
python dashboard\app.py
```

Open:

```text
http://127.0.0.1:8050
```

> Backend and dashboard entrypoints can vary with the final project layout. Keep the commands synchronized with the actual files committed to the repository.

---

## Analytical philosophy

The project deliberately favors **explainability and evidence**.

A risk score is treated as a signal for review, not an automated adjudication. Reviewers should be able to trace an alert to its observable analytical inputs and inspect the underlying work context.

---

## Data and privacy

Do not commit:

- credentials
- API tokens
- `.env`
- private datasets
- local virtual environments
- generated cache files
- machine-specific paths

Use `.env.example` to document required environment variables without exposing secrets.

---

## Team

**Team Fresh Minds**  
Smart India Hackathon 2026  
Problem Statement: **SIH26102**  
Team ID: **120613**

The dashboard includes a dedicated Team Fresh Minds interface with member roles and project responsibilities.

---

## Responsible use

This software is an analytical monitoring aid.

It should not be used to declare that an individual, agency, project, or record committed fraud based solely on a model output. Risk signals should be validated against the available evidence and reviewed by an authorized human analyst.

---

## Development principles

1. Keep analytical logic in the backend/ML layers.
2. Keep dashboard presentation synchronized with backend data.
3. Prefer explicit evidence over unexplained model outputs.
4. Never invent missing geographic coordinates or work attributes.
5. Keep UI changes isolated from data-processing logic.
6. Test syntax and critical API paths before pushing.
7. Keep documentation synchronized with actual commands and repository structure.

---

## Maintainers

**Abdul Hammad**  
GitHub: [@abdulhammad13](https://github.com/abdulhammad13)

Team Fresh Minds · Jamia Millia Islamia

---

## Status

This repository is an actively developed SIH 2026 project. Features, analytical logic, UI components and deployment details may evolve as validation and integration continue.
