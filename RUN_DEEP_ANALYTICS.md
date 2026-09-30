# MPLADS AI Monitor — Deep Analytics v6

## What changed

`dashboard/app.py` now renders a high-density dark analytics control center with 53 filter-synchronised overview charts plus deeper Risk, Financial & Execution, and Geography surfaces.

`backend/main.py` adds `/api/v1/deep-analytics`. Aggregated charts use the complete filtered population; only expensive scatter views use a seeded browser-performance sample capped at 3,500 rows.

The 500-row work queue remains a review surface and is not used as the denominator for the deep aggregated charts.

## Windows run

Activate the existing project environment, then run:

```powershell
python run_all.py
```

Or run the services separately:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

and in a second terminal:

```powershell
python dashboard/app.py
```

Open:

- FastAPI: `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`
- Dash: `http://127.0.0.1:8050`

## Dependencies

The updated `requirements.txt` explicitly includes Dash, Polars, FastAPI, Uvicorn, Requests and SQLAlchemy in addition to the existing ML/plotting stack.

## Important data behaviour

No latitude/longitude is fabricated. When the analytical dataset does not contain valid coordinates, the work-level map reports that coordinates are unavailable and the dashboard falls back to state/district concentration charts.

Risk signals remain analytical review signals. The interface does not turn an anomaly, similarity candidate, timing benchmark or data-quality issue into a finding of fraud or wrongdoing.
