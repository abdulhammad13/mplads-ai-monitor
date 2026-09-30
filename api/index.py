"""
MPLADS AI Monitor — Vercel serverless entrypoint (FastAPI).

Vercel detects the FastAPI `app` object and serves the whole API as one
Fluid Function. Designed for the free Hobby plan:

  • CSV-only by default (no SQLAlchemy / Postgres required)
  • Optional remote CSV via MPLADS_DATA_URL (GitHub raw / Blob / CDN)
  • Lazy load + in-memory cache (survives warm instances)
  • CORS open for the Dash dashboard origin
  • maxDuration handled in vercel.json (up to 300s on Hobby)

Local:  uvicorn api.index:app --reload --port 8000
Deploy: vercel --prod
"""

from __future__ import annotations

import math
import os
import re
import tempfile
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import polars as pl
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
CSV_CANDIDATES = (
    ROOT / "data" / "processed" / "final_risk_data.csv",
    ROOT / "data" / "final_risk_data.csv",
    Path("/tmp/mplads_final_risk_data.csv"),
)

SERVICE_VERSION = os.getenv("MPLADS_SERVICE_VERSION", "6.4-DeepAnalytics-Vercel")
DATA_SOURCE_MODE = os.getenv("MPLADS_DATA_SOURCE", "csv").strip().lower()
DATA_URL = os.getenv("MPLADS_DATA_URL", "").strip()
CORS_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "MPLADS_CORS_ORIGINS",
        "http://127.0.0.1:8050,http://localhost:8050,*",
    ).split(",")
    if o.strip()
]

WORKS: pl.DataFrame | None = None
WORKS_SOURCE: str | None = None

KEY_NUMERIC_COLUMNS = (
    "sanction_amount",
    "recommended_amount",
    "completed_amount",
    "total_expenditure",
    "allocated_amount",
    "utilization_pct",
    "final_risk_score",
    "priority_score",
    "ml_anomaly_percentile",
    "rule_risk_score",
    "financial_risk_score",
    "execution_risk_score",
    "duplicate_risk_score",
    "data_integrity_risk_score",
    "confidence_score",
    "priority_rank",
    "days_open_since_sanction",
    "days_rec_to_sanction",
    "days_sanction_to_complete",
    "independent_signal_count",
    "duplicate_group_count",
    "days_since_last_expenditure",
    "model_input_missing_pct",
    "signal_count",
    "data_quality_issue_count",
    "financial_exposure_percentile",
    "cost_robust_z",
    "duration_robust_z",
)

DATE_COLUMNS = (
    "recommended_date",
    "sanction_date",
    "completion_date",
    "first_expenditure_date",
    "last_expenditure_date",
    "monitoring_as_of_date",
)

BOOLEAN_COLUMNS = (
    "is_completed",
    "is_open",
    "is_duplicate_candidate",
    "flag_disbursement_over_sanction",
    "flag_bad_dates",
    "flag_overdue_open_work",
    "flag_stalled_expenditure",
    "flag_duplicate_candidate",
    "flag_cost_outlier",
    "flag_duration_outlier",
)

REQUIRED_COLUMNS = (
    "work_uid",
    "state",
    "sanction_amount",
    "final_risk_score",
    "risk_category",
    "priority_score",
)

RISK_ORDER = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
RISK_RANGES = {
    "LOW": {"min": 0.0, "max": 25.0},
    "MEDIUM": {"min": 25.0, "max": 50.0},
    "HIGH": {"min": 50.0, "max": 75.0},
    "CRITICAL": {"min": 75.0, "max": 100.0},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def to_records(frame: pl.DataFrame) -> list[dict[str, Any]]:
    if frame.is_empty():
        return []
    return [
        {key: _json_value(value) for key, value in row.items()}
        for row in frame.to_dicts()
    ]


def _sum(frame: pl.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    return _json_value(frame.select(pl.col(column).sum()).item())


def _mean(frame: pl.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    return _json_value(frame.select(pl.col(column).mean()).item())


def _median(frame: pl.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    return _json_value(frame.select(pl.col(column).median()).item())


def _count_true(frame: pl.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    value = frame.select(pl.col(column).fill_null(False).cast(pl.Int64).sum()).item()
    return int(value or 0)


def _safe_rate(numerator: float | int | None, denominator: int | float | None) -> float:
    if numerator is None or denominator in (None, 0):
        return 0.0
    return round(float(numerator) / float(denominator) * 100.0, 2)


def _deep_safe_rate(numerator: Any, denominator: Any) -> float:
    return _safe_rate(
        float(numerator) if numerator is not None else None,
        float(denominator) if denominator is not None else None,
    )


# ---------------------------------------------------------------------------
# Frame normalize + load
# ---------------------------------------------------------------------------


def _ensure_columns(frame: pl.DataFrame) -> pl.DataFrame:
    expressions: list[pl.Expr] = []
    numeric_defaults = {c: pl.Float64 for c in KEY_NUMERIC_COLUMNS}
    for column, dtype in numeric_defaults.items():
        if column not in frame.columns:
            expressions.append(pl.lit(None, dtype=dtype).alias(column))
    for column in BOOLEAN_COLUMNS:
        if column not in frame.columns:
            expressions.append(pl.lit(False).alias(column))
    if "risk_category" not in frame.columns:
        expressions.append(pl.lit(None, dtype=pl.String).alias("risk_category"))
    if expressions:
        frame = frame.with_columns(expressions)
    return frame


def _normalize_frame(frame: pl.DataFrame) -> pl.DataFrame:
    frame = frame.clone()
    frame = frame.rename({column: column.strip() for column in frame.columns})
    expressions: list[pl.Expr] = []

    for column in KEY_NUMERIC_COLUMNS:
        if column in frame.columns:
            expressions.append(
                pl.col(column)
                .cast(pl.String, strict=False)
                .str.strip_chars()
                .cast(pl.Float64, strict=False)
                .alias(column)
            )
    for column in DATE_COLUMNS:
        if column in frame.columns:
            expressions.append(
                pl.col(column)
                .cast(pl.String, strict=False)
                .str.strip_chars()
                .str.to_date(strict=False)
                .alias(column)
            )
    for column in BOOLEAN_COLUMNS:
        if column in frame.columns:
            expressions.append(
                pl.col(column)
                .cast(pl.String, strict=False)
                .str.strip_chars()
                .str.to_lowercase()
                .is_in(["true", "1", "yes", "y", "t"])
                .fill_null(False)
                .alias(column)
            )
    for column in (
        "state",
        "ida",
        "mp",
        "constituency",
        "work",
        "work_description",
        "work_category",
        "work_status",
        "financial_year",
        "primary_risk_reason",
        "risk_explanation",
        "risk_category",
        "work_uid",
        "evidence_json",
        "data_quality_status",
    ):
        if column in frame.columns:
            expressions.append(
                pl.col(column)
                .cast(pl.String, strict=False)
                .str.strip_chars()
                .alias(column)
            )
    if expressions:
        frame = frame.with_columns(expressions)
    frame = _ensure_columns(frame)
    finite_exprs = [
        pl.when(pl.col(column).is_finite())
        .then(pl.col(column))
        .otherwise(None)
        .alias(column)
        for column in KEY_NUMERIC_COLUMNS
        if column in frame.columns
    ]
    if finite_exprs:
        frame = frame.with_columns(finite_exprs)
    return frame


def _download_csv(url: str) -> Path:
    import urllib.request

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("MPLADS_DATA_URL must be http(s)")
    dest = Path(tempfile.gettempdir()) / "mplads_final_risk_data.csv"
    # Re-use cached file if already present and non-trivial
    if dest.is_file() and dest.stat().st_size > 10_000:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "MPLADS-Vercel/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, dest.open("wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    return dest


def _resolve_csv_path() -> Path:
    if DATA_URL:
        return _download_csv(DATA_URL)
    for candidate in CSV_CANDIDATES:
        if candidate.is_file() and candidate.stat().st_size > 256:
            return candidate
    raise FileNotFoundError(
        "No analytical CSV found. Place data/processed/final_risk_data.csv "
        "in the repo or set MPLADS_DATA_URL to a public HTTPS CSV."
    )


def _read_csv() -> pl.DataFrame:
    path = _resolve_csv_path()
    frame = pl.read_csv(
        path,
        infer_schema_length=10000,
        try_parse_dates=False,
        null_values=["", "NULL", "null", "NA", "N/A", "NaN"],
    )
    return _normalize_frame(frame)


def _has_required_schema(frame: pl.DataFrame) -> tuple[bool, list[str]]:
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    return not missing, missing


def load_data_once(force_refresh: bool = False) -> pl.DataFrame:
    global WORKS, WORKS_SOURCE
    if WORKS is not None and not force_refresh:
        return WORKS
    frame = _read_csv()
    valid, missing = _has_required_schema(frame)
    if not valid:
        raise RuntimeError(f"Analytical schema missing columns: {missing}")
    WORKS = frame
    WORKS_SOURCE = "url" if DATA_URL else "csv"
    return WORKS


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def _eq_filter(column: str, value: str) -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.String, strict=False)
        .str.strip_chars()
        .str.to_lowercase()
        == value.strip().lower()
    )


def _apply_filters_expr(
    frame: pl.DataFrame,
    *,
    state: str | None = None,
    district: str | None = None,
    mp: str | None = None,
    constituency: str | None = None,
    work_category: str | None = None,
    work_status: str | None = None,
    risk_category: str | None = None,
    completion_status: str | None = None,
    search: str | None = None,
    min_sanction: float | None = None,
    max_sanction: float | None = None,
    min_risk: float = 0.0,
    max_risk: float = 100.0,
) -> pl.DataFrame:
    result = frame
    predicates: list[pl.Expr] = []
    exact_filters = {
        "state": state,
        "ida": district,
        "mp": mp,
        "constituency": constituency,
        "work_category": work_category,
        "work_status": work_status,
        "risk_category": risk_category,
    }
    for column, value in exact_filters.items():
        if value and value != "All" and column in result.columns:
            predicates.append(_eq_filter(column, value))
    if completion_status and completion_status != "All" and "is_completed" in result.columns:
        if completion_status.casefold() == "completed":
            predicates.append(pl.col("is_completed").fill_null(False))
        elif completion_status.casefold() == "open":
            predicates.append(~pl.col("is_completed").fill_null(False))
    if search and search.strip():
        term = search.strip().lower()
        search_columns = [
            c
            for c in ("work", "work_description", "mp", "constituency", "ida", "work_uid")
            if c in result.columns
        ]
        if search_columns:
            predicates.append(
                pl.any_horizontal(
                    [
                        pl.col(c)
                        .cast(pl.String, strict=False)
                        .fill_null("")
                        .str.to_lowercase()
                        .str.contains(term, literal=True)
                        for c in search_columns
                    ]
                )
            )
    if "sanction_amount" in result.columns:
        if min_sanction is not None:
            predicates.append(pl.col("sanction_amount").ge(min_sanction))
        if max_sanction is not None:
            predicates.append(pl.col("sanction_amount").le(max_sanction))
    if "final_risk_score" in result.columns:
        predicates.append(pl.col("final_risk_score").is_between(min_risk, max_risk, closed="both"))
    if predicates:
        result = result.filter(pl.all_horizontal(predicates))
    return result


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _summary(frame: pl.DataFrame) -> dict[str, Any]:
    total = frame.height
    completed = _count_true(frame, "is_completed")
    sanctioned = _sum(frame, "sanction_amount")
    expenditure = _sum(frame, "total_expenditure")
    utilization = None
    if sanctioned is not None and expenditure is not None and sanctioned > 0:
        utilization = round(float(expenditure) / float(sanctioned) * 100.0, 2)
    high = 0
    critical = 0
    if "risk_category" in frame.columns:
        high = int(
            frame.select(pl.col("risk_category").is_in(["HIGH", "CRITICAL"]).sum()).item() or 0
        )
        critical = int(frame.select(pl.col("risk_category").eq("CRITICAL").sum()).item() or 0)
    return {
        "total_works": total,
        "completed_works": completed,
        "open_works": total - completed,
        "completion_rate_pct": _safe_rate(completed, total),
        "sanctioned_amount": sanctioned,
        "total_expenditure": expenditure,
        "portfolio_utilization_pct": utilization,
        "mean_risk": _mean(frame, "final_risk_score"),
        "median_risk": _median(frame, "final_risk_score"),
        "high_or_critical": high,
        "critical": critical,
        "high_or_critical_rate_pct": _safe_rate(high, total),
        "duplicate_candidates": _count_true(frame, "is_duplicate_candidate"),
        "overdue_open_works": _count_true(frame, "flag_overdue_open_work"),
        "multi_signal_cases": int(
            frame.select(pl.col("independent_signal_count").fill_null(0).ge(2).sum()).item() or 0
        ),
    }


def _risk_band_expr(column: str) -> pl.Expr:
    return (
        pl.when(pl.col(column).is_null())
        .then(None)
        .when(pl.col(column).le(25))
        .then(pl.lit("LOW"))
        .when(pl.col(column).le(50))
        .then(pl.lit("MEDIUM"))
        .when(pl.col(column).le(75))
        .then(pl.lit("HIGH"))
        .otherwise(pl.lit("CRITICAL"))
        .alias("risk_category")
    )


def _deep_group_stats(frame: pl.DataFrame, group_column: str) -> pl.DataFrame:
    if group_column not in frame.columns:
        return pl.DataFrame()
    return (
        frame.group_by(group_column, maintain_order=True)
        .agg(
            pl.len().alias("works"),
            pl.col("is_completed").fill_null(False).cast(pl.Int64).sum().alias("completed_works"),
            pl.col("sanction_amount").sum().alias("sanctioned_amount"),
            pl.col("total_expenditure").sum().alias("total_expenditure"),
            pl.col("final_risk_score").mean().alias("mean_risk"),
            pl.col("final_risk_score").median().alias("median_risk"),
            pl.col("priority_score").mean().alias("mean_priority"),
            pl.col("risk_category").is_in(["HIGH", "CRITICAL"]).sum().alias("high_or_critical"),
            pl.col("risk_category").eq("CRITICAL").sum().alias("critical"),
            pl.col("flag_overdue_open_work").fill_null(False).cast(pl.Int64).sum().alias("overdue_open_works"),
            pl.col("is_duplicate_candidate").fill_null(False).cast(pl.Int64).sum().alias("duplicate_candidates"),
        )
        .with_columns(
            pl.when(pl.col("sanctioned_amount") > 0)
            .then(pl.col("total_expenditure") / pl.col("sanctioned_amount") * 100)
            .otherwise(None)
            .alias("utilization_pct"),
            pl.when(pl.col("works") > 0)
            .then(pl.col("completed_works") / pl.col("works") * 100)
            .otherwise(0)
            .alias("completion_rate_pct"),
            pl.when(pl.col("works") > 0)
            .then(pl.col("high_or_critical") / pl.col("works") * 100)
            .otherwise(0)
            .alias("high_or_critical_rate_pct"),
            pl.when(pl.col("works") > 0)
            .then(pl.col("critical") / pl.col("works") * 100)
            .otherwise(0)
            .alias("critical_rate_pct"),
        )
    )


def _deep_json_records(frame: pl.DataFrame) -> list[dict[str, Any]]:
    return to_records(frame) if frame is not None and not frame.is_empty() else []


def _deep_flag_summary(frame: pl.DataFrame) -> list[dict[str, Any]]:
    flags = [
        ("Over sanction", "flag_disbursement_over_sanction"),
        ("Overdue open", "flag_overdue_open_work"),
        ("Stalled expenditure", "flag_stalled_expenditure"),
        ("Duplicate candidate", "flag_duplicate_candidate"),
        ("Cost outlier", "flag_cost_outlier"),
        ("Duration outlier", "flag_duration_outlier"),
        ("Bad dates", "flag_bad_dates"),
    ]
    total = frame.height
    rows = []
    for label, col in flags:
        if col not in frame.columns:
            continue
        count = _count_true(frame, col)
        rows.append(
            {
                "reason": label,
                "count": count,
                "share_pct": _deep_safe_rate(count, total),
            }
        )
    rows.sort(key=lambda r: r["count"], reverse=True)
    return rows


def _deep_distribution(
    frame: pl.DataFrame,
    column: str,
    bins: list[tuple[str, float | None, float | None]],
) -> list[dict[str, Any]]:
    if column not in frame.columns or frame.is_empty():
        return []
    total = frame.height
    rows = []
    for label, low, high in bins:
        expr = pl.col(column).is_not_null()
        if low is not None and low != float("-inf"):
            expr = expr & (pl.col(column) >= low)
        if high is not None:
            expr = expr & (pl.col(column) < high)
        count = int(frame.select(expr.sum()).item() or 0)
        rows.append({"bin": label, "count": count, "share_pct": _deep_safe_rate(count, total)})
    return rows


def _deep_profile(frame: pl.DataFrame, column: str) -> dict[str, Any]:
    if column not in frame.columns:
        return {"column": column, "available": False}
    series = frame.get_column(column)
    return {
        "column": column,
        "available": True,
        "count": int(series.len()),
        "nulls": int(series.null_count()),
        "mean": _json_value(series.mean()),
        "median": _json_value(series.median()),
        "min": _json_value(series.min()),
        "max": _json_value(series.max()),
    }


def _deep_time_frame(frame: pl.DataFrame, date_column: str) -> pl.DataFrame:
    if date_column not in frame.columns or frame.is_empty():
        return pl.DataFrame()
    tmp = frame.filter(pl.col(date_column).is_not_null()).with_columns(
        pl.col(date_column).dt.year().cast(pl.String).alias("financial_year")
    )
    if tmp.is_empty():
        return pl.DataFrame()
    return (
        tmp.group_by("financial_year", maintain_order=True)
        .agg(
            pl.len().alias("works"),
            pl.col("is_completed").fill_null(False).cast(pl.Int64).sum().alias("completed_works"),
            pl.col("sanction_amount").sum().alias("sanctioned_amount"),
            pl.col("total_expenditure").sum().alias("total_expenditure"),
            pl.col("final_risk_score").mean().alias("mean_risk"),
            pl.col("final_risk_score").median().alias("median_risk"),
            pl.col("priority_score").mean().alias("mean_priority"),
            pl.col("risk_category").is_in(["HIGH", "CRITICAL"]).sum().alias("high_or_critical"),
            pl.col("risk_category").eq("CRITICAL").sum().alias("critical"),
        )
        .with_columns(
            pl.when(pl.col("works") > 0)
            .then(pl.col("completed_works") / pl.col("works") * 100)
            .otherwise(0)
            .alias("completion_rate_pct"),
            pl.when(pl.col("sanctioned_amount") > 0)
            .then(pl.col("total_expenditure") / pl.col("sanctioned_amount") * 100)
            .otherwise(None)
            .alias("utilization_pct"),
            pl.when(pl.col("works") > 0)
            .then(pl.col("high_or_critical") / pl.col("works") * 100)
            .otherwise(0)
            .alias("high_or_critical_rate_pct"),
        )
        .sort("financial_year")
    )


def _deep_financial_band_stats(frame: pl.DataFrame) -> list[dict[str, Any]]:
    if frame.is_empty() or "sanction_amount" not in frame.columns:
        return []
    bands = [
        ("< ₹1 L", 0, 100000),
        ("₹1–5 L", 100000, 500000),
        ("₹5–10 L", 500000, 1000000),
        ("₹10–25 L", 1000000, 2500000),
        ("₹25–50 L", 2500000, 5000000),
        ("₹50 L–1 Cr", 5000000, 10000000),
        ("₹1–2 Cr", 10000000, 20000000),
        ("₹2–5 Cr", 20000000, 50000000),
        ("₹5 Cr+", 50000000, None),
    ]
    rows = []
    total = frame.height
    for label, low, high in bands:
        if high is None:
            subset = frame.filter(pl.col("sanction_amount") >= low)
        else:
            subset = frame.filter(
                (pl.col("sanction_amount") >= low) & (pl.col("sanction_amount") < high)
            )
        sanctioned = _sum(subset, "sanction_amount") or 0.0
        expenditure = _sum(subset, "total_expenditure") or 0.0
        works = subset.height
        completed = _count_true(subset, "is_completed")
        highcritical = (
            int(subset.select(pl.col("risk_category").is_in(["HIGH", "CRITICAL"]).sum()).item() or 0)
            if "risk_category" in subset.columns and works
            else 0
        )
        rows.append(
            {
                "band": label,
                "works": works,
                "share_pct": _deep_safe_rate(works, total),
                "sanctioned_amount": sanctioned,
                "total_expenditure": expenditure,
                "completion_rate_pct": _deep_safe_rate(completed, works),
                "high_or_critical_rate_pct": _deep_safe_rate(highcritical, works),
                "utilization_pct": _deep_safe_rate(expenditure, sanctioned),
            }
        )
    return rows


def _deep_risk_financial(frame: pl.DataFrame) -> list[dict[str, Any]]:
    if frame.is_empty() or "risk_category" not in frame.columns:
        return []
    result = (
        frame.group_by("risk_category", maintain_order=True)
        .agg(
            pl.len().alias("works"),
            pl.col("sanction_amount").sum().alias("sanctioned_amount"),
            pl.col("total_expenditure").sum().alias("total_expenditure"),
        )
        .sort("risk_category")
    )
    return to_records(result)


def _deep_scatter_sample(frame: pl.DataFrame, columns: list[str], limit: int = 2000) -> list[dict[str, Any]]:
    available = [c for c in columns if c in frame.columns]
    if frame.is_empty() or not available:
        return []
    sample = frame.select(available)
    if sample.height > limit:
        sample = sample.sample(n=limit, with_replacement=False, seed=42)
    return to_records(sample)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Warm the dataset on cold start so the first user request is fast.
    try:
        load_data_once()
    except Exception as exc:  # noqa: BLE001 — surface in /health
        print(f"[startup] data load deferred: {type(exc).__name__}: {exc}")
    yield


app = FastAPI(
    title="MPLADS AI Monitoring Platform",
    description=(
        "AI-assisted monitoring of MPLADS works. Analytical review signals only — "
        "does not adjudicate fraud or legal non-compliance."
    ),
    version=SERVICE_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if "*" not in CORS_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "MPLADS AI Monitoring API",
        "status": "running",
        "version": SERVICE_VERSION,
        "data_source": WORKS_SOURCE or DATA_SOURCE_MODE,
        "docs": "/docs",
        "health": "/health",
        "deep_analytics": "/api/v1/deep-analytics",
        "filter_options": "/api/v1/filter-options",
        "dashboard_summary": "/api/v1/dashboard-summary",
        "platform": "vercel-serverless",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        frame = load_data_once()
        as_of = None
        if "monitoring_as_of_date" in frame.columns:
            as_of = _json_value(frame.select(pl.col("monitoring_as_of_date").max()).item())
        return {
            "status": "ok",
            "service": "MPLADS AI Monitoring API",
            "version": SERVICE_VERSION,
            "works_loaded": frame.height,
            "data_source": WORKS_SOURCE or DATA_SOURCE_MODE,
            "data_as_of": as_of,
            "platform": "vercel",
        }
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "error": f"{type(exc).__name__}: {exc}",
                "hint": "Set MPLADS_DATA_URL or include data/processed/final_risk_data.csv",
            },
        )


@app.post("/api/v1/refresh")
def refresh() -> dict[str, Any]:
    global WORKS, WORKS_SOURCE
    WORKS = None
    WORKS_SOURCE = None
    frame = load_data_once(force_refresh=True)
    return {"status": "refreshed", "works_loaded": frame.height, "data_source": WORKS_SOURCE}


@app.get("/api/v1/filter-options")
def filter_options() -> dict[str, Any]:
    frame = load_data_once()

    def uniq(col: str, limit: int = 500) -> list[str]:
        if col not in frame.columns:
            return []
        vals = (
            frame.select(pl.col(col).cast(pl.String).str.strip_chars())
            .drop_nulls()
            .unique()
            .sort(col)
            .head(limit)
            .get_column(col)
            .to_list()
        )
        return [str(v) for v in vals if v and str(v).strip()]

    return {
        "states": uniq("state"),
        "districts": uniq("ida"),
        "mps": uniq("mp", 800),
        "constituencies": uniq("constituency", 800),
        "categories": uniq("work_category"),
        "statuses": uniq("work_status"),
        "risk_categories": list(RISK_ORDER),
        "completion_statuses": ["All", "Completed", "Open"],
    }


@app.get("/api/v1/dashboard-summary")
def dashboard_summary(
    state: str | None = None,
    district: str | None = None,
    mp: str | None = None,
    constituency: str | None = None,
    work_category: str | None = None,
    work_status: str | None = None,
    risk_category: str | None = None,
    completion_status: str | None = None,
    search: str | None = None,
    min_sanction: float | None = Query(default=None, ge=0),
    max_sanction: float | None = Query(default=None, ge=0),
    min_risk: float = Query(default=0, ge=0, le=100),
    max_risk: float = Query(default=100, ge=0, le=100),
) -> dict[str, Any]:
    frame = _apply_filters_expr(
        load_data_once(),
        state=state,
        district=district,
        mp=mp,
        constituency=constituency,
        work_category=work_category,
        work_status=work_status,
        risk_category=risk_category,
        completion_status=completion_status,
        search=search,
        min_sanction=min_sanction,
        max_sanction=max_sanction,
        min_risk=min_risk,
        max_risk=max_risk,
    )
    return {
        "scope": _summary(frame),
        "risk_band_definition": RISK_RANGES,
        "meta": {"rows": frame.height, "service": SERVICE_VERSION},
    }


@app.get("/api/v1/works")
def works_queue(
    state: str | None = None,
    district: str | None = None,
    mp: str | None = None,
    constituency: str | None = None,
    work_category: str | None = None,
    work_status: str | None = None,
    risk_category: str | None = None,
    completion_status: str | None = None,
    search: str | None = None,
    min_sanction: float | None = Query(default=None, ge=0),
    max_sanction: float | None = Query(default=None, ge=0),
    min_risk: float = Query(default=0, ge=0, le=100),
    max_risk: float = Query(default=100, ge=0, le=100),
    sort_by: str = Query(default="priority_score"),
    limit: int = Query(default=200, ge=1, le=500),
) -> dict[str, Any]:
    frame = _apply_filters_expr(
        load_data_once(),
        state=state,
        district=district,
        mp=mp,
        constituency=constituency,
        work_category=work_category,
        work_status=work_status,
        risk_category=risk_category,
        completion_status=completion_status,
        search=search,
        min_sanction=min_sanction,
        max_sanction=max_sanction,
        min_risk=min_risk,
        max_risk=max_risk,
    )
    sort_col = sort_by if sort_by in frame.columns else "priority_score"
    if sort_col in frame.columns:
        frame = frame.sort(sort_col, descending=True, nulls_last=True)
    cols = [
        c
        for c in (
            "work_uid",
            "state",
            "ida",
            "mp",
            "constituency",
            "work",
            "work_category",
            "work_status",
            "sanction_amount",
            "total_expenditure",
            "utilization_pct",
            "final_risk_score",
            "risk_category",
            "priority_score",
            "primary_risk_reason",
            "confidence_score",
            "is_completed",
        )
        if c in frame.columns
    ]
    page = frame.select(cols).head(limit)
    return {
        "total_matching": frame.height,
        "returned": page.height,
        "limit": limit,
        "records": to_records(page),
    }


@app.get("/api/v1/deep-analytics-status")
def deep_analytics_status() -> dict[str, Any]:
    frame = load_data_once()
    expected = [
        "risk_category",
        "work_category",
        "state",
        "ida",
        "sanction_amount",
        "total_expenditure",
        "final_risk_score",
        "priority_score",
    ]
    missing = [c for c in expected if c not in frame.columns]
    return {
        "status": "ready" if not missing else "degraded",
        "works_loaded": frame.height,
        "data_source": WORKS_SOURCE or DATA_SOURCE_MODE,
        "missing_columns": missing,
        "service_version": SERVICE_VERSION,
        "risk_band_definition": RISK_RANGES,
        "platform": "vercel",
        "guardrail": (
            "Signals are for human review only. The system does not adjudicate fraud."
        ),
    }


@app.get("/api/v1/deep-analytics")
def deep_analytics(
    state: str | None = None,
    district: str | None = None,
    mp: str | None = None,
    constituency: str | None = None,
    work_category: str | None = None,
    work_status: str | None = None,
    risk_category: str | None = None,
    completion_status: str | None = None,
    search: str | None = None,
    min_sanction: float | None = Query(default=None, ge=0),
    max_sanction: float | None = Query(default=None, ge=0),
    min_risk: float = Query(default=0, ge=0, le=100),
    max_risk: float = Query(default=100, ge=0, le=100),
) -> dict[str, Any]:
    if min_sanction is not None and max_sanction is not None and min_sanction > max_sanction:
        raise HTTPException(status_code=422, detail="min_sanction cannot exceed max_sanction")
    if min_risk > max_risk:
        raise HTTPException(status_code=422, detail="min_risk cannot exceed max_risk")

    frame = _apply_filters_expr(
        load_data_once(),
        state=state,
        district=district,
        mp=mp,
        constituency=constituency,
        work_category=work_category,
        work_status=work_status,
        risk_category=risk_category,
        completion_status=completion_status,
        search=search,
        min_sanction=min_sanction,
        max_sanction=max_sanction,
        min_risk=min_risk,
        max_risk=max_risk,
    )

    risk_final = []
    if "risk_category" in frame.columns:
        lookup: dict[str, int] = {}
        for row in frame.group_by("risk_category").agg(pl.len().alias("count")).to_dicts():
            lookup[str(row.get("risk_category"))] = safe_int(row.get("count"))
        total = frame.height
        risk_final = [
            {
                "risk_category": band,
                "count": lookup.get(band, 0),
                "pct": _deep_safe_rate(lookup.get(band, 0), total),
            }
            for band in RISK_ORDER
        ]

    state_group = _deep_group_stats(frame, "state")
    district_group = _deep_group_stats(frame, "ida")
    sector_group = _deep_group_stats(frame, "work_category")
    time_frame = _deep_time_frame(frame, "sanction_date")
    if time_frame.is_empty() and "financial_year" in frame.columns:
        time_frame = (
            frame.filter(pl.col("financial_year").is_not_null())
            .group_by("financial_year", maintain_order=True)
            .agg(
                pl.len().alias("works"),
                pl.col("is_completed").fill_null(False).cast(pl.Int64).sum().alias("completed_works"),
                pl.col("sanction_amount").sum().alias("sanctioned_amount"),
                pl.col("total_expenditure").sum().alias("total_expenditure"),
                pl.col("final_risk_score").mean().alias("mean_risk"),
                pl.col("final_risk_score").median().alias("median_risk"),
                pl.col("priority_score").mean().alias("mean_priority"),
                pl.col("risk_category").is_in(["HIGH", "CRITICAL"]).sum().alias("high_or_critical"),
            )
            .with_columns(
                pl.when(pl.col("works") > 0)
                .then(pl.col("completed_works") / pl.col("works") * 100)
                .otherwise(0)
                .alias("completion_rate_pct"),
                pl.when(pl.col("sanctioned_amount") > 0)
                .then(pl.col("total_expenditure") / pl.col("sanctioned_amount") * 100)
                .otherwise(None)
                .alias("utilization_pct"),
                pl.when(pl.col("works") > 0)
                .then(pl.col("high_or_critical") / pl.col("works") * 100)
                .otherwise(0)
                .alias("high_or_critical_rate_pct"),
            )
            .sort("financial_year")
        )

    scatter_columns = [
        "work_uid",
        "state",
        "work_category",
        "risk_category",
        "sanction_amount",
        "total_expenditure",
        "utilization_pct",
        "final_risk_score",
        "priority_score",
        "confidence_score",
        "ml_anomaly_percentile",
        "days_open_since_sanction",
    ]
    # Cap scatter lower on serverless to stay under response + time budgets
    scatter_sample = _deep_scatter_sample(frame, scatter_columns, limit=1500)

    return {
        "scope": _summary(frame),
        "risk_final": risk_final,
        "risk_reasons": _deep_flag_summary(frame),
        "state": _deep_json_records(
            state_group.sort("high_or_critical_rate_pct", descending=True, nulls_last=True)
            if not state_group.is_empty()
            else state_group
        ),
        "district": _deep_json_records(
            district_group.sort("high_or_critical_rate_pct", descending=True, nulls_last=True)
            if not district_group.is_empty()
            else district_group
        ),
        "sector": _deep_json_records(
            sector_group.sort("high_or_critical_rate_pct", descending=True, nulls_last=True)
            if not sector_group.is_empty()
            else sector_group
        ),
        "time": _deep_json_records(time_frame),
        "sanction_time": _deep_json_records(time_frame),
        "risk_financial": _deep_risk_financial(frame),
        "financial_bands": _deep_financial_band_stats(frame),
        "scatter_sample": scatter_sample,
        "distributions": {
            "risk_score": _deep_distribution(
                frame,
                "final_risk_score",
                [(f"{i}–{i+10}", float(i), None if i == 90 else float(i + 10)) for i in range(0, 100, 10)],
            ),
            "utilization": _deep_distribution(
                frame,
                "utilization_pct",
                [
                    ("<0%", float("-inf"), 0.0),
                    ("0–25%", 0.0, 25.0),
                    ("25–50%", 25.0, 50.0),
                    ("50–75%", 50.0, 75.0),
                    ("75–100%", 75.0, 100.0),
                    ("100–150%", 100.0, 150.0),
                    ("150%+", 150.0, None),
                ],
            ),
        },
        "profiles": {
            "final_risk_score": _deep_profile(frame, "final_risk_score"),
            "utilization_pct": _deep_profile(frame, "utilization_pct"),
            "priority_score": _deep_profile(frame, "priority_score"),
            "sanction_amount": _deep_profile(frame, "sanction_amount"),
        },
        "meta": {
            "analytics_scope_rows": frame.height,
            "scatter_sample_rows": len(scatter_sample),
            "scatter_sample_cap": 1500,
            "queue_cap": 500,
            "risk_band_definition": RISK_RANGES,
            "platform": "vercel",
            "note": (
                "Aggregations describe the active filtered population. "
                "Scatter uses a deterministic seeded sample for serverless performance."
            ),
        },
    }


# Vercel looks for `app` at supported entrypoints.
# Local: uvicorn api.index:app --host 0.0.0.0 --port 8000
