from __future__ import annotations

import math
import os
import re
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import text

from .database import engine, db_health, is_postgres, supports_percentile

# ============================================================
# MPLADS AI MONITOR — FASTAPI DATA/API LAYER
# ============================================================
# Architecture:
#   source CSV / SQL -> Polars -> FastAPI -> Dash/Plotly
#
# This module intentionally does NOT recompute feature engineering,
# ML anomaly detection, or final risk methodology. Those remain in
# the existing backend pipeline. The API exposes the resulting
# analytical dataset consistently to the dashboard.
#
# Signals are analytical review signals only. They do not establish
# fraud, corruption, legal non-compliance, misappropriation or causality.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "final_risk_data.csv"

SERVICE_VERSION = "6.4-DeepAnalytics"
DATA_SOURCE_MODE = os.getenv("MPLADS_DATA_SOURCE", "csv").strip().lower()
SQL_TABLE = os.getenv("MPLADS_SQL_TABLE", "dbo.works").strip()

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

SORT_COLUMNS = {
    "priority_score",
    "final_risk_score",
    "sanction_amount",
    "total_expenditure",
    "confidence_score",
    "days_open_since_sanction",
}

RISK_ORDER = ("LOW", "MEDIUM", "HIGH", "CRITICAL")

# Canonical score bands exposed by the deep-analytics API.
# Keep these boundaries aligned with _risk_band_expr().
RISK_RANGES = {
    "LOW": {"min": 0.0, "max": 25.0},
    "MEDIUM": {"min": 25.0, "max": 50.0},
    "HIGH": {"min": 50.0, "max": 75.0},
    "CRITICAL": {"min": 75.0, "max": 100.0},
}

# Environment-controlled SQL identifiers are validated before interpolation.
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_sql_table(value: str) -> str:
    parts = [part.strip() for part in value.split(".") if part.strip()]
    if not parts or len(parts) > 2 or any(not _IDENTIFIER.fullmatch(p) for p in parts):
        raise ValueError(
            "MPLADS_SQL_TABLE must be a table identifier such as 'dbo.works'."
        )
    return ".".join(f"[{part}]" for part in parts)


def _empty_series(dtype: pl.DataType = pl.String) -> pl.Series:
    return pl.Series([], dtype=dtype)


def _ensure_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Create only optional runtime columns needed by API calculations."""
    expressions: list[pl.Expr] = []

    numeric_defaults = {
        "sanction_amount": pl.Float64,
        "recommended_amount": pl.Float64,
        "completed_amount": pl.Float64,
        "total_expenditure": pl.Float64,
        "allocated_amount": pl.Float64,
        "utilization_pct": pl.Float64,
        "final_risk_score": pl.Float64,
        "priority_score": pl.Float64,
        "ml_anomaly_percentile": pl.Float64,
        "rule_risk_score": pl.Float64,
        "financial_risk_score": pl.Float64,
        "execution_risk_score": pl.Float64,
        "duplicate_risk_score": pl.Float64,
        "data_integrity_risk_score": pl.Float64,
        "confidence_score": pl.Float64,
        "priority_rank": pl.Int64,
        "days_open_since_sanction": pl.Float64,
        "days_rec_to_sanction": pl.Float64,
        "days_sanction_to_complete": pl.Float64,
        "independent_signal_count": pl.Int64,
        "duplicate_group_count": pl.Int64,
    }
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
    """Normalize analytical data once at the API boundary."""
    frame = frame.clone()

    # Standardize column names without changing business semantics.
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

    # Replace non-finite numeric values with null.
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


def _read_csv() -> pl.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing final analytical dataset: {CSV_PATH}")

    frame = pl.read_csv(
        CSV_PATH,
        infer_schema_length=10000,
        try_parse_dates=False,
        null_values=["", "NULL", "null", "NA", "N/A", "NaN"],
    )
    return _normalize_frame(frame)


def _read_sql() -> pl.DataFrame:
    table = _validate_sql_table(SQL_TABLE)
    query = text(f"SELECT * FROM {table}")

    # SQLAlchemy remains the connection layer. Polars owns the dataframe.
    with engine.connect() as connection:
        result = connection.execute(query)
        rows = result.mappings().all()
        if not rows:
            return _normalize_frame(pl.DataFrame())
        frame = pl.DataFrame([dict(row) for row in rows])

    return _normalize_frame(frame)


def _has_required_schema(frame: pl.DataFrame) -> tuple[bool, list[str]]:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    return not missing, missing


def load_data_once(force_refresh: bool = False) -> pl.DataFrame:
    global WORKS, WORKS_SOURCE

    if WORKS is not None and not force_refresh:
        return WORKS

    if DATA_SOURCE_MODE not in {"auto", "csv", "sql"}:
        raise RuntimeError("MPLADS_DATA_SOURCE must be one of: auto, csv, sql")

    candidates = ("csv", "sql") if DATA_SOURCE_MODE == "auto" else (DATA_SOURCE_MODE,)
    errors: list[str] = []

    for source in candidates:
        try:
            frame = _read_csv() if source == "csv" else _read_sql()
            valid, missing = _has_required_schema(frame)
            if not valid:
                errors.append(f"{source}: analytical schema missing {missing}")
                continue

            WORKS = frame
            WORKS_SOURCE = source
            print(f"Loaded {frame.height:,} works from {source}.")
            return WORKS
        except Exception as error:
            errors.append(f"{source}: {type(error).__name__}: {error}")

    raise RuntimeError("No valid analytical data source could be loaded.\n" + "\n".join(errors))


# ============================================================
# JSON / SERIALIZATION
# ============================================================


def _json_value(value: Any) -> Any:
    """Convert Polars/Python/NumPy values into strict JSON primitives."""
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


def _first_value(frame: pl.DataFrame, column: str) -> Any:
    if column not in frame.columns or frame.height == 0:
        return None
    return frame.get_column(column)[0]


def _sum(frame: pl.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    value = frame.select(pl.col(column).sum()).item()
    return _json_value(value)


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


# ============================================================
# FILTERING
# ============================================================


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
            column
            for column in (
                "work",
                "work_description",
                "mp",
                "constituency",
                "ida",
                "work_uid",
            )
            if column in result.columns
        ]
        if search_columns:
            predicates.append(
                pl.any_horizontal(
                    [
                        pl.col(column)
                        .cast(pl.String, strict=False)
                        .fill_null("")
                        .str.to_lowercase()
                        .str.contains(term, literal=True)
                        for column in search_columns
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


# ============================================================
# SUMMARY / ANALYTICS HELPERS
# ============================================================


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
        high = frame.select(
            pl.col("risk_category").is_in(["HIGH", "CRITICAL"]).sum()
        ).item() or 0
        critical = frame.select(
            pl.col("risk_category").eq("CRITICAL").sum()
        ).item() or 0

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
        "high_or_critical": int(high),
        "critical": int(critical),
        "high_or_critical_rate_pct": _safe_rate(high, total),
        "duplicate_candidates": _count_true(frame, "is_duplicate_candidate"),
        "overdue_open_works": _count_true(frame, "flag_overdue_open_work"),
        "multi_signal_cases": int(
            frame.select(
                pl.col("independent_signal_count").fill_null(0).ge(2).sum()
            ).item()
            or 0
        ),
    }


def _risk_band_expr(column: str) -> pl.Expr:
    return (
        pl.when(pl.col(column).is_null()).then(None)
        .when(pl.col(column).le(25)).then(pl.lit("LOW"))
        .when(pl.col(column).le(50)).then(pl.lit("MEDIUM"))
        .when(pl.col(column).le(75)).then(pl.lit("HIGH"))
        .otherwise(pl.lit("CRITICAL"))
        .alias("risk_category")
    )


def _group_risk_stats(frame: pl.DataFrame, group_column: str) -> pl.DataFrame:
    return (
        frame.group_by(group_column, maintain_order=True)
        .agg(
            pl.len().alias("works"),
            pl.col("is_completed").fill_null(False).cast(pl.Int64).sum().alias("completed_works"),
            pl.col("sanction_amount").sum().alias("sanctioned_amount"),
            pl.col("total_expenditure").sum().alias("total_expenditure"),
            pl.col("final_risk_score").median().alias("median_risk"),
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


# ============================================================
# APPLICATION
# ============================================================


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_data_once()
    yield


app = FastAPI(
    title="MPLADS AI Monitoring Platform",
    description=(
        "AI-assisted monitoring of MPLADS works using transparent risk signals, "
        "peer-relative anomaly detection, financial/execution indicators, "
        "duplicate candidates and explainable investigation prioritization."
    ),
    version=SERVICE_VERSION,
    lifespan=lifespan,
)


# ============================================================
# ROOT / HEALTH / REFRESH
# ============================================================


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
        "deep_analytics_status": "/api/v1/deep-analytics-status",
        "filter_options": "/api/v1/filter-options",
        "dashboard_summary": "/api/v1/dashboard-summary",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    frame = load_data_once()
    as_of = None
    if "monitoring_as_of_date" in frame.columns:
        value = frame.select(pl.col("monitoring_as_of_date").max()).item()
        as_of = _json_value(value)

    db_info = db_health()

    return {
        "status": "ok",
        "service": "MPLADS AI Monitoring API",
        "version": SERVICE_VERSION,
        "works_loaded": frame.height,
        "data_source": WORKS_SOURCE or DATA_SOURCE_MODE,
        "data_as_of": as_of,
        "database": db_info,
        "supports_percentile": supports_percentile(),
        "is_postgres": is_postgres(),
        "risk_band_definition": RISK_RANGES,
        "guardrail": (
            "An anomaly is a review signal; a similarity candidate is not a confirmed "
            "duplicate; data-integrity issues are data-quality signals; timing benchmarks "
            "require rule-specific verification; the system does not establish fraud, "
            "corruption, legal non-compliance, misappropriation or causality."
        ),
    }


@app.post("/api/v1/refresh")
def refresh_data() -> dict[str, Any]:
    frame = load_data_once(force_refresh=True)
    return {
        "status": "ok",
        "works_loaded": frame.height,
        "data_source": WORKS_SOURCE,
    }


# ============================================================
# FILTER OPTIONS
# ============================================================


@app.get("/api/v1/filter-options")
def filter_options() -> dict[str, list[str]]:
    """Return synchronized, deterministic filter values from WORKS."""
    frame = load_data_once()

    def options(column: str) -> list[str]:
        if column not in frame.columns:
            return []

        values = (
            frame
            .select(
                pl.col(column)
                .cast(pl.String, strict=False)
                .str.strip_chars()
                .alias("_value")
            )
            .filter(
                pl.col("_value").is_not_null()
                & pl.col("_value").ne("")
            )
            .unique()
            .sort("_value")
            .get_column("_value")
            .to_list()
        )
        return [str(value) for value in values if value is not None]

    return {
        "states": options("state"),
        "districts": options("ida"),
        "mps": options("mp"),
        "constituencies": options("constituency"),
        "work_categories": options("work_category"),
        "work_statuses": options("work_status"),
        "risk_categories": [
            value
            for value in RISK_ORDER
            if value in options("risk_category")
        ],
        "financial_years": options("financial_year"),
    }


# ============================================================
# OVERVIEW / RISK
# ============================================================


@app.get("/api/v1/dashboard-summary")
def dashboard_summary() -> dict[str, Any]:
    frame = load_data_once()
    result = _summary(frame)

    if "risk_category" in frame.columns:
        distribution = (
            frame.group_by("risk_category")
            .agg(pl.len().alias("count"))
            .with_columns(
                pl.when(pl.lit(frame.height) > 0)
                .then(pl.col("count") / frame.height * 100)
                .otherwise(0)
                .alias("pct")
            )
            .sort("risk_category")
        )
        result["risk_distribution"] = to_records(distribution)

    return result


@app.get("/api/v1/filtered-summary")
def filtered_summary(
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
    return _summary(frame)


@app.get("/api/v1/risk-distribution")
def risk_distribution(
    risk_basis: str = Query(default="final", pattern="^(final|rule|ml)$"),
) -> list[dict[str, Any]]:
    frame = load_data_once()
    column = {
        "final": "final_risk_score",
        "rule": "rule_risk_score",
        "ml": "ml_anomaly_percentile",
    }[risk_basis]

    if column not in frame.columns:
        return []

    result = (
        frame
        .with_columns(_risk_band_expr(column))
        .filter(pl.col("risk_category").is_not_null())
        .group_by("risk_category")
        .agg(pl.len().alias("count"))
        .with_columns(
            pl.when(pl.lit(frame.height) > 0)
            .then(pl.col("count") / frame.height * 100)
            .otherwise(0)
            .alias("pct")
        )
    )

    return to_records(result)


@app.get("/api/v1/risk-reasons")
def risk_reasons() -> list[dict[str, Any]]:
    frame = load_data_once()
    definitions = (
        ("flag_disbursement_over_sanction", "Expenditure exceeds sanction"),
        ("flag_overdue_open_work", "Open beyond general one-year benchmark"),
        ("flag_stalled_expenditure", "Long expenditure gap"),
        ("flag_duplicate_candidate", "Potential duplicate/similar work"),
        ("flag_cost_outlier", "Cost outlier within category"),
        ("flag_duration_outlier", "Duration outlier within category"),
        ("flag_bad_dates", "Date integrity issue"),
    )

    rows = []
    for column, label in definitions:
        if column not in frame.columns:
            continue
        count = _count_true(frame, column)
        rows.append({
            "reason": label,
            "count": count,
            "pct_of_all_works": _safe_rate(count, frame.height),
        })
    return rows


@app.get("/api/v1/filtered-analytics")
def filtered_analytics(
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
    """
    Return ALL dashboard analytics from the same filtered dataframe.

    This endpoint is intentionally separate from /api/v1/works because the
    work explorer is capped/paginated, while charts must describe the complete
    active scope. This prevents a 500-row queue from being mistaken for the
    entire filtered population.
    """
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

    def risk_dist(column: str) -> list[dict[str, Any]]:
        if column not in frame.columns:
            return []
        result = (
            frame.with_columns(_risk_band_expr(column))
            .filter(pl.col("risk_category").is_not_null())
            .group_by("risk_category")
            .agg(pl.len().alias("count"))
            .with_columns(
                pl.when(pl.lit(frame.height) > 0)
                .then(pl.col("count") / frame.height * 100)
                .otherwise(0)
                .alias("pct")
            )
        )
        return to_records(result)

    reason_defs = (
        ("flag_disbursement_over_sanction", "Expenditure exceeds sanction"),
        ("flag_overdue_open_work", "Open beyond general one-year benchmark"),
        ("flag_stalled_expenditure", "Long expenditure gap"),
        ("flag_duplicate_candidate", "Potential duplicate/similar work"),
        ("flag_cost_outlier", "Cost outlier within category"),
        ("flag_duration_outlier", "Duration outlier within category"),
        ("flag_bad_dates", "Date integrity issue"),
    )
    reasons=[]
    for column,label in reason_defs:
        if column in frame.columns:
            count=_count_true(frame,column)
            reasons.append({
                "reason": label,
                "count": count,
                "pct_of_filtered_works": _safe_rate(count, frame.height),
            })

    def group_stats(column: str) -> list[dict[str, Any]]:
        if column not in frame.columns:
            return []
        return to_records(_group_risk_stats(frame,column).sort("high_or_critical_rate_pct",descending=True,nulls_last=True))

    sector=[]
    if "work_category" in frame.columns:
        sector_frame=(
            frame.group_by("work_category",maintain_order=True)
            .agg(
                pl.len().alias("works"),
                pl.col("is_completed").fill_null(False).cast(pl.Int64).sum().alias("completed_works"),
                pl.col("sanction_amount").sum().alias("sanctioned_amount"),
                pl.col("total_expenditure").sum().alias("total_expenditure"),
                pl.col("final_risk_score").median().alias("median_risk"),
                pl.col("sanction_amount").median().alias("median_cost"),
                pl.col("risk_category").is_in(["HIGH","CRITICAL"]).sum().alias("high_or_critical"),
            )
            .with_columns(
                pl.when(pl.col("sanctioned_amount")>0).then(pl.col("total_expenditure")/pl.col("sanctioned_amount")*100).otherwise(None).alias("utilization_pct"),
                pl.when(pl.col("works")>0).then(pl.col("completed_works")/pl.col("works")*100).otherwise(0).alias("completion_rate_pct"),
                pl.when(pl.col("works")>0).then(pl.col("high_or_critical")/pl.col("works")*100).otherwise(0).alias("high_or_critical_rate_pct"),
            )
            .sort("high_or_critical_rate_pct",descending=True,nulls_last=True)
        )
        sector=to_records(sector_frame)

    time=[]
    if "financial_year" in frame.columns:
        time_frame=(
            frame.filter(pl.col("financial_year").is_not_null())
            .group_by("financial_year",maintain_order=True)
            .agg(
                pl.len().alias("works"),
                pl.col("is_completed").fill_null(False).cast(pl.Int64).sum().alias("completed_works"),
                pl.col("sanction_amount").sum().alias("sanctioned_amount"),
                pl.col("total_expenditure").sum().alias("total_expenditure"),
                pl.col("final_risk_score").median().alias("median_risk"),
                pl.col("risk_category").is_in(["HIGH","CRITICAL"]).sum().alias("high_or_critical"),
                pl.col("risk_category").eq("CRITICAL").sum().alias("critical"),
            )
            .with_columns(
                pl.when(pl.col("sanctioned_amount")>0).then(pl.col("total_expenditure")/pl.col("sanctioned_amount")*100).otherwise(None).alias("utilization_pct"),
                pl.when(pl.col("works")>0).then(pl.col("completed_works")/pl.col("works")*100).otherwise(0).alias("completion_rate_pct"),
                pl.when(pl.col("works")>0).then(pl.col("high_or_critical")/pl.col("works")*100).otherwise(0).alias("high_or_critical_rate_pct"),
                pl.when(pl.col("works")>0).then(pl.col("critical")/pl.col("works")*100).otherwise(0).alias("critical_rate_pct"),
            )
        )
        time=to_records(time_frame.sort("financial_year"))

    return {
        "scope": _summary(frame),
        "risk_final": risk_dist("final_risk_score"),
        "risk_rule": risk_dist("rule_risk_score"),
        "risk_ml": risk_dist("ml_anomaly_percentile"),
        "risk_reasons": reasons,
        "state": group_stats("state"),
        "district": group_stats("ida"),
        "sector": sector,
        "time": time,
    }


# ============================================================
# WORK EXPLORER
# ============================================================


@app.get("/api/v1/works")
def works(
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
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
    sort_by: str = Query(default="priority_score"),
    sort_order: str = Query(default="desc", pattern="^(asc|desc)$"),
) -> dict[str, Any]:
    if sort_by not in SORT_COLUMNS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported sort_by. Allowed values: {sorted(SORT_COLUMNS)}",
        )
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

    total = frame.height
    descending = sort_order == "desc"
    if sort_by in frame.columns:
        frame = frame.sort(sort_by, descending=descending, nulls_last=True)

    start = (page - 1) * page_size
    page_frame = frame.slice(start, page_size)

    preferred = (
        "work_uid",
        "work",
        "work_description",
        "state",
        "ida",
        "mp",
        "constituency",
        "work_category",
        "work_status",
        "financial_year",
        "sanction_amount",
        "total_expenditure",
        "utilization_pct",
        "is_completed",
        "days_open_since_sanction",
        "final_risk_score",
        "risk_category",
        "priority_score",
        "priority_rank",
        "confidence_score",
        "primary_risk_reason",
        "risk_explanation",
    )
    selected = [column for column in preferred if column in page_frame.columns]

    return {
        "items": to_records(page_frame.select(selected)),
        "page": page,
        "page_size": page_size,
        "total_matching": total,
        "total_pages": math.ceil(total / page_size) if total else 0,
    }


@app.get("/api/v1/works/{work_uid}")
def work_detail(work_uid: str) -> dict[str, Any]:
    frame = load_data_once()
    if "work_uid" not in frame.columns:
        raise HTTPException(status_code=500, detail="work_uid is missing from analytical dataset")

    match = frame.filter(pl.col("work_uid").cast(pl.String, strict=False) == str(work_uid))
    if match.is_empty():
        raise HTTPException(status_code=404, detail="Work not found")

    row = match.row(0, named=True)
    component_columns = (
        "ml_anomaly_percentile",
        "rule_risk_score",
        "financial_risk_score",
        "execution_risk_score",
        "duplicate_risk_score",
        "data_integrity_risk_score",
        "final_risk_score",
        "confidence_score",
        "priority_score",
        "priority_rank",
    )

    return {
        "work": {key: _json_value(value) for key, value in row.items()},
        "risk_components": {
            key: _json_value(row[key])
            for key in component_columns
            if key in row
        },
        "evidence": row.get("evidence_json"),
    }


@app.get("/api/v1/works/{work_uid}/similar")
def work_similar(
    work_uid: str,
    limit: int = Query(default=10, ge=1, le=50),
) -> list[dict[str, Any]]:
    frame = load_data_once()
    if "work_uid" not in frame.columns:
        raise HTTPException(status_code=500, detail="work_uid is missing")

    match = frame.filter(pl.col("work_uid").cast(pl.String, strict=False) == str(work_uid))
    if match.is_empty():
        raise HTTPException(status_code=404, detail="Work not found")

    row = match.row(0, named=True)
    candidates = frame.filter(pl.col("work_uid").cast(pl.String, strict=False) != str(work_uid))

    # Conservative exact-context matching only. No fabricated semantic score.
    keys = ["state", "ida", "constituency", "work_category"]
    predicates = []
    for column in keys:
        value = row.get(column)
        if column in candidates.columns and value not in (None, ""):
            predicates.append(
                pl.col(column).cast(pl.String, strict=False) == str(value)
            )

    if predicates:
        subset = candidates.filter(pl.all_horizontal(predicates))
    else:
        subset = candidates.head(0)

    columns = [
        column
        for column in (
            "work_uid",
            "work",
            "work_description",
            "state",
            "ida",
            "constituency",
            "work_category",
            "sanction_amount",
            "final_risk_score",
            "priority_score",
        )
        if column in subset.columns
    ]

    if "priority_score" in subset.columns:
        subset = subset.sort("priority_score", descending=True, nulls_last=True)
    return to_records(subset.head(limit).select(columns))


# ============================================================
# AGGREGATED ANALYTICS
# ============================================================


@app.get("/api/v1/state-analytics")
def state_analytics() -> list[dict[str, Any]]:
    frame = load_data_once()
    if "state" not in frame.columns:
        return []

    result = _group_risk_stats(frame, "state")
    return to_records(result.sort("high_or_critical_rate_pct", descending=True, nulls_last=True))


@app.get("/api/v1/district-analytics")
def district_analytics(state: str | None = None) -> list[dict[str, Any]]:
    frame = load_data_once()
    if state and state != "All" and "state" in frame.columns:
        frame = frame.filter(_eq_filter("state", state))

    if "ida" not in frame.columns:
        return []

    result = _group_risk_stats(frame, "ida")
    return to_records(result.sort("high_or_critical_rate_pct", descending=True, nulls_last=True))


@app.get("/api/v1/sector-analytics")
def sector_analytics() -> list[dict[str, Any]]:
    frame = load_data_once()
    if "work_category" not in frame.columns:
        return []

    result = (
        frame.group_by("work_category", maintain_order=True)
        .agg(
            pl.len().alias("works"),
            pl.col("is_completed").fill_null(False).cast(pl.Int64).sum().alias("completed_works"),
            pl.col("sanction_amount").sum().alias("sanctioned_amount"),
            pl.col("total_expenditure").sum().alias("total_expenditure"),
            pl.col("final_risk_score").median().alias("median_risk"),
            pl.col("sanction_amount").median().alias("median_cost"),
            pl.col("risk_category").is_in(["HIGH", "CRITICAL"]).sum().alias("high_or_critical"),
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
        )
        .sort("high_or_critical_rate_pct", descending=True, nulls_last=True)
    )
    return to_records(result)


@app.get("/api/v1/time-series")
def time_series() -> list[dict[str, Any]]:
    frame = load_data_once()
    if "financial_year" not in frame.columns:
        return []

    result = (
        frame.filter(pl.col("financial_year").is_not_null())
        .group_by("financial_year", maintain_order=True)
        .agg(
            pl.len().alias("works"),
            pl.col("is_completed").fill_null(False).cast(pl.Int64).sum().alias("completed_works"),
            pl.col("sanction_amount").sum().alias("sanctioned_amount"),
            pl.col("total_expenditure").sum().alias("total_expenditure"),
            pl.col("final_risk_score").median().alias("median_risk"),
            pl.col("risk_category").is_in(["HIGH", "CRITICAL"]).sum().alias("high_or_critical"),
            pl.col("risk_category").eq("CRITICAL").sum().alias("critical"),
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

    # Prefer chronological FY ordering when labels are YYYY-YY.
    return to_records(result.sort("financial_year"))


@app.get("/api/v1/geography-points")
def geography_points(
    risk_min: float = Query(default=0, ge=0, le=100),
) -> dict[str, Any]:
    frame = load_data_once()
    lat_col = next(
        (column for column in ("latitude", "lat", "work_latitude") if column in frame.columns),
        None,
    )
    lon_col = next(
        (column for column in ("longitude", "lon", "lng", "work_longitude") if column in frame.columns),
        None,
    )

    if not lat_col or not lon_col:
        return {
            "available": False,
            "reason": "No latitude/longitude columns in analytical dataset",
            "points": [],
        }

    subset = (
        frame
        .filter(pl.col("final_risk_score").ge(risk_min))
        .with_columns(
            pl.col(lat_col).cast(pl.Float64, strict=False).alias("lat"),
            pl.col(lon_col).cast(pl.Float64, strict=False).alias("lon"),
        )
        .filter(
            pl.col("lat").is_not_null()
            & pl.col("lon").is_not_null()
            & pl.col("lat").is_between(-90, 90, closed="both")
            & pl.col("lon").is_between(-180, 180, closed="both")
        )
        .head(5000)
    )

    preferred = [
        "work_uid",
        "work",
        "state",
        "ida",
        "constituency",
        "work_category",
        "sanction_amount",
        "final_risk_score",
        "risk_category",
        "priority_score",
        "lat",
        "lon",
    ]
    selected = [column for column in preferred if column in subset.columns]
    return {"available": True, "points": to_records(subset.select(selected))}


# ============================================================
# DEEP ANALYTICS EXTENSION — FILTER-SYNCHRONISED CONTROL CENTER
# ============================================================
# Purpose:
#   Provide a single authoritative analytics payload for the Dash
#   frontend. All tables below are calculated from the COMPLETE
#   active filtered population, not the 500-row investigation queue.
#
# Design principles:
#   1. Never infer a fraud/adjudication conclusion from an anomaly.
#   2. Keep denominator definitions explicit in the response fields.
#   3. Prefer rates + counts + monetary exposure together.
#   4. Cap only expensive scatter samples; aggregate charts are exact.
#   5. Keep every result synchronized with the active dashboard filter.
# ============================================================


def safe_int(value: Any, default: int = 0) -> int:
    """Safely coerce a scalar to a JSON-safe int for API aggregation rows."""
    try:
        if value is None:
            return default
        if isinstance(value, float) and not math.isfinite(value):
            return default
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely coerce a scalar to a finite float for API aggregation rows."""
    try:
        if value is None:
            return default
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _deep_safe_rate(numerator: Any, denominator: Any) -> float:
    try:
        n = float(numerator or 0)
        d = float(denominator or 0)
        if not math.isfinite(n) or not math.isfinite(d) or d <= 0:
            return 0.0
        return round(n / d * 100.0, 3)
    except (TypeError, ValueError):
        return 0.0


def _deep_json_records(frame: pl.DataFrame) -> list[dict[str, Any]]:
    if frame.is_empty():
        return []
    return to_records(frame)


def _deep_numeric_values(frame: pl.DataFrame, column: str) -> list[float]:
    if frame.is_empty() or column not in frame.columns:
        return []
    values = []
    for value in frame.get_column(column).cast(pl.Float64, strict=False).to_list():
        try:
            number = float(value)
            if math.isfinite(number):
                values.append(number)
        except (TypeError, ValueError):
            continue
    return values


def _deep_profile(frame: pl.DataFrame, column: str) -> dict[str, Any]:
    values = _deep_numeric_values(frame, column)
    if not values:
        return {
            "column": column,
            "count": 0,
            "min": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
            "mean": None,
        }
    values_sorted = sorted(values)

    def percentile(q: float) -> float:
        if len(values_sorted) == 1:
            return round(values_sorted[0], 4)
        position = (len(values_sorted) - 1) * q
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return round(values_sorted[lower], 4)
        weight = position - lower
        return round(
            values_sorted[lower] * (1.0 - weight)
            + values_sorted[upper] * weight,
            4,
        )

    return {
        "column": column,
        "count": len(values_sorted),
        "min": round(values_sorted[0], 4),
        "p10": percentile(0.10),
        "p25": percentile(0.25),
        "median": percentile(0.50),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "max": round(values_sorted[-1], 4),
        "mean": round(sum(values_sorted) / len(values_sorted), 4),
    }


def _deep_distribution(
    frame: pl.DataFrame,
    column: str,
    buckets: list[tuple[str, float, float | None]],
) -> list[dict[str, Any]]:
    """Compute exact bucket counts with vectorised Polars expressions.

    The previous implementation converted the full filtered column into a
    Python list and then compared every value against every bucket. That is
    needlessly expensive for a 36,000-row monitoring universe and can make
    the dashboard client hit its HTTP timeout even though FastAPI itself is
    healthy. This implementation keeps the same half-open bucket semantics
    but performs the work inside Polars.
    """
    if frame.is_empty() or column not in frame.columns:
        return [
            {"bucket": label, "count": 0, "share_pct": 0.0}
            for label, _, _ in buckets
        ]

    numeric = (
        pl.col(column)
        .cast(pl.Float64, strict=False)
        .alias("_deep_numeric")
    )
    classified = frame.select(numeric).filter(
        pl.col("_deep_numeric").is_not_null()
        & pl.col("_deep_numeric").is_finite()
    )
    total = classified.height

    if total == 0:
        return [
            {"bucket": label, "count": 0, "share_pct": 0.0}
            for label, _, _ in buckets
        ]

    expressions: list[pl.Expr] = []
    for label, lower, upper in buckets:
        condition = pl.col("_deep_numeric") >= lower
        if upper is not None:
            condition = condition & (pl.col("_deep_numeric") < upper)
        expressions.append(
            pl.when(condition)
            .then(pl.lit(label))
            .otherwise(None)
            .alias("_bucket")
        )

    # The bucket definitions are intentionally ordered and non-overlapping.
    # Assign the first matching bucket, then count once per bucket.
    bucket_expr = expressions[0]
    for expression in expressions[1:]:
        bucket_expr = pl.coalesce([bucket_expr, expression])

    grouped = (
        classified
        .with_columns(bucket_expr)
        .filter(pl.col("_bucket").is_not_null())
        .group_by("_bucket", maintain_order=True)
        .agg(pl.len().alias("count"))
    )
    counts = {str(row["_bucket"]): int(row["count"]) for row in grouped.to_dicts()}

    return [
        {
            "bucket": label,
            "count": counts.get(label, 0),
            "share_pct": _deep_safe_rate(counts.get(label, 0), total),
        }
        for label, _, _ in buckets
    ]


def _deep_flag_summary(frame: pl.DataFrame) -> list[dict[str, Any]]:
    definitions = (
        ("flag_disbursement_over_sanction", "Expenditure exceeds sanction", "financial"),
        ("flag_overdue_open_work", "Open beyond one-year benchmark", "timing"),
        ("flag_stalled_expenditure", "Long expenditure gap", "execution"),
        ("flag_duplicate_candidate", "Potential duplicate / similar work", "similarity"),
        ("flag_cost_outlier", "Cost outlier within category", "financial"),
        ("flag_duration_outlier", "Duration outlier within category", "execution"),
        ("flag_bad_dates", "Date integrity issue", "data_quality"),
    )
    rows = []
    for column, label, family in definitions:
        count = _count_true(frame, column) if column in frame.columns else 0
        rows.append(
            {
                "signal": column,
                "label": label,
                "family": family,
                "count": count,
                "rate_pct": _deep_safe_rate(count, frame.height),
            }
        )
    return rows


def _deep_group_stats(frame: pl.DataFrame, group_column: str) -> pl.DataFrame:
    if frame.is_empty() or group_column not in frame.columns:
        return pl.DataFrame()

    result = (
        frame.group_by(group_column, maintain_order=True)
        .agg(
            pl.len().alias("works"),
            pl.col("is_completed").fill_null(False).cast(pl.Int64).sum().alias("completed_works"),
            pl.col("sanction_amount").sum().alias("sanctioned_amount"),
            pl.col("total_expenditure").sum().alias("total_expenditure"),
            pl.col("final_risk_score").mean().alias("mean_risk"),
            pl.col("final_risk_score").median().alias("median_risk"),
            pl.col("priority_score").mean().alias("mean_priority"),
            pl.col("confidence_score").mean().alias("mean_confidence"),
            pl.col("ml_anomaly_percentile").mean().alias("mean_ml_anomaly"),
            pl.col("sanction_amount").median().alias("median_sanction"),
            pl.col("total_expenditure").median().alias("median_expenditure"),
            pl.col("risk_category").is_in(["HIGH", "CRITICAL"]).sum().alias("high_or_critical"),
            pl.col("risk_category").eq("CRITICAL").sum().alias("critical"),
            pl.col("flag_overdue_open_work").fill_null(False).cast(pl.Int64).sum().alias("overdue_open_works"),
            pl.col("flag_stalled_expenditure").fill_null(False).cast(pl.Int64).sum().alias("stalled_works"),
            pl.col("flag_duplicate_candidate").fill_null(False).cast(pl.Int64).sum().alias("duplicate_candidates"),
            pl.col("flag_cost_outlier").fill_null(False).cast(pl.Int64).sum().alias("cost_outliers"),
            pl.col("flag_duration_outlier").fill_null(False).cast(pl.Int64).sum().alias("duration_outliers"),
            pl.col("flag_bad_dates").fill_null(False).cast(pl.Int64).sum().alias("bad_date_issues"),
            pl.col("independent_signal_count").mean().alias("mean_independent_signals"),
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
            pl.when(pl.col("works") > 0)
            .then(pl.col("overdue_open_works") / pl.col("works") * 100)
            .otherwise(0)
            .alias("overdue_rate_pct"),
            pl.when(pl.col("works") > 0)
            .then(pl.col("stalled_works") / pl.col("works") * 100)
            .otherwise(0)
            .alias("stalled_rate_pct"),
            pl.when(pl.col("works") > 0)
            .then(pl.col("duplicate_candidates") / pl.col("works") * 100)
            .otherwise(0)
            .alias("duplicate_rate_pct"),
            pl.when(pl.col("works") > 0)
            .then(pl.col("bad_date_issues") / pl.col("works") * 100)
            .otherwise(0)
            .alias("bad_date_rate_pct"),
        )
    )
    return result


def _deep_risk_financial(frame: pl.DataFrame) -> list[dict[str, Any]]:
    if frame.is_empty() or "risk_category" not in frame.columns:
        return []
    result = (
        frame.group_by("risk_category", maintain_order=True)
        .agg(
            pl.len().alias("works"),
            pl.col("sanction_amount").sum().alias("sanctioned_amount"),
            pl.col("total_expenditure").sum().alias("total_expenditure"),
            pl.col("priority_score").sum().alias("priority_total"),
            pl.col("final_risk_score").mean().alias("mean_risk"),
            pl.col("is_completed").fill_null(False).cast(pl.Int64).sum().alias("completed_works"),
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
        )
    )
    rows = result.to_dicts()
    order = {value: index for index, value in enumerate(RISK_ORDER)}
    rows.sort(key=lambda r: order.get(str(r.get("risk_category")), 99))
    total_sanction = sum(safe_float(row.get("sanctioned_amount")) for row in rows)
    for row in rows:
        row["sanction_share_pct"] = _deep_safe_rate(row.get("sanctioned_amount"), total_sanction)
    return rows


def _deep_signal_cooccurrence(frame: pl.DataFrame) -> list[dict[str, Any]]:
    signal_columns = [
        ("Over sanction", "flag_disbursement_over_sanction"),
        ("Overdue", "flag_overdue_open_work"),
        ("Stalled", "flag_stalled_expenditure"),
        ("Duplicate", "flag_duplicate_candidate"),
        ("Cost outlier", "flag_cost_outlier"),
        ("Duration outlier", "flag_duration_outlier"),
        ("Bad dates", "flag_bad_dates"),
    ]
    available = [(label, column) for label, column in signal_columns if column in frame.columns]
    rows = []
    for i, (label_a, column_a) in enumerate(available):
        for j, (label_b, column_b) in enumerate(available):
            if j < i:
                continue
            both = (
                frame.select(
                    (pl.col(column_a).fill_null(False) & pl.col(column_b).fill_null(False)).sum()
                ).item()
                or 0
            )
            rows.append(
                {
                    "signal_a": label_a,
                    "signal_b": label_b,
                    "count": int(both),
                    "rate_pct": _deep_safe_rate(both, frame.height),
                }
            )
    return rows


def _deep_time_frame(frame: pl.DataFrame, date_column: str) -> pl.DataFrame:
    if frame.is_empty() or date_column not in frame.columns:
        return pl.DataFrame()
    temp = frame.filter(pl.col(date_column).is_not_null()).with_columns(
        pl.col(date_column).dt.year().cast(pl.Int64, strict=False).alias("financial_year")
    )
    if temp.is_empty():
        return temp
    result = (
        temp.group_by("financial_year", maintain_order=True)
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
            pl.col("flag_duplicate_candidate").fill_null(False).cast(pl.Int64).sum().alias("duplicate_candidates"),
        )
        .with_columns(
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
            pl.when(pl.col("sanctioned_amount") > 0)
            .then(pl.col("total_expenditure") / pl.col("sanctioned_amount") * 100)
            .otherwise(None)
            .alias("utilization_pct"),
            pl.when(pl.col("works") > 0)
            .then(pl.col("overdue_open_works") / pl.col("works") * 100)
            .otherwise(0)
            .alias("overdue_rate_pct"),
            pl.when(pl.col("works") > 0)
            .then(pl.col("duplicate_candidates") / pl.col("works") * 100)
            .otherwise(0)
            .alias("duplicate_rate_pct"),
        )
        .sort("financial_year")
    )
    return result


def _deep_category_risk_long(frame: pl.DataFrame, limit: int = 20) -> list[dict[str, Any]]:
    if frame.is_empty() or "work_category" not in frame.columns or "risk_category" not in frame.columns:
        return []
    top_categories = (
        frame.group_by("work_category")
        .agg(pl.len().alias("works"))
        .sort("works", descending=True)
        .head(limit)
        .get_column("work_category")
        .to_list()
    )
    subset = frame.filter(pl.col("work_category").is_in(top_categories))
    result = (
        subset.group_by(["work_category", "risk_category"], maintain_order=True)
        .agg(pl.len().alias("count"))
        .sort(["work_category", "risk_category"])
    )
    return to_records(result)


def _deep_status_risk_long(frame: pl.DataFrame) -> list[dict[str, Any]]:
    if frame.is_empty() or "work_status" not in frame.columns or "risk_category" not in frame.columns:
        return []
    result = (
        frame.group_by(["work_status", "risk_category"], maintain_order=True)
        .agg(pl.len().alias("count"))
        .sort(["work_status", "risk_category"])
    )
    return to_records(result)


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
                (pl.col("sanction_amount") >= low)
                & (pl.col("sanction_amount") < high)
            )
        sanctioned = _sum(subset, "sanction_amount") or 0.0
        expenditure = _sum(subset, "total_expenditure") or 0.0
        works = subset.height
        completed = _count_true(subset, "is_completed")
        highcritical = (
            subset.select(pl.col("risk_category").is_in(["HIGH", "CRITICAL"]).sum()).item()
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


def _deep_risk_signal_matrix(frame: pl.DataFrame) -> list[dict[str, Any]]:
    # Long-format state x signal table; exact counts, filter-synchronised.
    if frame.is_empty() or "state" not in frame.columns:
        return []
    signal_defs = (
        ("Over sanction", "flag_disbursement_over_sanction"),
        ("Overdue", "flag_overdue_open_work"),
        ("Stalled", "flag_stalled_expenditure"),
        ("Duplicate", "flag_duplicate_candidate"),
        ("Cost outlier", "flag_cost_outlier"),
        ("Duration outlier", "flag_duration_outlier"),
        ("Bad dates", "flag_bad_dates"),
    )
    aggregations = [pl.len().alias("works")]
    for _, column in signal_defs:
        if column in frame.columns:
            aggregations.append(
                pl.col(column).fill_null(False).cast(pl.Int64).sum().alias(column)
            )
    grouped = frame.group_by("state", maintain_order=True).agg(aggregations)
    rows = []
    for row in grouped.to_dicts():
        state = str(row.get("state") or "Unknown")
        works = safe_int(row.get("works"))
        for label, column in signal_defs:
            count = safe_int(row.get(column))
            rows.append(
                {
                    "state": state,
                    "signal": label,
                    "count": count,
                    "rate_pct": _deep_safe_rate(count, works),
                }
            )
    return rows


def _deep_exposure_by_group(frame: pl.DataFrame, group_column: str, limit: int = 15) -> list[dict[str, Any]]:
    if frame.is_empty() or group_column not in frame.columns:
        return []
    subset = frame.filter(pl.col("risk_category").is_in(["HIGH", "CRITICAL"]))
    if subset.is_empty():
        return []
    result = (
        subset.group_by(group_column, maintain_order=True)
        .agg(
            pl.len().alias("high_critical_works"),
            pl.col("sanction_amount").sum().alias("high_critical_sanction"),
            pl.col("total_expenditure").sum().alias("high_critical_expenditure"),
            pl.col("final_risk_score").mean().alias("mean_risk"),
        )
        .sort("high_critical_sanction", descending=True, nulls_last=True)
        .head(limit)
    )
    return to_records(result)


def _deep_scatter_sample(frame: pl.DataFrame, columns: list[str], limit: int = 3500) -> list[dict[str, Any]]:
    available = [column for column in columns if column in frame.columns]
    if frame.is_empty() or not available:
        return []
    sample = frame.select(available)
    if sample.height > limit:
        # Seeded sampling makes repeated browser refreshes stable and therefore
        # easier to compare visually during a demo.
        sample = sample.sample(n=limit, with_replacement=False, seed=42)
    return to_records(sample)


@app.get("/api/v1/deep-analytics-status")
def deep_analytics_status() -> dict[str, Any]:
    """Lightweight readiness probe for the deep analytics layer.

    This endpoint intentionally performs only the data/schema checks needed to
    establish that the analytical dataset can support the deep analytics
    contract. It does not execute the complete chart payload.
    """
    frame = load_data_once()
    expected_optional = [
        "risk_category",
        "work_category",
        "state",
        "ida",
        "sanction_amount",
        "total_expenditure",
        "final_risk_score",
        "priority_score",
        "ml_anomaly_percentile",
        "independent_signal_count",
        "is_duplicate_candidate",
        "flag_overdue_open_work",
    ]
    missing_optional = [column for column in expected_optional if column not in frame.columns]
    db_info = db_health()
    return {
        "status": "ready" if not missing_optional else "degraded",
        "works_loaded": frame.height,
        "data_source": WORKS_SOURCE or DATA_SOURCE_MODE,
        "missing_columns": missing_optional,
        "deep_analytics_route": "/api/v1/deep-analytics",
        "service_version": SERVICE_VERSION,
        "database": db_info,
        "supports_percentile": supports_percentile(),
        "risk_band_definition": RISK_RANGES,
        "guardrail": (
            "Anomaly / similarity / timing / data-quality signals are for human "
            "review and verification only. The system does not adjudicate fraud."
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
    """Deep, filter-synchronised analytics for the Dash control center."""
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
        lookup = {}
        grouped = frame.group_by("risk_category").agg(pl.len().alias("count"))
        for row in grouped.to_dicts():
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

    def _deep_risk_distribution(column: str) -> list[dict[str, Any]]:
        if column not in frame.columns:
            return []
        temp = frame.select(pl.col(column).alias("score"))
        result = (
            temp.with_columns(_risk_band_expr("score"))
            .filter(pl.col("risk_category").is_not_null())
            .group_by("risk_category")
            .agg(pl.len().alias("count"))
        )
        counts = {str(row.get("risk_category")): safe_int(row.get("count")) for row in result.to_dicts()}
        return [
            {
                "risk_category": band,
                "count": counts.get(band, 0),
                "pct": _deep_safe_rate(counts.get(band, 0), frame.height),
            }
            for band in RISK_ORDER
        ]

    state_group = _deep_group_stats(frame, "state")
    district_group = _deep_group_stats(frame, "ida")
    sector_group = _deep_group_stats(frame, "work_category")

    time_frame = _deep_time_frame(frame, "sanction_date")
    if time_frame.is_empty() and "financial_year" in frame.columns:
        # Fallback when dates are unparsed but financial_year is present in the CSV.
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
    recommendation_time_frame = _deep_time_frame(frame, "recommended_date")
    completion_time_frame = _deep_time_frame(frame, "completion_date")

    component_columns = [
        "final_risk_score",
        "rule_risk_score",
        "ml_anomaly_percentile",
        "financial_risk_score",
        "execution_risk_score",
        "duplicate_risk_score",
        "data_integrity_risk_score",
        "priority_score",
        "confidence_score",
    ]
    component_profiles = [_deep_profile(frame, column) for column in component_columns]

    distributions = {
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
                ("100–125%", 100.0, 125.0),
                ("125–150%", 125.0, 150.0),
                ("150–200%", 150.0, 200.0),
                ("200–300%", 200.0, 300.0),
                ("300%+", 300.0, None),
            ],
        ),
        "open_age": _deep_distribution(
            frame,
            "days_open_since_sanction",
            [("0–90d", 0, 91), ("91–180d", 91, 181), ("181–365d", 181, 366), ("366–730d", 366, 731), ("731–1095d", 731, 1096), ("1096–1825d", 1096, 1826), ("1826d+", 1826, None)],
        ),
        "expenditure_gap": _deep_distribution(
            frame,
            "days_since_last_expenditure",
            [("0–30d", 0, 31), ("31–90d", 31, 91), ("91–180d", 91, 181), ("181–365d", 181, 366), ("366–730d", 366, 731), ("731d+", 731, None)],
        ),
        "sanction_to_complete": _deep_distribution(
            frame,
            "days_sanction_to_complete",
            [("0–90d", 0, 91), ("91–180d", 91, 181), ("181–365d", 181, 366), ("366–730d", 366, 731), ("731–1095d", 731, 1096), ("1096d+", 1096, None)],
        ),
        "rec_to_sanction": _deep_distribution(
            frame,
            "days_rec_to_sanction",
            [("0–7d", 0, 8), ("8–30d", 8, 31), ("31–90d", 31, 91), ("91–180d", 91, 181), ("181d+", 181, None)],
        ),
        "confidence": _deep_distribution(
            frame,
            "confidence_score",
            [("0–50", 0, 50), ("50–60", 50, 60), ("60–70", 60, 70), ("70–80", 70, 80), ("80–90", 80, 90), ("90–100", 90, 100.0001)],
        ),
        "model_missing": _deep_distribution(
            frame,
            "model_input_missing_pct",
            [("0%", 0, 0.0001), (">0–10%", 0.0001, 10), ("10–25%", 10, 25), ("25–50%", 25, 50), ("50%+", 50, None)],
        ),
        "signal_count": _deep_distribution(
            frame,
            "signal_count",
            [(str(i), float(i), float(i + 1)) for i in range(0, 7)] + [("7+", 7.0, None)],
        ),
        "independent_signal_count": _deep_distribution(
            frame,
            "independent_signal_count",
            [(str(i), float(i), float(i + 1)) for i in range(0, 7)] + [("7+", 7.0, None)],
        ),
        "data_quality_issues": _deep_distribution(
            frame,
            "data_quality_issue_count",
            [("0", 0, 1), ("1", 1, 2), ("2", 2, 3), ("3", 3, 4), ("4+", 4, None)],
        ),
        "duplicate_group": _deep_distribution(
            frame,
            "duplicate_group_count",
            [("0", 0, 1), ("1", 1, 2), ("2", 2, 3), ("3", 3, 4), ("4+", 4, None)],
        ),
    }

    financial_bands = _deep_financial_band_stats(frame)
    risk_financial = _deep_risk_financial(frame)
    status_risk = _deep_status_risk_long(frame)
    category_risk = _deep_category_risk_long(frame, limit=20)
    state_signal = _deep_risk_signal_matrix(frame)
    signal_cooccurrence = _deep_signal_cooccurrence(frame)

    status_distribution = []
    if "work_status" in frame.columns:
        status_grouped = frame.group_by("work_status").agg(pl.len().alias("works"))
        for row in status_grouped.sort("works", descending=True).to_dicts():
            status_distribution.append(
                {
                    "work_status": str(row.get("work_status") or "Unknown"),
                    "works": safe_int(row.get("works")),
                    "share_pct": _deep_safe_rate(row.get("works"), frame.height),
                }
            )

    quality_distribution = []
    if "data_quality_status" in frame.columns:
        quality_grouped = frame.group_by("data_quality_status").agg(pl.len().alias("works"))
        for row in quality_grouped.sort("works", descending=True).to_dicts():
            quality_distribution.append(
                {
                    "data_quality_status": str(row.get("data_quality_status") or "Unknown"),
                    "works": safe_int(row.get("works")),
                    "share_pct": _deep_safe_rate(row.get("works"), frame.height),
                }
            )

    high_critical_state_exposure = _deep_exposure_by_group(frame, "state", limit=15)
    high_critical_category_exposure = _deep_exposure_by_group(frame, "work_category", limit=15)

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
        "financial_risk_score",
        "execution_risk_score",
        "duplicate_risk_score",
        "data_integrity_risk_score",
        "days_open_since_sanction",
        "days_since_last_expenditure",
        "financial_exposure_percentile",
        "cost_robust_z",
        "duration_robust_z",
    ]
    scatter_sample = _deep_scatter_sample(frame, scatter_columns, limit=3500)

    return {
        "scope": _summary(frame),
        "risk_final": risk_final,
        "risk_rule": _deep_risk_distribution("rule_risk_score"),
        "risk_ml": _deep_risk_distribution("ml_anomaly_percentile"),
        "risk_reasons": _deep_flag_summary(frame),
        "state": _deep_json_records(state_group.sort("high_or_critical_rate_pct", descending=True, nulls_last=True)),
        "district": _deep_json_records(district_group.sort("high_or_critical_rate_pct", descending=True, nulls_last=True)),
        "sector": _deep_json_records(sector_group.sort("high_or_critical_rate_pct", descending=True, nulls_last=True)),
        "time": _deep_json_records(time_frame),
        "sanction_time": _deep_json_records(time_frame),
        "recommendation_time": _deep_json_records(recommendation_time_frame),
        "completion_time": _deep_json_records(completion_time_frame),
        "component_profiles": component_profiles,
        "risk_financial": risk_financial,
        "financial_bands": financial_bands,
        "status_distribution": status_distribution,
        "status_risk": status_risk,
        "category_risk": category_risk,
        "state_signal": state_signal,
        "signal_cooccurrence": signal_cooccurrence,
        "quality_distribution": quality_distribution,
        "high_critical_state_exposure": high_critical_state_exposure,
        "high_critical_category_exposure": high_critical_category_exposure,
        "scatter_sample": scatter_sample,
        "distributions": distributions,
        "profiles": {
            "final_risk_score": _deep_profile(frame, "final_risk_score"),
            "utilization_pct": _deep_profile(frame, "utilization_pct"),
            "days_open_since_sanction": _deep_profile(frame, "days_open_since_sanction"),
            "days_since_last_expenditure": _deep_profile(frame, "days_since_last_expenditure"),
            "days_sanction_to_complete": _deep_profile(frame, "days_sanction_to_complete"),
            "days_rec_to_sanction": _deep_profile(frame, "days_rec_to_sanction"),
            "confidence_score": _deep_profile(frame, "confidence_score"),
            "ml_anomaly_percentile": _deep_profile(frame, "ml_anomaly_percentile"),
            "priority_score": _deep_profile(frame, "priority_score"),
            "sanction_amount": _deep_profile(frame, "sanction_amount"),
            "total_expenditure": _deep_profile(frame, "total_expenditure"),
        },
        "meta": {
            "analytics_scope_rows": frame.height,
            "scatter_sample_rows": len(scatter_sample),
            "scatter_sample_cap": 3500,
            "queue_cap": 500,
            "risk_band_definition": RISK_RANGES,
            "note": "Aggregations describe the active filtered population. Scatter charts may use a deterministic seeded sample for browser performance.",
        },
    }

if __name__ == "__main__":
    # Direct execution is intentionally simple; use Uvicorn for the server.
    print("MPLADS AI MONITOR — FastAPI")
    print("API: http://127.0.0.1:8000")
    print("Docs: http://127.0.0.1:8000/docs")
