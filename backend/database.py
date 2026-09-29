"""
MPLADS AI Monitor — Database layer (backend-agnostic)

Priority order:
1. DATABASE_URL          → Postgres
2. PG_* environment vars → Postgres
3. Nothing set           → local SQLite at data/processed/mplads.db
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import Session, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SQLITE_PATH = PROCESSED_DIR / "mplads.db"

try:
    from dotenv import load_dotenv
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass


def _build_database_url() -> tuple[str, str]:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if url:
        if url.startswith("postgres://"):
            url = "postgresql://" + url[len("postgres://"):]
        dialect = "postgresql" if url.startswith("postgresql") else "sqlite"
        return url, dialect

    pg_host = (os.getenv("PG_HOST") or os.getenv("POSTGRES_HOST") or "").strip()
    if pg_host:
        pg_port = (os.getenv("PG_PORT") or os.getenv("POSTGRES_PORT") or "5432").strip()
        pg_user = (os.getenv("PG_USER") or os.getenv("POSTGRES_USER") or "postgres").strip()
        pg_pass = (os.getenv("PG_PASSWORD") or os.getenv("POSTGRES_PASSWORD") or "").strip()
        pg_db   = (os.getenv("PG_DATABASE") or os.getenv("POSTGRES_DB") or "mplads").strip()
        url = f"postgresql+psycopg2://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"
        return url, "postgresql"

    return f"sqlite:///{SQLITE_PATH}", "sqlite"


DATABASE_URL, DB_DIALECT = _build_database_url()


def _create_engine() -> Engine:
    if DB_DIALECT == "sqlite":
        eng = create_engine(
            DATABASE_URL,
            connect_args={"check_same_thread": False, "timeout": 30},
            pool_pre_ping=True,
        )

        @event.listens_for(eng, "connect")
        def _sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.close()

        return eng

    return create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
        pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
        pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "1800")),
    )


engine: Engine = _create_engine()

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


def is_postgres() -> bool:
    return DB_DIALECT == "postgresql"


def supports_percentile() -> bool:
    return is_postgres()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_connection() -> Connection:
    return engine.connect()


def test_connection() -> bool:
    try:
        with engine.connect() as conn:
            val = conn.execute(text("SELECT 1")).scalar_one()
            if val != 1:
                raise ConnectionError("Health-check returned unexpected value")
        print(f"Database connected successfully ({DB_DIALECT}).")
        return True
    except Exception as exc:
        raise ConnectionError(
            f"Unable to connect to database.\n"
            f"Dialect : {DB_DIALECT}\n"
            f"Error   : {exc}"
        ) from exc


def db_health() -> dict:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {
            "status": "ok",
            "dialect": DB_DIALECT,
            "supports_percentile": supports_percentile(),
            "url_hint": DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else str(SQLITE_PATH.name),
        }
    except Exception as exc:
        return {
            "status": "error",
            "dialect": DB_DIALECT,
            "error": str(exc),
        }


def dispose_engine() -> None:
    engine.dispose()


if __name__ == "__main__":
    test_connection()
    print(db_health())