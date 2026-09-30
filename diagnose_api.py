from __future__ import annotations

import sys
import time
from typing import Any

import requests

BASE = "http://127.0.0.1:8000"
TIMEOUT = 120


def show(path: str, params: dict[str, Any] | None = None) -> None:
    print(f"\n{path}")
    started = time.perf_counter()
    try:
        response = requests.get(f"{BASE}{path}", params=params or {}, timeout=TIMEOUT)
        elapsed = time.perf_counter() - started
        print(f"  HTTP {response.status_code}")
        print(f"  elapsed={elapsed:.2f}s")
        print(f"  bytes={len(response.content):,}")
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                payload = response.json()
                if response.status_code == 200:
                    text = repr(payload)
                    print(f"  payload={text[:1200]}")
                else:
                    print(f"  body={repr(payload)[:2000]}")
            except Exception:
                print(f"  body={response.text[:2000]}")
        else:
            print(f"  body={response.text[:2000]}")
    except requests.RequestException as exc:
        elapsed = time.perf_counter() - started
        print(f"  ERROR after {elapsed:.2f}s: {exc}")


def main() -> int:
    print("MPLADS AI MONITOR — API DIAGNOSTIC")
    print(f"Base: {BASE}")
    show("/health")
    show("/api/v1/deep-analytics-status")
    show("/api/v1/filter-options")
    show("/api/v1/dashboard-summary")
    show("/api/v1/filtered-summary", {"min_risk": 0, "max_risk": 100})
    show(
        "/api/v1/works",
        {
            "page": 1,
            "page_size": 1,
            "sort_by": "priority_score",
            "sort_order": "desc",
            "min_risk": 0,
            "max_risk": 100,
        },
    )
    show("/api/v1/deep-analytics", {"min_risk": 0, "max_risk": 100})
    print("\nDiagnostic complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
