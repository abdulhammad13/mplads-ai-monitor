from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def start_process(command: list[str]) -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("MPLADS_API_URL", "http://127.0.0.1:8000")
    env.setdefault("MPLADS_DASH_HOST", "127.0.0.1")
    env.setdefault("MPLADS_DASH_PORT", "8050")
    return subprocess.Popen(command, cwd=PROJECT_ROOT, env=env)


def main() -> int:
    print("=" * 76)
    print("MPLADS AI MONITOR — ONE-COMMAND LOCAL RUNNER")
    print("=" * 76)
    print("Starting FastAPI on http://127.0.0.1:8000 ...")
    backend = start_process(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ]
    )

    time.sleep(2.0)

    print("Starting Dash on http://127.0.0.1:8050 ...")
    dashboard = start_process([sys.executable, "dashboard/app.py"])
    print("=" * 76)
    print("FASTAPI : http://127.0.0.1:8000")
    print("DOCS    : http://127.0.0.1:8000/docs")
    print("DASH    : http://127.0.0.1:8050")
    print("=" * 76)
    print("Press Ctrl+C once to stop both services.")

    try:
        while True:
            backend_code = backend.poll()
            dashboard_code = dashboard.poll()
            if backend_code is not None:
                print(f"FastAPI exited with code {backend_code}.")
                if dashboard.poll() is None:
                    dashboard.terminate()
                return backend_code or 0
            if dashboard_code is not None:
                print(f"Dash exited with code {dashboard_code}.")
                if backend.poll() is None:
                    backend.terminate()
                return dashboard_code or 0
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping MPLADS AI Monitor services...")
        for process in (dashboard, backend):
            if process.poll() is None:
                try:
                    process.send_signal(signal.SIGINT)
                except Exception:
                    process.terminate()
        for process in (dashboard, backend):
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
        print("Services stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
