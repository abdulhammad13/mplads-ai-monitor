#!/usr/bin/env bash
set -e

export MPLADS_API_URL="http://127.0.0.1:8000"
export MPLADS_DASH_HOST="0.0.0.0"
export MPLADS_DASH_PORT="${PORT:-8050}"

echo "Starting FastAPI on 127.0.0.1:8000 ..."
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!

sleep 5

echo "Starting Dash on ${MPLADS_DASH_HOST}:${MPLADS_DASH_PORT} ..."
python dashboard/app.py &
DASH_PID=$!

wait -n $API_PID $DASH_PID
EXIT_CODE=$?
kill $API_PID $DASH_PID 2>/dev/null || true
exit $EXIT_CODE
