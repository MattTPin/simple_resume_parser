# entrypoint.sh
#!/usr/bin/env bash
set -euo pipefail

API_AUTO_START="${API_AUTO_START:-true}"
HOST="0.0.0.0"
START_PORT=8000

# Function to find a free port
find_free_port() {
    local port=$START_PORT
    while lsof -i :"$port" >/dev/null 2>&1; do
        port=$((port + 1))
    done
    echo "$port"
}

if [[ "${API_AUTO_START}" == "true" ]]; then
    PORT=$(find_free_port)
    echo "[entrypoint] Starting FastAPI server on ${HOST}:${PORT} ..."

    # Start Uvicorn in the background
    uvicorn api.server:app --host "$HOST" --port "$PORT" --reload &
    PID=$!

    # Wait until server responds
    until curl -s "http://localhost:${PORT}/docs" > /dev/null; do
        sleep 1.0
    done

    # Server is live, print message
    echo "================================================================="
    echo "🚀 API is running at: http://localhost:${PORT}"
    echo "   Swagger docs:     http://localhost:${PORT}/docs"
    echo "================================================================="

    # Keep container alive while Uvicorn is running
    wait $PID
else
    echo "[entrypoint] API_AUTO_START=false — container is idle for manual start."
    tail -f /dev/null
fi
