# entrypoint.sh
#!/usr/bin/env bash
set -euo pipefail

API_AUTO_START="${API_AUTO_START:-true}"

if [[ "${API_AUTO_START}" == "true" ]]; then
    echo "[entrypoint] Starting FastAPI server on 0.0.0.0:8000"
    echo "================================================================="
    echo "🚀 API is running at: http://localhost:8000"
    echo "   Swagger docs:     http://localhost:8000/docs"
    echo "================================================================="
    # exec env uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
    # Supress warnings (torch / numpy compatibility)
    exec env PYTHONWARNINGS="ignore" uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload 2>/dev/null
else
    echo "[entrypoint] API_AUTO_START=false — container is idle for manual start."
    tail -f /dev/null
fi
