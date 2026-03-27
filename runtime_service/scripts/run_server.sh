#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PILL_MODEL_BUNDLE="${PILL_MODEL_BUNDLE:-runtime_service/models/current}"
python -m uvicorn runtime_service.api:app --host 127.0.0.1 --port 9000
