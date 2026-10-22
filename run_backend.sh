#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec uv run uvicorn serving.main:app --reload --port 8000 --reload-dir serving
