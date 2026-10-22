#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec uv run main.py step=run_on_kaggle "$@"
