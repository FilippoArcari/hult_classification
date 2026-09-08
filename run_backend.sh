#!/usr/bin/env bash
python -m venv .venv && source .venv/bin/activate
pip install -r serving/requirements.txt
uvicorn serving.main:app --reload --port 8000 --reload-dir serving