@echo off
setlocal

REM Ensure dependencies are installed and run via uv
uv sync
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
