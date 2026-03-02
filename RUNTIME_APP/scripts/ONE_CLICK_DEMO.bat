@echo off
cd /d %~dp0\..
if not exist .venv ( python -m venv .venv )
call .\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
start "PillAI Server" cmd /k python -m uvicorn app.core.api:app --host 127.0.0.1 --port 9000
ping 127.0.0.1 -n 3 >nul
start "PillAI GUI" cmd /k python -m app.main
