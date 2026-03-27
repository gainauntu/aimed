@echo off
cd /d %~dp0\..\..
set PILL_MODEL_BUNDLE=runtime_service\models\current
python -m uvicorn runtime_service.api:app --host 127.0.0.1 --port 9000
