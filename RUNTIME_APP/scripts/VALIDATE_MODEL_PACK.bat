@echo off
setlocal
cd /d %~dp0\..
call .\.venv\Scripts\activate.bat
python -c "from app.core.pack import load_pack, validate_pack; p=load_pack(); ok,errs=validate_pack(p); print('OK' if ok else 'FAIL'); [print('-',e) for e in errs]"
pause
