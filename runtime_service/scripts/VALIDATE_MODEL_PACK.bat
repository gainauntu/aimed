@echo off
cd /d %~dp0\..\..
python -c "from runtime_service.pack import load_pack, validate_pack; p=load_pack('runtime_service/models/current'); ok,errs=validate_pack(p); print('OK' if ok else 'FAIL'); [print(e) for e in errs]"
pause
