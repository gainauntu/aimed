# Pill Production System - exact code-side architecture package

This package is the rebuilt code-side implementation of your proposed architecture.
It is meant to finish **everything from my side**:

- production runtime with your two-image interface style
- exact reject-first pipeline stages
- Kaggle dual-T4 training/orchestration code
- export/runtime pack wiring
- architecture coverage report with no hidden simplifications

Read these first:
- `ARCHITECTURE_COVERAGE.md`
- `trainer/README.md`
- `runtime_service/README.md`

What still requires the real dataset run:
- trained artifacts
- prototype library
- OOD index
- class profiles
- calibrator outputs
- final runtime bundle generated from those outputs

## Main entry points

Runtime server:
- `runtime_service/scripts/RUN_SERVER.bat`
- `runtime_service/scripts/run_server.sh`

Kaggle orchestrator:
- `trainer/notebook_launcher.py`
- `notebooks/kaggle_t4x2_train_all.ipynb`

Exact API request shape:
```json
{
  "request_id": "REQ-001",
  "image_paths": ["C:\\a.png", "C:\\b.png"],
  "machine_id": "DISP-01"
}
```
