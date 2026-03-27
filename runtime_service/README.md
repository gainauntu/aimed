# Runtime Service

Windows-style deployment wrapper for the production pill pipeline.

## API

### POST /predict
```json
{
  "request_id": "REQ-001",
  "image_paths": ["C:\\pill_a.png", "C:\\pill_b.png"],
  "machine_id": "DISP-01"
}
```

### POST /predict_debug
Same payload, but returns the full debug object from the production decision engine.

## Bundle location

By default the service looks for:
- `runtime_service/models/current`

You can override with:
- `PILL_MODEL_BUNDLE=/path/to/runtime_bundle`

## Run

### Windows
- `scripts\RUN_SERVER.bat`

### Linux
```bash
bash scripts/run_server.sh
```

## Validate pack

### Windows
- `scripts\VALIDATE_MODEL_PACK.bat`

### Linux
```bash
bash scripts/validate_model_pack.sh
```
