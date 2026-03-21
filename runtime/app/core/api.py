from __future__ import annotations
from pathlib import Path
import json
import logging
import os
import sys
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.core.predictor import MultiImagePredictor
from app.core.schemas import PredictRequest


def setup_logging() -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"pill_runtime_{ts}.log"

    logger = logging.getLogger("pill_runtime")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.propagate = False
    logger.info("Log file: %s", str(log_path))
    return logger


LOGGER = setup_logging()
app = FastAPI(title="Pill FullRes Runtime API")
PREDICTOR = None


def _load_runtime_cfg():
    cfg_path = os.environ.get("PILL_RUNTIME_CONFIG", "").strip()
    if not cfg_path:
        return {}
    p = Path(cfg_path)
    if not p.exists():
        raise FileNotFoundError(f"PILL_RUNTIME_CONFIG not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


@app.on_event("startup")
def startup_event():
    global PREDICTOR
    bundle_dir = os.environ.get("PILL_MODEL_BUNDLE", "models/runtime_bundle")
    runtime_cfg = _load_runtime_cfg()
    PREDICTOR = MultiImagePredictor(bundle_dir=bundle_dir, runtime_cfg=runtime_cfg)
    LOGGER.info(
        "[PACK] global=%s tile=%s num_tiles=%s min_images=%s max_images=%s classes=%s",
        runtime_cfg.get("global_size", 1024),
        runtime_cfg.get("tile_size", 512),
        runtime_cfg.get("num_tiles", 4),
        runtime_cfg.get("min_images", 2),
        runtime_cfg.get("max_images", 8),
        len(PREDICTOR.adapter.labels),
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    rid = req.request_id or str(uuid.uuid4())
    try:
        result = PREDICTOR.predict(rid, req.image_paths)
        return JSONResponse(result)
    except Exception as e:
        LOGGER.exception("predict failed")
        return JSONResponse(
            {
                "request_id": rid,
                "status": "ERROR",
                "class_name": None,
                "confidence_score": 0.0,
                "reason": str(e),
                "reject_code": "INTERNAL_EXCEPTION",
                "debug": {},
            },
            status_code=200,
        )
