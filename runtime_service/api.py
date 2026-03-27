from __future__ import annotations

from pathlib import Path
import logging
import os
import sys
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from runtime_service.pack import load_pack, sync_pack_into_root, validate_pack
from runtime_service.predictor import RuntimePredictor
from runtime_service.schemas import PredictRequest


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
app = FastAPI(title="Pill Runtime API")
PREDICTOR: RuntimePredictor | None = None


@app.on_event("startup")
def startup_event():
    global PREDICTOR
    bundle_dir = os.environ.get("PILL_MODEL_BUNDLE", "runtime_service/models/current")
    pack = load_pack(bundle_dir)
    ok, errs = validate_pack(pack)
    if not ok:
        raise RuntimeError("Invalid runtime pack: " + "; ".join(errs))
    repo_root = Path(__file__).resolve().parents[1]
    config_path = sync_pack_into_root(pack, repo_root)
    PREDICTOR = RuntimePredictor(config_path)
    LOGGER.info("[PACK] bundle=%s classes=%s", bundle_dir, len(pack.labels))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    rid = req.request_id or str(uuid.uuid4())
    if PREDICTOR is None:
        return JSONResponse({
            "request_id": rid,
            "status": "ERROR",
            "class_name": None,
            "confidence_score": 0.0,
            "reason": "predictor not initialized",
            "reject_code": "NOT_READY",
        })
    return JSONResponse(PREDICTOR.predict(rid, req.image_paths, machine_id=req.machine_id, return_debug=False))


@app.post("/predict_debug")
def predict_debug(req: PredictRequest):
    rid = req.request_id or str(uuid.uuid4())
    if PREDICTOR is None:
        return JSONResponse({
            "request_id": rid,
            "status": "ERROR",
            "class_name": None,
            "confidence_score": 0.0,
            "reason": "predictor not initialized",
            "reject_code": "NOT_READY",
            "debug": {},
        })
    return JSONResponse(PREDICTOR.predict(rid, req.image_paths, machine_id=req.machine_id, return_debug=True))
