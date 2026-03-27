from __future__ import annotations

import os
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from app.api.schemas import HealthResponse, PredictResponse
from app.core.bootstrap import build_pipeline
from app.core.logging import configure_logging

pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    configure_logging()
    config_path = os.environ.get("PILL_APP_CONFIG", "configs/default.yaml")
    pipeline = build_pipeline(config_path)
    yield


app = FastAPI(title="Pill Dispensing Service", lifespan=lifespan)


def _decode_image(content: bytes) -> np.ndarray:
    arr = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image payload")
    return image


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse)
async def predict(image_a: UploadFile = File(...), image_b: UploadFile = File(...)) -> PredictResponse:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    frame_a = _decode_image(await image_a.read())
    frame_b = _decode_image(await image_b.read())
    result = pipeline.predict(frame_a, frame_b)
    return PredictResponse(
        status=result.status.value,
        predicted_class=result.predicted_class,
        calibrated_p=result.calibrated_p,
        failure_gate=result.failure_gate.value if result.failure_gate else None,
        failure_reason=result.failure_reason,
        elapsed_ms=result.elapsed_ms,
        details=result.to_dict(),
    )
