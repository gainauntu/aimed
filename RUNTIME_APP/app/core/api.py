from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from app.core.predictor import UnifiedPredictor
from app.core.logger import init_logger

logger = init_logger("pill_runtime")
app = FastAPI(title="Pill AI Runtime", version="unified")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

predictor = UnifiedPredictor(prefer="auto")

class PredictRequest(BaseModel):
    request_id: str = Field(...)
    side_a_path: str = Field(...)
    side_b_path: str = Field(...)

class TopKItem(BaseModel):
    name: str
    confidence: float

class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    request_id: str
    status: str
    class_name: Optional[str] = None
    confidence_percent: Optional[float] = None
    top3: List[TopKItem] = []
    reason: str = ""
    ocr_text: str = ""
    ocr_conf: float = 0.0
    ocr_quality: float = 0.0
    verifier_score: float = 0.0

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    out = predictor.predict(req.side_a_path, req.side_b_path)
    logger.info(f"{req.request_id} {out.status} best={out.best_name} conf={out.conf_percent:.1f} margin={out.margin:.1f} ocr='{out.ocr_text}'")
    return PredictResponse(
        request_id=req.request_id,
        status=out.status,
        class_name=out.best_name if out.status=="OK" else None,
        confidence_percent=out.conf_percent if out.status=="OK" else None,
        top3=[TopKItem(name=n, confidence=c) for (n,c) in out.top3],
        reason=out.reason,
        ocr_text=out.ocr_text,
        ocr_conf=out.ocr_conf,
        ocr_quality=out.ocr_quality,
        verifier_score=out.verifier_score
    )
