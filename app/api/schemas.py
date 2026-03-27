from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class PredictResponse(BaseModel):
    status: str
    predicted_class: str | None
    calibrated_p: float
    failure_gate: str | None
    failure_reason: str | None
    elapsed_ms: float
    details: dict
