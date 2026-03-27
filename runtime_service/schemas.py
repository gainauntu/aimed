from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    request_id: str = ""
    image_paths: List[str] = Field(default_factory=list, min_length=2, max_length=2)
    machine_id: str = ""


class PredictResponse(BaseModel):
    request_id: str
    status: str
    class_name: Optional[str]
    confidence_score: float
    reason: str
    reject_code: Optional[str] = None
    debug: Dict[str, Any] = Field(default_factory=dict)
