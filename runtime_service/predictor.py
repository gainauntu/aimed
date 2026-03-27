from __future__ import annotations

from pathlib import Path
import sys
import cv2
import numpy as np

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.bootstrap import build_pipeline  # noqa: E402


def read_image_unicode(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to decode image: {path}")
    return image


class RuntimePredictor:
    def __init__(self, config_path: str | Path):
        self.pipeline = build_pipeline(config_path)

    def predict(self, request_id: str, image_paths: list[str], machine_id: str = "", return_debug: bool = False) -> dict:
        if len(image_paths) != 2:
            payload = {
                "request_id": request_id,
                "status": "ERROR",
                "class_name": None,
                "confidence_score": 0.0,
                "reason": "Exactly 2 images are required",
                "reject_code": "BAD_IMAGE_COUNT",
            }
            if return_debug:
                payload["debug"] = {}
            return payload

        a = read_image_unicode(image_paths[0])
        b = read_image_unicode(image_paths[1])
        result = self.pipeline.predict(a, b)

        status = "OK" if result.status.value == "CLASSIFIED" else "UNDECIDED"
        class_name = result.predicted_class if status == "OK" else None
        reject_code = result.failure_gate.value if result.failure_gate else None
        reason = result.failure_reason if result.failure_reason else "판정 기준 충족"
        confidence_score = float(result.calibrated_p * 100.0) if status == "OK" else 0.0

        payload = {
            "request_id": request_id,
            "status": status,
            "class_name": class_name,
            "confidence_score": round(confidence_score, 2),
            "reason": reason,
            "reject_code": reject_code,
        }
        if return_debug:
            payload["debug"] = result.to_dict()
            payload["debug"]["machine_id"] = machine_id
            payload["debug"]["image_paths"] = image_paths
        return payload
