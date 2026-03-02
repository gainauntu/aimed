from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class OCRResult:
    text: str
    conf: float
    quality: float
    raw: dict

def _quality_score(gray: np.ndarray) -> float:
    import cv2
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(max(0.0, min(1.0, lap / 500.0)))

class OCRProvider:
    def __init__(self, lang: str = "korean", use_gpu: bool = False):
        self._impl = None
        try:
            from paddleocr import PaddleOCR
            self._impl = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu)
        except Exception:
            self._impl = None
    def available(self) -> bool:
        return self._impl is not None
    def run(self, img_bgr: np.ndarray) -> OCRResult:
        import cv2
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        q = _quality_score(gray)
        if self._impl is None:
            return OCRResult(text="", conf=0.0, quality=q, raw={"provider":"none"})
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        out = self._impl.ocr(rgb, cls=True)
        best_text=""; best_conf=0.0
        try:
            for block in out:
                for line in block:
                    txt=line[1][0]; conf=float(line[1][1])
                    if conf>best_conf:
                        best_conf=conf; best_text=txt
        except Exception:
            pass
        return OCRResult(text=best_text, conf=best_conf, quality=q, raw={"provider":"paddleocr"})
