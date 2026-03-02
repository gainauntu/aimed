from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import cv2

@dataclass
class VerifyResult:
    ok: bool
    score: float
    best_ref: str

def patch_match_score(query_bgr: np.ndarray, ref_bgr: np.ndarray) -> float:
    orb = cv2.ORB_create(nfeatures=800)
    qg = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2GRAY)
    rg = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    kq, dq = orb.detectAndCompute(qg, None)
    kr, dr = orb.detectAndCompute(rg, None)
    if dq is None or dr is None:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(dq, dr)
    if not matches:
        return 0.0
    matches = sorted(matches, key=lambda m: m.distance)[:100]
    good = [m for m in matches if m.distance < 50]
    return float(len(good) / max(1, len(matches)))

def verify_against_gallery(query_roi: np.ndarray, gallery: Dict[str, List[np.ndarray]], candidates: List[str]) -> VerifyResult:
    best_score=-1.0; best_ref=""
    for cls in candidates:
        for ref in gallery.get(cls, []):
            s = patch_match_score(query_roi, ref)
            if s>best_score:
                best_score=s; best_ref=cls
    return VerifyResult(ok=(best_score>=0.18), score=float(best_score), best_ref=best_ref)
