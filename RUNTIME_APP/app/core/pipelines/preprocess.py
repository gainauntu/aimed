from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import cv2

@dataclass
class PreprocessOutput:
    roi_bgr: np.ndarray
    mask: np.ndarray
    angle: float

def segment_pill(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]
    _, mask = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8), iterations=2)
    return mask

def crop_to_mask(img_bgr: np.ndarray, mask: np.ndarray, pad: int = 10) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return img_bgr
    x1,x2=xs.min(),xs.max(); y1,y2=ys.min(),ys.max()
    h,w = img_bgr.shape[:2]
    x1=max(0,x1-pad); y1=max(0,y1-pad); x2=min(w-1,x2+pad); y2=min(h-1,y2+pad)
    return img_bgr[y1:y2+1, x1:x2+1]

def estimate_angle(mask: np.ndarray) -> float:
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    angle = rect[-1]
    if angle < -45:
        angle += 90
    return float(angle)

def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h,w=img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def preprocess(img_bgr: np.ndarray) -> PreprocessOutput:
    mask = segment_pill(img_bgr)
    roi = crop_to_mask(img_bgr, mask)
    mask2 = segment_pill(roi)
    angle = estimate_angle(mask2)
    roi_rot = rotate(roi, angle)
    mask_rot = segment_pill(roi_rot)
    return PreprocessOutput(roi_bgr=roi_rot, mask=mask_rot, angle=angle)
