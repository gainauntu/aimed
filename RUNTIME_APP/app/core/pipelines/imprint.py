from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2

@dataclass
class ImprintROI:
    roi_bgr: np.ndarray
    bbox: tuple

def enhance_for_ocr(roi_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    top = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, kernel)
    sharp = cv2.addWeighted(g, 1.0, top, 1.2, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

def propose_imprint_roi(roi_bgr: np.ndarray, mask: np.ndarray) -> ImprintROI:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mag = cv2.bitwise_and(mag, mag, mask=mask)
    h,w = mag.shape
    win = max(40, min(h,w)//3)
    ii = cv2.integral(mag)
    best=(0,0,win,win,-1)
    step = max(5, win//10)
    for y in range(0, max(1,h-win), step):
        y2=y+win
        for x in range(0, max(1,w-win), step):
            x2=x+win
            s = ii[y2,x2]-ii[y,x2]-ii[y2,x]+ii[y,x]
            if s>best[-1]:
                best=(x,y,win,win,int(s))
    x,y,ww,hh,_=best
    return ImprintROI(roi_bgr=roi_bgr[y:y+hh, x:x+ww], bbox=(x,y,ww,hh))
