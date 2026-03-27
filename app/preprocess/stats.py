from __future__ import annotations

import math

import cv2
import numpy as np

from app.domain.models import PhysicalStats, ShapeClass


def _circular_mean_deg(hues: np.ndarray) -> float:
    radians = hues * (2 * math.pi / 180.0)
    mean_angle = math.atan2(np.sin(radians).mean(), np.cos(radians).mean())
    deg = mean_angle * 180.0 / (2 * math.pi)
    return deg % 180.0


def _circular_std_deg(hues: np.ndarray) -> float:
    radians = hues * (2 * math.pi / 180.0)
    r = np.sqrt(np.sin(radians).mean() ** 2 + np.cos(radians).mean() ** 2)
    if r <= 1e-8:
        return 90.0
    return math.sqrt(max(0.0, -2.0 * math.log(r))) * 180.0 / (2 * math.pi)


def classify_shape(aspect_ratio: float) -> ShapeClass:
    if aspect_ratio < 1.15:
        return ShapeClass.ROUND
    if aspect_ratio < 1.80:
        return ShapeClass.OVAL
    if aspect_ratio < 2.50:
        return ShapeClass.OBLONG
    return ShapeClass.CAPSULE


def extract_physical_stats(crop_bgr: np.ndarray, mask: np.ndarray) -> PhysicalStats:
    if crop_bgr.shape[:2] != mask.shape[:2]:
        raise ValueError("crop and mask must share HxW")

    foreground = mask > 0
    if foreground.sum() == 0:
        raise ValueError("mask has no foreground")

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    fg_h = hsv[..., 0][foreground].astype(np.float32)
    fg_s = hsv[..., 1][foreground].astype(np.float32)
    fg_v = hsv[..., 2][foreground].astype(np.float32)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = max(w, h) / max(1.0, min(w, h))
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    convexity_ratio = area / max(hull_area, 1.0)

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edge_density = float(edges[foreground].sum()) / max(float(foreground.sum()), 1.0)

    return PhysicalStats(
        hue_mean=float(_circular_mean_deg(fg_h)),
        hue_std=float(_circular_std_deg(fg_h)),
        sat_mean=float(fg_s.mean()),
        sat_std=float(fg_s.std()),
        val_mean=float(fg_v.mean()),
        val_std=float(fg_v.std()),
        pixel_area_proxy=float(foreground.sum()),
        aspect_ratio=float(aspect_ratio),
        convexity_ratio=float(convexity_ratio),
        edge_density=float(edge_density),
        shape_class=classify_shape(float(aspect_ratio)),
    )
