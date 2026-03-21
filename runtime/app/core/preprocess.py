from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class LocalizationResult:
    ok: bool
    bbox: tuple[int, int, int, int]
    crop_ratio: float


def read_image_unicode(path: str) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.uint8)
    if arr.size == 0:
        raise ValueError(f"failed to read bytes: {path}")
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"failed to decode image: {path}")
    return img


def compute_quality_score(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F).var()
    std = float(gray.std())
    h, w = gray.shape[:2]
    area_norm = min(1.0, (h * w) / float(352 * 288))
    score = 0.35 * min(100.0, lap / 4.0) + 0.45 * min(100.0, std * 2.0) + 20.0 * area_norm
    return float(max(0.0, min(100.0, score)))


def _best_component_bbox(mask: np.ndarray, image_area: int):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best = None
    best_score = -1.0
    for idx in range(1, num_labels):
        x, y, w, h, area = stats[idx]
        if area < 0.01 * image_area or area > 0.80 * image_area:
            continue
        aspect = max(w, h) / max(1.0, min(w, h))
        if aspect > 3.5:
            continue
        fill = float(area) / max(1.0, float(w * h))
        score = area * (1.0 + fill)
        if score > best_score:
            best_score = score
            best = (x, y, w, h)
    return best


def simple_localize(bgr: np.ndarray) -> LocalizationResult:
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, th1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    th1 = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel, iterations=1)
    th1 = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel, iterations=2)
    th2 = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel, iterations=1)
    th2 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel, iterations=2)

    image_area = h * w
    c1 = _best_component_bbox(th1, image_area)
    c2 = _best_component_bbox(th2, image_area)

    cand = c1
    if c2 is not None and (cand is None or (c2[2] * c2[3]) > (cand[2] * cand[3])):
        cand = c2

    if cand is None:
        return LocalizationResult(False, (0, 0, w, h), 1.0)

    x, y, bw, bh = cand
    pad_x = int(round(bw * 0.12))
    pad_y = int(round(bh * 0.12))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)
    crop_ratio = ((x2 - x1) * (y2 - y1)) / float(max(1, image_area))
    ok = 0.08 <= crop_ratio <= 0.90
    if not ok:
        return LocalizationResult(False, (0, 0, w, h), 1.0)

    return LocalizationResult(True, (x1, y1, x2, y2), float(crop_ratio))


def resize_with_pad(rgb: np.ndarray, size: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = min(size / max(1, w), size / max(1, h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(rgb, (nw, nh), interpolation=interp)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    x = (size - nw) // 2
    y = (size - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def to_model_input(bgr: np.ndarray, size: int) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = resize_with_pad(rgb, size)
    arr = rgb.astype(np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


def extract_tiles_and_boxes(bgr: np.ndarray, tile_size: int, num_tiles: int):
    h, w = bgr.shape[:2]
    loc = simple_localize(bgr)
    x1, y1, x2, y2 = loc.bbox
    roi_w = x2 - x1
    roi_h = y2 - y1

    centers = []
    centers.append((x1 + roi_w // 2, y1 + roi_h // 2))
    centers.append((x1 + roi_w // 3, y1 + roi_h // 3))
    centers.append((x1 + 2 * roi_w // 3, y1 + roi_h // 3))
    centers.append((x1 + roi_w // 3, y1 + 2 * roi_h // 3))
    centers.append((x1 + 2 * roi_w // 3, y1 + 2 * roi_h // 3))

    tiles = []
    boxes = []
    half = max(32, min(h, w) // 5)

    for cx, cy in centers[: max(num_tiles, 4)]:
        sx1 = max(0, cx - half)
        sy1 = max(0, cy - half)
        sx2 = min(w, cx + half)
        sy2 = min(h, cy + half)
        crop = bgr[sy1:sy2, sx1:sx2]
        if crop.size == 0:
            continue
        tiles.append(to_model_input(crop, tile_size))
        boxes.append((sx1, sy1, sx2, sy2))

    while len(tiles) < num_tiles:
        tiles.append(to_model_input(bgr, tile_size))
        boxes.append((0, 0, w, h))

    return np.stack(tiles[:num_tiles], axis=0), boxes[:num_tiles], loc
