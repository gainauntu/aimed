from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import math
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class Sample:
    path: Path
    class_name: str
    subgroup_name: str
    label: int


def read_image_unicode(path: str) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.uint8)
    if arr.size == 0:
        raise ValueError(f"cannot read bytes: {path}")
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot decode image: {path}")
    return img


def resize_with_pad(rgb: np.ndarray, size: int, pad_value: int = 114) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = min(size / max(1, w), size / max(1, h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(rgb, (nw, nh), interpolation=interp)
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    x = (size - nw) // 2
    y = (size - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def to_tensor_uint8_rgb(rgb: np.ndarray, normalize: bool = True) -> torch.Tensor:
    arr = rgb.astype(np.float32) / 255.0
    if normalize:
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).float()


def _safe_rot90(bgr: np.ndarray) -> np.ndarray:
    k = random.choice([0, 1, 2, 3])
    if k == 0:
        return bgr
    return np.ascontiguousarray(np.rot90(bgr, k=k))


def _jpeg_simulate(bgr: np.ndarray, qmin: int = 70, qmax: int = 95) -> np.ndarray:
    q = random.randint(qmin, qmax)
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return bgr
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec if dec is not None else bgr


def _random_global_crop(bgr: np.ndarray, min_keep: float = 0.88) -> np.ndarray:
    h, w = bgr.shape[:2]
    if h < 32 or w < 32:
        return bgr

    keep_h = random.uniform(min_keep, 1.0)
    keep_w = random.uniform(min_keep, 1.0)
    ch = max(16, int(round(h * keep_h)))
    cw = max(16, int(round(w * keep_w)))

    if ch >= h and cw >= w:
        return bgr

    y1 = 0 if ch >= h else random.randint(0, h - ch)
    x1 = 0 if cw >= w else random.randint(0, w - cw)
    return bgr[y1:y1 + ch, x1:x1 + cw]


def maybe_augment(bgr: np.ndarray) -> np.ndarray:
    img = bgr.copy()

    if random.random() < 0.80:
        img = _safe_rot90(img)

    if random.random() < 0.45:
        img = _random_global_crop(img, min_keep=0.90)

    if random.random() < 0.60:
        alpha = random.uniform(0.88, 1.14)
        beta = random.uniform(-12, 12)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    if random.random() < 0.20:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= random.uniform(0.90, 1.12)
        hsv[..., 2] *= random.uniform(0.92, 1.10)
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if random.random() < 0.25:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    if random.random() < 0.20:
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
        img = cv2.addWeighted(img, 1.15, blur, -0.15, 0)

    if random.random() < 0.20:
        img = _jpeg_simulate(img, qmin=70, qmax=95)

    return img


def _window_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten().astype(np.float32)
    s = float(hist.sum())
    if s <= 0:
        return 0.0
    p = hist / s
    p = p[p > 1e-8]
    return float(-(p * np.log2(p)).sum())


def _window_score(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    lap = float(cv2.Laplacian(gray, cv2.CV_32F).var())

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = float(np.mean(np.sqrt(gx * gx + gy * gy)))

    ent = _window_entropy(gray)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = float(np.std(hsv[..., 1]))

    score = (
        0.42 * math.log1p(max(0.0, lap))
        + 0.28 * math.log1p(max(0.0, grad))
        + 0.20 * ent
        + 0.10 * math.log1p(max(0.0, sat))
    )
    return score


def _box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1, (bx2 - bx1) * (by2 - by1))
    return float(inter) / float(a_area + b_area - inter)


def _generate_anchor_boxes(h: int, w: int, side: int) -> List[Tuple[int, int, int, int]]:
    side = max(16, min(side, h, w))
    coords = []

    xs = [0, max(0, (w - side) // 2), max(0, w - side)]
    ys = [0, max(0, (h - side) // 2), max(0, h - side)]

    for y in ys:
        for x in xs:
            coords.append((x, y, x + side, y + side))

    # extra anchors
    coords.extend([
        (max(0, w // 4 - side // 2), max(0, h // 4 - side // 2),
         max(0, w // 4 - side // 2) + side, max(0, h // 4 - side // 2) + side),
        (max(0, 3 * w // 4 - side // 2), max(0, h // 4 - side // 2),
         max(0, 3 * w // 4 - side // 2) + side, max(0, h // 4 - side // 2) + side),
        (max(0, w // 4 - side // 2), max(0, 3 * h // 4 - side // 2),
         max(0, w // 4 - side // 2) + side, max(0, 3 * h // 4 - side // 2) + side),
        (max(0, 3 * w // 4 - side // 2), max(0, 3 * h // 4 - side // 2),
         max(0, 3 * w // 4 - side // 2) + side, max(0, 3 * h // 4 - side // 2) + side),
    ])

    out = []
    for x1, y1, x2, y2 in coords:
        x1 = max(0, min(x1, w - side))
        y1 = max(0, min(y1, h - side))
        out.append((x1, y1, x1 + side, y1 + side))
    return out


def _generate_dense_boxes(
    h: int,
    w: int,
    side: int,
    train: bool,
) -> List[Tuple[int, int, int, int]]:
    side = max(16, min(side, h, w))
    stride = max(8, side // (5 if train else 4))

    boxes = []
    y = 0
    while y + side <= h:
        x = 0
        while x + side <= w:
            boxes.append((x, y, x + side, y + side))
            x += stride
        if x < w and w - side >= 0:
            boxes.append((w - side, y, w, y + side))
        y += stride

    if y < h and h - side >= 0:
        yy = h - side
        x = 0
        while x + side <= w:
            boxes.append((x, yy, x + side, h))
            x += stride
        if x < w and w - side >= 0:
            boxes.append((w - side, yy, w, h))

    return boxes


def _select_best_boxes(
    bgr: np.ndarray,
    num_tiles: int,
    train: bool,
) -> List[Tuple[int, int, int, int]]:
    h, w = bgr.shape[:2]
    short = min(h, w)

    if short < 32:
        return [(0, 0, w, h)] * num_tiles

    scales = [0.95, 0.80, 0.65, 0.50, 0.38]
    candidates: List[Tuple[float, Tuple[int, int, int, int]]] = []

    img_cx = w / 2.0
    img_cy = h / 2.0
    img_diag = math.hypot(w, h) + 1e-6

    for sc in scales:
        side = int(round(short * sc))
        side = max(32, min(side, short))

        for box in _generate_anchor_boxes(h, w, side):
            x1, y1, x2, y2 = box
            crop = bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            s = _window_score(crop)

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            dist = math.hypot(cx - img_cx, cy - img_cy) / img_diag
            center_bonus = 0.08 * (1.0 - dist)
            size_bonus = 0.06 * sc
            candidates.append((s + center_bonus + size_bonus, box))

        dense_boxes = _generate_dense_boxes(h, w, side, train=train)
        if train:
            random.shuffle(dense_boxes)
            dense_boxes = dense_boxes[: min(len(dense_boxes), 80)]
        else:
            dense_boxes = dense_boxes[: min(len(dense_boxes), 120)]

        for box in dense_boxes:
            x1, y1, x2, y2 = box
            crop = bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            s = _window_score(crop)

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            dist = math.hypot(cx - img_cx, cy - img_cy) / img_diag
            center_bonus = 0.05 * (1.0 - dist)
            size_bonus = 0.05 * sc
            candidates.append((s + center_bonus + size_bonus, box))

    candidates.sort(key=lambda x: x[0], reverse=True)

    selected: List[Tuple[int, int, int, int]] = []
    iou_thr = 0.45 if train else 0.40

    for _, box in candidates:
        if all(_box_iou(box, prev) < iou_thr for prev in selected):
            selected.append(box)
            if len(selected) >= num_tiles:
                break

    while len(selected) < num_tiles:
        selected.append((0, 0, w, h))

    return selected[:num_tiles]


def _jitter_box(
    box: Tuple[int, int, int, int],
    h: int,
    w: int,
    ratio: float = 0.06,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    dx = int(round(bw * ratio))
    dy = int(round(bh * ratio))

    nx1 = x1 + random.randint(-dx, dx)
    ny1 = y1 + random.randint(-dy, dy)
    nx2 = x2 + random.randint(-dx, dx)
    ny2 = y2 + random.randint(-dy, dy)

    nx1 = max(0, min(nx1, w - 2))
    ny1 = max(0, min(ny1, h - 2))
    nx2 = max(nx1 + 1, min(nx2, w))
    ny2 = max(ny1 + 1, min(ny2, h))

    # keep near-square
    side = min(nx2 - nx1, ny2 - ny1)
    cx = (nx1 + nx2) // 2
    cy = (ny1 + ny2) // 2
    half = max(8, side // 2)
    nx1 = max(0, cx - half)
    ny1 = max(0, cy - half)
    nx2 = min(w, nx1 + side)
    ny2 = min(h, ny1 + side)

    if nx2 - nx1 < 8 or ny2 - ny1 < 8:
        return box
    return nx1, ny1, nx2, ny2


def make_global_view(bgr: np.ndarray, global_size: int, train: bool) -> np.ndarray:
    img = bgr
    if train and random.random() < 0.35:
        img = _random_global_crop(img, min_keep=0.92)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return resize_with_pad(rgb, global_size)


def extract_tiles(
    bgr: np.ndarray,
    tile_size: int,
    num_tiles: int,
    train: bool = False,
) -> List[np.ndarray]:
    h, w = bgr.shape[:2]
    boxes = _select_best_boxes(bgr, num_tiles=num_tiles, train=train)

    out: List[np.ndarray] = []
    for box in boxes:
        if train:
            box = _jitter_box(box, h=h, w=w, ratio=0.06)

        x1, y1, x2, y2 = box
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            crop = bgr

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        out.append(resize_with_pad(rgb, tile_size))

    while len(out) < num_tiles:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out.append(resize_with_pad(rgb, tile_size))

    return out[:num_tiles]


def scan_dataset(root: str):
    root_p = Path(root)
    class_names = sorted([p.name for p in root_p.iterdir() if p.is_dir()])
    label_map = {name: idx for idx, name in enumerate(class_names)}
    samples: List[Sample] = []

    for cls in class_names:
        class_dir = root_p / cls
        for p in class_dir.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                continue
            rel = p.relative_to(class_dir)
            subgroup = rel.parts[0] if len(rel.parts) >= 2 else class_dir.name
            samples.append(
                Sample(
                    path=p,
                    class_name=cls,
                    subgroup_name=subgroup,
                    label=label_map[cls],
                )
            )

    return samples, class_names


def split_samples(samples: List[Sample], val_ratio: float = 0.15, seed: int = 42):
    rng = random.Random(seed)
    by_group: Dict[Tuple[str, str], List[Sample]] = {}
    for s in samples:
        by_group.setdefault((s.class_name, s.subgroup_name), []).append(s)

    train, val = [], []
    class_to_groups: Dict[str, List[Tuple[str, str]]] = {}
    for key in by_group:
        class_to_groups.setdefault(key[0], []).append(key)

    for cls, groups in class_to_groups.items():
        groups = groups[:]
        rng.shuffle(groups)

        if len(groups) >= 3:
            n_val = max(1, int(round(len(groups) * val_ratio)))
        elif len(groups) == 2:
            n_val = 1
        else:
            n_val = 0

        val_groups = set(groups[:n_val])

        for g in groups:
            if g in val_groups:
                val.extend(by_group[g])
            else:
                train.extend(by_group[g])

    if not train:
        train = samples[:]
    if not val:
        val = samples[:]

    return train, val


class PillFullResDataset(Dataset):
    def __init__(
        self,
        samples: List[Sample],
        global_size: int = 1280,
        tile_size: int = 640,
        num_tiles: int = 6,
        train: bool = True,
        normalize: bool = True,
    ):
        self.samples = samples
        self.global_size = global_size
        self.tile_size = tile_size
        self.num_tiles = num_tiles
        self.train = train
        self.normalize = normalize

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        bgr = read_image_unicode(str(s.path))

        if self.train:
            bgr = maybe_augment(bgr)

        global_rgb = make_global_view(bgr, self.global_size, train=self.train)
        tiles_rgb = extract_tiles(
            bgr,
            tile_size=self.tile_size,
            num_tiles=self.num_tiles,
            train=self.train,
        )

        global_tensor = to_tensor_uint8_rgb(global_rgb, normalize=self.normalize)
        tile_tensor = torch.stack(
            [to_tensor_uint8_rgb(t, normalize=self.normalize) for t in tiles_rgb],
            dim=0,
        )

        return {
            "global_img": global_tensor,
            "tiles": tile_tensor,
            "label": torch.tensor(s.label, dtype=torch.long),
            "path": str(s.path),
            "class_name": s.class_name,
            "subgroup_name": s.subgroup_name,
        }