from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


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


def simple_localize(bgr: np.ndarray):
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
        return False, (0, 0, w, h), 1.0

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
        return False, (0, 0, w, h), 1.0
    return True, (x1, y1, x2, y2), crop_ratio


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


def to_tensor_uint8_rgb(rgb: np.ndarray) -> torch.Tensor:
    arr = rgb.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).float()


def maybe_augment(bgr: np.ndarray) -> np.ndarray:
    img = bgr.copy()
    if random.random() < 0.50:
        alpha = random.uniform(0.85, 1.15)
        beta = random.uniform(-10, 10)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    if random.random() < 0.25:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if random.random() < 0.60:
        rot = random.choice([0, 1, 2, 3])
        if rot > 0:
            img = np.ascontiguousarray(np.rot90(img, k=rot))
    return img


def extract_tiles(bgr: np.ndarray, tile_size: int, num_tiles: int):
    h, w = bgr.shape[:2]
    ok, (x1, y1, x2, y2), _ = simple_localize(bgr)
    roi_w = x2 - x1
    roi_h = y2 - y1

    centers = []
    centers.append((x1 + roi_w // 2, y1 + roi_h // 2))
    centers.append((x1 + roi_w // 3, y1 + roi_h // 3))
    centers.append((x1 + 2 * roi_w // 3, y1 + roi_h // 3))
    centers.append((x1 + roi_w // 3, y1 + 2 * roi_h // 3))
    centers.append((x1 + 2 * roi_w // 3, y1 + 2 * roi_h // 3))

    out = []
    half = max(32, min(h, w) // 5)
    for cx, cy in centers[: max(num_tiles, 4)]:
        sx1 = max(0, cx - half)
        sy1 = max(0, cy - half)
        sx2 = min(w, cx + half)
        sy2 = min(h, cy + half)
        crop = bgr[sy1:sy2, sx1:sx2]
        if crop.size == 0:
            continue
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
    samples = []

    for cls in class_names:
        class_dir = root_p / cls
        for p in class_dir.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                continue
            rel = p.relative_to(class_dir)
            subgroup = rel.parts[0] if len(rel.parts) >= 2 else class_dir.name
            samples.append(Sample(path=p, class_name=cls, subgroup_name=subgroup, label=label_map[cls]))

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
        n_val = max(1, int(round(len(groups) * val_ratio))) if len(groups) >= 3 else (1 if len(groups) >= 2 else 0)
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
    def __init__(self, samples: List[Sample], global_size: int = 1024, tile_size: int = 512, num_tiles: int = 4, train: bool = True):
        self.samples = samples
        self.global_size = global_size
        self.tile_size = tile_size
        self.num_tiles = num_tiles
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        bgr = read_image_unicode(str(s.path))
        if self.train:
            bgr = maybe_augment(bgr)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        global_rgb = resize_with_pad(rgb, self.global_size)
        tiles_rgb = extract_tiles(bgr, self.tile_size, self.num_tiles)

        global_tensor = to_tensor_uint8_rgb(global_rgb)
        tile_tensor = torch.stack([to_tensor_uint8_rgb(t) for t in tiles_rgb], dim=0)

        return {
            "global_img": global_tensor,
            "tiles": tile_tensor,
            "label": torch.tensor(s.label, dtype=torch.long),
            "path": str(s.path),
            "class_name": s.class_name,
            "subgroup_name": s.subgroup_name,
        }
