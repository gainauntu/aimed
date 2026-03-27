from __future__ import annotations

# ---------------------------------------------------------------------------
# FIX (Issue 6): Training previously operated on raw full-frame images while
# inference operates on preprocessed 288×288 ROI crops.  This train/inference
# distribution mismatch meant models could learn background cues that vanish
# at inference time.
#
# LightROIPreprocessor (added here) replicates the inference pipeline steps
# that normalise the pill view — background subtraction, localization,
# PCA rotation canonicalization, and resize — without the quality-rejection
# gates that would discard legitimate training samples.
#
# PillPairDataset now runs each image through LightROIPreprocessor before
# augmentation and fusion so training and inference see the same distribution.
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(slots=True)
class PairSample:
    class_name: str
    class_idx: int
    pair_id: str
    image_paths: list[str]


@dataclass(slots=True)
class SingleImageSample:
    class_name: str
    class_idx: int
    pair_id: str
    image_path: str


# ---------------------------------------------------------------------------
# LightROIPreprocessor
# ---------------------------------------------------------------------------

class LightROIPreprocessor:
    """Lightweight preprocessing that mirrors inference Stage 1 without gates.

    Steps performed:
      1. Background subtraction → binary mask
      2. Largest connected component → bounding box + 10% margin → crop
      3. PCA rotation canonicalization (ROUND pills skipped)
      4. Resize to target_size × target_size (Lanczos)

    On any failure (no foreground, very small component, etc.) the method
    falls back to a plain center-crop + resize so training never crashes.

    background_bgr: reference frame.  Pass None (or omit) to use a pure
                    black frame — correct for black-tray pharmacy setups.
    """

    def __init__(self, background_bgr: np.ndarray | None = None,
                 target_size: int = 288, diff_threshold: int = 15):
        if background_bgr is None:
            # Black tray: pure zero reference so absdiff = brightness threshold
            self.reference = np.zeros((target_size, 352, 3), dtype=np.uint8)
        else:
            self.reference = background_bgr.astype(np.uint8)
        self.target_size = target_size
        self.diff_threshold = diff_threshold

    # ------------------------------------------------------------------
    def process(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return a (target_size, target_size, 3) BGR ROI crop."""
        try:
            return self._preprocess(image_bgr)
        except Exception:
            return self._direct_resize(image_bgr)

    # ------------------------------------------------------------------
    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        # 1. Background subtraction
        diff = cv2.absdiff(image_bgr, self.reference)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.diff_threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))

        # 2. Localize largest component
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self._direct_resize(image_bgr)
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 500:
            return self._direct_resize(image_bgr)

        x, y, w, h = cv2.boundingRect(contour)
        margin = int(max(w, h) * 0.10)
        H, W = image_bgr.shape[:2]
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(W, x + w + margin)
        y1 = min(H, y + h + margin)
        crop = image_bgr[y0:y1, x0:x1].copy()
        crop_mask = mask[y0:y1, x0:x1].copy()

        # 3. PCA canonicalization
        crop = self._canonicalize(crop, crop_mask)

        # 4. Resize
        return self._resize_to_canvas(crop)

    # ------------------------------------------------------------------
    def _canonicalize(self, crop_bgr: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
        ys, xs = np.where(crop_mask > 0)
        if len(xs) < 5:
            return crop_bgr
        coords = np.column_stack([xs, ys]).astype(np.float32)
        _, _, vt = np.linalg.svd(coords - coords.mean(axis=0), full_matrices=False)
        major_axis = vt[0]
        # Skip canonicalization for round pills (aspect ratio near 1)
        bx, by, bw, bh = cv2.boundingRect(
            np.column_stack([xs, ys]).reshape(-1, 1, 2).astype(np.int32))
        aspect = max(bw, bh) / max(1.0, min(bw, bh))
        if aspect < 1.15:
            return crop_bgr
        angle = float(np.degrees(np.arctan2(major_axis[1], major_axis[0])))
        center = (crop_bgr.shape[1] / 2, crop_bgr.shape[0] / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(crop_bgr, M, (crop_bgr.shape[1], crop_bgr.shape[0]),
                              borderMode=cv2.BORDER_REPLICATE)

    # ------------------------------------------------------------------
    def _resize_to_canvas(self, crop_bgr: np.ndarray) -> np.ndarray:
        t = self.target_size
        h, w = crop_bgr.shape[:2]
        scale = min(t / max(w, 1), t / max(h, 1))
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        resized = cv2.resize(crop_bgr, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        canvas = np.zeros((t, t, 3), dtype=np.uint8)   # black padding
        x0 = (t - nw) // 2
        y0 = (t - nh) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        return canvas

    # ------------------------------------------------------------------
    def _direct_resize(self, image_bgr: np.ndarray) -> np.ndarray:
        return self._resize_to_canvas(image_bgr)



def read_image_unicode(path: str) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'failed to decode image: {path}')
    return image


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


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = (arr - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return arr.astype(np.float32)


def symmetric_fusion_chw(a_bgr: np.ndarray, b_bgr: np.ndarray, size: int = 288) -> np.ndarray:
    a = normalize_rgb(resize_with_pad(cv2.cvtColor(a_bgr, cv2.COLOR_BGR2RGB), size))
    b = normalize_rgb(resize_with_pad(cv2.cvtColor(b_bgr, cv2.COLOR_BGR2RGB), size))
    return np.concatenate([a + b, np.abs(a - b), a * b], axis=0).astype(np.float32)


def infer_pair_id(root: Path, path: Path) -> str:
    rel = path.relative_to(root)
    if len(rel.parts) >= 3:
        return rel.parts[1]
    stem = path.stem
    if '_' in stem:
        return stem.split('_')[0]
    return stem


def scan_pair_samples(data_root: str | Path):
    root = Path(data_root)
    class_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    class_names = [p.name for p in class_dirs]
    samples: list[PairSample] = []
    for class_idx, class_dir in enumerate(class_dirs):
        by_pair: dict[str, list[str]] = {}
        for p in class_dir.rglob('*'):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                pid = infer_pair_id(root, p)
                by_pair.setdefault(pid, []).append(str(p))
        for pid, paths in sorted(by_pair.items()):
            if paths:
                samples.append(PairSample(class_dir.name, class_idx, pid, sorted(paths)))
    return samples, class_names


def scan_single_image_samples(data_root: str | Path):
    pair_samples, class_names = scan_pair_samples(data_root)
    out: list[SingleImageSample] = []
    for ps in pair_samples:
        for p in ps.image_paths:
            out.append(SingleImageSample(ps.class_name, ps.class_idx, ps.pair_id, p))
    return out, class_names


def loocv_n_folds(samples: list[PairSample]) -> int:
    """Return the correct number of LOOCV folds for this dataset.

    Uses class-balanced leave-one-per-class LOOCV.  The number of folds
    equals the size of the largest class so every pair in every class is
    used as validation at least once.  Smaller classes cycle via modulo so
    every fold always has a validation sample from every class.
    """
    by_class: dict[str, list] = {}
    for s in samples:
        by_class.setdefault(s.class_name, []).append(s)
    if not by_class:
        return 1
    return max(len(items) for items in by_class.values())


def build_loocv_fold(samples: list[PairSample], fold_index: int):
    """Build one fold of class-balanced leave-one-per-class LOOCV.

    FIX (ChatGPT review): The original implementation used a hardcoded
    n_folds=20 with fold_index % len(items).  This caused two problems:
      1. Classes with <20 pairs: some pairs were held out multiple times
         (folds 0..K-1 are unique, folds K..19 repeat indices 0..19-K).
      2. Classes with >20 pairs: pairs 20+ were NEVER held out.
    Both biased phase selection and calibration signal generation.

    Fix: n_folds is derived from the data via loocv_n_folds(), which equals
    max(class_size).  When all classes have the same size K (the common case
    here: ~20 pairs per class), fold_index never reaches K so modulo is a
    no-op and we get true, correct, non-repeating LOOCV.
    """
    by_class: dict[str, list[PairSample]] = {}
    for s in samples:
        by_class.setdefault(s.class_name, []).append(s)

    train: list[PairSample] = []
    val:   list[PairSample] = []

    for cls, items in by_class.items():
        items_sorted = sorted(items, key=lambda x: x.pair_id)
        hold_out_idx = fold_index % len(items_sorted)
        for i, item in enumerate(items_sorted):
            (val if i == hold_out_idx else train).append(item)

    return train, val


def elastic_warp(image: np.ndarray, alpha: float = 30.0, sigma: float = 5.0) -> np.ndarray:
    h, w = image.shape[:2]
    dx = cv2.GaussianBlur((np.random.rand(h, w).astype(np.float32) * 2.0 - 1.0), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(h, w).astype(np.float32) * 2.0 - 1.0), (0, 0), sigma) * alpha
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = np.clip(xx + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(yy + dy, 0, h - 1).astype(np.float32)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


class PairAugment:
    def __init__(self, size: int = 288, train: bool = True):
        self.size = size
        self.train = train

    def _aug(self, bgr: np.ndarray) -> np.ndarray:
        image = bgr.copy()
        if not self.train:
            return image
        angle = random.uniform(0.0, 360.0)
        h, w = image.shape[:2]
        mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        image = cv2.warpAffine(image, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
        if random.random() < 0.20:
            image = elastic_warp(image, alpha=30.0, sigma=5.0)
        if random.random() < 0.30:
            noise = np.random.normal(0.0, 2.0, image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if random.random() < 0.30:
            image = cv2.GaussianBlur(image, (3, 3), random.uniform(0.0, 0.5))
        if random.random() < 0.30:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + random.uniform(-3.0, 3.0)) % 180.0
            hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.95, 1.05), 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(0.97, 1.03), 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return image

    def __call__(self, a_bgr: np.ndarray, b_bgr: np.ndarray):
        if self.train and random.random() < 0.5:
            a_bgr, b_bgr = b_bgr, a_bgr
        return self._aug(a_bgr), self._aug(b_bgr)


class PillPairDataset(Dataset):
    def __init__(self, samples: list[PairSample], image_size: int = 288,
                 train: bool = True,
                 light_preprocessor: LightROIPreprocessor | None = None):
        self.samples = list(samples)
        self.augment = PairAugment(image_size, train=train)
        self.image_size = image_size
        self.train = train
        # Issue 6 fix: apply lightweight ROI preprocessing before augmentation
        # so models see the same view distribution as the inference pipeline.
        self.light_preprocessor = light_preprocessor
        self.class_to_paths: dict[str, list[str]] = {}
        for s in self.samples:
            self.class_to_paths.setdefault(s.class_name, []).extend(s.image_paths)

    def __len__(self):
        return len(self.samples)

    def _same_class_donor(self, class_name: str) -> np.ndarray | None:
        pool = self.class_to_paths.get(class_name, [])
        if not pool:
            return None
        return read_image_unicode(random.choice(pool))

    def _same_class_mix(self, image: np.ndarray, class_name: str) -> np.ndarray:
        donor = self._same_class_donor(class_name)
        if donor is None:
            return image
        donor = cv2.resize(donor, (image.shape[1], image.shape[0]),
                           interpolation=cv2.INTER_AREA if donor.shape[0] > image.shape[0] else cv2.INTER_CUBIC)
        if random.random() < 0.15:
            lam = np.random.beta(0.2, 0.2)
            mixed = image.astype(np.float32) * lam + donor.astype(np.float32) * (1.0 - lam)
            image = np.clip(mixed, 0, 255).astype(np.uint8)
        if random.random() < 0.15:
            h, w = image.shape[:2]
            cut_w = max(8, int(round(w * random.uniform(0.20, 0.45))))
            cut_h = max(8, int(round(h * random.uniform(0.20, 0.45))))
            x1 = random.randint(0, max(0, w - cut_w))
            y1 = random.randint(0, max(0, h - cut_h))
            image = image.copy()
            image[y1:y1 + cut_h, x1:x1 + cut_w] = donor[y1:y1 + cut_h, x1:x1 + cut_w]
        return image

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if len(sample.image_paths) >= 2:
            pa, pb = random.sample(sample.image_paths, 2) if self.train else sample.image_paths[:2]
        else:
            pa = pb = sample.image_paths[0]
        a = read_image_unicode(pa)
        b = read_image_unicode(pb)
        # Issue 6: normalise each raw frame to ROI crop before augmentation
        if self.light_preprocessor is not None:
            a = self.light_preprocessor.process(a)
            b = self.light_preprocessor.process(b)
        a, b = self.augment(a, b)
        if self.train:
            a = self._same_class_mix(a, sample.class_name)
            b = self._same_class_mix(b, sample.class_name)
        fused = symmetric_fusion_chw(a, b, size=self.image_size)
        return {
            'fused': torch.from_numpy(fused),
            'label': torch.tensor(sample.class_idx, dtype=torch.long),
            'class_name': sample.class_name,
            'pair_id': sample.pair_id,
            'paths': [pa, pb],
        }


class PillSingleImageDataset(Dataset):
    def __init__(self, samples: list[SingleImageSample], image_size: int = 288,
                 train: bool = True,
                 light_preprocessor: LightROIPreprocessor | None = None):
        self.samples = list(samples)
        self.image_size = image_size
        self.train = train
        self.light_preprocessor = light_preprocessor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = read_image_unicode(sample.image_path)
        if self.light_preprocessor is not None:
            image = self.light_preprocessor.process(image)
        if self.train:
            angle = random.uniform(0.0, 360.0)
            h, w = image.shape[:2]
            mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
            image = cv2.warpAffine(image, mat, (w, h),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            if random.random() < 0.20:
                image = elastic_warp(image, alpha=30.0, sigma=5.0)
        rgb = resize_with_pad(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), self.image_size)
        chw = normalize_rgb(rgb)
        return {
            'image': torch.from_numpy(chw),
            'label': torch.tensor(sample.class_idx, dtype=torch.long),
            'pair_id': sample.pair_id,
            'class_name': sample.class_name,
            'path': sample.image_path,
        }
