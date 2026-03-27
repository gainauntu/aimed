from __future__ import annotations

# ---------------------------------------------------------------------------
# Build class constraint profiles and OOD feature index.
#
# FIX (Bug 2): Constraint profiles are now computed from preprocessed ROI
# crops (via LightROIPreprocessor) instead of raw full-frame images.
# Previously, stats like aspect_ratio and HSV were computed on 352×288 frames
# that included background, giving values completely different from those
# computed on 288×288 centered ROI crops at inference.
#
# FIX (Bug 3): OOD index features are also extracted from preprocessed ROI
# crops — single-image, 3-channel — using the ImageNet-pretrained
# DINOFeatureBackbone.  Previously, raw frame features were indexed and the
# p99 threshold was meaningless for the ROI features seen at inference.
# ---------------------------------------------------------------------------

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from models import DINOFeatureBackbone
from pair_dataset import (
    scan_pair_samples,
    scan_single_image_samples,
    PillSingleImageDataset,
    LightROIPreprocessor,
    read_image_unicode,
)

try:
    import faiss  # type: ignore
except Exception:
    faiss = None


# ---------------------------------------------------------------------------
# Helper: stat extraction that exactly mirrors app/preprocess/stats.py
# so that profile bands match what the inference ConstraintGate receives.
# ---------------------------------------------------------------------------

def _circular_mean_deg(hues: np.ndarray) -> float:
    r = hues * (2 * math.pi / 180.0)
    d = math.atan2(np.sin(r).mean(), np.cos(r).mean()) * 180.0 / (2 * math.pi)
    return d % 180.0


def _circular_std_deg(hues: np.ndarray) -> float:
    r = hues * (2 * math.pi / 180.0)
    rv = math.sqrt(np.sin(r).mean() ** 2 + np.cos(r).mean() ** 2)
    if rv <= 1e-8:
        return 90.0
    return math.sqrt(max(0.0, -2.0 * math.log(rv))) * 180.0 / (2 * math.pi)


def classify_shape(aspect: float) -> str:
    if aspect < 1.15:
        return 'ROUND'
    if aspect < 1.80:
        return 'OVAL'
    if aspect < 2.50:
        return 'OBLONG'
    return 'CAPSULE'


def extract_roi_stats(roi_bgr: np.ndarray) -> dict:
    """Extract the same 7 stats that ConstraintGate checks at inference."""
    # Build a foreground mask: anything that is not the padded background.
    # For a centered crop the pill is the non-background region. We use
    # simple Otsu thresholding on the value channel as a robust approximation.
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological clean-up
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))

    fg = mask > 0
    if fg.sum() < 100:
        # Fallback: treat whole image as foreground
        fg = np.ones(roi_bgr.shape[:2], dtype=bool)

    fg_h = hsv[..., 0][fg].astype(np.float32)
    fg_s = hsv[..., 1][fg].astype(np.float32)
    fg_v = hsv[..., 2][fg].astype(np.float32)

    # Aspect ratio and area from contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        _, _, w, h = cv2.boundingRect(contour)
        aspect = max(w, h) / max(1.0, min(w, h))
        area = float(fg.sum())
    else:
        aspect = 1.0
        area = float(fg.sum())

    edges = cv2.Canny(gray, 40, 120)
    edge_density = float(edges[fg].sum()) / max(float(fg.sum()), 1.0)

    return {
        'hue':    float(_circular_mean_deg(fg_h)),
        'sat':    float(fg_s.mean()),
        'val':    float(fg_v.mean()),
        'aspect': float(aspect),
        'area':   float(area),
        'edge':   float(edge_density),
        'shape':  classify_shape(float(aspect)),
    }


def class_profile_from_rois(roi_list: list[np.ndarray]) -> dict:
    """Build ±3σ constraint profile from preprocessed ROI crops."""
    stats = [extract_roi_stats(r) for r in roi_list]
    out = {}
    for key in ['hue', 'sat', 'val', 'aspect', 'area', 'edge']:
        vals = np.asarray([s[key] for s in stats], dtype=np.float32)
        out[f'{key}_mean'] = float(vals.mean())
        out[f'{key}_std']  = float(max(vals.std(), 1e-4))   # minimum std avoids zero-width bands
    shapes = [s['shape'] for s in stats]
    out['expected_shape'] = max(set(shapes), key=shapes.count)
    return out


# ---------------------------------------------------------------------------
# Normalize single BGR image for DINOFeatureBackbone (3-channel)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_single(bgr: np.ndarray) -> np.ndarray:
    rgb = bgr[..., ::-1].astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    return ((chw - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root',           required=True)
    ap.add_argument('--output-profiles',     required=True)
    ap.add_argument('--output-ood-prefix',   required=True)
    ap.add_argument('--batch-size',          type=int, default=32)
    ns = ap.parse_args()

    # Black background — no reference file needed.
    # absdiff against a zero frame = simple brightness threshold,
    # which correctly isolates the pill on a black tray.
    bg_bgr = np.zeros((288, 352, 3), dtype=np.uint8)
    preprocessor = LightROIPreprocessor(background_bgr=bg_bgr, target_size=288)

    pair_samples, class_names = scan_pair_samples(ns.data_root)

    # ------------------------------------------------------------------
    # Bug 2 fix: build class profiles from preprocessed ROI images
    # ------------------------------------------------------------------
    print('building constraint profiles from preprocessed ROI crops ...')
    by_class_paths: dict[str, list[str]] = {name: [] for name in class_names}
    for s in pair_samples:
        by_class_paths[s.class_name].extend(s.image_paths)

    profiles: dict[str, dict] = {}
    for cls_name, paths in tqdm(by_class_paths.items(), desc='class profiles'):
        roi_list = []
        for p in paths:
            try:
                raw = read_image_unicode(p)
                roi = preprocessor.process(raw)
                roi_list.append(roi)
            except Exception:
                continue
        if not roi_list:
            print(f'  WARNING: no valid ROIs for class {cls_name}, skipping profile')
            continue
        profiles[cls_name] = class_profile_from_rois(roi_list)

    Path(ns.output_profiles).write_text(
        json.dumps(profiles, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'saved {len(profiles)} class profiles → {ns.output_profiles}')

    # ------------------------------------------------------------------
    # Bug 3 fix: build OOD index from preprocessed single-image ROI features
    # using ImageNet-pretrained DINOFeatureBackbone (3-channel).
    # ------------------------------------------------------------------
    print('building OOD index from preprocessed ROI features ...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # No checkpoint needed — ImageNet pretrained weights are already strong
    # OOD detectors.  The key correctness requirement is that build-time and
    # inference-time features come from the same view (preprocessed ROI crop).
    ood_backbone = DINOFeatureBackbone(model_name='convnext_base', feature_dim=512)
    ood_backbone.eval().to(device)

    all_features: list[np.ndarray] = []
    single_samples, _ = scan_single_image_samples(ns.data_root)
    batch_imgs: list[np.ndarray] = []

    def flush_batch():
        if not batch_imgs:
            return
        x = torch.from_numpy(np.stack(batch_imgs, axis=0)).to(device)
        with torch.inference_mode():
            emb = ood_backbone(x)
        arr = emb.detach().cpu().numpy().astype(np.float32)
        arr /= np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-8, None)
        all_features.append(arr)
        batch_imgs.clear()

    for s in tqdm(single_samples, desc='ood features'):
        try:
            raw = read_image_unicode(s.image_path)
            roi = preprocessor.process(raw)              # preprocessed 288×288 BGR ROI
            tensor = normalize_single(roi)               # 3-channel CHW float32
            batch_imgs.append(tensor)
            if len(batch_imgs) >= ns.batch_size:
                flush_batch()
        except Exception:
            continue
    flush_batch()

    if not all_features:
        raise RuntimeError('No features extracted for OOD index — check data-root and background-reference')

    vecs = np.concatenate(all_features, axis=0)
    vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-8, None)

    # Compute p99 of self-distances (k=5 nearest excluding self)
    if faiss is not None:
        db = vecs.copy()
        faiss.normalize_L2(db)
        index = faiss.IndexFlatIP(db.shape[1])
        index.add(db)
        sims, _ = index.search(db, min(6, db.shape[0]))
        # column 0 is self (similarity ≈ 1); columns 1-5 are neighbours
        d = 1.0 - sims[:, 1:].mean(axis=1)
    else:
        sims = vecs @ vecs.T
        np.fill_diagonal(sims, -1.0)
        topk = np.sort(sims, axis=1)[:, -5:]
        d = 1.0 - topk.mean(axis=1)

    threshold = float(np.percentile(d, 99))

    prefix = Path(ns.output_ood_prefix)
    np.save(prefix.with_suffix('.npy'), vecs.astype(np.float32))
    prefix.with_suffix('.json').write_text(
        json.dumps({'threshold': threshold}, ensure_ascii=False, indent=2),
        encoding='utf-8')
    print(f'saved OOD index ({len(vecs)} vectors, p99 threshold={threshold:.4f}) → {prefix}')


if __name__ == '__main__':
    main()
