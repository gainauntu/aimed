from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.morphology import skeletonize

from app.core.config import AppConfig
from app.domain.models import CrossImageReport, FrameQualityReport, PreprocessedImage, ShapeClass
from app.preprocess.background import BackgroundReference
from app.preprocess.stats import extract_physical_stats

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ComponentInfo:
    contour: np.ndarray
    area: float
    solidity: float
    bbox: tuple[int, int, int, int]


class PreprocessingError(RuntimeError):
    pass


class PillPreprocessor:
    def __init__(self, config: AppConfig, background: BackgroundReference) -> None:
        self.cfg = config
        self.background = background

    def process(self, frame_bgr: np.ndarray) -> PreprocessedImage:
        quality = self._triage(frame_bgr)
        if not quality.passed:
            raise PreprocessingError('; '.join(quality.reasons))

        mask = self.background.subtract(frame_bgr, self.cfg.thresholds.background_diff_threshold)
        component = self._analyze_components(mask)
        crop_bgr, crop_mask = self._extract_crop(frame_bgr, mask, component)
        self._check_damage(crop_bgr, crop_mask)
        canonical_bgr, canonical_mask = self._canonicalize(crop_bgr, crop_mask)
        resized_bgr, resized_mask = self._resize_to_canvas(canonical_bgr, canonical_mask)
        stats = extract_physical_stats(resized_bgr, resized_mask)
        return PreprocessedImage(
            image_bgr=resized_bgr,
            mask=resized_mask,
            stats=stats,
            quality=quality,
            metadata={
                'component_area': component.area,
                'solidity': component.solidity,
            },
        )

    def cross_check(self, image_a: PreprocessedImage, image_b: PreprocessedImage) -> CrossImageReport:
        reasons: list[str] = []
        stats_a = image_a.stats
        stats_b = image_b.stats
        shape_ok = self._shape_consistent(stats_a.shape_class, stats_b.shape_class)
        if not shape_ok:
            reasons.append(f'shape mismatch: {stats_a.shape_class.value} vs {stats_b.shape_class.value}')

        color_distance = float(np.linalg.norm(
            np.array([stats_a.hue_mean, stats_a.sat_mean, stats_a.val_mean], dtype=np.float32)
            - np.array([stats_b.hue_mean, stats_b.sat_mean, stats_b.val_mean], dtype=np.float32)
        ))
        if color_distance > self.cfg.thresholds.max_color_distance:
            reasons.append(f'color distance {color_distance:.3f} exceeds {self.cfg.thresholds.max_color_distance}')

        size_ratio = max(stats_a.pixel_area_proxy, stats_b.pixel_area_proxy) / max(1.0, min(stats_a.pixel_area_proxy, stats_b.pixel_area_proxy))
        if size_ratio > self.cfg.thresholds.max_area_ratio:
            reasons.append(f'size ratio {size_ratio:.3f} exceeds {self.cfg.thresholds.max_area_ratio}')

        gray_a = cv2.cvtColor(image_a.image_bgr, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(image_b.image_bgr, cv2.COLOR_BGR2GRAY)
        ssim_value = float(ssim(gray_a, gray_b, data_range=255))
        if ssim_value < self.cfg.thresholds.min_ssim:
            reasons.append(f'SSIM {ssim_value:.3f} below {self.cfg.thresholds.min_ssim}')

        return CrossImageReport(
            passed=not reasons,
            reasons=reasons,
            color_distance=color_distance,
            size_ratio=size_ratio,
            ssim=ssim_value,
        )

    def _triage(self, frame_bgr: np.ndarray) -> FrameQualityReport:
        reasons: list[str] = []
        h, w = frame_bgr.shape[:2]
        if abs(w - self.cfg.runtime.expected_width) > 2 or abs(h - self.cfg.runtime.expected_height) > 2:
            reasons.append(f'resolution {w}x{h} outside expected bounds')

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_variance < self.cfg.thresholds.min_blur_variance:
            reasons.append(f'blur variance {blur_variance:.3f} below threshold')

        frame_mean = float(gray.mean())
        if frame_mean < self.cfg.thresholds.min_frame_mean:
            reasons.append(f'frame mean {frame_mean:.3f} below threshold')
        if frame_mean > self.cfg.thresholds.max_frame_mean:
            reasons.append(f'frame mean {frame_mean:.3f} above threshold')

        col_std = gray.std(axis=0)
        stuck_column_ratio = float((col_std < 1e-3).sum() / max(1, gray.shape[1]))
        if stuck_column_ratio > self.cfg.thresholds.max_stuck_column_ratio:
            reasons.append(f'stuck column ratio {stuck_column_ratio:.3f} above threshold')

        mask = self.background.subtract(frame_bgr, self.cfg.thresholds.background_diff_threshold)
        has_signal = bool((mask > 0).any())
        if not has_signal:
            reasons.append('no foreground signal after background subtraction')

        return FrameQualityReport(
            blur_variance=blur_variance,
            frame_mean=frame_mean,
            stuck_column_ratio=stuck_column_ratio,
            has_signal=has_signal,
            passed=not reasons,
            reasons=reasons,
        )

    def _analyze_components(self, mask: np.ndarray) -> ComponentInfo:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        components: list[ComponentInfo] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area <= 0:
                continue
            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            solidity = area / max(hull_area, 1.0)
            bbox = cv2.boundingRect(contour)
            components.append(ComponentInfo(contour=contour, area=area, solidity=solidity, bbox=bbox))

        if not components:
            raise PreprocessingError('no pill component found')

        large_components = [c for c in components if c.area > self.cfg.thresholds.min_large_component_area]
        if len(large_components) > 1:
            raise PreprocessingError('multiple large components detected')

        component = max(components, key=lambda c: c.area)
        if component.area < self.cfg.thresholds.min_component_area:
            raise PreprocessingError(f'component area too small: {component.area:.1f}')
        if component.area > self.cfg.thresholds.max_component_area:
            raise PreprocessingError(f'component area too large: {component.area:.1f}')
        if component.solidity < self.cfg.thresholds.min_solidity:
            raise PreprocessingError(f'component solidity too low: {component.solidity:.3f}')
        return component

    def _extract_crop(self, frame_bgr: np.ndarray, mask: np.ndarray, component: ComponentInfo) -> tuple[np.ndarray, np.ndarray]:
        x, y, w, h = component.bbox
        margin = int(max(w, h) * 0.10)
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(frame_bgr.shape[1], x + w + margin)
        y1 = min(frame_bgr.shape[0], y + h + margin)
        return frame_bgr[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy()

    def _check_damage(self, crop_bgr: np.ndarray, crop_mask: np.ndarray) -> None:
        contours, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        hull_perimeter = cv2.arcLength(hull, True)
        actual_perimeter = cv2.arcLength(contour, True)
        perimeter_ratio = float(actual_perimeter / max(hull_perimeter, 1.0))
        if perimeter_ratio > self.cfg.thresholds.max_perimeter_ratio:
            raise PreprocessingError(f'perimeter ratio too high: {perimeter_ratio:.3f}')

        # Crack scan: Laplacian -> threshold -> skeleton length inside mask.
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        abs_lap = np.abs(lap)
        pill_vals = abs_lap[crop_mask > 0]
        if pill_vals.size == 0:
            raise PreprocessingError('empty crop mask after localization')
        lap_th = float(np.percentile(pill_vals, 93))
        crack_mask = ((abs_lap >= lap_th).astype(np.uint8) * 255)
        crack_mask = cv2.bitwise_and(crack_mask, crack_mask, mask=crop_mask)
        crack_mask = cv2.morphologyEx(crack_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        skel = skeletonize(crack_mask > 0)
        line_length = float(skel.sum())
        pill_area = float((crop_mask > 0).sum())
        crack_ratio = line_length / max(pill_area, 1.0)
        # conservative threshold tuned to architecture intent: visible residual crack pattern only.
        if crack_ratio > 0.025:
            raise PreprocessingError(f'crack scan residual too high: {crack_ratio:.4f}')

        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        vals = hsv[..., 2][crop_mask > 0].astype(np.float32)
        val_mean = float(vals.mean())
        val_std = float(vals.std())
        defects = np.abs(vals - val_mean) > (3.0 * max(val_std, 1.0))
        defect_ratio = float(defects.sum() / max(1, vals.shape[0]))
        if defect_ratio > self.cfg.thresholds.max_coating_defect_ratio:
            raise PreprocessingError(f'coating defect ratio too high: {defect_ratio:.3f}')

    def _canonicalize(self, crop_bgr: np.ndarray, crop_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ys, xs = np.where(crop_mask > 0)
        if len(xs) < 5:
            raise PreprocessingError('not enough foreground pixels for canonicalization')
        coords = np.column_stack([xs, ys]).astype(np.float32)
        cov = np.cov(coords, rowvar=False)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        major_axis = eig_vecs[:, np.argmax(eig_vals)]
        angle = float(np.degrees(np.arctan2(major_axis[1], major_axis[0])))

        x, y, w, h = cv2.boundingRect(np.column_stack([xs, ys]).reshape(-1, 1, 2).astype(np.int32))
        aspect_ratio = max(w, h) / max(1.0, min(w, h))
        if aspect_ratio < 1.15:
            return crop_bgr, crop_mask

        center = (crop_bgr.shape[1] / 2, crop_bgr.shape[0] / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_bgr = cv2.warpAffine(crop_bgr, matrix, (crop_bgr.shape[1], crop_bgr.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        rotated_mask = cv2.warpAffine(crop_mask, matrix, (crop_mask.shape[1], crop_mask.shape[0]), borderMode=cv2.BORDER_CONSTANT)
        return rotated_bgr, rotated_mask

    def _resize_to_canvas(self, crop_bgr: np.ndarray, crop_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        target = self.cfg.runtime.target_size
        h, w = crop_bgr.shape[:2]
        scale = min(target / max(w, 1), target / max(h, 1))
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized_bgr = cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        resized_mask = cv2.resize(crop_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Black background: zero canvas is correct for black-tray setups.
        # If a non-black reference is loaded from file the mean will still be
        # used so that padding matches the actual tray colour.
        if self.background.is_black:
            canvas = np.zeros((target, target, 3), dtype=np.uint8)
        else:
            bg_mean = self.background.reference_bgr.mean(axis=(0, 1)).astype(np.uint8)
            canvas = np.full((target, target, 3), bg_mean, dtype=np.uint8)
        canvas_mask = np.zeros((target, target), dtype=np.uint8)
        x0 = (target - new_w) // 2
        y0 = (target - new_h) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized_bgr
        canvas_mask[y0:y0 + new_h, x0:x0 + new_w] = resized_mask
        return canvas, canvas_mask

    @staticmethod
    def _shape_consistent(shape_a: ShapeClass, shape_b: ShapeClass) -> bool:
        if shape_a == shape_b:
            return True
        allowed = {
            (ShapeClass.OVAL, ShapeClass.OBLONG),
            (ShapeClass.OBLONG, ShapeClass.OVAL),
        }
        return (shape_a, shape_b) in allowed
