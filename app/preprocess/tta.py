from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class ViewPair:
    image_a: np.ndarray
    image_b: np.ndarray
    rotation_deg: int
    variant_name: str


class TTAEngine:
    ROTATIONS = (0, 45, 90, 135, 180, 225, 270, 315)
    VARIANTS = ("clean", "blur", "noise", "hsv")

    def generate(self, image_a: np.ndarray, image_b: np.ndarray) -> list[ViewPair]:
        views: list[ViewPair] = []
        for angle in self.ROTATIONS:
            rot_a = self._rotate(image_a, angle)
            rot_b = self._rotate(image_b, angle)
            for variant in self.VARIANTS:
                views.append(
                    ViewPair(
                        image_a=self._variant(rot_a, variant),
                        image_b=self._variant(rot_b, variant),
                        rotation_deg=angle,
                        variant_name=variant,
                    )
                )
        return views

    @staticmethod
    def _rotate(image: np.ndarray, angle: float) -> np.ndarray:
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    @staticmethod
    def _variant(image: np.ndarray, variant: str) -> np.ndarray:
        if variant == "clean":
            return image.copy()
        if variant == "blur":
            return cv2.GaussianBlur(image, (3, 3), 0.3)
        if variant == "noise":
            noise = np.random.default_rng(42).normal(0, 1.5, image.shape)
            return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if variant == "hsv":
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + 2.0) % 180.0
            hsv[..., 1] = np.clip(hsv[..., 1] * 1.03, 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] * 1.02, 0, 255)
            return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        raise ValueError(f"Unsupported TTA variant: {variant}")
