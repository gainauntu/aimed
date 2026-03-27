from __future__ import annotations

import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_image(image_bgr: np.ndarray) -> np.ndarray:
    image = image_bgr.astype(np.float32) / 255.0
    rgb = image[..., ::-1]
    chw = np.transpose(rgb, (2, 0, 1))
    chw = (chw - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return chw.astype(np.float32)


def symmetric_fusion(image_a_bgr: np.ndarray, image_b_bgr: np.ndarray) -> np.ndarray:
    a = normalize_image(image_a_bgr)
    b = normalize_image(image_b_bgr)
    return np.concatenate([a + b, np.abs(a - b), a * b], axis=0).astype(np.float32)


def batch_fusion(pairs: list[tuple[np.ndarray, np.ndarray]]) -> torch.Tensor:
    fused = [symmetric_fusion(a, b) for a, b in pairs]
    return torch.from_numpy(np.stack(fused, axis=0))
