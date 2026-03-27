from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.preprocess.stats import classify_shape, extract_physical_stats


def build_class_profiles(dataset_dir: str | Path) -> dict[str, dict]:
    dataset_root = Path(dataset_dir)
    profiles: dict[str, dict] = {}
    for class_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        stats = []
        for image_path in sorted(class_dir.glob("*")):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            stats.append(extract_physical_stats(image, mask))
        if not stats:
            continue
        profiles[class_dir.name] = {
            "hue_mean": float(np.mean([s.hue_mean for s in stats])),
            "hue_std": float(np.std([s.hue_mean for s in stats])),
            "sat_mean": float(np.mean([s.sat_mean for s in stats])),
            "sat_std": float(np.std([s.sat_mean for s in stats])),
            "val_mean": float(np.mean([s.val_mean for s in stats])),
            "val_std": float(np.std([s.val_mean for s in stats])),
            "aspect_mean": float(np.mean([s.aspect_ratio for s in stats])),
            "aspect_std": float(np.std([s.aspect_ratio for s in stats])),
            "area_mean": float(np.mean([s.pixel_area_proxy for s in stats])),
            "area_std": float(np.std([s.pixel_area_proxy for s in stats])),
            "edge_mean": float(np.mean([s.edge_density for s in stats])),
            "edge_std": float(np.std([s.edge_density for s in stats])),
            "expected_shape": max(set([s.shape_class.value for s in stats]), key=[s.shape_class.value for s in stats].count),
        }
    return profiles
