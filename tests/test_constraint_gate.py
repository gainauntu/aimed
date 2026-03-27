from __future__ import annotations

import numpy as np

from app.domain.models import FrameQualityReport, PhysicalStats, PreprocessedImage, ShapeClass
from app.verifiers.constraints import ConstraintGate, ConstraintProfiles


def test_constraint_gate_passes_within_profile() -> None:
    profiles = ConstraintProfiles({
        "pill": {
            "hue_mean": 10.0,
            "hue_std": 1.0,
            "sat_mean": 20.0,
            "sat_std": 1.0,
            "val_mean": 30.0,
            "val_std": 1.0,
            "aspect_mean": 1.3,
            "aspect_std": 0.1,
            "area_mean": 1000.0,
            "area_std": 10.0,
            "edge_mean": 3.0,
            "edge_std": 1.0,
            "expected_shape": "OVAL",
        }
    })
    gate = ConstraintGate(profiles)
    stats = PhysicalStats(10.0, 1.0, 20.0, 1.0, 30.0, 1.0, 1000.0, 1.3, 0.95, 3.0, ShapeClass.OVAL)
    img = PreprocessedImage(np.zeros((10, 10, 3), dtype=np.uint8), np.zeros((10, 10), dtype=np.uint8), stats, FrameQualityReport(200, 30, 0.0, True, True))
    report = gate.verify("pill", img, img)
    assert report.passed
    assert report.pass_ratio == 1.0
