from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression


def main() -> None:
    root = Path("artifacts")
    root.mkdir(parents=True, exist_ok=True)

    # No background_reference.png needed — black reference used automatically.

    # Minimal class profiles
    profiles = {
        "demo_pill": {
            "hue_mean": 0.0,
            "hue_std": 10.0,
            "sat_mean": 0.0,
            "sat_std": 10.0,
            "val_mean": 220.0,
            "val_std": 30.0,
            "aspect_mean": 1.4,
            "aspect_std": 0.4,
            "area_mean": 12000.0,
            "area_std": 6000.0,
            "edge_mean": 4.0,
            "edge_std": 5.0,
            "expected_shape": "OVAL",
        }
    }
    (root / "class_profiles.json").write_text(json.dumps(profiles, indent=2), encoding="utf-8")

    # Prototype library (matching tower embedding dim 256)
    proto = {
        "demo_pill": {
            "centroid": [1.0] * 256,
            "p99_distance": 25.0,
            "min_similarity": -1.0,
            "second_class_margin": -10.0,
        }
    }
    (root / "prototype_library.json").write_text(json.dumps(proto, indent=2), encoding="utf-8")

    # OOD index
    np.save(root / "ood_index.npy", np.array([[0.5, 0.5, 0.5]], dtype=np.float32))
    (root / "ood_index.json").write_text(json.dumps({"threshold": 5.0}, indent=2), encoding="utf-8")

    # Calibrator
    iso = IsotonicRegression(out_of_bounds="clip")
    xs = np.linspace(0, 1, 20)
    ys = np.linspace(0.995, 1.0, 20)
    iso.fit(xs, ys)
    joblib.dump({"weights": np.ones(12, dtype=np.float32) / 12.0, "intercept": 0.0, "isotonic_model": iso}, root / "calibrator.joblib")

    # Note: dummy towers are used only in tests; production checkpoints are not generated here.
    print("Mock artifacts created under ./artifacts")


if __name__ == "__main__":
    main()
