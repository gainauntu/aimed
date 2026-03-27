from __future__ import annotations

import numpy as np

from app.decision.calibration import MultivariateIsotonicCalibrator
from app.domain.models import MetaSignals


def test_multivariate_isotonic_preserves_monotone_lookup() -> None:
    train = np.asarray([
        [0.0] * 12,
        [0.5] * 12,
        [1.0] * 12,
    ], dtype=np.float32)
    fitted = np.asarray([0.1, 0.5, 0.9], dtype=np.float32)
    cal = MultivariateIsotonicCalibrator(
        train_features=train,
        fitted_values=fitted,
        feature_mins=np.zeros(12, dtype=np.float32),
        feature_maxs=np.ones(12, dtype=np.float32),
    )
    lo = cal.predict_proba(MetaSignals(*([0.1] * 12)))
    hi = cal.predict_proba(MetaSignals(*([0.9] * 12)))
    assert 0.0 <= lo <= hi <= 1.0
