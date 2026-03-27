from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from app.domain.models import MetaSignals


class CalibratorBase:
    def predict_proba(self, signals: MetaSignals) -> float:  # pragma: no cover
        raise NotImplementedError


class LinearScoreIsotonicCalibrator(CalibratorBase):
    def __init__(self, weights: np.ndarray, intercept: float, isotonic_model: object) -> None:
        self.weights = weights.astype(np.float32)
        self.intercept = float(intercept)
        self.isotonic_model = isotonic_model

    def predict_proba(self, signals: MetaSignals) -> float:
        vector = np.asarray(signals.to_vector(), dtype=np.float32)
        raw = float(vector @ self.weights + self.intercept)
        return float(self.isotonic_model.predict([raw])[0])


class MonotoneScoreIsotonicCalibrator(CalibratorBase):
    def __init__(self, weights: np.ndarray, bias: float, feature_mins: np.ndarray, feature_maxs: np.ndarray, isotonic_model: object) -> None:
        self.weights = weights.astype(np.float32)
        self.bias = float(bias)
        self.feature_mins = feature_mins.astype(np.float32)
        self.feature_maxs = feature_maxs.astype(np.float32)
        self.isotonic_model = isotonic_model

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        denom = np.maximum(self.feature_maxs - self.feature_mins, 1e-6)
        out = (vector - self.feature_mins) / denom
        return np.clip(out, 0.0, 1.0)

    def predict_proba(self, signals: MetaSignals) -> float:
        vector = np.asarray(signals.to_vector(), dtype=np.float32)
        vector = self._normalize(vector)
        raw = float(vector @ self.weights + self.bias)
        return float(self.isotonic_model.predict([raw])[0])


class MultivariateIsotonicCalibrator(CalibratorBase):
    """
    Product-order multivariate isotonic calibrator.

    Training fits a value f_i to each calibration sample under the exact partial-order
    constraint used by multivariate isotonic regression: if x_i <= x_j elementwise,
    then f_i <= f_j. Prediction on unseen points uses the fitted lower/upper envelope
    induced by the training set:

      lower(x) = max{f_i : x_i <= x}
      upper(x) = min{f_i : x <= x_i}

    If both exist, the prediction is the midpoint of the feasible interval.
    If only one side exists, that side is used. If neither side exists, the nearest
    fitted sample is used as a conservative fallback.

    This preserves the core mathematical requirement from the architecture without
    collapsing the 12-signal space into a single learned score.
    """

    def __init__(
        self,
        train_features: np.ndarray,
        fitted_values: np.ndarray,
        feature_mins: np.ndarray,
        feature_maxs: np.ndarray,
        eps: float = 1e-6,
    ) -> None:
        self.train_features = np.asarray(train_features, dtype=np.float32)
        self.fitted_values = np.asarray(fitted_values, dtype=np.float32)
        self.feature_mins = np.asarray(feature_mins, dtype=np.float32)
        self.feature_maxs = np.asarray(feature_maxs, dtype=np.float32)
        self.eps = float(eps)

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        denom = np.maximum(self.feature_maxs - self.feature_mins, 1e-6)
        out = (vector - self.feature_mins) / denom
        return np.clip(out, 0.0, 1.0)

    def predict_proba(self, signals: MetaSignals) -> float:
        x = self._normalize(np.asarray(signals.to_vector(), dtype=np.float32))
        lower_mask = np.all(self.train_features <= (x[None, :] + self.eps), axis=1)
        upper_mask = np.all(self.train_features >= (x[None, :] - self.eps), axis=1)

        lower = float(np.max(self.fitted_values[lower_mask])) if np.any(lower_mask) else None
        upper = float(np.min(self.fitted_values[upper_mask])) if np.any(upper_mask) else None

        if lower is not None and upper is not None:
            if lower <= upper + 1e-6:
                return float(np.clip(0.5 * (lower + upper), 0.0, 1.0))
            # Numerical slack fallback: use clipped average while preserving [0,1].
            return float(np.clip(0.5 * (lower + upper), 0.0, 1.0))
        if lower is not None:
            return float(np.clip(lower, 0.0, 1.0))
        if upper is not None:
            return float(np.clip(upper, 0.0, 1.0))

        d = np.linalg.norm(self.train_features - x[None, :], axis=1)
        idx = int(np.argmin(d))
        return float(np.clip(self.fitted_values[idx], 0.0, 1.0))


class CalibratorFactory:
    @staticmethod
    def from_joblib(path: str | Path) -> CalibratorBase:
        payload = joblib.load(path)
        kind = payload.get('kind', 'linear_isotonic')
        if kind == 'linear_isotonic':
            return LinearScoreIsotonicCalibrator(
                weights=np.asarray(payload['weights'], dtype=np.float32),
                intercept=float(payload.get('intercept', 0.0)),
                isotonic_model=payload['isotonic_model'],
            )
        if kind == 'monotone_score_isotonic':
            return MonotoneScoreIsotonicCalibrator(
                weights=np.asarray(payload['weights'], dtype=np.float32),
                bias=float(payload.get('bias', 0.0)),
                feature_mins=np.asarray(payload['feature_mins'], dtype=np.float32),
                feature_maxs=np.asarray(payload['feature_maxs'], dtype=np.float32),
                isotonic_model=payload['isotonic_model'],
            )
        if kind == 'multivariate_isotonic':
            return MultivariateIsotonicCalibrator(
                train_features=np.asarray(payload['train_features'], dtype=np.float32),
                fitted_values=np.asarray(payload['fitted_values'], dtype=np.float32),
                feature_mins=np.asarray(payload['feature_mins'], dtype=np.float32),
                feature_maxs=np.asarray(payload['feature_maxs'], dtype=np.float32),
                eps=float(payload.get('eps', 1e-6)),
            )
        raise ValueError(f'Unsupported calibrator kind: {kind}')
