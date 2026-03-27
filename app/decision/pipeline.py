from __future__ import annotations

import logging
import time
from typing import Iterable

import numpy as np

from app.core.config import AppConfig
from app.domain.models import DecisionResult, DecisionStatus, FailureGate, MetaSignals, TowerPrediction
from app.inference.towers import TowerAdapter
from app.observability.audit import AuditLogger
from app.preprocess.pipeline import PillPreprocessor, PreprocessingError
from app.preprocess.tta import TTAEngine
from app.verifiers.constraints import ConstraintGate
from app.verifiers.ood import OODVerifier
from app.verifiers.prototype import PrototypeVerifier
from .calibration import CalibratorBase

logger = logging.getLogger(__name__)


class PillDecisionPipeline:
    def __init__(
        self,
        config: AppConfig,
        preprocessor: PillPreprocessor,
        tta_engine: TTAEngine,
        towers: Iterable[TowerAdapter],
        ood_verifier: OODVerifier,
        prototype_verifier: PrototypeVerifier,
        constraint_gate: ConstraintGate,
        calibrator: CalibratorBase,
        audit_logger: AuditLogger,
    ) -> None:
        self.cfg = config
        self.preprocessor = preprocessor
        self.tta_engine = tta_engine
        self.towers = list(towers)
        self.ood_verifier = ood_verifier
        self.prototype_verifier = prototype_verifier
        self.constraint_gate = constraint_gate
        self.calibrator = calibrator
        self.audit_logger = audit_logger

    def predict(self, frame_a_bgr: np.ndarray, frame_b_bgr: np.ndarray) -> DecisionResult:
        started = time.perf_counter()
        towers: list[TowerPrediction] = []
        cross_image = None
        ood = None
        prototype = None
        constraint = None
        signals = None
        try:
            prep_a = self.preprocessor.process(frame_a_bgr)
            prep_b = self.preprocessor.process(frame_b_bgr)
            cross_image = self.preprocessor.cross_check(prep_a, prep_b)
            if not cross_image.passed:
                return self._finalize(DecisionResult(DecisionStatus.UNDECIDED, None, 0.0, FailureGate.CROSS_IMAGE, '; '.join(cross_image.reasons), towers, cross_image, ood, prototype, constraint, signals, 0.0), started)

            views = self.tta_engine.generate(prep_a.image_bgr, prep_b.image_bgr)
            towers = [tower.predict(views) for tower in self.towers]

            classes = {t.predicted_class for t in towers}
            if len(classes) != 1:
                return self._finalize(DecisionResult(DecisionStatus.UNDECIDED, None, 0.0, FailureGate.AGREEMENT, f'tower disagreement: {[t.predicted_class for t in towers]}', towers, cross_image, ood, prototype, constraint, signals, 0.0), started)

            predicted_class = next(iter(classes))
            var_threshold = self.cfg.class_specific_variance_thresholds.get(predicted_class, 0.025)
            unstable = [t for t in towers if t.tta_variance >= var_threshold]
            if unstable:
                return self._finalize(DecisionResult(DecisionStatus.UNDECIDED, predicted_class, 0.0, FailureGate.TTA_STABILITY, f'TTA variance threshold exceeded for {[t.tower_name for t in unstable]} (threshold={var_threshold:.4f})', towers, cross_image, ood, prototype, constraint, signals, 0.0), started)

            ood = self.ood_verifier.verify(prep_a.image_bgr, prep_b.image_bgr)
            if not ood.passed:
                return self._finalize(DecisionResult(DecisionStatus.UNDECIDED, predicted_class, 0.0, FailureGate.OOD, '; '.join(ood.reasons), towers, cross_image, ood, prototype, constraint, signals, 0.0), started)

            prototype = self.prototype_verifier.verify(predicted_class, views=views, embedding=np.asarray(towers[0].embedding, dtype=np.float32) if self.prototype_verifier.model is None else None)
            if not prototype.passed:
                return self._finalize(DecisionResult(DecisionStatus.UNDECIDED, predicted_class, 0.0, FailureGate.PROTOTYPE, '; '.join(prototype.reasons), towers, cross_image, ood, prototype, constraint, signals, 0.0), started)

            constraint = self.constraint_gate.verify(predicted_class, prep_a, prep_b)
            if not constraint.passed:
                return self._finalize(DecisionResult(DecisionStatus.UNDECIDED, predicted_class, 0.0, FailureGate.CONSTRAINT, '; '.join(constraint.reasons), towers, cross_image, ood, prototype, constraint, signals, 0.0), started)

            signals = self._build_signals(towers, cross_image.ssim, constraint.pass_ratio, prototype, ood, [prep_a.quality, prep_b.quality])
            calibrated_p = self.calibrator.predict_proba(signals)
            if calibrated_p < self.cfg.thresholds.accept_probability:
                return self._finalize(DecisionResult(DecisionStatus.UNDECIDED, predicted_class, calibrated_p, FailureGate.CALIBRATION, f'calibrated probability {calibrated_p:.4f} below {self.cfg.thresholds.accept_probability:.4f}', towers, cross_image, ood, prototype, constraint, signals, 0.0), started)

            return self._finalize(DecisionResult(DecisionStatus.CLASSIFIED, predicted_class, calibrated_p, None, None, towers, cross_image, ood, prototype, constraint, signals, 0.0), started)
        except PreprocessingError as exc:
            return self._finalize(DecisionResult(DecisionStatus.UNDECIDED, None, 0.0, FailureGate.STAGE1, str(exc), towers, cross_image, ood, prototype, constraint, signals, 0.0), started)
        except Exception as exc:
            logger.exception('Unhandled error during prediction')
            return self._finalize(DecisionResult(DecisionStatus.UNDECIDED, None, 0.0, FailureGate.SYSTEM, str(exc), towers, cross_image, ood, prototype, constraint, signals, 0.0), started)

    def _build_signals(self, towers: list[TowerPrediction], ssim_value: float, pass_ratio: float, prototype, ood, quality_reports) -> MetaSignals:
        confs = [t.top1_confidence for t in towers]
        margins = [t.top1_top2_margin for t in towers]
        variances = [t.tta_variance for t in towers]
        inv_ood = 1.0 / (1.0 + float((ood.mean_distance_a + ood.mean_distance_b) / 2.0))
        mean_quality = float(np.mean([q.blur_variance / max(q.blur_variance, 120.0) if q.blur_variance > 0 else 0.0 for q in quality_reports]))
        return MetaSignals(
            s1=float(confs[0]),
            s2=float(confs[1]),
            s3=float(confs[2]),
            s4=float(np.mean(margins)),
            s5=float(np.mean(variances)),
            s6=float(np.var(variances)),
            s7=float(prototype.similarity),
            s8=float(prototype.second_class_gap),
            s9=inv_ood,
            s10=float(ssim_value),
            s11=float(pass_ratio),
            s12=float(mean_quality),
        )

    def _finalize(self, result: DecisionResult, started: float) -> DecisionResult:
        result.elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.audit_logger.write(result)
        return result
