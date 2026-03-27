from __future__ import annotations

from pathlib import Path
import os

from app.core.config import AppConfig
from app.decision.calibration import CalibratorFactory
from app.decision.pipeline import PillDecisionPipeline
from app.inference.towers import DummyTower, TorchScriptTower
from app.observability.audit import AuditLogger
from app.preprocess.background import BackgroundReference
from app.preprocess.pipeline import PillPreprocessor
from app.preprocess.tta import TTAEngine
from app.verifiers.constraints import ConstraintGate, ConstraintProfiles
from app.verifiers.ood import OODIndex, OODRuntimeBackbone, OODVerifier
from app.verifiers.prototype import PrototypeLibrary, PrototypeModel, PrototypeVerifier


def build_pipeline(config_path: str | Path) -> PillDecisionPipeline:
    cfg = AppConfig.from_yaml(config_path)
    cfg.ensure_dirs()

    background = (
        BackgroundReference.from_file(cfg.paths.background_reference)
        if cfg.paths.background_reference
        else BackgroundReference.black(
            height=cfg.runtime.expected_height,
            width=cfg.runtime.expected_width,
        )
    )
    preprocessor = PillPreprocessor(cfg, background)
    tta_engine = TTAEngine()

    allow_dummy = os.environ.get('PILL_ALLOW_DUMMY', '0') == '1'
    if allow_dummy:
        towers = [
            DummyTower('tower_a', cfg.class_labels or ['demo_pill'], bias_index=0),
            DummyTower('tower_b', cfg.class_labels or ['demo_pill'], bias_index=0),
            DummyTower('tower_c', cfg.class_labels or ['demo_pill'], bias_index=0),
        ]
    else:
        towers = [
            TorchScriptTower('tower_a', cfg.class_labels, cfg.paths.tower_a_checkpoint),
            TorchScriptTower('tower_b', cfg.class_labels, cfg.paths.tower_b_checkpoint),
            TorchScriptTower('tower_c', cfg.class_labels, cfg.paths.tower_c_checkpoint),
        ]

    ood_backbone = None
    if Path(cfg.paths.ood_backbone_checkpoint).exists():
        ood_backbone = OODRuntimeBackbone(cfg.paths.ood_backbone_checkpoint)
    ood = OODVerifier(OODIndex.from_prefix(cfg.paths.ood_index), backbone=ood_backbone)

    proto_model = None
    if Path(cfg.paths.prototype_model_checkpoint).exists():
        proto_model = PrototypeModel(cfg.paths.prototype_model_checkpoint)
    prototype = PrototypeVerifier(PrototypeLibrary.from_json(cfg.paths.prototype_library), model=proto_model)

    constraints = ConstraintGate(ConstraintProfiles.from_json(cfg.paths.class_profiles))
    calibrator = CalibratorFactory.from_joblib(cfg.paths.calibrator)
    audit = AuditLogger(cfg.paths.audit_log)

    return PillDecisionPipeline(
        config=cfg,
        preprocessor=preprocessor,
        tta_engine=tta_engine,
        towers=towers,
        ood_verifier=ood,
        prototype_verifier=prototype,
        constraint_gate=constraints,
        calibrator=calibrator,
        audit_logger=audit,
    )
