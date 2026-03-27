from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ShapeClass(str, Enum):
    ROUND = "ROUND"
    OVAL = "OVAL"
    OBLONG = "OBLONG"
    CAPSULE = "CAPSULE"
    UNKNOWN = "UNKNOWN"


class DecisionStatus(str, Enum):
    CLASSIFIED = "CLASSIFIED"
    UNDECIDED = "UNDECIDED"
    REJECTED = "REJECTED"


class FailureGate(str, Enum):
    STAGE0 = "stage0"
    STAGE1 = "stage1"
    CROSS_IMAGE = "cross_image"
    AGREEMENT = "agreement"
    TTA_STABILITY = "tta_stability"
    OOD = "ood"
    PROTOTYPE = "prototype"
    CONSTRAINT = "constraint"
    CALIBRATION = "calibration"
    SYSTEM = "system"


@dataclass(slots=True)
class PhysicalStats:
    hue_mean: float
    hue_std: float
    sat_mean: float
    sat_std: float
    val_mean: float
    val_std: float
    pixel_area_proxy: float
    aspect_ratio: float
    convexity_ratio: float
    edge_density: float
    shape_class: ShapeClass


@dataclass(slots=True)
class FrameQualityReport:
    blur_variance: float
    frame_mean: float
    stuck_column_ratio: float
    has_signal: bool
    passed: bool
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PreprocessedImage:
    image_bgr: Any
    mask: Any
    stats: PhysicalStats
    quality: FrameQualityReport
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CrossImageReport:
    passed: bool
    reasons: list[str]
    color_distance: float
    size_ratio: float
    ssim: float


@dataclass(slots=True)
class TowerPrediction:
    tower_name: str
    predicted_class: str
    top1_confidence: float
    top1_top2_margin: float
    tta_variance: float
    probabilities: dict[str, float]
    embedding: list[float]


@dataclass(slots=True)
class OODReport:
    passed: bool
    mean_distance_a: float
    mean_distance_b: float
    threshold: float
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PrototypeReport:
    passed: bool
    similarity: float
    distance_to_centroid: float
    second_class_gap: float
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ConstraintCheck:
    name: str
    passed: bool
    observed: float | str
    expected: str


@dataclass(slots=True)
class ConstraintReport:
    passed: bool
    pass_ratio: float
    checks_a: list[ConstraintCheck]
    checks_b: list[ConstraintCheck]
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MetaSignals:
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float
    s7: float
    s8: float
    s9: float
    s10: float
    s11: float
    s12: float

    def to_vector(self) -> list[float]:
        return [
            self.s1,
            self.s2,
            self.s3,
            self.s4,
            self.s5,
            self.s6,
            self.s7,
            self.s8,
            self.s9,
            self.s10,
            self.s11,
            self.s12,
        ]


@dataclass(slots=True)
class DecisionResult:
    status: DecisionStatus
    predicted_class: str | None
    calibrated_p: float
    failure_gate: FailureGate | None
    failure_reason: str | None
    towers: list[TowerPrediction]
    cross_image: CrossImageReport | None
    ood: OODReport | None
    prototype: PrototypeReport | None
    constraint: ConstraintReport | None
    signals: MetaSignals | None
    elapsed_ms: float
    audit_extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["failure_gate"] = self.failure_gate.value if self.failure_gate else None
        return payload
