from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class Thresholds:
    min_blur_variance: float = 120.0
    min_frame_mean: float = 5.0
    max_frame_mean: float = 250.0
    max_stuck_column_ratio: float = 0.05
    background_diff_threshold: int = 15
    min_component_area: int = 2000
    max_component_area: int = 55000
    max_small_debris_area: int = 500
    min_large_component_area: int = 1500
    min_solidity: float = 0.75
    max_perimeter_ratio: float = 1.15
    max_coating_defect_ratio: float = 0.03
    max_area_ratio: float = 1.30
    max_color_distance: float = 60.0
    min_ssim: float = 0.18
    accept_probability: float = 0.995


@dataclass(slots=True)
class Paths:
    # background_reference is optional — empty string means use black reference
    background_reference: str = ''
    class_profiles: str = 'artifacts/class_profiles.json'
    prototype_library: str = 'artifacts/prototype_library.json'
    prototype_model_checkpoint: str = 'artifacts/prototype_model.pt'
    calibrator: str = 'artifacts/calibrator.joblib'
    ood_index: str = 'artifacts/ood_index'
    ood_backbone_checkpoint: str = 'artifacts/ood_backbone.pt'
    audit_log: str = 'logs/audit.jsonl'
    tower_a_checkpoint: str = 'artifacts/tower_a.pt'
    tower_b_checkpoint: str = 'artifacts/tower_b.pt'
    tower_c_checkpoint: str = 'artifacts/tower_c.pt'


@dataclass(slots=True)
class Runtime:
    target_size: int = 288
    expected_width: int = 352
    expected_height: int = 288
    use_gpu: bool = True
    api_title: str = 'Pill Dispensing Service'


@dataclass(slots=True)
class AppConfig:
    thresholds: Thresholds = field(default_factory=Thresholds)
    paths: Paths = field(default_factory=Paths)
    runtime: Runtime = field(default_factory=Runtime)
    class_labels: list[str] = field(default_factory=list)
    class_specific_variance_thresholds: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'AppConfig':
        with open(path, 'r', encoding='utf-8') as f:
            raw: dict[str, Any] = yaml.safe_load(f)
        return cls(
            thresholds=Thresholds(**raw.get('thresholds', {})),
            paths=Paths(**raw.get('paths', {})),
            runtime=Runtime(**raw.get('runtime', {})),
            class_labels=raw.get('class_labels', []),
            class_specific_variance_thresholds=raw.get('class_specific_variance_thresholds', {}),
        )

    def ensure_dirs(self) -> None:
        Path(self.paths.audit_log).parent.mkdir(parents=True, exist_ok=True)
