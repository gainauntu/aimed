from __future__ import annotations
"""Write a temporary runtime config for calibration signal generation.
No background_reference path needed — empty string means black reference.
"""

import json
from pathlib import Path


def write_calibration_config(
    bundle_dir: str,
    output_path: str,
    labels_path: str,
) -> None:
    labels = json.loads(Path(labels_path).read_text(encoding='utf-8'))
    bundle = Path(bundle_dir)

    yaml_text = f"""thresholds:
  min_blur_variance: 120.0
  min_frame_mean: 5.0
  max_frame_mean: 250.0
  max_stuck_column_ratio: 0.05
  background_diff_threshold: 15
  min_component_area: 2000
  max_component_area: 55000
  max_small_debris_area: 500
  min_large_component_area: 1500
  min_solidity: 0.75
  max_perimeter_ratio: 1.15
  max_coating_defect_ratio: 0.03
  max_area_ratio: 1.30
  max_color_distance: 60.0
  min_ssim: 0.18
  accept_probability: 0.5
paths:
  background_reference: ''
  class_profiles: {bundle / 'class_profiles.json'}
  prototype_library: {bundle / 'prototype_library.json'}
  calibrator: {bundle / 'calibrator.joblib'}
  ood_index: {bundle / 'ood_index'}
  audit_log: /tmp/pill_cal_audit.jsonl
  tower_a_checkpoint: {bundle / 'tower_a.pt'}
  tower_b_checkpoint: {bundle / 'tower_b.pt'}
  tower_c_checkpoint: {bundle / 'tower_c.pt'}
  ood_backbone_checkpoint: {bundle / 'ood_backbone.pt'}
  prototype_model_checkpoint: {bundle / 'prototype_model.pt'}
runtime:
  target_size: 288
  expected_width: 352
  expected_height: 288
  use_gpu: true
  api_title: Pill Cal Config
class_labels: {json.dumps(labels)}
class_specific_variance_thresholds: {{}}
"""
    Path(output_path).write_text(yaml_text, encoding='utf-8')
