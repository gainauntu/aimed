from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import shutil
import yaml


@dataclass(slots=True)
class RuntimePack:
    bundle_dir: Path
    labels: list[str]
    variance_thresholds: dict[str, float]


REQUIRED_FILES = [
    'tower_a.pt',
    'tower_b.pt',
    'tower_c.pt',
    'prototype_model.pt',
    'ood_backbone.pt',
    'labels.json',
    'variance_thresholds.json',
    'prototype_library.json',
    'class_profiles.json',
    'ood_index.npy',
    'ood_index.json',
    'calibrator.joblib',
    'pack_manifest.json',
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_pack(bundle_dir: str | Path) -> RuntimePack:
    bundle = Path(bundle_dir)
    labels = json.loads((bundle / 'labels.json').read_text(encoding='utf-8'))
    variance_thresholds = json.loads((bundle / 'variance_thresholds.json').read_text(encoding='utf-8'))
    return RuntimePack(bundle_dir=bundle, labels=labels, variance_thresholds=variance_thresholds)


def validate_pack(pack: RuntimePack) -> tuple[bool, list[str]]:
    errs: list[str] = []
    if not pack.bundle_dir.exists():
        return False, [f'bundle directory not found: {pack.bundle_dir}']
    for name in REQUIRED_FILES:
        if not (pack.bundle_dir / name).exists():
            errs.append(f'missing file: {name}')
    manifest_path = pack.bundle_dir / 'pack_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        files = manifest.get('files', {})
        for name, meta in files.items():
            p = pack.bundle_dir / name
            if not p.exists():
                errs.append(f'manifest references missing file: {name}')
                continue
            actual = sha256_of(p)
            expected = str(meta.get('sha256', ''))
            if expected and expected != actual:
                errs.append(f'sha256 mismatch: {name}')
    return len(errs) == 0, errs


def sync_pack_into_root(pack: RuntimePack, repo_root: str | Path) -> Path:
    repo_root = Path(repo_root)
    artifacts = repo_root / 'artifacts'
    artifacts.mkdir(parents=True, exist_ok=True)

    mapping = {
        'tower_a.pt':             'tower_a.pt',
        'tower_b.pt':             'tower_b.pt',
        'tower_c.pt':             'tower_c.pt',
        'prototype_model.pt':     'prototype_model.pt',
        'ood_backbone.pt':        'ood_backbone.pt',
        'prototype_library.json': 'prototype_library.json',
        'class_profiles.json':    'class_profiles.json',
        'ood_index.npy':          'ood_index.npy',
        'ood_index.json':         'ood_index.json',
        'calibrator.joblib':      'calibrator.joblib',
    }
    for src_name, dst_name in mapping.items():
        shutil.copy2(pack.bundle_dir / src_name, artifacts / dst_name)

    cfg = {
        'thresholds': {},
        'paths': {
            'background_reference': '',
            'class_profiles': 'artifacts/class_profiles.json',
            'prototype_library': 'artifacts/prototype_library.json',
            'prototype_model_checkpoint': 'artifacts/prototype_model.pt',
            'calibrator': 'artifacts/calibrator.joblib',
            'ood_index': 'artifacts/ood_index',
            'ood_backbone_checkpoint': 'artifacts/ood_backbone.pt',
            'audit_log': 'logs/runtime_audit.jsonl',
            'tower_a_checkpoint': 'artifacts/tower_a.pt',
            'tower_b_checkpoint': 'artifacts/tower_b.pt',
            'tower_c_checkpoint': 'artifacts/tower_c.pt',
        },
        'runtime': {
            'target_size': 288,
            'expected_width': 352,
            'expected_height': 288,
            'use_gpu': True,
            'api_title': 'Pill Runtime API',
        },
        'class_labels': pack.labels,
        'class_specific_variance_thresholds': pack.variance_thresholds,
    }
    cfg_path = repo_root / 'configs' / 'runtime_bridge.yaml'
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    return cfg_path
