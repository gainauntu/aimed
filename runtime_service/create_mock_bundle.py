from __future__ import annotations

from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import cv2
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression


class DummyTower(nn.Module):
    def __init__(self, num_classes: int, emb_dim: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim

    def forward(self, x):
        b = x.shape[0]
        logits = torch.zeros((b, self.num_classes), dtype=torch.float32, device=x.device)
        logits[:, 0] = 5.0
        emb = torch.ones((b, self.emb_dim), dtype=torch.float32, device=x.device)
        return {'logits': logits, 'embedding': emb}


class DummyBackbone(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        b = x.shape[0]
        feat = torch.ones((b, self.dim), dtype=torch.float32, device=x.device)
        return {'feature': feat}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    out = Path(__file__).resolve().parent / 'models/current'
    if out.exists():
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    labels = ['demo_pill']
    fused_ex = torch.randn(1, 9, 288, 288)
    img_ex = torch.randn(1, 3, 288, 288)
    for name in ['tower_a', 'tower_b', 'tower_c', 'prototype_model']:
        model = torch.jit.trace(DummyTower(len(labels)).eval(), (fused_ex,), strict=False)
        model.save(str(out / f'{name}.pt'))
    ood_model = torch.jit.trace(DummyBackbone(256).eval(), (img_ex,), strict=False)
    ood_model.save(str(out / 'ood_backbone.pt'))

    (out / 'labels.json').write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'variance_thresholds.json').write_text(json.dumps({'demo_pill': 1.0}, ensure_ascii=False, indent=2), encoding='utf-8')

    proto = {
        'demo_pill': {'centroid': [1.0] * 256, 'p95_distance': 999.0, 'p99_distance': 1000.0, 'min_similarity': -1.0, 'second_class_margin': -1000.0},
    }
    (out / 'prototype_library.json').write_text(json.dumps(proto, ensure_ascii=False, indent=2), encoding='utf-8')

    profiles = {
        'demo_pill': {'hue_mean': 0, 'hue_std': 180, 'sat_mean': 127, 'sat_std': 127, 'val_mean': 127, 'val_std': 127, 'aspect_mean': 1.0, 'aspect_std': 10.0, 'area_mean': 10000, 'area_std': 100000, 'edge_mean': 10, 'edge_std': 1000, 'expected_shape': 'OVAL'},
    }
    (out / 'class_profiles.json').write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding='utf-8')

    np.save(out / 'ood_index.npy', np.array([[0.5] * 256], dtype=np.float32))
    (out / 'ood_index.json').write_text(json.dumps({'threshold': 5.0}, ensure_ascii=False, indent=2), encoding='utf-8')

    iso = IsotonicRegression(out_of_bounds='clip')
    xs = np.linspace(0, 1, 10)
    iso.fit(xs, xs)
    joblib.dump({'weights': np.ones(12, dtype=np.float32) / 12.0, 'intercept': 0.0, 'isotonic_model': iso}, out / 'calibrator.joblib')

    # No background_reference.png needed — black reference used automatically.

    manifest = {'files': {}}
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name != 'pack_manifest.json':
            manifest['files'][p.name] = {'bytes': p.stat().st_size, 'sha256': sha256_of(p)}
    (out / 'pack_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('mock bundle created:', out)


if __name__ == '__main__':
    main()
