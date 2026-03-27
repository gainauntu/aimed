from __future__ import annotations

# ---------------------------------------------------------------------------
# Export trained artifacts as a self-contained runtime bundle.
#
# FIX: Added required --prototype-checkpoint argument (the .ckpt file saved
# by train_prototype.py).  The previous version only accepted
# --prototype-library (the JSON) and therefore export_backbone was called
# without any prototype model file, causing the runtime_service to fail at
# startup when it tried to load prototype_model.pt.
#
# The OOD backbone is exported from ImageNet-pretrained DINOFeatureBackbone
# (3-channel) — this is correct because at inference OODRuntimeBackbone
# receives single preprocessed ROI images (3-channel normalized).
# ---------------------------------------------------------------------------

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch

from models import DINOFeatureBackbone, TorchScriptBackboneWrapper, TorchScriptWrapper, build_model


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def export_tower(checkpoint_path: str, out_path: Path, num_classes: int, tower_name: str):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model = build_model(tower_name, num_classes=num_classes)
    model.load_state_dict(ckpt['model_state'], strict=False)
    model.eval()
    wrapper = TorchScriptWrapper(model).eval()
    example = torch.randn(1, 9, 288, 288)
    ts = torch.jit.trace(wrapper, (example,), strict=False)
    ts.save(str(out_path))
    print(f'  exported {tower_name} → {out_path}')


def export_prototype_model(checkpoint_path: str, out_path: Path, num_classes: int):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model = build_model('prototype', num_classes=num_classes)
    model.load_state_dict(ckpt['model_state'], strict=False)
    model.eval()
    wrapper = TorchScriptWrapper(model).eval()
    example = torch.randn(1, 9, 288, 288)
    ts = torch.jit.trace(wrapper, (example,), strict=False)
    ts.save(str(out_path))
    print(f'  exported prototype_model → {out_path}')


def export_ood_backbone(out_path: Path):
    """Export ImageNet-pretrained DINOFeatureBackbone (3-channel) for OOD.

    No checkpoint argument needed: ImageNet pretrained weights are strong
    OOD detectors.  Using pretrained (not DINO-adapted) here is intentional
    — the OOD backbone is trained on single 3-channel images while the
    DINOAdaptBackbone was trained on 9-channel fused pairs.
    """
    model = DINOFeatureBackbone(model_name='convnext_base', feature_dim=512)
    model.eval()
    wrapper = TorchScriptBackboneWrapper(model).eval()
    example = torch.randn(1, 3, 288, 288)   # 3-channel single ROI crop
    ts = torch.jit.trace(wrapper, (example,), strict=False)
    ts.save(str(out_path))
    print(f'  exported ood_backbone (3-ch ImageNet) → {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tower-a',              required=True)
    ap.add_argument('--tower-b',              required=True)
    ap.add_argument('--tower-c',              required=True)
    ap.add_argument('--prototype-checkpoint', required=True,
                    help='prototype_best.ckpt saved by train_prototype.py')
    ap.add_argument('--prototype-library',    required=True)
    ap.add_argument('--labels',               required=True)
    ap.add_argument('--class-profiles',       required=True)
    ap.add_argument('--variance-thresholds',  required=True)
    ap.add_argument('--ood-index-prefix',     required=True)
    ap.add_argument('--calibrator',           required=True)
    ap.add_argument('--output-dir',           required=True)
    ns = ap.parse_args()

    outdir = Path(ns.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    class_names = json.loads(Path(ns.labels).read_text(encoding='utf-8'))
    n = len(class_names)
    print(f'exporting runtime pack for {n} classes → {outdir}')

    export_tower(ns.tower_a, outdir / 'tower_a.pt', n, 'tower_a')
    export_tower(ns.tower_b, outdir / 'tower_b.pt', n, 'tower_b')
    export_tower(ns.tower_c, outdir / 'tower_c.pt', n, 'tower_c')
    export_prototype_model(ns.prototype_checkpoint, outdir / 'prototype_model.pt', n)
    export_ood_backbone(outdir / 'ood_backbone.pt')

    shutil.copy2(ns.labels,               outdir / 'labels.json')
    shutil.copy2(ns.prototype_library,    outdir / 'prototype_library.json')
    shutil.copy2(ns.class_profiles,       outdir / 'class_profiles.json')
    shutil.copy2(ns.variance_thresholds,  outdir / 'variance_thresholds.json')
    shutil.copy2(ns.calibrator,           outdir / 'calibrator.joblib')
    shutil.copy2(Path(ns.ood_index_prefix).with_suffix('.npy'), outdir / 'ood_index.npy')
    shutil.copy2(Path(ns.ood_index_prefix).with_suffix('.json'), outdir / 'ood_index.json')

    manifest = {'files': {}}
    for p in sorted(outdir.iterdir()):
        if p.is_file() and p.name != 'pack_manifest.json':
            manifest['files'][p.name] = {
                'bytes':  p.stat().st_size,
                'sha256': sha256_of(p),
            }
    (outdir / 'pack_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'runtime pack saved: {outdir}')


if __name__ == '__main__':
    main()
