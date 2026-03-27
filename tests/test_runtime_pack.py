from __future__ import annotations

from runtime_service.pack import load_pack, validate_pack


def test_mock_runtime_bundle_is_valid(tmp_path):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    required = [
        'tower_a.pt', 'tower_b.pt', 'tower_c.pt', 'prototype_model.pt', 'ood_backbone.pt',
        'labels.json', 'variance_thresholds.json', 'prototype_library.json', 'class_profiles.json',
        'ood_index.npy', 'ood_index.json', 'calibrator.joblib', 'pack_manifest.json',
    ]
    for name in required:
        p = bundle / name
        if name.endswith('.json'):
            p.write_text('{}', encoding='utf-8')
        else:
            p.write_bytes(b'x')
    (bundle / 'labels.json').write_text('["a"]', encoding='utf-8')
    (bundle / 'variance_thresholds.json').write_text('{"a": 1.0}', encoding='utf-8')
    pack = load_pack(bundle)
    ok, errs = validate_pack(pack)
    assert ok
    assert errs == []
