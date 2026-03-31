from __future__ import annotations

# ---------------------------------------------------------------------------
# Kaggle T4x2 training orchestrator — no background_reference.png required.
#
# The system now uses a pure black reference (np.zeros) for background
# subtraction throughout training and inference.  No reference file needs
# to be captured or provided.
# ---------------------------------------------------------------------------

import argparse
import json
import subprocess
from pathlib import Path

import sys
from pathlib import Path as _Path
_TRAINER_DIR = _Path(__file__).resolve().parent
if str(_TRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINER_DIR))

import joblib
import numpy as np
import shutil


def run(cmd: list[str]):
    print('\nRUN:', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def disk_free_gb() -> float:
    try:
        stat = shutil.disk_usage('/kaggle/working')
        return stat.free / (1024 ** 3)
    except Exception:
        return float('inf')


def check_disk(min_gb: float = 1.0, label: str = ''):
    free = disk_free_gb()
    tag = f' [{label}]' if label else ''
    print(f'  disk free{tag}: {free:.1f} GB')
    if free < min_gb:
        raise RuntimeError(
            f'Disk nearly full ({free:.2f} GB free). Cannot continue safely.')


def clear_pip_cache():
    """Remove pip download/wheel cache (~1-3 GB freed)."""
    run(['pip', 'cache', 'purge'])


def collect_fold_metrics(base: Path, tower: str, phases: list[str], n_folds: int):
    rows = []
    for phase in phases:
        for fold in range(n_folds):
            p = base / tower / phase / f'fold_{fold}' / 'metrics.json'
            if p.exists():
                hist = json.loads(p.read_text(encoding='utf-8'))
                best = max(hist, key=lambda r: r.get('acc', 0.0))
                rows.append({
                    'tower': tower, 'phase': phase, 'fold': fold,
                    'best_acc': best.get('acc', 0.0),
                    'best_loss': best.get('loss', 0.0),
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root',   required=True)
    ap.add_argument('--workdir',     default='/kaggle/working/pill_run')
    ap.add_argument('--dino-backbone', default='convnext_base')
    ns = ap.parse_args()

    work = Path(ns.workdir)
    work.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Derive n_folds from the dataset — correct LOOCV fix.
    # n_folds = max class size so every pair in the largest class is
    # held out exactly once.  With ~20 pairs per class this equals ~20.
    # This replaces the previous hardcoded --n-folds 20 which caused
    # repeat evaluations for smaller classes and skipped pairs in larger ones.
    # ---------------------------------------------------------------
    from pair_dataset import scan_pair_samples as _scan, loocv_n_folds as _loocv_n
    _all_samples, _ = _scan(ns.data_root)
    n_folds = _loocv_n(_all_samples)
    print(f'[LOOCV] derived n_folds={n_folds} from dataset (max class size)')

    # ---------------------------------------------------------------
    # Phase T1 — DINO self-supervised domain adaptation (9-channel)
    # ---------------------------------------------------------------
    run([
        'python', 'train_dino_ssl.py',
        '--data-root', ns.data_root,
        '--outdir',    str(work / 'dino'),
        '--backbone',  ns.dino_backbone,
    ])
    dino_ckpt = str(work / 'dino' / 'dino_adapted_backbone.ckpt')

    # Free pip cache immediately after install — recovers ~1-3 GB.
    clear_pip_cache()
    check_disk(min_gb=2.0, label='after DINO')

    # ---------------------------------------------------------------
    # Phases T2+T3 — LOOCV tower training
    # LOOCV folds save ONLY metrics.json, never model weights.
    # This keeps disk usage at ~0 MB per fold instead of ~300 MB.
    # ---------------------------------------------------------------
    phases = [
        ('frozen_head',       20, '1e-3'),
        ('unfreeze_top10',    20, '1e-3'),
        ('unfreeze_top10_30', 20, '1e-3'),
    ]

    for tower in ['tower_a', 'tower_b', 'tower_c']:
        for phase, epochs, lr in phases:
            for fold in range(n_folds):
                outdir = work / tower / phase / f'fold_{fold}'
                run([
                    'torchrun', '--standalone', '--nproc_per_node=2', 'train_tower.py',
                    '--tower',         tower,
                    '--data-root',     ns.data_root,
                    '--outdir',        str(outdir),
                    '--split-mode',    'loocv',
                    '--fold-index',    str(fold),
                    '--phase',         phase,
                    '--epochs',        str(epochs),
                    '--batch-size',    '8',
                    '--accum-steps',   '2',
                    '--num-workers',   '4',
                    '--lr',            lr,
                    '--backbone-init', dino_ckpt,
                ])

    # Choose best phase per tower from LOOCV results
    summary = []
    for tower in ['tower_a', 'tower_b', 'tower_c']:
        summary.extend(collect_fold_metrics(work, tower, [p for p, _, _ in phases], n_folds))
    (work / 'loocv_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    # Delete LOOCV fold directories — we only needed metrics.json from each.
    # LOOCV folds never saved model weights (train_tower.py skips torch.save
    # in loocv mode) so these directories are very small, but clean them up
    # anyway to keep the working directory tidy and disk usage clear.
    for tower in ['tower_a', 'tower_b', 'tower_c']:
        for phase, _, _ in phases:
            for fold in range(n_folds):
                fold_dir = work / tower / phase / f'fold_{fold}'
                if fold_dir.exists():
                    shutil.rmtree(fold_dir)
    print('LOOCV fold directories cleaned up')
    check_disk(min_gb=2.0, label='after LOOCV cleanup')

    final_choices = {}
    for tower in ['tower_a', 'tower_b', 'tower_c']:
        candidates: dict[str, list[float]] = {}
        for row in summary:
            if row['tower'] != tower:
                continue
            candidates.setdefault(row['phase'], []).append(row['best_acc'])
        best_phase = max(candidates.items(), key=lambda kv: sum(kv[1]) / max(1, len(kv[1])))[0]
        final_choices[tower] = best_phase
        run([
            'torchrun', '--standalone', '--nproc_per_node=2', 'train_tower.py',
            '--tower',         tower,
            '--data-root',     ns.data_root,
            '--outdir',        str(work / tower / 'final'),
            '--split-mode',    'random',
            '--phase',         best_phase,
            '--epochs',        '20',
            '--batch-size',    '8',
            '--accum-steps',   '2',
            '--num-workers',   '4',
            '--lr',            '1e-3',
            '--backbone-init', dino_ckpt,
        ])

    (work / 'final_training_choices.json').write_text(
        json.dumps(final_choices, ensure_ascii=False, indent=2), encoding='utf-8')

    check_disk(min_gb=2.0, label='after final tower retrain')

    # ---------------------------------------------------------------
    # Phase T4 — Prototypical Network episodic training
    # ---------------------------------------------------------------
    run([
        'python', 'train_prototype.py',
        '--data-root',          ns.data_root,
        '--outdir',             str(work / 'prototype'),
        '--episodes',           '50000',
        '--episodes-per-epoch', '1000',
        '--backbone-init',      dino_ckpt,
    ])

    # ---------------------------------------------------------------
    # Build class constraint profiles and OOD index
    # No --background-reference needed — black reference is used automatically.
    # ---------------------------------------------------------------
    run([
        'python', 'build_profiles_and_ood.py',
        '--data-root',         ns.data_root,
        '--output-profiles',   str(work / 'profiles.json'),
        '--output-ood-prefix', str(work / 'ood_index'),
    ])

    # ---------------------------------------------------------------
    # Pre-export with placeholder calibrator for signal generation
    # ---------------------------------------------------------------
    pre_export = work / 'runtime_bundle_pre'

    placeholder_cal = {
        'kind': 'multivariate_isotonic',
        'train_features': np.zeros((1, 12), dtype=np.float32),
        'fitted_values': np.asarray([0.5], dtype=np.float32),
        'feature_mins': np.zeros(12, dtype=np.float32),
        'feature_maxs': np.ones(12, dtype=np.float32),
        'eps': 1e-9,
    }
    placeholder_cal_path = str(work / 'calibrator_placeholder.joblib')
    joblib.dump(placeholder_cal, placeholder_cal_path)
    placeholder_var_path = str(work / 'variance_thresholds_placeholder.json')
    (work / 'variance_thresholds_placeholder.json').write_text('{}', encoding='utf-8')

    run([
        'python', 'export_runtime_pack.py',
        '--tower-a',              str(work / 'tower_a' / 'final' / 'best.ckpt'),
        '--tower-b',              str(work / 'tower_b' / 'final' / 'best.ckpt'),
        '--tower-c',              str(work / 'tower_c' / 'final' / 'best.ckpt'),
        '--prototype-checkpoint', str(work / 'prototype' / 'prototype_best.ckpt'),
        '--prototype-library',    str(work / 'prototype' / 'prototype_library.json'),
        '--labels',               str(work / 'tower_a' / 'final' / 'labels.json'),
        '--class-profiles',       str(work / 'profiles.json'),
        '--variance-thresholds',  placeholder_var_path,
        '--ood-index-prefix',     str(work / 'ood_index'),
        '--calibrator',           placeholder_cal_path,
        '--output-dir',           str(pre_export),
    ])

    # Sync into runtime_service/models/current for calibration signal generation
    runtime_current = Path(__file__).resolve().parents[1] / 'runtime_service' / 'models' / 'current'
    runtime_current.mkdir(parents=True, exist_ok=True)
    for f in pre_export.iterdir():
        shutil.copy2(f, runtime_current / f.name)

    from configs_for_calibration import write_calibration_config
    cal_config_path = str(work / 'cal_config.yaml')
    write_calibration_config(
        str(runtime_current), cal_config_path,
        labels_path=str(work / 'tower_a' / 'final' / 'labels.json'),
    )

    # ---------------------------------------------------------------
    # Generate calibration signals (LOOCV inference pass)
    # ---------------------------------------------------------------
    signals_path = str(work / 'signals.jsonl')
    run([
        'python', 'generate_calibration_signals.py',
        '--data-root',     ns.data_root,
        '--config-path',   cal_config_path,
        '--output-signals', signals_path,
    ])

    # ---------------------------------------------------------------
    # Fit the calibrated meta-decision engine
    # ---------------------------------------------------------------
    run([
        'python', 'build_calibrator.py',
        '--signals-jsonl',              signals_path,
        '--output',                     str(work / 'calibrator.joblib'),
        '--variance-thresholds-output', str(work / 'variance_thresholds.json'),
        '--reliability-output',         str(work / 'reliability.json'),
    ])

    # ---------------------------------------------------------------
    # Final runtime pack — no background file included
    # ---------------------------------------------------------------
    run([
        'python', 'export_runtime_pack.py',
        '--tower-a',              str(work / 'tower_a' / 'final' / 'best.ckpt'),
        '--tower-b',              str(work / 'tower_b' / 'final' / 'best.ckpt'),
        '--tower-c',              str(work / 'tower_c' / 'final' / 'best.ckpt'),
        '--prototype-checkpoint', str(work / 'prototype' / 'prototype_best.ckpt'),
        '--prototype-library',    str(work / 'prototype' / 'prototype_library.json'),
        '--labels',               str(work / 'tower_a' / 'final' / 'labels.json'),
        '--class-profiles',       str(work / 'profiles.json'),
        '--variance-thresholds',  str(work / 'variance_thresholds.json'),
        '--ood-index-prefix',     str(work / 'ood_index'),
        '--calibrator',           str(work / 'calibrator.joblib'),
        '--output-dir',           str(work / 'runtime_bundle'),
    ])

    print('\n=== training and export complete ===')
    print('runtime bundle:', work / 'runtime_bundle')


if __name__ == '__main__':
    main()
