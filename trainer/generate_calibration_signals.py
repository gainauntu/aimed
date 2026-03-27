from __future__ import annotations

# ---------------------------------------------------------------------------
# Generate calibration signals via LOOCV inference pass.
#
# FIX (Bug 4): The original notebook_launcher.py passed an empty signals.jsonl
# to build_calibrator.py because no script ever generated the 12-signal records
# that the calibrator expects.  As a result the calibrator was fitted on zero
# samples and returned a constant P≈0 at inference, causing every valid
# classification to fail Gate 7.
#
# No background_reference.png required — black reference used automatically.
# ---------------------------------------------------------------------------

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Add repo root to path so we can import from app/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pair_dataset import scan_pair_samples, read_image_unicode


def _build_pipeline(config_path: str, allow_dummy: bool = False):
    """Bootstrap the inference pipeline from a config file."""
    if allow_dummy:
        os.environ['PILL_ALLOW_DUMMY'] = '1'
    from app.core.bootstrap import build_pipeline
    return build_pipeline(config_path)


def run_inference_on_pair(pipeline, img_a_bgr: np.ndarray, img_b_bgr: np.ndarray):
    """Run the full pipeline and return the DecisionResult."""
    return pipeline.predict(img_a_bgr, img_b_bgr)


def build_signals_list(result) -> list[float] | None:
    """Extract the 12-signal list from a DecisionResult.
    Returns None if the result was rejected before signals were computed."""
    if result.signals is None:
        return None
    s = result.signals
    return [s.s1, s.s2, s.s3, s.s4, s.s5, s.s6,
            s.s7, s.s8, s.s9, s.s10, s.s11, s.s12]


def main():
    ap = argparse.ArgumentParser(
        description='LOOCV inference pass to produce signals.jsonl for calibration')
    ap.add_argument('--data-root',           required=True,
                    help='Same data root used for training')
    ap.add_argument('--config-path',         required=True,
                    help='Runtime config YAML (must point to trained artifacts)')
    ap.add_argument('--output-signals',      required=True,
                    help='Output path for signals.jsonl')
    ap.add_argument('--min-pairs-per-class', type=int, default=2,
                    help='Skip classes with fewer than this many pairs')
    ns = ap.parse_args()

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    pair_samples, class_names = scan_pair_samples(ns.data_root)
    by_class: dict[str, list] = {c: [] for c in class_names}
    for s in pair_samples:
        by_class[s.class_name].append(s)

    # ------------------------------------------------------------------
    # Build inference pipeline (black background used automatically via
    # bootstrap.py when background_reference path is empty string)
    # ------------------------------------------------------------------
    print(f'loading inference pipeline from {ns.config_path} ...')
    pipeline = _build_pipeline(ns.config_path)
    print('pipeline loaded.')

    # ------------------------------------------------------------------
    # LOOCV inference: hold out each pair, classify, record signals
    # ------------------------------------------------------------------
    output_path = Path(ns.output_signals)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    total_skipped = 0

    with output_path.open('w', encoding='utf-8') as fout:
        for cls_name, samples in by_class.items():
            if len(samples) < ns.min_pairs_per_class:
                print(f'  skip {cls_name}: only {len(samples)} pair(s)')
                continue

            print(f'  {cls_name}: {len(samples)} pairs → LOOCV ...')
            for hold_idx in range(len(samples)):
                held = samples[hold_idx]
                paths = held.image_paths

                if len(paths) < 2:
                    total_skipped += 1
                    continue

                try:
                    img_a = read_image_unicode(paths[0])
                    img_b = read_image_unicode(paths[1])
                except Exception as exc:
                    print(f'    skip pair {held.pair_id}: {exc}')
                    total_skipped += 1
                    continue

                result = run_inference_on_pair(pipeline, img_a, img_b)
                signals = build_signals_list(result)
                if signals is None:
                    signals = [0.0] * 12

                predicted = result.predicted_class
                correct = int(predicted == cls_name) if predicted is not None else 0

                tta_variance_mean = 0.0
                if result.tower_predictions:
                    variances = [t.tta_variance for t in result.tower_predictions
                                 if t is not None]
                    if variances:
                        tta_variance_mean = float(np.mean(variances))

                record = {
                    'signals':           signals,
                    'correct':           correct,
                    'predicted_class':   predicted or 'UNDECIDED',
                    'true_class':        cls_name,
                    'pair_id':           held.pair_id,
                    'tta_variance_mean': tta_variance_mean,
                    'decision':          result.status.value,
                    'failure_gate':      (result.failure_gate.value
                                         if result.failure_gate else None),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                total_written += 1

    print(f'\nsignals written: {total_written}  skipped: {total_skipped}')
    print(f'saved → {output_path}')


if __name__ == '__main__':
    main()
