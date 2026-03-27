from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


def load_rows(path: str | Path):
    rows = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        ts = row.get('timestamp')
        try:
            row['_dt'] = datetime.fromisoformat(ts.replace('Z', '+00:00')) if ts else None
        except Exception:
            row['_dt'] = None
        rows.append(row)
    return rows


def mean_tta(row: dict) -> float | None:
    towers = row.get('towers', {})
    vals = [float(v.get('tta_var', 0.0)) for v in towers.values() if v is not None]
    return float(np.mean(vals)) if vals else None


def reviewed_accuracy(rows: list[dict]) -> tuple[float | None, float | None]:
    probs, accs = [], []
    for r in rows:
        if 'review_correct' not in r:
            continue
        probs.append(float(r.get('calibrated_p', 0.0)))
        accs.append(float(1.0 if r['review_correct'] else 0.0))
    if not probs:
        return None, None
    return float(np.mean(probs)), float(np.mean(accs))


def within(rows: list[dict], start: datetime, end: datetime):
    out = []
    for r in rows:
        dt = r.get('_dt')
        if dt is None:
            continue
        if start <= dt < end:
            out.append(r)
    return out


def class_rejection_alerts(rows: list[dict], now: datetime):
    alerts = []
    recent = within(rows, now - timedelta(days=7), now)
    baseline = within(rows, now - timedelta(days=35), now - timedelta(days=7))
    by_cls_recent = defaultdict(list)
    by_cls_base = defaultdict(list)
    for r in recent:
        by_cls_recent[r.get('predicted_class') or 'UNDECIDED'].append(r)
    for r in baseline:
        by_cls_base[r.get('predicted_class') or 'UNDECIDED'].append(r)
    for cls, items in by_cls_recent.items():
        if len(items) < 10:
            continue
        rej_recent = sum(1 for x in items if x.get('decision') != 'CLASSIFIED') / max(1, len(items))
        base_items = by_cls_base.get(cls, [])
        rej_base = sum(1 for x in base_items if x.get('decision') != 'CLASSIFIED') / max(1, len(base_items)) if base_items else 0.0
        if rej_recent > max(0.10, 2.0 * max(rej_base, 1e-6)):
            tta_vals = [mean_tta(x) for x in items]
            ood_vals = [np.mean([float(x.get('ood_distance_A', 0.0)), float(x.get('ood_distance_B', 0.0))]) for x in items if x.get('ood_distance_A') is not None]
            alerts.append({
                'type': 'class_rejection_rate_rising',
                'class': cls,
                'recent_rate': float(rej_recent),
                'baseline_rate': float(rej_base),
                'diagnostic_hint': 'tta_variance rising -> distribution drift; ood_distance rising -> genuinely new variant',
                'recent_mean_tta_variance': float(np.nanmean(tta_vals)) if tta_vals else None,
                'recent_mean_ood_distance': float(np.mean(ood_vals)) if ood_vals else None,
                'recommended_response': 'collect new lot images; update constraint profile and prototype library',
            })
    return alerts


def global_tta_alert(rows: list[dict], now: datetime):
    recent = [mean_tta(r) for r in within(rows, now - timedelta(days=14), now)]
    baseline = [mean_tta(r) for r in within(rows, now - timedelta(days=42), now - timedelta(days=14))]
    recent = [x for x in recent if x is not None]
    baseline = [x for x in baseline if x is not None]
    if len(recent) < 20 or len(baseline) < 20:
        return []
    mr, mb = float(np.mean(recent)), float(np.mean(baseline))
    if mr > 1.20 * max(mb, 1e-6):
        return [{
            'type': 'global_tta_variance_rising',
            'recent_mean_tta_variance': mr,
            'baseline_mean_tta_variance': mb,
            'recommended_response': 'hardware inspection; recapture background; verify camera mount and lighting',
        }]
    return []


def tower_disagreement_alert(rows: list[dict], now: datetime):
    recent = within(rows, now - timedelta(days=14), now)
    baseline = within(rows, now - timedelta(days=42), now - timedelta(days=14))

    def disagreement_counts(items: list[dict]):
        counts = Counter()
        total = Counter()
        for r in items:
            towers = r.get('towers', {})
            preds = {k: v.get('top1') for k, v in towers.items() if isinstance(v, dict)}
            if len(preds) < 3:
                continue
            maj = Counter(preds.values()).most_common(1)[0][0]
            for name, pred in preds.items():
                total[name] += 1
                if pred != maj:
                    counts[name] += 1
        return {k: counts[k] / max(1, total[k]) for k in total}

    rr = disagreement_counts(recent)
    rb = disagreement_counts(baseline)
    alerts = []
    for tower, rate in rr.items():
        base = rb.get(tower, 0.0)
        if rate > max(0.05, 2.0 * max(base, 1e-6)):
            alerts.append({
                'type': 'tower_disagreement_rate_rising',
                'tower': tower,
                'recent_rate': float(rate),
                'baseline_rate': float(base),
                'recommended_response': f'inspect {tower} error patterns; retrain {tower} only if confirmed',
            })
    return alerts


def ood_spike_alert(rows: list[dict], now: datetime):
    recent = within(rows, now - timedelta(days=1), now)
    baseline = within(rows, now - timedelta(days=31), now - timedelta(days=1))
    rr = sum(1 for r in recent if r.get('rejection_gate') == 'ood') / max(1, len(recent)) if recent else 0.0
    rb = sum(1 for r in baseline if r.get('rejection_gate') == 'ood') / max(1, len(baseline)) if baseline else 0.0
    if recent and rr > max(0.05, 3.0 * max(rb, 1e-6)):
        return [{
            'type': 'ood_rejection_rate_spike',
            'recent_rate': float(rr),
            'baseline_rate': float(rb),
            'recommended_response': 'sample rejected images; verify preprocessing; check reference background for drift',
        }]
    return []


def constraint_after_agreement_alert(rows: list[dict], now: datetime):
    recent = within(rows, now - timedelta(days=14), now)
    by_cls = defaultdict(list)
    for r in recent:
        if r.get('rejection_gate') == 'constraint':
            towers = r.get('towers', {})
            preds = [v.get('top1') for v in towers.values() if isinstance(v, dict)]
            if preds and len(set(preds)) == 1:
                by_cls[preds[0]].append(r)
    alerts = []
    for cls, items in by_cls.items():
        if len(items) >= 5:
            alerts.append({
                'type': 'constraint_gate_failing_after_tower_agreement',
                'class': cls,
                'count_recent': len(items),
                'recommended_response': 'update class constraint profile with new lot images; prototype refresh if needed',
            })
    return alerts


def calibration_inflation_alert(rows: list[dict], now: datetime):
    recent = within(rows, now - timedelta(days=30), now)
    mean_prob, mean_acc = reviewed_accuracy(recent)
    if mean_prob is None or mean_acc is None:
        return []
    if mean_prob - mean_acc > 0.02:
        return [{
            'type': 'calibration_p_value_inflation',
            'mean_predicted_probability': mean_prob,
            'mean_reviewed_accuracy': mean_acc,
            'recommended_response': 'recalibrate multivariate isotonic regression using reviewed audit samples',
        }]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audit-log', required=True)
    ap.add_argument('--output', default='alerts_summary.json')
    ns = ap.parse_args()

    rows = load_rows(ns.audit_log)
    if not rows:
        Path(ns.output).write_text(json.dumps({'alerts': []}, ensure_ascii=False, indent=2), encoding='utf-8')
        print('saved', ns.output)
        return

    dated = [r['_dt'] for r in rows if r.get('_dt') is not None]
    now = max(dated) if dated else datetime.now(timezone.utc)
    alerts = []
    alerts.extend(class_rejection_alerts(rows, now))
    alerts.extend(global_tta_alert(rows, now))
    alerts.extend(tower_disagreement_alert(rows, now))
    alerts.extend(ood_spike_alert(rows, now))
    alerts.extend(constraint_after_agreement_alert(rows, now))
    alerts.extend(calibration_inflation_alert(rows, now))

    payload = {'alerts': alerts, 'generated_at': now.isoformat()}
    Path(ns.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print('saved', ns.output)


if __name__ == '__main__':
    main()
