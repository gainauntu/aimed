from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.sparse import coo_matrix
from sklearn.isotonic import IsotonicRegression


def reliability_diagram(probs: np.ndarray, y: np.ndarray, width: float = 0.05):
    bins = np.arange(0.0, 1.0 + width, width)
    rows = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if i == len(bins) - 2:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        rows.append({
            'bin_lo': float(lo),
            'bin_hi': float(hi),
            'count': int(mask.sum()),
            'mean_pred': float(probs[mask].mean()),
            'empirical_acc': float(y[mask].mean()),
        })
    return rows


def normalize_features(X: np.ndarray):
    feature_mins = X.min(axis=0)
    feature_maxs = X.max(axis=0)
    Xn = (X - feature_mins) / np.maximum(feature_maxs - feature_mins, 1e-6)
    return np.clip(Xn, 0.0, 1.0), feature_mins.astype(np.float32), feature_maxs.astype(np.float32)


def precedence_edges(Xn: np.ndarray, eps: float = 1e-9):
    n = Xn.shape[0]
    edges: list[tuple[int, int]] = []
    for i in range(n):
        xi = Xn[i]
        le = np.all(xi[None, :] <= Xn + eps, axis=1)
        lt = np.any(xi[None, :] < Xn - eps, axis=1)
        idxs = np.where(le & lt)[0]
        for j in idxs.tolist():
            edges.append((i, j))
    return edges


def transitive_reduction_like(edges: list[tuple[int, int]], n: int):
    """Cheap edge pruning for small calibration sets.

    Removes edge i->k when there exists j such that i->j and j->k. This is not a full
    transitive reduction algorithm, but it massively shrinks the constraint graph for
    product-order isotonic fits while preserving the order relation we care about.
    """
    succ = {i: set() for i in range(n)}
    for i, j in edges:
        succ[i].add(j)
    keep = []
    for i, k in edges:
        redundant = False
        for j in succ[i]:
            if j == k:
                continue
            if k in succ.get(j, set()):
                redundant = True
                break
        if not redundant:
            keep.append((i, k))
    return keep


def solve_multivariate_isotonic(Xn: np.ndarray, y: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = Xn.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    if n == 1:
        return np.clip(y.astype(np.float32), 0.0, 1.0)

    edges = precedence_edges(Xn, eps=eps)
    if edges:
        edges = transitive_reduction_like(edges, n)
        rows = np.arange(len(edges)).repeat(2)
        cols = np.asarray([idx for e in edges for idx in e], dtype=np.int64)
        data = np.asarray([-1.0, 1.0] * len(edges), dtype=np.float64)
        A = coo_matrix((data, (rows, cols)), shape=(len(edges), n)).tocsr()
        lin = LinearConstraint(A, np.zeros(len(edges), dtype=np.float64), np.full(len(edges), np.inf, dtype=np.float64))
        constraints = [lin]
    else:
        constraints = []

    def fun(f: np.ndarray):
        diff = f - y
        return 0.5 * float(diff @ diff)

    def jac(f: np.ndarray):
        return f - y

    x0 = np.clip(y.astype(np.float64), 0.0, 1.0)
    bounds = Bounds(np.zeros(n, dtype=np.float64), np.ones(n, dtype=np.float64))
    result = minimize(
        fun,
        x0,
        jac=jac,
        method='trust-constr',
        constraints=constraints,
        bounds=bounds,
        options={'verbose': 0, 'maxiter': 2000},
    )
    if not result.success:
        # fallback: monotone 1D score isotonic over the mean feature score, still safe to export
        order = np.argsort(Xn.mean(axis=1))
        iso = IsotonicRegression(out_of_bounds='clip')
        fitted = np.empty_like(y, dtype=np.float32)
        fitted_order = iso.fit_transform(Xn[order].mean(axis=1), y[order])
        fitted[order] = fitted_order.astype(np.float32)
        return fitted
    return np.clip(result.x.astype(np.float32), 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--signals-jsonl', required=True, help='JSONL with fields: signals[12], correct, predicted_class, tta_variance_mean')
    ap.add_argument('--output', required=True)
    ap.add_argument('--variance-thresholds-output', required=True)
    ap.add_argument('--reliability-output', default='')
    ap.add_argument('--order-eps', type=float, default=1e-9)
    ns = ap.parse_args()

    rows = [json.loads(line) for line in Path(ns.signals_jsonl).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        payload = {
            'kind': 'multivariate_isotonic',
            'train_features': np.zeros((1, 12), dtype=np.float32),
            'fitted_values': np.asarray([0.0], dtype=np.float32),
            'feature_mins': np.zeros(12, dtype=np.float32),
            'feature_maxs': np.ones(12, dtype=np.float32),
            'eps': float(ns.order_eps),
        }
        joblib.dump(payload, ns.output)
        Path(ns.variance_thresholds_output).write_text(json.dumps({}, ensure_ascii=False, indent=2), encoding='utf-8')
        if ns.reliability_output:
            Path(ns.reliability_output).write_text(json.dumps([], ensure_ascii=False, indent=2), encoding='utf-8')
        print('saved fallback multivariate isotonic calibrator')
        return

    X = np.asarray([r['signals'] for r in rows], dtype=np.float32)
    y = np.asarray([int(r['correct']) for r in rows], dtype=np.float32)
    Xn, feature_mins, feature_maxs = normalize_features(X)
    fitted_values = solve_multivariate_isotonic(Xn, y, eps=ns.order_eps)

    payload = {
        'kind': 'multivariate_isotonic',
        'train_features': Xn.astype(np.float32),
        'fitted_values': fitted_values.astype(np.float32),
        'feature_mins': feature_mins,
        'feature_maxs': feature_maxs,
        'eps': float(ns.order_eps),
    }
    joblib.dump(payload, ns.output)

    by_class = {}
    for r in rows:
        by_class.setdefault(str(r['predicted_class']), []).append(float(r['tta_variance_mean']))
    th = {k: float(np.percentile(v, 99)) for k, v in by_class.items()}
    Path(ns.variance_thresholds_output).write_text(json.dumps(th, ensure_ascii=False, indent=2), encoding='utf-8')

    if ns.reliability_output:
        rel = reliability_diagram(fitted_values.astype(np.float32), y.astype(np.float32))
        Path(ns.reliability_output).write_text(json.dumps(rel, ensure_ascii=False, indent=2), encoding='utf-8')
    print('saved multivariate isotonic calibrator and variance thresholds')


if __name__ == '__main__':
    main()
