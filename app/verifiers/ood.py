from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from app.domain.models import OODReport
from app.inference.fusion import normalize_image

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


class OODIndex:
    def __init__(self, vectors: np.ndarray, threshold: float) -> None:
        self.vectors = vectors.astype(np.float32)
        self.threshold = float(threshold)
        self.index = None
        if faiss is not None:
            db = self.vectors.copy()
            faiss.normalize_L2(db)
            self.index = faiss.IndexFlatIP(db.shape[1])
            self.index.add(db)

    @classmethod
    def from_prefix(cls, prefix: str | Path) -> 'OODIndex':
        prefix = Path(prefix)
        meta_path = prefix.with_suffix('.json')
        vec_path = prefix.with_suffix('.npy')
        if not meta_path.exists() or not vec_path.exists():
            raise FileNotFoundError(f'OOD artifacts not found: {prefix}')
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        vectors = np.load(vec_path)
        return cls(vectors=vectors, threshold=float(meta['threshold']))

    def mean_distance(self, query_vector: np.ndarray, k: int = 5) -> float:
        query = query_vector.astype(np.float32)[None, :]
        if self.index is not None:
            faiss.normalize_L2(query)
            sims, _ = self.index.search(query, min(k, self.vectors.shape[0]))
            return float(1.0 - sims.mean())
        db = self.vectors / np.clip(np.linalg.norm(self.vectors, axis=1, keepdims=True), 1e-8, None)
        qq = query / np.clip(np.linalg.norm(query, axis=1, keepdims=True), 1e-8, None)
        sims = db @ qq.T
        topk = np.sort(sims[:, 0])[::-1][: min(k, sims.shape[0])]
        return float(1.0 - topk.mean())


class OODRuntimeBackbone:
    def __init__(self, checkpoint_path: str, device: str | None = None) -> None:
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.jit.load(checkpoint_path, map_location=self.device)
        self.model.eval()

    def extract(self, image_bgr: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(normalize_image(image_bgr)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            out = self.model(x)
        if isinstance(out, dict):
            feat = out.get('feature', out.get('embedding', out.get('features')))
        else:
            feat = out
        if feat is None:
            raise RuntimeError('OOD backbone output missing feature tensor')
        feat = feat.detach().cpu().numpy()[0].astype(np.float32)
        feat /= np.clip(np.linalg.norm(feat), 1e-8, None)
        return feat


class OODVerifier:
    def __init__(self, index: OODIndex, backbone: OODRuntimeBackbone | None = None) -> None:
        self.index = index
        self.backbone = backbone

    def _extract_feature(self, image_bgr: np.ndarray) -> np.ndarray:
        if self.backbone is not None:
            return self.backbone.extract(image_bgr)
        norm = normalize_image(image_bgr)
        feat = norm.mean(axis=(1, 2)).astype(np.float32)
        feat /= np.clip(np.linalg.norm(feat), 1e-8, None)
        return feat

    def verify(self, image_a_bgr: np.ndarray, image_b_bgr: np.ndarray) -> OODReport:
        dist_a = self.index.mean_distance(self._extract_feature(image_a_bgr))
        dist_b = self.index.mean_distance(self._extract_feature(image_b_bgr))
        reasons: list[str] = []
        if dist_a > self.index.threshold:
            reasons.append(f'image A OOD distance {dist_a:.4f} exceeds {self.index.threshold:.4f}')
        if dist_b > self.index.threshold:
            reasons.append(f'image B OOD distance {dist_b:.4f} exceeds {self.index.threshold:.4f}')
        return OODReport(
            passed=not reasons,
            mean_distance_a=dist_a,
            mean_distance_b=dist_b,
            threshold=self.index.threshold,
            reasons=reasons,
        )
