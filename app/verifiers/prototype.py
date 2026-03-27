from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from app.domain.models import PrototypeReport
from app.inference.fusion import batch_fusion
from app.preprocess.tta import ViewPair


@dataclass(slots=True)
class PrototypeEntry:
    centroid: np.ndarray
    p95_distance: float
    p99_distance: float
    min_similarity: float
    second_class_margin: float


class PrototypeLibrary:
    def __init__(self, entries: dict[str, PrototypeEntry]) -> None:
        self.entries = entries

    @classmethod
    def from_json(cls, path: str | Path) -> 'PrototypeLibrary':
        raw = json.loads(Path(path).read_text(encoding='utf-8'))
        entries = {
            key: PrototypeEntry(
                centroid=np.asarray(value['centroid'], dtype=np.float32),
                p95_distance=float(value.get('p95_distance', value['p99_distance'])),
                p99_distance=float(value['p99_distance']),
                min_similarity=float(value['min_similarity']),
                second_class_margin=float(value['second_class_margin']),
            )
            for key, value in raw.items()
        }
        return cls(entries)


class PrototypeModel:
    def __init__(self, checkpoint_path: str, device: str | None = None) -> None:
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.jit.load(checkpoint_path, map_location=self.device)
        self.model.eval()

    def embed_views(self, views: list[ViewPair]) -> np.ndarray:
        fused = batch_fusion([(v.image_a, v.image_b) for v in views]).to(self.device)
        with torch.inference_mode():
            out = self.model(fused)
        if isinstance(out, dict):
            emb = out['embedding']
        else:
            emb = out[1]
        emb = emb.detach().cpu().numpy().astype(np.float32)
        mean_emb = emb.mean(axis=0)
        mean_emb /= np.clip(np.linalg.norm(mean_emb), 1e-8, None)
        return mean_emb


class PrototypeVerifier:
    def __init__(self, library: PrototypeLibrary, model: PrototypeModel | None = None) -> None:
        self.library = library
        self.model = model

    def verify(self, predicted_class: str, views: list[ViewPair] | None = None, embedding: np.ndarray | None = None) -> PrototypeReport:
        if embedding is None:
            if self.model is None or views is None:
                raise ValueError('Prototype verifier requires either embedding or (model + views)')
            emb = self.model.embed_views(views)
        else:
            emb = embedding.astype(np.float32)
            emb /= np.clip(np.linalg.norm(emb), 1e-8, None)

        entry = self.library.entries[predicted_class]
        centroid = entry.centroid.astype(np.float32)
        centroid /= np.clip(np.linalg.norm(centroid), 1e-8, None)

        similarity = float(np.dot(emb, centroid))
        distance = float(np.linalg.norm(emb - centroid))
        other_distances = []
        for cls_name, other in self.library.entries.items():
            if cls_name == predicted_class:
                continue
            c = other.centroid.astype(np.float32)
            c /= np.clip(np.linalg.norm(c), 1e-8, None)
            other_distances.append(float(np.linalg.norm(emb - c)))
        nearest_other = min(other_distances) if other_distances else float('inf')
        second_gap = float(nearest_other - distance) if np.isfinite(nearest_other) else float('inf')

        reasons: list[str] = []
        if similarity < entry.min_similarity:
            reasons.append(f'similarity {similarity:.4f} below {entry.min_similarity:.4f}')
        if distance > entry.p99_distance:
            reasons.append(f'distance {distance:.4f} exceeds {entry.p99_distance:.4f}')
        if second_gap < entry.second_class_margin:
            reasons.append(f'second-class gap {second_gap:.4f} below {entry.second_class_margin:.4f}')

        return PrototypeReport(
            passed=not reasons,
            similarity=similarity,
            distance_to_centroid=distance,
            second_class_gap=second_gap,
            reasons=reasons,
        )
