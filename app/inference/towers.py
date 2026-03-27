from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from app.domain.models import TowerPrediction
from app.preprocess.tta import ViewPair
from app.inference.fusion import batch_fusion


class TowerAdapter(ABC):
    def __init__(self, name: str, class_labels: list[str]) -> None:
        self.name = name
        self.class_labels = class_labels

    @abstractmethod
    def forward_batch(self, fused_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def predict(self, views: list[ViewPair]) -> TowerPrediction:
        pairs = [(view.image_a, view.image_b) for view in views]
        fused_batch = batch_fusion(pairs)
        logits, embeddings = self.forward_batch(fused_batch)
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
        embeddings_np = F.normalize(embeddings, dim=-1).detach().cpu().numpy()

        mean_prediction = probs.mean(axis=0)
        top1_idx = int(np.argmax(mean_prediction))
        sorted_probs = np.sort(mean_prediction)[::-1]
        top1_conf = float(mean_prediction[top1_idx])
        margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else float(sorted_probs[0])
        tta_variance = float(np.std(probs[:, top1_idx]))
        mean_embedding = embeddings_np.mean(axis=0).tolist()

        return TowerPrediction(
            tower_name=self.name,
            predicted_class=self.class_labels[top1_idx],
            top1_confidence=top1_conf,
            top1_top2_margin=margin,
            tta_variance=tta_variance,
            probabilities={label: float(value) for label, value in zip(self.class_labels, mean_prediction)},
            embedding=mean_embedding,
        )


class TorchScriptTower(TowerAdapter):
    def __init__(self, name: str, class_labels: list[str], checkpoint_path: str, device: str | None = None) -> None:
        super().__init__(name, class_labels)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint missing for tower {name}: {checkpoint_path}")
        self.model = torch.jit.load(str(ckpt), map_location=self.device)
        self.model.eval()

    def forward_batch(self, fused_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fused_batch = fused_batch.to(self.device)
        with torch.inference_mode():
            output = self.model(fused_batch)
        if isinstance(output, tuple) and len(output) == 2:
            return output[0], output[1]
        if isinstance(output, dict):
            return output["logits"], output["embedding"]
        raise TypeError("Expected model output to be (logits, embedding) or dict")


class DummyTower(TowerAdapter):
    """Useful for tests and smoke runs only."""

    def __init__(self, name: str, class_labels: list[str], bias_index: int = 0) -> None:
        super().__init__(name, class_labels)
        self.bias_index = bias_index

    def forward_batch(self, fused_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = fused_batch.shape[0]
        num_classes = len(self.class_labels)
        logits = torch.zeros((batch, num_classes), dtype=torch.float32)
        logits[:, self.bias_index] = 5.0
        logits += torch.linspace(0, 0.2, steps=batch).unsqueeze(1)
        embeddings = torch.ones((batch, 256), dtype=torch.float32)
        return logits, embeddings
