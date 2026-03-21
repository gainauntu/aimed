from __future__ import annotations
import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        centers_batch = self.centers[target]
        return ((feat - centers_batch) ** 2).sum(dim=1).mean()


def topk_accuracy(logits: torch.Tensor, target: torch.Tensor, k: int = 1) -> float:
    with torch.no_grad():
        pred = logits.topk(k, dim=1).indices
        ok = pred.eq(target.unsqueeze(1)).any(dim=1)
        return float(ok.float().mean().item())
