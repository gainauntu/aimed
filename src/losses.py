from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets)


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be [B,D], got {tuple(embeddings.shape)}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be [B], got {tuple(labels.shape)}")

    device = embeddings.device
    sim = torch.matmul(embeddings, embeddings.T) / temperature

    labels = labels.contiguous().view(-1, 1)
    pos_mask = torch.eq(labels, labels.T).float().to(device)

    logits_mask = torch.ones_like(pos_mask) - torch.eye(pos_mask.size(0), device=device)
    pos_mask = pos_mask * logits_mask

    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    pos_count = pos_mask.sum(dim=1)
    valid = pos_count > 0
    if not valid.any():
        return embeddings.new_tensor(0.0)

    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-8)
    loss = -mean_log_prob_pos[valid].mean()
    return loss
