from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, s: float = 64.0, m: float = 0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = float(s)
        self.m = float(m)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - cosine * cosine).clamp(0, 1))
        cos_m = torch.cos(torch.tensor(self.m, device=cosine.device))
        sin_m = torch.sin(torch.tensor(self.m, device=cosine.device))
        th = torch.cos(torch.tensor(torch.pi - self.m, device=cosine.device))
        mm = torch.sin(torch.tensor(torch.pi - self.m, device=cosine.device)) * self.m
        phi = cosine * cos_m - sine * sin_m
        phi = torch.where(cosine > th, phi, cosine - mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.s


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        features = F.normalize(features, dim=-1)
        logits = torch.matmul(features, features.T) / self.temperature
        logits_mask = torch.ones_like(logits) - torch.eye(logits.shape[0], device=logits.device)
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float() * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
        return -mean_log_prob_pos.mean()


class ProxyNCALoss(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.proxies = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.proxies)

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        x = F.normalize(features, dim=-1)
        p = F.normalize(self.proxies, dim=-1)
        d = torch.cdist(x, p, p=2)
        return F.cross_entropy(-d, labels)
