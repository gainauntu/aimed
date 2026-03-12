from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class PillEncoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str = "convnext_tiny",
        emb_dim: int = 256,
        dropout: float = 0.0,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.emb_dim = emb_dim

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features

        self.emb_head = nn.Sequential(
            nn.Linear(feat_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.cls_head = nn.Linear(emb_dim, num_classes)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        emb_raw = self.emb_head(feat)
        emb = F.normalize(emb_raw, dim=1)
        logits = self.cls_head(emb_raw)
        return logits, emb
