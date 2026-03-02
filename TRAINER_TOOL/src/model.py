from __future__ import annotations
import torch.nn as nn
import timm

class PillClassifier(nn.Module):
    def __init__(self, n_classes: int, backbone: str = "convnext_small"):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=n_classes)

    def forward(self, x):
        return self.backbone(x)
