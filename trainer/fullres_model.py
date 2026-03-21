from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class ResNetBackbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = resnet50(weights=weights)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return x


class TileAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.score(x).squeeze(-1)
        w = torch.softmax(w, dim=1)
        return torch.sum(x * w.unsqueeze(-1), dim=1)


class FullResPillNet(nn.Module):
    def __init__(self, num_classes: int, emb_dim: int = 512, pretrained: bool = True):
        super().__init__()
        self.backbone = ResNetBackbone(pretrained=pretrained)
        d = self.backbone.out_dim
        self.tile_attn = TileAttention(d)
        self.fuse = nn.Sequential(
            nn.Linear(d * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(1024, emb_dim),
        )
        self.classifier = nn.Linear(emb_dim, num_classes)

    def encode(self, global_img: torch.Tensor, tiles: torch.Tensor) -> torch.Tensor:
        g = self.backbone(global_img)
        b, t, c, h, w = tiles.shape
        tf = self.backbone(tiles.reshape(b * t, c, h, w)).reshape(b, t, -1)
        tp = self.tile_attn(tf)
        fused = torch.cat([g, tp], dim=1)
        emb = self.fuse(fused)
        return F.normalize(emb, dim=1)

    def forward(self, global_img: torch.Tensor, tiles: torch.Tensor):
        emb = self.encode(global_img, tiles)
        logits = self.classifier(emb)
        return logits, emb


class TorchScriptWrapper(nn.Module):
    def __init__(self, model: FullResPillNet):
        super().__init__()
        self.model = model

    def forward(self, global_img: torch.Tensor, tiles: torch.Tensor):
        logits, emb = self.model(global_img, tiles)
        return {"logits": logits, "embedding": emb}
