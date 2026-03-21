from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from torchvision.models import resnet101, ResNet101_Weights


def _ckpt(module, x: torch.Tensor, use_checkpoint: bool):
    if use_checkpoint and x.requires_grad:
        return checkpoint(module, x, use_reentrant=False)
    return module(x)


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.pow(1.0 / self.p)
        return x.flatten(1)


class ResNetBackbone(nn.Module):
    """
    Shared backbone for global image and local tiles.
    Uses ResNet101 + GeM pooling.
    """
    def __init__(self, pretrained: bool = True, use_checkpoint: bool = True):
        super().__init__()
        weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        m = resnet101(weights=weights)

        self.stem = nn.Sequential(
            m.conv1,
            m.bn1,
            m.relu,
            m.maxpool,
        )
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4

        self.pool = GeM()
        self.out_dim = 2048
        self.use_checkpoint = use_checkpoint

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = _ckpt(self.stem, x, self.use_checkpoint)
        x = _ckpt(self.layer1, x, self.use_checkpoint)
        x = _ckpt(self.layer2, x, self.use_checkpoint)
        x = _ckpt(self.layer3, x, self.use_checkpoint)
        x = _ckpt(self.layer4, x, self.use_checkpoint)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.pool(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TileTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 2.0, drop: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=drop,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class TileAttentionPool(nn.Module):
    """
    Learnable query attention pooling over tile embeddings.
    """
    def __init__(self, dim: int, num_heads: int = 8, drop: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=drop,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        q = self.query.expand(b, -1, -1)
        pooled, _ = self.attn(q, x, x, need_weights=False)
        pooled = pooled.squeeze(1)
        return self.norm(pooled)


class CrossGating(nn.Module):
    """
    Global <-> Tile interaction.
    Each branch gates the other.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.g_from_t = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.t_from_g = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, g: torch.Tensor, t: torch.Tensor):
        g_gate = self.g_from_t(t)
        t_gate = self.t_from_g(g)

        g2 = g * (1.0 + g_gate)
        t2 = t * (1.0 + t_gate)
        return g2, t2


class FullResPillNet(nn.Module):
    """
    Stronger full-resolution pill classifier.

    Input:
      global_img: [B, 3, H, W]
      tiles:      [B, T, 3, h, w]

    Output:
      logits: [B, num_classes]
      emb:    [B, emb_dim]  (L2-normalized)
    """
    def __init__(
        self,
        num_classes: int,
        emb_dim: int = 768,
        pretrained: bool = True,
    ):
        super().__init__()

        self.backbone = ResNetBackbone(
            pretrained=pretrained,
            use_checkpoint=True,
        )
        d = self.backbone.out_dim

        self.global_norm = nn.LayerNorm(d)
        self.tile_in_norm = nn.LayerNorm(d)

        self.tile_blocks = nn.Sequential(
            TileTransformerBlock(d, num_heads=8, mlp_ratio=2.0, drop=0.10),
            TileTransformerBlock(d, num_heads=8, mlp_ratio=2.0, drop=0.10),
        )
        self.tile_pool = TileAttentionPool(d, num_heads=8, drop=0.10)

        self.cross_gate = CrossGating(d)

        fusion_dim = d * 4
        self.fuse = nn.Sequential(
            nn.Linear(fusion_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.20),

            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.15),

            nn.Linear(1024, emb_dim),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes),
        )

    def encode(self, global_img: torch.Tensor, tiles: torch.Tensor) -> torch.Tensor:
        # global branch
        g = self.backbone(global_img)          # [B, 2048]
        g = self.global_norm(g)

        # tile branch
        b, t, c, h, w = tiles.shape
        tf = self.backbone(tiles.reshape(b * t, c, h, w))   # [B*T, 2048]
        tf = tf.reshape(b, t, -1)                           # [B, T, 2048]
        tf = self.tile_in_norm(tf)
        tf = self.tile_blocks(tf)
        tp = self.tile_pool(tf)                             # [B, 2048]

        # cross interaction
        g, tp = self.cross_gate(g, tp)

        # stronger fusion
        fused = torch.cat(
            [
                g,
                tp,
                torch.abs(g - tp),
                g * tp,
            ],
            dim=1,
        )

        emb = self.fuse(fused)
        emb = F.normalize(emb, dim=1)
        return emb

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