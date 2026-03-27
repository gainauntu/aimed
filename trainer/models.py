from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except Exception:  # pragma: no cover
    timm = None


def _require_timm():
    if timm is None:
        raise ImportError('timm is required for trainer models. Install trainer/requirements.txt first.')


def _load_partial(module: nn.Module, checkpoint_path: str | None):
    if not checkpoint_path:
        return
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location='cpu')
    state = payload.get('model_state', payload)
    missing, unexpected = module.load_state_dict(state, strict=False)
    return {'missing': missing, 'unexpected': unexpected}


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden_dim, dim), nn.Dropout(drop))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TileAttentionPool(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, drop: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=drop, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        q = self.query.expand(b, -1, -1)
        pooled, _ = self.attn(q, x, x, need_weights=False)
        return self.norm(pooled.squeeze(1))


class CrossGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.g_from_t = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.Sigmoid())
        self.t_from_g = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.Sigmoid())

    def forward(self, g: torch.Tensor, t: torch.Tensor):
        return g * (1.0 + self.g_from_t(t)), t * (1.0 + self.t_from_g(g))


class TowerA(nn.Module):
    """ConvNeXt-Base global + 5x5 tile branch."""
    def __init__(self, num_classes: int, emb_dim: int = 256, backbone_checkpoint: str | None = None):
        super().__init__()
        _require_timm()
        self.global_backbone = timm.create_model('convnext_base', pretrained=backbone_checkpoint is None, in_chans=9, num_classes=0, global_pool='avg')
        _load_partial(self.global_backbone, backbone_checkpoint)
        self.tile_encoder = nn.Sequential(
            nn.Conv2d(9, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.tile_proj = nn.Linear(128, self.global_backbone.num_features)
        self.tile_pool = TileAttentionPool(self.global_backbone.num_features)
        self.cross_gate = CrossGate(self.global_backbone.num_features)
        fusion_dim = self.global_backbone.num_features * 4
        self.fuse = nn.Sequential(
            nn.Linear(fusion_dim, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(),
        )
        self.embedding = nn.Linear(512, emb_dim)
        self.classifier = nn.Linear(512, num_classes)

    def _tiles(self, x: torch.Tensor):
        b, c, h, w = x.shape
        assert h == 288 and w == 288, 'TowerA expects 288x288 fused image'
        step = 57
        tiles = []
        for yi in range(5):
            for xi in range(5):
                y0 = yi * step
                x0 = xi * step
                y1 = min(h, y0 + step)
                x1 = min(w, x0 + step)
                t = x[:, :, y0:y1, x0:x1]
                if t.shape[-2:] != (57, 57):
                    t = F.interpolate(t, size=(57, 57), mode='bilinear', align_corners=False)
                tiles.append(t)
        return torch.stack(tiles, dim=1)

    def forward(self, x: torch.Tensor):
        g = self.global_backbone(x)
        tiles = self._tiles(x).reshape(-1, 9, 57, 57)
        tf = self.tile_encoder(tiles).flatten(1)
        tf = self.tile_proj(tf).reshape(x.shape[0], 25, -1)
        tp = self.tile_pool(tf)
        g, tp = self.cross_gate(g, tp)
        fused = torch.cat([g, tp, torch.abs(g - tp), g * tp], dim=1)
        h = self.fuse(fused)
        emb = F.normalize(self.embedding(h), dim=-1)
        logits = self.classifier(h)
        return logits, emb


class TowerB(nn.Module):
    """EfficientNet-B5 global only."""
    def __init__(self, num_classes: int, emb_dim: int = 256, backbone_checkpoint: str | None = None):
        super().__init__()
        _require_timm()
        self.backbone = timm.create_model('tf_efficientnet_b5', pretrained=backbone_checkpoint is None, in_chans=9, num_classes=0, global_pool='avg')
        _load_partial(self.backbone, backbone_checkpoint)
        fdim = self.backbone.num_features
        self.fc = nn.Sequential(nn.Linear(fdim, 768), nn.LayerNorm(768), nn.GELU(), nn.Dropout(0.15), nn.Linear(768, 512), nn.LayerNorm(512), nn.GELU())
        self.embedding = nn.Linear(512, emb_dim)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor):
        h = self.fc(self.backbone(x))
        emb = F.normalize(self.embedding(h), dim=-1)
        logits = self.classifier(h)
        return logits, emb


class TowerC(nn.Module):
    """ViT-B/8 tower with DINOv2 pretrained weights.

    Architecture spec requires DINOv2 ViT-B/8, not standard supervised ViT.
    DINOv2 self-supervised pretraining produces features that spontaneously
    attend to object parts rather than background — critically important for
    pill imprint and fine-feature recognition at 288×288 resolution.

    timm model string: 'vit_base_patch8_224.dino' loads the DINOv2 weights.
    img_size=288 overrides the 224 default; timm interpolates position
    embeddings automatically so 288 works cleanly.

    in_chans=9 accepts the symmetric-fused tensor [A+B, |A-B|, A*B].
    When backbone_checkpoint is provided (DINO-adapted weights from Phase T1),
    _load_partial loads all matching layers including the stem conv (in_chans=9
    matches because DINOAdaptBackbone also uses in_chans=9).
    """
    def __init__(self, num_classes: int, emb_dim: int = 512, backbone_checkpoint: str | None = None):
        super().__init__()
        _require_timm()
        # Use DINOv2 pretrained weights as the base. pretrained=True here
        # loads DINOv2 weights when backbone_checkpoint is None (cold start).
        # When backbone_checkpoint is set (Phase T1 adapted weights), those
        # are loaded via _load_partial and override the DINOv2 weights.
        self.backbone = timm.create_model(
            'vit_base_patch8_224.dino',
            pretrained=backbone_checkpoint is None,
            in_chans=9,
            img_size=288,
            num_classes=0,
        )
        _load_partial(self.backbone, backbone_checkpoint)
        fdim = self.backbone.num_features
        self.fc = nn.Sequential(nn.Linear(fdim, 768), nn.LayerNorm(768), nn.GELU(), nn.Dropout(0.10))
        self.embedding = nn.Linear(768, emb_dim)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x: torch.Tensor):
        h = self.fc(self.backbone(x))
        emb = F.normalize(self.embedding(h), dim=-1)
        logits = self.classifier(h)
        return logits, emb


class PrototypeNet(nn.Module):
    """DenseNet-169 episodic verifier."""
    def __init__(self, num_classes: int, emb_dim: int = 512, backbone_checkpoint: str | None = None):
        super().__init__()
        _require_timm()
        self.backbone = timm.create_model('densenet169', pretrained=backbone_checkpoint is None, in_chans=9, num_classes=0, global_pool='avg')
        _load_partial(self.backbone, backbone_checkpoint)
        fdim = self.backbone.num_features
        self.fc = nn.Sequential(nn.Linear(fdim, 768), nn.LayerNorm(768), nn.GELU(), nn.Dropout(0.10))
        self.embedding = nn.Linear(768, emb_dim)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x: torch.Tensor):
        h = self.fc(self.backbone(x))
        emb = F.normalize(self.embedding(h), dim=-1)
        logits = self.classifier(h)
        return logits, emb


class DINOFeatureBackbone(nn.Module):
    """3-channel backbone for Stage 3 OOD feature extraction.
    Accepts single preprocessed ROI images (BGR→RGB normalised, 3-channel).
    Used at both OOD index build time and runtime inference.
    """
    def __init__(self, model_name: str = 'convnext_base', feature_dim: int = 512, checkpoint_path: str | None = None):
        super().__init__()
        _require_timm()
        self.backbone = timm.create_model(model_name, pretrained=checkpoint_path is None, in_chans=3, num_classes=0, global_pool='avg')
        _load_partial(self.backbone, checkpoint_path)
        out_dim = self.backbone.num_features
        self.proj = nn.Sequential(nn.Linear(out_dim, out_dim), nn.GELU(), nn.Linear(out_dim, feature_dim))

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        feat = F.normalize(self.proj(feat), dim=-1)
        return feat


class DINOAdaptBackbone(nn.Module):
    """9-channel backbone for DINO self-supervised domain adaptation.

    Accepts the symmetric-fused tensor [A+B, |A-B|, A*B] (9 channels).
    After DINO training the adapted weights are used to initialise all three
    classification towers (TowerA, TowerB, TowerC all take 9-channel input).
    The first conv layer channel count matches, so _load_partial can copy
    every layer cleanly — including the stem — which was the root cause of
    Issue 5 when the 3-channel DINOFeatureBackbone was used.
    """
    def __init__(self, model_name: str = 'convnext_base', feature_dim: int = 512, checkpoint_path: str | None = None):
        super().__init__()
        _require_timm()
        self.backbone = timm.create_model(model_name, pretrained=checkpoint_path is None, in_chans=9, num_classes=0, global_pool='avg')
        _load_partial(self.backbone, checkpoint_path)
        out_dim = self.backbone.num_features
        self.proj = nn.Sequential(nn.Linear(out_dim, out_dim), nn.GELU(), nn.Linear(out_dim, feature_dim))
        self.backbone_name = model_name

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        feat = F.normalize(self.proj(feat), dim=-1)
        return feat


class TorchScriptWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.eval()

    def forward(self, x):
        logits, emb = self.model(x)
        return {'logits': logits, 'embedding': emb}


class TorchScriptBackboneWrapper(nn.Module):
    """Wraps DINOFeatureBackbone (3-ch) for TorchScript export as the OOD backbone."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.eval()

    def forward(self, x):
        feat = self.model(x)
        return {'feature': feat}


def build_model(name: str, num_classes: int, backbone_checkpoint: str | None = None):
    if name == 'tower_a':
        return TowerA(num_classes, backbone_checkpoint=backbone_checkpoint)
    if name == 'tower_b':
        return TowerB(num_classes, backbone_checkpoint=backbone_checkpoint)
    if name == 'tower_c':
        return TowerC(num_classes, backbone_checkpoint=backbone_checkpoint)
    if name == 'prototype':
        return PrototypeNet(num_classes, backbone_checkpoint=backbone_checkpoint)
    raise ValueError(f'Unknown model: {name}')
