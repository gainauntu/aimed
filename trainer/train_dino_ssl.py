from __future__ import annotations

# ---------------------------------------------------------------------------
# Phase T1 — DINO self-supervised domain adaptation
#
# FIX (Issue 5): trained DINOAdaptBackbone (9-channel) on fused pairs so that
# adapted weights load cleanly (including the stem conv) into all three towers.
# ---------------------------------------------------------------------------

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ddp_utils import seed_everything, seed_worker
from models import DINOAdaptBackbone
from pair_dataset import scan_pair_samples, PillPairDataset


class DINOProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 512, out_dim: int = 65536, bottleneck_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim), nn.GELU(),
            nn.Linear(bottleneck_dim, bottleneck_dim), nn.GELU(),
            nn.Linear(bottleneck_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class DINOWrapper(nn.Module):
    def __init__(self, backbone: DINOAdaptBackbone, out_dim: int = 65536):
        super().__init__()
        self.backbone = backbone
        self.head = DINOProjectionHead(512, out_dim)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat), feat


class DINOCenter(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer('center', torch.zeros(1, dim))

    @torch.no_grad()
    def update(self, teacher_out: torch.Tensor, momentum: float = 0.9):
        batch_center = teacher_out.mean(dim=0, keepdim=True)
        self.center.mul_(momentum).add_(batch_center, alpha=1.0 - momentum)


@torch.no_grad()
def update_teacher(student: nn.Module, teacher: nn.Module, momentum: float):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)


def random_resized_crop_tensor(x, scale_low, scale_high, out_size):
    b, c, h, w = x.shape
    outs = []
    for i in range(b):
        size = int(round(min(h, w) * np.random.uniform(scale_low, scale_high)))
        size = max(16, min(size, min(h, w)))
        y  = 0 if h == size else np.random.randint(0, h - size + 1)
        x0 = 0 if w == size else np.random.randint(0, w - size + 1)
        crop = x[i:i+1, :, y:y+size, x0:x0+size]
        crop = F.interpolate(crop, size=(out_size, out_size), mode='bilinear', align_corners=False)
        outs.append(crop)
    return torch.cat(outs, dim=0)


def tensor_aug(x):
    out = x.clone()
    if torch.rand(1).item() < 0.8:
        out = out * torch.empty((x.size(0), 1, 1, 1), device=x.device).uniform_(0.92, 1.08)
    if torch.rand(1).item() < 0.8:
        out = out + torch.empty((x.size(0), 1, 1, 1), device=x.device).uniform_(-0.05, 0.05)
    if torch.rand(1).item() < 0.2:
        out = out + torch.randn_like(out) * 0.02
    return out.clamp(-4.0, 4.0)


def multi_crop_views(x):
    g1 = tensor_aug(random_resized_crop_tensor(x, 0.60, 1.00, 288))
    g2 = tensor_aug(random_resized_crop_tensor(x, 0.60, 1.00, 288))
    l1 = tensor_aug(random_resized_crop_tensor(x, 0.35, 0.55, 288))
    l2 = tensor_aug(random_resized_crop_tensor(x, 0.35, 0.55, 288))
    return [g1, g2], [l1, l2]


def dino_loss(student_outs, teacher_outs, center, student_temp=0.1, teacher_temp=0.04):
    total = 0.0
    count = 0
    teacher_probs = [F.softmax((t - center) / teacher_temp, dim=-1).detach() for t in teacher_outs]
    for s in student_outs:
        logp = F.log_softmax(s / student_temp, dim=-1)
        for tprob in teacher_probs:
            total = total + torch.mean(torch.sum(-tprob * logp, dim=-1))
            count += 1
    return total / max(1, count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--backbone', default='convnext_base')
    ap.add_argument('--teacher-momentum', type=float, default=0.996)
    ns = ap.parse_args()

    seed_everything(ns.seed)
    outdir = Path(ns.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    samples, _ = scan_pair_samples(ns.data_root)
    # PillPairDataset already produces 9-channel symmetric-fused tensors.
    ds = PillPairDataset(samples, image_size=288, train=True)
    loader = DataLoader(ds, batch_size=ns.batch_size, shuffle=True,
                        num_workers=ns.num_workers, pin_memory=True,
                        persistent_workers=ns.num_workers > 0,
                        worker_init_fn=seed_worker)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # DINOAdaptBackbone: in_chans=9 matches all three tower first-conv stems.
    student = DINOWrapper(DINOAdaptBackbone(model_name=ns.backbone)).to(device)
    teacher = copy.deepcopy(student).to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    center = DINOCenter(dim=65536).to(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=ns.lr,
                                   weight_decay=1e-4, fused=torch.cuda.is_available())
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    history = []

    for epoch in range(ns.epochs):
        student.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f'[DINO-9ch] epoch {epoch+1}/{ns.epochs}')
        for batch in pbar:
            x = batch['fused'].to(device)          # 9-channel fused tensor
            global_views, local_views = multi_crop_views(x)
            student_views = global_views + local_views
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.float16):
                s_outs = [student(v)[0] for v in student_views]
                with torch.no_grad():
                    t_outs = [teacher(v)[0] for v in global_views]
                loss = dino_loss(s_outs, t_outs, center.center)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                teacher_concat = torch.cat([t.float() for t in t_outs], dim=0)
                center.update(teacher_concat)
                update_teacher(student, teacher, momentum=ns.teacher_momentum)
            epoch_loss += float(loss.item())
            pbar.set_postfix(loss=f'{epoch_loss / max(1, pbar.n):.4f}')
        history.append({'epoch': epoch + 1, 'loss': epoch_loss / max(1, len(loader))})

    # The saved backbone state matches the tower first-conv (in_chans=9).
    torch.save({
        'model_state': student.backbone.state_dict(),
        'backbone_name': ns.backbone,
        'in_chans': 9,
        'history': history,
    }, outdir / 'dino_adapted_backbone.ckpt')
    (outdir / 'history.json').write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding='utf-8')
    print('saved 9-ch DINO-adapted backbone:', outdir / 'dino_adapted_backbone.ckpt')


if __name__ == '__main__':
    main()
