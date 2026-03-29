from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from ddp_utils import setup_distributed, cleanup_distributed, barrier, is_main_process, seed_everything, seed_worker
from pair_dataset import scan_pair_samples, build_loocv_fold, loocv_n_folds, PillPairDataset, LightROIPreprocessor
from models import build_model
from losses import ArcMarginProduct, SupConLoss, ProxyNCALoss


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def _backbone_param_groups(core: nn.Module, tower: str):
    bb = getattr(core, 'global_backbone', None) or getattr(core, 'backbone', None)
    params = list(bb.named_parameters()) if bb is not None else []
    if tower == 'tower_c':
        # keep deterministic ordering by transformer block index for ViT
        ordered = []
        for n, p in core.named_parameters():
            if n.startswith('backbone.'):
                ordered.append((n, p))
        return ordered
    return params


def apply_phase_freeze(core: nn.Module, tower: str, phase: str):
    for p in core.parameters():
        p.requires_grad = False

    head_keywords = ['classifier', 'embedding', 'fc', 'fuse', 'tile_', 'cross_gate']
    for name, p in core.named_parameters():
        if any(k in name for k in head_keywords):
            p.requires_grad = True

    named = _backbone_param_groups(core, tower)
    if phase == 'frozen_head':
        return

    total = len(named)
    if total == 0:
        return

    if phase == 'unfreeze_top10':
        start = int(total * 0.90)
        for _, p in named[start:]:
            p.requires_grad = True
        return

    if phase == 'unfreeze_top10_30':
        start10 = int(total * 0.90)
        start30 = int(total * 0.70)
        for _, p in named[start30:]:
            p.requires_grad = True
        return

    if phase == 'full':
        for _, p in named:
            p.requires_grad = True
        return

    raise ValueError(f'Unsupported phase: {phase}')


def build_optimizer(model: nn.Module, metric_head, metric_loss, tower: str, phase: str, head_lr: float, wd: float):
    core = _unwrap(model)
    head_params = []
    named_backbone = _backbone_param_groups(core, tower)
    backbone_params = {id(p) for _, p in named_backbone}
    for p in core.parameters():
        if p.requires_grad and id(p) not in backbone_params:
            head_params.append(p)

    groups = []
    if head_params:
        groups.append({'params': head_params, 'lr': head_lr, 'weight_decay': wd})

    if phase == 'unfreeze_top10':
        trainable = [p for _, p in named_backbone if p.requires_grad]
        if trainable:
            groups.append({'params': trainable, 'lr': 1e-4, 'weight_decay': wd})
    elif phase == 'unfreeze_top10_30':
        total = len(named_backbone)
        start10 = int(total * 0.90)
        start30 = int(total * 0.70)
        top10 = [p for _, p in named_backbone[start10:] if p.requires_grad]
        next20 = [p for _, p in named_backbone[start30:start10] if p.requires_grad]
        if top10:
            groups.append({'params': top10, 'lr': 1e-4, 'weight_decay': wd})
        if next20:
            groups.append({'params': next20, 'lr': 1e-5, 'weight_decay': wd})
    elif phase == 'full':
        trainable = [p for _, p in named_backbone if p.requires_grad]
        if trainable:
            groups.append({'params': trainable, 'lr': 1e-4, 'weight_decay': wd})

    if metric_head is not None:
        groups.append({'params': list(metric_head.parameters()), 'lr': head_lr, 'weight_decay': wd})
    if metric_loss is not None:
        groups.append({'params': list(metric_loss.parameters()), 'lr': head_lr, 'weight_decay': wd})
    return torch.optim.AdamW(groups, fused=torch.cuda.is_available())


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    losses = []
    ce = nn.CrossEntropyLoss()
    for batch in loader:
        x = batch['fused'].to(device, non_blocking=True)
        y = batch['label'].to(device, non_blocking=True)
        logits, emb = model(x)
        loss = ce(logits, y)
        pred = logits.argmax(dim=1)
        total += y.numel()
        correct += int((pred == y).sum().item())
        losses.append(float(loss.item()))
    return {'loss': float(np.mean(losses) if losses else 0.0), 'acc': float(correct / max(1, total))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tower', choices=['tower_a', 'tower_b', 'tower_c'], required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--image-size', type=int, default=288)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--accum-steps', type=int, default=2)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--split-mode', choices=['random', 'loocv'], default='loocv')
    ap.add_argument('--fold-index', type=int, default=0)
    ap.add_argument('--phase', choices=['frozen_head', 'unfreeze_top10', 'unfreeze_top10_30', 'full'], default='frozen_head')
    ap.add_argument('--backbone-init', default='', help='DINO-adapted backbone checkpoint')
    ns = ap.parse_args()

    distributed, rank, world_size, local_rank = setup_distributed()
    seed_everything(ns.seed, rank)
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')
    outdir = Path(ns.outdir)
    if is_main_process():
        outdir.mkdir(parents=True, exist_ok=True)

    samples, class_names = scan_pair_samples(ns.data_root)
    if ns.split_mode == 'loocv':
        train_samples, val_samples = build_loocv_fold(samples, ns.fold_index)
    else:
        rng = np.random.default_rng(ns.seed)
        idx = np.arange(len(samples))
        rng.shuffle(idx)
        split = max(1, int(round(len(idx) * 0.85)))
        train_samples = [samples[i] for i in idx[:split].tolist()]
        val_samples = [samples[i] for i in idx[split:].tolist() or idx[-1:].tolist()]

    # Issue 6 fix: always use LightROIPreprocessor with black reference so
    # training images go through the same background-subtract → localize →
    # canonicalize → resize pipeline that inference uses.
    # Black reference (np.zeros) is correct for black-tray setups and requires
    # no external file — LightROIPreprocessor() with no arguments uses it.
    from pair_dataset import LightROIPreprocessor
    light_preprocessor = LightROIPreprocessor(target_size=ns.image_size)
    if is_main_process():
        print('LightROIPreprocessor active with black reference (Issue 6 fix)')

    train_ds = PillPairDataset(train_samples, image_size=ns.image_size, train=True,
                               light_preprocessor=light_preprocessor)
    val_ds = PillPairDataset(val_samples, image_size=ns.image_size, train=False,
                             light_preprocessor=light_preprocessor)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None
    train_loader = DataLoader(train_ds, batch_size=ns.batch_size, shuffle=train_sampler is None, sampler=train_sampler, num_workers=ns.num_workers, pin_memory=True, persistent_workers=ns.num_workers > 0, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_ds, batch_size=ns.batch_size, shuffle=False, sampler=val_sampler, num_workers=ns.num_workers, pin_memory=True, persistent_workers=ns.num_workers > 0, worker_init_fn=seed_worker)

    model = build_model(ns.tower, len(class_names), backbone_checkpoint=ns.backbone_init or None).to(device)
    if distributed:
        # find_unused_parameters=True is required because during frozen_head
        # and partial-unfreeze phases most backbone parameters have
        # requires_grad=False and produce no gradients.  DDP's default
        # (find_unused_parameters=False) assumes every parameter participates
        # in every backward pass — that assumption is wrong here and causes
        # "Expected to have finished reduction in the prior iteration" errors.
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)
    core = _unwrap(model)
    apply_phase_freeze(core, ns.tower, ns.phase)

    if ns.tower == 'tower_a':
        metric_head = ArcMarginProduct(256, len(class_names)).to(device)
        metric_loss = None
    elif ns.tower == 'tower_b':
        metric_head = None
        metric_loss = SupConLoss().to(device)
    else:
        metric_head = None
        metric_loss = ProxyNCALoss(len(class_names), 512).to(device)

    optimizer = build_optimizer(model, metric_head, metric_loss, ns.tower, ns.phase, ns.lr, ns.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, ns.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    ce = nn.CrossEntropyLoss()

    best_acc = -1.0
    history = []
    for epoch in range(ns.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        run_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        iterator = tqdm(train_loader, disable=not is_main_process(), desc=f'[{ns.tower}:{ns.phase}] epoch {epoch+1}/{ns.epochs}')
        for step, batch in enumerate(iterator, start=1):
            x = batch['fused'].to(device, non_blocking=True)
            y = batch['label'].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.float16):
                logits, emb = model(x)
                loss = ce(logits, y)
                if metric_head is not None:
                    arc_logits = metric_head(emb, y)
                    loss = 0.5 * loss + 0.5 * ce(arc_logits, y)
                elif metric_loss is not None:
                    loss = 0.5 * loss + 0.5 * metric_loss(emb, y)
                loss = loss / max(1, ns.accum_steps)
            scaler.scale(loss).backward()
            if step % ns.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            run_loss += float(loss.item()) * max(1, ns.accum_steps)
            iterator.set_postfix(loss=f'{run_loss / step:.4f}')
        scheduler.step()
        barrier()
        val_metrics = evaluate(_unwrap(model), val_loader, device)
        record = {'epoch': epoch + 1, 'train_loss': run_loss / max(1, len(train_loader)), **val_metrics}
        history.append(record)
        if is_main_process():
            if val_metrics['acc'] > best_acc:
                best_acc = val_metrics['acc']
                (outdir / 'labels.json').write_text(json.dumps(class_names, ensure_ascii=False, indent=2), encoding='utf-8')
            # Always write metrics — needed by collect_fold_metrics in the launcher.
            (outdir / 'metrics.json').write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding='utf-8')

            # LOOCV folds: never save model weights.
            # 180 folds × ~300 MB per checkpoint = ~54 GB — instantly exceeds Kaggle 20 GB limit.
            # The fold job only needs to produce metrics.json so the launcher can
            # choose the best training phase.  The actual model is saved only once
            # during the final retrain (split_mode='random') which runs after all folds.
            if ns.split_mode != 'loocv':
                ckpt = {
                    'model_state': _unwrap(model).state_dict(),
                    'tower': ns.tower,
                    'class_names': class_names,
                    'args': vars(ns),
                    'history': history,
                }
                torch.save(ckpt, outdir / 'last.ckpt')
                if val_metrics['acc'] >= best_acc:
                    torch.save(ckpt, outdir / 'best.ckpt')
    cleanup_distributed()


if __name__ == '__main__':
    main()
