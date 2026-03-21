from __future__ import annotations
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from fullres_dataset import scan_dataset, split_samples, PillFullResDataset
from fullres_model import FullResPillNet
from losses import CenterLoss, topk_accuracy


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    return {
        "global_img": torch.stack([b["global_img"] for b in batch], dim=0),
        "tiles": torch.stack([b["tiles"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "path": [b["path"] for b in batch],
        "class_name": [b["class_name"] for b in batch],
        "subgroup_name": [b["subgroup_name"] for b in batch],
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc1 = 0.0
    count = 0
    for batch in tqdm(loader, desc="valid", leave=False):
        g = batch["global_img"].to(device, non_blocking=True)
        t = batch["tiles"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        logits, emb = model(g, t)
        loss = loss_fn(logits, y)
        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_acc1 += topk_accuracy(logits, y, k=1) * bs
        count += bs
    return {"loss": total_loss / max(1, count), "acc1": total_acc1 / max(1, count)}


@torch.no_grad()
def build_prototypes(model, loader, device, num_classes: int, emb_dim: int):
    model.eval()
    embs = [[] for _ in range(num_classes)]
    for batch in tqdm(loader, desc="prototypes", leave=False):
        g = batch["global_img"].to(device, non_blocking=True)
        t = batch["tiles"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        _, emb = model(g, t)
        emb = emb.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        for e, label in zip(emb, y):
            embs[int(label)].append(e)

    prototypes = np.zeros((num_classes, emb_dim), dtype=np.float32)
    thresholds = np.zeros((num_classes,), dtype=np.float32)
    for idx in range(num_classes):
        cls_embs = np.asarray(embs[idx], dtype=np.float32)
        if len(cls_embs) == 0:
            continue
        proto = cls_embs.mean(axis=0)
        dists = np.linalg.norm(cls_embs - proto[None, :], axis=1)
        th = float(np.percentile(dists, 95) * 1.10 + 1e-6)
        prototypes[idx] = proto
        thresholds[idx] = th
    return prototypes, thresholds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--accum-steps", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--global-size", type=int, default=1024)
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--num-tiles", type=int, default=4)
    ap.add_argument("--emb-dim", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--center-loss-weight", type=float, default=0.02)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ns = ap.parse_args()

    set_seed(ns.seed)
    outdir = Path(ns.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    all_samples, class_names = scan_dataset(ns.data_root)
    train_samples, val_samples = split_samples(all_samples, val_ratio=ns.val_ratio, seed=ns.seed)

    train_ds = PillFullResDataset(train_samples, ns.global_size, ns.tile_size, ns.num_tiles, train=True)
    val_ds = PillFullResDataset(val_samples, ns.global_size, ns.tile_size, ns.num_tiles, train=False)

    train_loader = DataLoader(train_ds, batch_size=ns.batch_size, shuffle=True, num_workers=ns.num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=max(1, ns.batch_size), shuffle=False, num_workers=ns.num_workers, pin_memory=True, collate_fn=collate_fn)

    model = FullResPillNet(num_classes=len(class_names), emb_dim=ns.emb_dim, pretrained=True).to(device)
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.05)
    center_loss = CenterLoss(num_classes=len(class_names), feat_dim=ns.emb_dim).to(device)

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(center_loss.parameters()), lr=ns.lr, weight_decay=ns.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ns.epochs)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    meta = {"class_names": class_names, "train_count": len(train_ds), "val_count": len(val_ds), "args": vars(ns)}
    (outdir / "labels.json").write_text(json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "train_config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    best_acc = -1.0
    for epoch in range(1, ns.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        seen = 0
        run_loss = 0.0
        run_acc = 0.0

        pbar = tqdm(train_loader, desc=f"train {epoch}/{ns.epochs}")
        for step, batch in enumerate(pbar, start=1):
            g = batch["global_img"].to(device, non_blocking=True)
            t = batch["tiles"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            with autocast(enabled=torch.cuda.is_available()):
                logits, emb = model(g, t)
                loss_ce = ce_loss(logits, y)
                loss_ctr = center_loss(emb, y)
                loss = (loss_ce + ns.center_loss_weight * loss_ctr) / ns.accum_steps

            scaler.scale(loss).backward()
            if step % ns.accum_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            bs = y.size(0)
            seen += bs
            run_loss += float(loss_ce.item()) * bs
            run_acc += topk_accuracy(logits, y, k=1) * bs
            pbar.set_postfix({"loss": f"{run_loss/max(1,seen):.4f}", "acc1": f"{run_acc/max(1,seen):.4f}"})

        valid = evaluate(model, val_loader, device)
        scheduler.step()

        ckpt = {
            "model_state": model.state_dict(),
            "class_names": class_names,
            "args": vars(ns),
            "epoch": epoch,
            "val_acc1": valid["acc1"],
        }
        torch.save(ckpt, outdir / "last.ckpt")
        if valid["acc1"] > best_acc:
            best_acc = valid["acc1"]
            torch.save(ckpt, outdir / "best.ckpt")

        print({
            "epoch": epoch,
            "train_loss": run_loss / max(1, seen),
            "train_acc1": run_acc / max(1, seen),
            "val_loss": valid["loss"],
            "val_acc1": valid["acc1"],
        })

    best = torch.load(outdir / "best.ckpt", map_location="cpu")
    model.load_state_dict(best["model_state"], strict=True)
    model.to(device)

    proto_ds = PillFullResDataset(all_samples, ns.global_size, ns.tile_size, ns.num_tiles, train=False)
    proto_loader = DataLoader(proto_ds, batch_size=max(1, ns.batch_size), shuffle=False, num_workers=ns.num_workers, pin_memory=True, collate_fn=collate_fn)
    prototypes, thresholds = build_prototypes(model, proto_loader, device, len(class_names), ns.emb_dim)
    np.savez(outdir / "prototypes.npz", prototypes=prototypes, thresholds=thresholds)

    print("saved:", outdir)


if __name__ == "__main__":
    main()
