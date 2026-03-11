from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from sklearn.metrics import f1_score

from src.dataset_images import (
    scan_class_folders,
    split_samples_stratified,
    PillImageDataset,
    build_train_transform,
    build_val_transform,
)
from src.losses import cross_entropy_loss, supervised_contrastive_loss
from src.model import PillEncoder
from src.utils import seed_everything, save_json, rank0


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--img", type=int, default=320)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--emb-dim", type=int, default=256)
    ap.add_argument("--backbone", type=str, default="convnext_tiny")
    ap.add_argument("--lambda-supcon", type=float, default=0.15)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true")
    return ap.parse_args()


def init_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return True, local_rank
    return False, 0


@torch.no_grad()
def validate(model, loader, device, amp_enabled):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0
    total = 0

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits, emb = model(x)
            loss_ce = cross_entropy_loss(logits, y)

        pred = logits.argmax(dim=1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
        total_loss += float(loss_ce.item()) * y.numel()
        total += y.numel()

    acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / max(1, len(y_true))
    macro_f1 = f1_score(y_true, y_pred, average="macro") if y_true else 0.0
    return {
        "val_loss_ce": total_loss / max(1, total),
        "val_acc": acc,
        "val_macro_f1": macro_f1,
    }


def main():
    args = parse_args()
    seed_everything(args.seed)
    ddp, local_rank = init_ddp()

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes, class_to_idx, samples = scan_class_folders(args.data)
    train_samples, val_samples = split_samples_stratified(samples, val_ratio=args.val_ratio, seed=args.seed)

    train_ds = PillImageDataset(train_samples, build_train_transform(args.img))
    val_ds = PillImageDataset(val_samples, build_val_transform(args.img))

    train_sampler = DistributedSampler(train_ds, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if ddp else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = PillEncoder(
        num_classes=len(classes),
        backbone=args.backbone,
        emb_dim=args.emb_dim,
        pretrained=True,
    ).to(device)

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    best_acc = -1.0
    history = []

    if rank0():
        save_json(classes, out_dir / "classes.json")

    for epoch in range(args.epochs):
        if ddp:
            train_sampler.set_epoch(epoch)

        model.train()
        t0 = time.time()
        running = {"loss": 0.0, "ce": 0.0, "supcon": 0.0, "n": 0}

        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                logits, emb = model(x)
                loss_ce = cross_entropy_loss(logits, y)
                loss_sc = supervised_contrastive_loss(emb, y)
                loss = loss_ce + args.lambda_supcon * loss_sc

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = y.numel()
            running["loss"] += float(loss.item()) * bs
            running["ce"] += float(loss_ce.item()) * bs
            running["supcon"] += float(loss_sc.item()) * bs
            running["n"] += bs

        val_metrics = validate(model, val_loader, device, args.amp)
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": running["loss"] / max(1, running["n"]),
            "train_loss_ce": running["ce"] / max(1, running["n"]),
            "train_loss_supcon": running["supcon"] / max(1, running["n"]),
            **val_metrics,
            "seconds": time.time() - t0,
        }
        history.append(epoch_metrics)

        if rank0():
            print(
                f"epoch={epoch+1}/{args.epochs} "
                f"train_loss={epoch_metrics['train_loss']:.4f} "
                f"val_acc={epoch_metrics['val_acc']:.4f} "
                f"val_macro_f1={epoch_metrics['val_macro_f1']:.4f}"
            )

            ckpt = {
                "state_dict": model.module.state_dict() if ddp else model.state_dict(),
                "classes": classes,
                "img_size": args.img,
                "backbone": args.backbone,
                "emb_dim": args.emb_dim,
                "lambda_supcon": args.lambda_supcon,
            }
            torch.save(ckpt, out_dir / "last.pt")
            save_json(history, out_dir / "metrics.json")

            if epoch_metrics["val_acc"] > best_acc:
                best_acc = epoch_metrics["val_acc"]
                torch.save(ckpt, out_dir / "best.pt")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
