# TRAINER_TOOL/src/train.py
from __future__ import annotations

import argparse, json, os, random
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from src.dataset import FolderDataset
from src.model import PillClassifier


def is_dist() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def ddp_init():
    # NCCL is correct on Linux GPU (Kaggle)
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(get_local_rank())


def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_classes(root: Path) -> List[str]:
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--img", type=int, default=448)
    ap.add_argument("--batch", type=int, default=16, help="per-GPU batch size in DDP")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backbone", default="convnext_small")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--grad_clip", type=float, default=0.0)
    args = ap.parse_args()

    distributed = is_dist()
    rank = get_rank()
    world = get_world_size()
    local_rank = get_local_rank()

    if distributed:
        ddp_init()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("This DDP script expects CUDA on Kaggle/Colab GPU runtime.")

    # Speed for fixed shapes
    torch.backends.cudnn.benchmark = True

    # seed per-rank (so augmentation differs between ranks but reproducible)
    set_seed(args.seed + rank)

    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)

    root = Path(args.data)
    classes_all = list_classes(root)
    classes = classes_all[: args.classes] if args.classes > 0 else classes_all

    ds = FolderDataset(str(root), classes, img_size=args.img, train=True, seed=args.seed)
    if len(ds) < 10:
        raise SystemExit("Dataset too small or wrong path. Check --data folder.")

    n_val = max(1, int(len(ds) * args.val_ratio))
    n_train = max(1, len(ds) - n_val)
    train_ds, val_ds = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    # no augmentation for val
    val_ds.dataset.train = False

    # Distributed samplers
    train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
    )

    model = PillClassifier(n_classes=len(classes), backbone=args.backbone).to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
        )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)

    use_amp = bool(args.amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_acc = -1.0
    best_path = out / "best.pt"

    def reduce_mean(x: float) -> float:
        # average a scalar across ranks
        t = torch.tensor([x], device=device, dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= world
        return float(t.item())

    for epoch in range(1, args.epochs + 1):
        if distributed:
            train_sampler.set_epoch(epoch)

        model.train()
        tl = 0.0
        correct = 0
        total = 0

        it = train_loader
        if rank == 0:
            it = tqdm(train_loader, desc=f"train {epoch}/{args.epochs}")

        for x, y in it:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = ce(logits, y)

            scaler.scale(loss).backward()

            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(opt)
            scaler.update()

            tl += float(loss.item()) * x.size(0)
            pred = logits.argmax(1)
            correct += int((pred == y).sum().item())
            total += int(x.size(0))

            if rank == 0 and hasattr(it, "set_postfix"):
                it.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/max(1,total):.3f}")

        # compute local metrics
        train_loss = tl / max(1, total)
        train_acc = correct / max(1, total)

        # Validation
        model.eval()
        vl = 0.0
        vcorrect = 0
        vtotal = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(x)
                    loss = ce(logits, y)

                vl += float(loss.item()) * x.size(0)
                pred = logits.argmax(1)
                vcorrect += int((pred == y).sum().item())
                vtotal += int(x.size(0))

        val_loss = vl / max(1, vtotal)
        val_acc = vcorrect / max(1, vtotal)

        # reduce metrics across ranks so rank0 prints global numbers
        if distributed:
            train_loss = reduce_mean(train_loss)
            train_acc = reduce_mean(train_acc)
            val_loss = reduce_mean(val_loss)
            val_acc = reduce_mean(val_acc)

        if rank == 0:
            print(
                f"Epoch {epoch} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                # unwrap DDP
                state = model.module.state_dict() if distributed else model.state_dict()
                torch.save(
                    {
                        "model": state,
                        "classes": classes,
                        "img": args.img,
                        "backbone": args.backbone,
                    },
                    best_path,
                )
                print("Saved best:", best_path, "val_acc=", best_acc)

    if rank == 0:
        (out / "classes.json").write_text(
            json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out / "calibration.json").write_text(
            json.dumps(
                {
                    "img": args.img,
                    "min_conf_ok": 85.0,
                    "min_margin_ok": 12.0,
                    "ocr_min_quality": 0.75,
                    "ocr_min_conf": 0.85,
                    "verifier_trigger_margin": 15.0,
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "color_space": "RGB",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    if distributed:
        ddp_cleanup()


if __name__ == "__main__":
    main()