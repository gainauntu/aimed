from __future__ import annotations
import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from fullres_dataset import scan_dataset, split_samples, PillFullResDataset
from fullres_model import FullResPillNet
from losses import CenterLoss, topk_accuracy


def set_seed(seed: int, rank: int = 0):
    seed = int(seed) + int(rank)
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


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if distributed:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)

    return distributed, rank, local_rank, world_size, device


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def ddp_sum_scalar(value: float, device: torch.device) -> float:
    t = torch.tensor([value], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def ddp_mean_scalar(value: float, device: torch.device) -> float:
    t = torch.tensor([value], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return float(t.item())


def get_amp_dtype(device: torch.device, prefer_bf16: bool = True):
    if device.type != "cuda":
        return torch.float32
    if prefer_bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss_local = 0.0
    total_acc1_local = 0.0
    count_local = 0

    iterator = loader
    if is_main_process():
        iterator = tqdm(loader, desc="valid", leave=False)

    for batch in iterator:
        g = batch["global_img"].to(device, non_blocking=True)
        t = batch["tiles"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        with autocast(device_type="cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits, emb = model(g, t)
            loss = loss_fn(logits, y)

        bs = y.size(0)
        total_loss_local += float(loss.item()) * bs
        total_acc1_local += topk_accuracy(logits, y, k=1) * bs
        count_local += bs

    total_loss = ddp_sum_scalar(total_loss_local, device)
    total_acc1 = ddp_sum_scalar(total_acc1_local, device)
    count = ddp_sum_scalar(count_local, device)

    return {
        "loss": total_loss / max(1.0, count),
        "acc1": total_acc1 / max(1.0, count),
    }


@torch.no_grad()
def build_prototypes(model, loader, device, num_classes: int, emb_dim: int, amp_dtype):
    model.eval()

    local_embs = [[] for _ in range(num_classes)]

    iterator = loader
    if is_main_process():
        iterator = tqdm(loader, desc="prototypes", leave=False)

    for batch in iterator:
        g = batch["global_img"].to(device, non_blocking=True)
        t = batch["tiles"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        with autocast(device_type="cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            _, emb = model(g, t)

        emb = emb.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        for e, label in zip(emb, y):
            local_embs[int(label)].append(e)

    if dist.is_available() and dist.is_initialized():
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local_embs)
    else:
        gathered = [local_embs]

    if not is_main_process():
        return None, None

    merged = [[] for _ in range(num_classes)]
    for rank_embs in gathered:
        for cls_idx in range(num_classes):
            merged[cls_idx].extend(rank_embs[cls_idx])

    prototypes = np.zeros((num_classes, emb_dim), dtype=np.float32)
    thresholds = np.zeros((num_classes,), dtype=np.float32)

    for idx in range(num_classes):
        cls_embs = np.asarray(merged[idx], dtype=np.float32)
        if len(cls_embs) == 0:
            continue

        proto = cls_embs.mean(axis=0)
        dists = np.linalg.norm(cls_embs - proto[None, :], axis=1)

        # 좀 더 robust 하게 threshold
        p95 = float(np.percentile(dists, 95))
        p99 = float(np.percentile(dists, 99))
        th = max(p95 * 1.10, p99 * 1.03) + 1e-6

        prototypes[idx] = proto
        thresholds[idx] = th

    return prototypes, thresholds


def maybe_enable_checkpointing(model: nn.Module):
    core = unwrap_model(model)
    backbone = getattr(core, "backbone", None)
    if backbone is None:
        return

    for m in [backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4]:
        for block in m:
            if hasattr(block, "gradient_checkpointing"):
                block.gradient_checkpointing = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=2, help="per-GPU batch size")
    ap.add_argument("--accum-steps", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)

    ap.add_argument("--global-size", type=int, default=1280)
    ap.add_argument("--tile-size", type=int, default=640)
    ap.add_argument("--num-tiles", type=int, default=6)
    ap.add_argument("--emb-dim", type=int, default=512)

    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--backbone-lr-mul", type=float, default=0.35)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--center-loss-weight", type=float, default=0.03)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--prefer-bf16", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true")
    ns = ap.parse_args()

    distributed, rank, local_rank, world_size, device = setup_ddp()
    set_seed(ns.seed, rank)

    torch.backends.cudnn.benchmark = True
    amp_dtype = get_amp_dtype(device, prefer_bf16=ns.prefer_bf16)

    outdir = Path(ns.outdir)
    if is_main_process():
        outdir.mkdir(parents=True, exist_ok=True)

    if is_main_process():
        print(f"[DDP] distributed={distributed} world_size={world_size} rank={rank} local_rank={local_rank}")
        print(f"[CUDA] device_count={torch.cuda.device_count()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"[CUDA] gpu{i}={torch.cuda.get_device_name(i)}")
        print(f"[AMP] dtype={amp_dtype}")

    all_samples, class_names = scan_dataset(ns.data_root)
    train_samples, val_samples = split_samples(all_samples, val_ratio=ns.val_ratio, seed=ns.seed)

    train_ds = PillFullResDataset(
        train_samples,
        global_size=ns.global_size,
        tile_size=ns.tile_size,
        num_tiles=ns.num_tiles,
        train=True,
        normalize=True,
    )
    val_ds = PillFullResDataset(
        val_samples,
        global_size=ns.global_size,
        tile_size=ns.tile_size,
        num_tiles=ns.num_tiles,
        train=False,
        normalize=True,
    )

    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False,
    ) if distributed else None

    val_sampler = DistributedSampler(
        val_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    ) if distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=ns.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=ns.num_workers,
        pin_memory=True,
        persistent_workers=(ns.num_workers > 0),
        collate_fn=collate_fn,
        drop_last=False,
        prefetch_factor=2 if ns.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, ns.batch_size),
        shuffle=False,
        sampler=val_sampler,
        num_workers=ns.num_workers,
        pin_memory=True,
        persistent_workers=(ns.num_workers > 0),
        collate_fn=collate_fn,
        drop_last=False,
        prefetch_factor=2 if ns.num_workers > 0 else None,
    )

    model = FullResPillNet(
        num_classes=len(class_names),
        emb_dim=ns.emb_dim,
        pretrained=True,
    ).to(device)

    if ns.grad_ckpt:
        maybe_enable_checkpointing(model)

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.05).to(device)
    center_loss = CenterLoss(num_classes=len(class_names), feat_dim=ns.emb_dim).to(device)

    core = unwrap_model(model)
    backbone_params = list(core.backbone.parameters())
    head_params = (
        list(core.tile_attn.parameters())
        + list(core.fuse.parameters())
        + list(core.classifier.parameters())
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": ns.lr * ns.backbone_lr_mul},
            {"params": head_params, "lr": ns.lr},
            {"params": center_loss.parameters(), "lr": ns.lr},
        ],
        weight_decay=ns.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ns.epochs)

    scaler = GradScaler(
        "cuda",
        enabled=(device.type == "cuda" and amp_dtype == torch.float16),
    )

    if is_main_process():
        meta = {
            "class_names": class_names,
            "train_count": len(train_ds),
            "val_count": len(val_ds),
            "args": vars(ns),
        }
        (outdir / "labels.json").write_text(
            json.dumps(class_names, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (outdir / "train_config.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    best_score = -1.0

    for epoch in range(1, ns.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        seen_local = 0
        run_loss_local = 0.0
        run_acc_local = 0.0

        iterator = train_loader
        if is_main_process():
            iterator = tqdm(train_loader, desc=f"train {epoch}/{ns.epochs}")

        for step, batch in enumerate(iterator, start=1):
            g = batch["global_img"].to(device, non_blocking=True)
            t = batch["tiles"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            with autocast(device_type="cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                logits, emb = model(g, t)
                loss_ce = ce_loss(logits, y)
                loss_ctr = center_loss(emb, y)
                loss_total = loss_ce + ns.center_loss_weight * loss_ctr
                loss = loss_total / ns.accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % ns.accum_steps == 0 or step == len(train_loader):
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            bs = y.size(0)
            seen_local += bs
            run_loss_local += float(loss_ce.item()) * bs
            run_acc_local += topk_accuracy(logits, y, k=1) * bs

            if is_main_process():
                iterator.set_postfix({
                    "loss": f"{run_loss_local / max(1, seen_local):.4f}",
                    "acc1": f"{run_acc_local / max(1, seen_local):.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                })

        train_seen = ddp_sum_scalar(seen_local, device)
        train_loss = ddp_sum_scalar(run_loss_local, device) / max(1.0, train_seen)
        train_acc1 = ddp_sum_scalar(run_acc_local, device) / max(1.0, train_seen)

        valid = evaluate(model, val_loader, device, amp_dtype)
        scheduler.step()

        score = valid["acc1"] - 0.05 * valid["loss"]

        if is_main_process():
            ckpt = {
                "model_state": unwrap_model(model).state_dict(),
                "class_names": class_names,
                "args": vars(ns),
                "epoch": epoch,
                "val_acc1": valid["acc1"],
                "val_loss": valid["loss"],
            }
            torch.save(ckpt, outdir / "last.ckpt")
            if score > best_score:
                best_score = score
                torch.save(ckpt, outdir / "best.ckpt")

            print({
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc1": round(train_acc1, 6),
                "val_loss": round(valid["loss"], 6),
                "val_acc1": round(valid["acc1"], 6),
                "score": round(score, 6),
            })

    if is_main_process():
        best = torch.load(outdir / "best.ckpt", map_location="cpu")
        unwrap_model(model).load_state_dict(best["model_state"], strict=True)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    proto_ds = PillFullResDataset(
        all_samples,
        global_size=ns.global_size,
        tile_size=ns.tile_size,
        num_tiles=ns.num_tiles,
        train=False,
        normalize=True,
    )

    proto_sampler = DistributedSampler(
        proto_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    ) if distributed else None

    proto_loader = DataLoader(
        proto_ds,
        batch_size=max(1, ns.batch_size),
        shuffle=False,
        sampler=proto_sampler,
        num_workers=ns.num_workers,
        pin_memory=True,
        persistent_workers=(ns.num_workers > 0),
        collate_fn=collate_fn,
        drop_last=False,
        prefetch_factor=2 if ns.num_workers > 0 else None,
    )

    prototypes, thresholds = build_prototypes(
        model=model,
        loader=proto_loader,
        device=device,
        num_classes=len(class_names),
        emb_dim=ns.emb_dim,
        amp_dtype=amp_dtype,
    )

    if is_main_process():
        np.savez(outdir / "prototypes.npz", prototypes=prototypes, thresholds=thresholds)
        print("saved:", outdir)

    cleanup_ddp()


if __name__ == "__main__":
    main()