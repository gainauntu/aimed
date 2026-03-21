from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from fullres_dataset import scan_dataset, split_samples, PillFullResDataset
from fullres_model import FullResPillNet


# =========================================================
# basic utils
# =========================================================
def set_seed(seed: int, rank: int = 0) -> None:
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_dist():
        dist.barrier()


def cleanup_dist() -> None:
    if is_dist():
        dist.destroy_process_group()


def setup_distributed() -> Tuple[bool, int, int, int]:
    """
    torchrun sets:
      RANK, WORLD_SIZE, LOCAL_RANK
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return True, rank, world_size, local_rank

    return False, 0, 1, 0


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def collate_fn(batch: List[Dict]) -> Dict:
    return {
        "global_img": torch.stack([b["global_img"] for b in batch], dim=0),
        "tiles": torch.stack([b["tiles"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "path": [b["path"] for b in batch],
        "class_name": [b["class_name"] for b in batch],
        "subgroup_name": [b.get("subgroup_name", "") for b in batch],
    }


# =========================================================
# losses / metrics
# =========================================================
class CenterLoss(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        centers_batch = self.centers.index_select(0, labels)
        return ((features - centers_batch) ** 2).sum(dim=1).mean()


@torch.no_grad()
def topk_accuracy(logits: torch.Tensor, target: torch.Tensor, k: int = 1) -> float:
    k = min(k, logits.size(1))
    pred = logits.topk(k, dim=1).indices
    correct = pred.eq(target.view(-1, 1)).any(dim=1).float()
    return float(correct.mean().item())


# =========================================================
# model helpers
# =========================================================
def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def maybe_enable_grad_ckpt(model: nn.Module) -> None:
    core = unwrap_model(model)
    candidates = [
        "gradient_checkpointing_enable",
        "enable_gradient_checkpointing",
        "set_gradient_checkpointing",
    ]
    for fn_name in candidates:
        fn = getattr(core, fn_name, None)
        if callable(fn):
            try:
                fn()
                if is_main_process():
                    print(f"[GRAD_CKPT] enabled by {fn_name}()")
                return
            except TypeError:
                try:
                    fn(True)
                    if is_main_process():
                        print(f"[GRAD_CKPT] enabled by {fn_name}(True)")
                    return
                except Exception:
                    pass
            except Exception:
                pass

    backbone = getattr(core, "backbone", None)
    if backbone is not None:
        for fn_name in candidates:
            fn = getattr(backbone, fn_name, None)
            if callable(fn):
                try:
                    fn()
                    if is_main_process():
                        print(f"[GRAD_CKPT] backbone enabled by {fn_name}()")
                    return
                except TypeError:
                    try:
                        fn(True)
                        if is_main_process():
                            print(f"[GRAD_CKPT] backbone enabled by {fn_name}(True)")
                        return
                    except Exception:
                        pass
                except Exception:
                    pass

    if is_main_process():
        print("[GRAD_CKPT] requested, but model/backbone has no supported API. ignored.")


def forward_model(model: nn.Module, global_img: torch.Tensor, tiles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    out = model(global_img, tiles)

    if isinstance(out, tuple):
        if len(out) >= 2:
            return out[0], out[1]
        raise RuntimeError("Model returned tuple with insufficient length.")

    if isinstance(out, dict):
        logits = out.get("logits", None)
        emb = out.get("embedding", None)
        if emb is None:
            emb = out.get("emb", None)
        if emb is None:
            emb = out.get("features", None)
        if logits is None or emb is None:
            raise RuntimeError("Model returned dict but logits/embedding keys were not found.")
        return logits, emb

    raise RuntimeError(f"Unsupported model output type: {type(out)}")


def collect_named_module_params(module: nn.Module, names: List[str]) -> List[nn.Parameter]:
    params: List[nn.Parameter] = []
    for name in names:
        if hasattr(module, name):
            m = getattr(module, name)
            if isinstance(m, nn.Module):
                params.extend(list(m.parameters()))
    return params


def build_optimizer(model: nn.Module, center_loss: nn.Module, lr: float, backbone_lr_mul: float, weight_decay: float):
    core = unwrap_model(model)

    backbone_params: List[nn.Parameter] = []
    if hasattr(core, "backbone") and isinstance(core.backbone, nn.Module):
        backbone_params = list(core.backbone.parameters())

    # supports both old and new FullResPillNet variants
    head_candidates = [
        # old
        "tile_attn",
        "fuse",
        "classifier",

        # newer possible blocks
        "global_norm",
        "tile_in_norm",
        "tile_norm",
        "tile_blocks",
        "tile_pool",
        "cross_gate",
        "global_proj",
        "tile_proj",
        "proj",
        "bnneck",
        "head",
    ]

    head_params = collect_named_module_params(core, head_candidates)

    used = {id(p) for p in backbone_params + head_params}
    remaining_params = [p for p in core.parameters() if id(p) not in used]

    if is_main_process():
        present = [n for n in head_candidates if hasattr(core, n)]
        print(f"[MODEL] backbone={'yes' if hasattr(core, 'backbone') else 'no'}")
        print(f"[MODEL] head modules detected={present}")

    param_groups = []
    if len(backbone_params) > 0:
        param_groups.append({
            "params": backbone_params,
            "lr": lr * backbone_lr_mul,
            "weight_decay": weight_decay,
        })

    head_all = head_params + remaining_params + list(center_loss.parameters())
    param_groups.append({
        "params": head_all,
        "lr": lr,
        "weight_decay": weight_decay,
    })

    return torch.optim.AdamW(param_groups)


def choose_amp_dtype(prefer_bf16: bool) -> Tuple[bool, torch.dtype]:
    if not torch.cuda.is_available():
        return False, torch.float32

    if prefer_bf16 and torch.cuda.is_bf16_supported():
        return True, torch.bfloat16

    return True, torch.float16


# =========================================================
# reduction helpers
# =========================================================
def reduce_sum_scalar(value: float, device: torch.device) -> float:
    t = torch.tensor([value], dtype=torch.float64, device=device)
    if is_dist():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def reduce_sum_count(total: float, count: int, device: torch.device) -> Tuple[float, int]:
    t = torch.tensor([total, float(count)], dtype=torch.float64, device=device)
    if is_dist():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t[0].item()), int(round(float(t[1].item())))


# =========================================================
# eval / prototype
# =========================================================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp_enabled: bool, amp_dtype: torch.dtype) -> Dict[str, float]:
    model.eval()
    ce_loss = nn.CrossEntropyLoss()

    local_loss_sum = 0.0
    local_acc1_sum = 0.0
    local_count = 0

    iterator = loader
    if is_main_process():
        iterator = tqdm(loader, desc="valid", leave=False)

    for batch in iterator:
        g = batch["global_img"].to(device, non_blocking=True)
        t = batch["tiles"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
            logits, emb = forward_model(model, g, t)
            loss = ce_loss(logits, y)

        bs = y.size(0)
        local_loss_sum += float(loss.item()) * bs
        local_acc1_sum += topk_accuracy(logits, y, k=1) * bs
        local_count += bs

    total_loss_sum, total_count = reduce_sum_count(local_loss_sum, local_count, device)
    total_acc1_sum = reduce_sum_scalar(local_acc1_sum, device)

    return {
        "loss": total_loss_sum / max(1, total_count),
        "acc1": total_acc1_sum / max(1, total_count),
    }


@torch.no_grad()
def build_prototypes(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    emb_dim: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    embs = [[] for _ in range(num_classes)]

    iterator = tqdm(loader, desc="prototypes", leave=False) if is_main_process() else loader
    for batch in iterator:
        g = batch["global_img"].to(device, non_blocking=True)
        t = batch["tiles"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
            logits, emb = forward_model(model, g, t)

        emb_np = emb.detach().float().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        for e, label in zip(emb_np, y_np):
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


# =========================================================
# train loop
# =========================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--accum-steps", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)

    ap.add_argument("--global-size", type=int, default=1024)
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--num-tiles", type=int, default=6)

    ap.add_argument("--emb-dim", type=int, default=512)

    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--backbone-lr-mul", type=float, default=0.25)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--center-loss-weight", type=float, default=0.02)

    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--prefer-bf16", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true")

    ns = ap.parse_args()

    distributed, rank, world_size, local_rank = setup_distributed()

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(ns.seed, rank=rank)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    amp_enabled, amp_dtype = choose_amp_dtype(ns.prefer_bf16)

    if is_main_process():
        print(f"[DDP] distributed={distributed} world_size={world_size} rank={rank} local_rank={local_rank}")
        if torch.cuda.is_available():
            print(f"[CUDA] device_count={torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"[CUDA] gpu{i}={torch.cuda.get_device_name(i)}")
        print(f"[AMP] enabled={amp_enabled} dtype={amp_dtype}")
        print(f"[PATH] data_root={ns.data_root}")
        print(f"[PATH] outdir={ns.outdir}")

    outdir = Path(ns.outdir)
    if is_main_process():
        outdir.mkdir(parents=True, exist_ok=True)
    barrier()

    # dataset
    all_samples, class_names = scan_dataset(ns.data_root)
    train_samples, val_samples = split_samples(all_samples, val_ratio=ns.val_ratio, seed=ns.seed)

    train_ds = PillFullResDataset(
        train_samples,
        ns.global_size,
        ns.tile_size,
        ns.num_tiles,
        train=True,
    )
    val_ds = PillFullResDataset(
        val_samples,
        ns.global_size,
        ns.tile_size,
        ns.num_tiles,
        train=False,
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=ns.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=ns.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        persistent_workers=(ns.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, ns.batch_size),
        shuffle=False,
        sampler=val_sampler,
        num_workers=ns.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        persistent_workers=(ns.num_workers > 0),
    )

    # model
    model = FullResPillNet(
        num_classes=len(class_names),
        emb_dim=ns.emb_dim,
        pretrained=True,
    ).to(device)

    if ns.grad_ckpt:
        maybe_enable_grad_ckpt(model)

    center_loss = CenterLoss(num_classes=len(class_names), feat_dim=ns.emb_dim).to(device)
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.05)

    optimizer = build_optimizer(
        model=model,
        center_loss=center_loss,
        lr=ns.lr,
        backbone_lr_mul=ns.backbone_lr_mul,
        weight_decay=ns.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ns.epochs)

    use_grad_scaler = amp_enabled and amp_dtype == torch.float16 and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
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

    best_acc = -1.0

    for epoch in range(1, ns.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        local_loss_sum = 0.0
        local_acc1_sum = 0.0
        local_count = 0

        iterator = train_loader
        if is_main_process():
            iterator = tqdm(train_loader, desc=f"train {epoch}/{ns.epochs}")

        for step, batch in enumerate(iterator, start=1):
            g = batch["global_img"].to(device, non_blocking=True)
            t = batch["tiles"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            sync_now = (step % ns.accum_steps == 0) or (step == len(train_loader))
            sync_context = contextlib.nullcontext()
            if distributed and not sync_now:
                sync_context = model.no_sync()

            with sync_context:
                with torch.amp.autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
                    logits, emb = forward_model(model, g, t)
                    loss_ce = ce_loss(logits, y)
                    loss_ctr = center_loss(emb.float(), y)
                    loss_total = loss_ce + ns.center_loss_weight * loss_ctr
                    loss_for_backward = loss_total / ns.accum_steps

                if use_grad_scaler:
                    scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()

            if sync_now:
                if use_grad_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            bs = y.size(0)
            local_loss_sum += float(loss_ce.item()) * bs
            local_acc1_sum += topk_accuracy(logits, y, k=1) * bs
            local_count += bs

            if is_main_process():
                iterator.set_postfix({
                    "loss": f"{local_loss_sum / max(1, local_count):.4f}",
                    "acc1": f"{local_acc1_sum / max(1, local_count):.4f}",
                })

        train_loss_sum, train_count = reduce_sum_count(local_loss_sum, local_count, device)
        train_acc1_sum = reduce_sum_scalar(local_acc1_sum, device)

        train_metrics = {
            "loss": train_loss_sum / max(1, train_count),
            "acc1": train_acc1_sum / max(1, train_count),
        }

        val_metrics = evaluate(model, val_loader, device, amp_enabled, amp_dtype)
        scheduler.step()

        if is_main_process():
            core = unwrap_model(model)
            ckpt = {
                "model_state": core.state_dict(),
                "class_names": class_names,
                "args": vars(ns),
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc1": train_metrics["acc1"],
                "val_loss": val_metrics["loss"],
                "val_acc1": val_metrics["acc1"],
            }
            torch.save(ckpt, outdir / "last.ckpt")
            if val_metrics["acc1"] > best_acc:
                best_acc = val_metrics["acc1"]
                torch.save(ckpt, outdir / "best.ckpt")

            print(json.dumps({
                "epoch": epoch,
                "train_loss": round(train_metrics["loss"], 6),
                "train_acc1": round(train_metrics["acc1"], 6),
                "val_loss": round(val_metrics["loss"], 6),
                "val_acc1": round(val_metrics["acc1"], 6),
                "best_val_acc1": round(max(best_acc, val_metrics["acc1"]), 6),
            }, ensure_ascii=False))

        barrier()

    # prototype export on rank0 only
    if is_main_process():
        best_ckpt = torch.load(outdir / "best.ckpt", map_location="cpu")

        proto_model = FullResPillNet(
            num_classes=len(class_names),
            emb_dim=ns.emb_dim,
            pretrained=False,
        )
        proto_model.load_state_dict(best_ckpt["model_state"], strict=True)
        proto_model = proto_model.to(device).eval()

        proto_ds = PillFullResDataset(
            all_samples,
            ns.global_size,
            ns.tile_size,
            ns.num_tiles,
            train=False,
        )
        proto_loader = DataLoader(
            proto_ds,
            batch_size=max(1, ns.batch_size),
            shuffle=False,
            num_workers=ns.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
            persistent_workers=(ns.num_workers > 0),
        )

        prototypes, thresholds = build_prototypes(
            proto_model,
            proto_loader,
            device,
            num_classes=len(class_names),
            emb_dim=ns.emb_dim,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        np.savez(outdir / "prototypes.npz", prototypes=prototypes, thresholds=thresholds)

        print(f"[DONE] saved to: {outdir}")

    barrier()
    cleanup_dist()


if __name__ == "__main__":
    main()