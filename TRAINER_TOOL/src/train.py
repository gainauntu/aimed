from __future__ import annotations
import argparse, json, random
from pathlib import Path
from typing import List
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.dataset import FolderDataset
from src.model import PillClassifier

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
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
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backbone", default="convnext_small")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    root = Path(args.data)

    classes_all = list_classes(root)
    classes = classes_all[:args.classes] if args.classes > 0 else classes_all

    ds = FolderDataset(str(root), classes, img_size=args.img, train=True, seed=args.seed)
    if len(ds) < 10:
        raise SystemExit("Dataset too small or wrong path. Check --data folder.")
    n_val = max(1, int(len(ds)*args.val_ratio))
    n_train = max(1, len(ds) - n_val)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    # turn off augmentation for validation
    val_ds.dataset.train = False

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    model = PillClassifier(n_classes=len(classes), backbone=args.backbone).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_acc = -1.0
    best_path = out/"best.pt"

    for epoch in range(1, args.epochs+1):
        model.train()
        tl=0.0; correct=0; total=0
        for x,y in tqdm(train_loader, desc=f"train {epoch}/{args.epochs}"):
            x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            tl += float(loss.item())*x.size(0)
            pred = logits.argmax(1)
            correct += int((pred==y).sum().item())
            total += int(x.size(0))
        train_loss = tl/max(1,total)
        train_acc = correct/max(1,total)

        model.eval()
        vl=0.0; vcorrect=0; vtotal=0
        with torch.no_grad():
            for x,y in tqdm(val_loader, desc="val"):
                x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True)
                logits = model(x)
                loss = ce(logits, y)
                vl += float(loss.item())*x.size(0)
                pred = logits.argmax(1)
                vcorrect += int((pred==y).sum().item())
                vtotal += int(x.size(0))
        val_loss = vl/max(1,vtotal)
        val_acc = vcorrect/max(1,vtotal)

        print(f"Epoch {epoch} train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "classes": classes, "img": args.img, "backbone": args.backbone}, best_path)
            print("Saved best:", best_path, "val_acc=", best_acc)

    (out/"classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")
    (out/"calibration.json").write_text(json.dumps({
        "img": args.img,
        "min_conf_ok": 85.0,
        "min_margin_ok": 12.0,
        "ocr_min_quality": 0.75,
        "ocr_min_conf": 0.85,
        "verifier_trigger_margin": 15.0,
        "mean":[0.485,0.456,0.406],
        "std":[0.229,0.224,0.225],
        "color_space":"RGB"
    }, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
