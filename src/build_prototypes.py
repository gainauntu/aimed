from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset_images import scan_class_folders, PillImageDataset, build_val_transform
from src.model import PillEncoder


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    return ap.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    classes = ckpt["classes"]

    model = PillEncoder(
        num_classes=len(classes),
        backbone=ckpt["backbone"],
        emb_dim=ckpt["emb_dim"],
        pretrained=False,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    _, _, samples = scan_class_folders(args.data)
    ds = PillImageDataset(samples, build_val_transform(ckpt["img_size"]))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2)

    all_embs = {i: [] for i in range(len(classes))}
    with torch.no_grad():
        for x, y, _ in loader:
            logits, emb = model(x)
            emb = emb.cpu().numpy()
            for i in range(len(y)):
                all_embs[int(y[i])].append(emb[i])

    out = {}
    for cls_idx, cls_name in enumerate(classes):
        arr = np.stack(all_embs[cls_idx], axis=0)
        centroid = arr.mean(axis=0)
        dists = np.linalg.norm(arr - centroid[None, :], axis=1)

        out[f"{cls_name}__centroid"] = centroid.astype(np.float32)
        out[f"{cls_name}__mean_dist"] = np.array([dists.mean()], dtype=np.float32)
        out[f"{cls_name}__std_dist"] = np.array([dists.std()], dtype=np.float32)
        out[f"{cls_name}__p95_dist"] = np.array([np.percentile(dists, 95)], dtype=np.float32)
        out[f"{cls_name}__p99_dist"] = np.array([np.percentile(dists, 99)], dtype=np.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
