from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=8)
    ap.add_argument("--img", type=int, default=320)
    return ap.parse_args()


def main():
    args = parse_args()
    root = Path(args.data)
    if not root.exists():
        raise FileNotFoundError(root)

    out = {}
    for cls_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        imgs = []
        for p in sorted(cls_dir.rglob("*")):
            if p.suffix.lower() not in IMG_EXTS:
                continue
            img = Image.open(p).convert("RGB").resize((args.img, args.img))
            arr = np.asarray(img, dtype=np.uint8)
            imgs.append(arr)
            if len(imgs) >= args.per_class:
                break
        if imgs:
            out[cls_dir.name] = np.stack(imgs, axis=0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
