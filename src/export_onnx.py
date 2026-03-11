from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from src.model import PillEncoder


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--img", type=int, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu")

    classes = ckpt["classes"]
    img_size = args.img or ckpt["img_size"]
    backbone = ckpt["backbone"]
    emb_dim = ckpt["emb_dim"]

    model = PillEncoder(
        num_classes=len(classes),
        backbone=backbone,
        emb_dim=emb_dim,
        pretrained=False,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, img_size, img_size)

    onnx_path = out_dir / "pill_cls.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["image"],
        output_names=["logits", "embedding"],
        opset_version=18,
        do_constant_folding=True,
    )

    with open(out_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    manifest = {
        "schema_version": 2,
        "model_type": "single_image_encoder_multi_fusion",
        "onnx": {
            "file": "pill_cls.onnx",
            "input_name": "image",
            "logits_output": "logits",
            "embedding_output": "embedding",
        },
        "preprocess": {
            "img_size": img_size,
            "color_space": "RGB",
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "runtime": {
            "min_images": 2,
            "max_images": 12,
        },
    }
    with open(out_dir / "pack_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {onnx_path}")


if __name__ == "__main__":
    main()
