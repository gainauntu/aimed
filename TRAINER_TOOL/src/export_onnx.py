from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from src.model import PillClassifier

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    classes = ckpt["classes"]
    img = int(ckpt.get("img", 448))
    backbone = ckpt.get("backbone","convnext_small")

    model = PillClassifier(n_classes=len(classes), backbone=backbone)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    onnx_path = out/"pill_cls.onnx"
    dummy = torch.zeros(1,3,img,img, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["image"],
        output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes={"image": {0:"batch"}, "logits": {0:"batch"}},
    )
    (out/"classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote:", onnx_path)

if __name__ == "__main__":
    main()
