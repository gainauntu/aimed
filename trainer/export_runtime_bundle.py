from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import torch

from fullres_model import FullResPillNet, TorchScriptWrapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--prototypes", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--global-size", type=int, default=1024)
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--num-tiles", type=int, default=4)
    ap.add_argument("--min-images", type=int, default=2)
    ap.add_argument("--max-images", type=int, default=8)
    ns = ap.parse_args()

    outdir = Path(ns.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    class_names = json.loads(Path(ns.labels).read_text(encoding="utf-8"))
    ckpt = torch.load(ns.checkpoint, map_location="cpu")
    emb_dim = int(ckpt["args"].get("emb_dim", 512))

    model = FullResPillNet(num_classes=len(class_names), emb_dim=emb_dim, pretrained=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    wrapper = TorchScriptWrapper(model).eval()
    example_g = torch.randn(1, 3, ns.global_size, ns.global_size)
    example_t = torch.randn(1, ns.num_tiles, 3, ns.tile_size, ns.tile_size)
    ts = torch.jit.trace(wrapper, (example_g, example_t), strict=False)
    ts.save(str(outdir / "model.ts"))

    shutil.copy2(ns.labels, outdir / "labels.json")
    shutil.copy2(ns.prototypes, outdir / "prototypes.npz")

    runtime_cfg = {
        "global_size": ns.global_size,
        "tile_size": ns.tile_size,
        "num_tiles": ns.num_tiles,
        "min_images": ns.min_images,
        "max_images": ns.max_images,
        "decision": {
            "min_avg_quality": 20.0,
            "default_top1_cut": 60.0,
            "default_margin_cut": 20.0,
            "default_consensus_cut": 60.0,
            "default_final_conf_cut": 70.0,
            "class_overrides": {
                "SHP 50": {"top1_cut": 57.0, "margin_cut": 18.0, "final_conf_cut": 65.0},
                "SPP RM": {"top1_cut": 57.0, "margin_cut": 18.0, "final_conf_cut": 65.0},
            },
        },
        "set_purity": {
            "enabled": True,
            "min_quality": 25.0,
            "min_support_conf": 58.0,
            "min_support_margin": 10.0,
            "min_soft_quality": 18.0,
            "min_soft_support_conf": 45.0,
            "min_soft_support_margin": 5.0,
            "min_contradict_conf": 72.0,
            "min_contradict_margin": 12.0,
            "max_contradictions": 0,
            "min_support_ratio": 0.50,
            "min_support_frames": 1,
        },
    }
    (outdir / "runtime_config.json").write_text(json.dumps(runtime_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved runtime bundle:", outdir)


if __name__ == "__main__":
    main()
