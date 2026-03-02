from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime_models_dir", required=True)
    ap.add_argument("--pill_cls_onnx", required=True)
    ap.add_argument("--classes_json", required=True)
    ap.add_argument("--calibration_json", required=True)
    ap.add_argument("--imprint_db_json", required=False)
    ap.add_argument("--gallery_refs_npz", required=False)
    ap.add_argument("--version", default="exported-pack-v1")
    ap.add_argument("--input_mode", default="auto")
    ap.add_argument("--input_names", default="")
    ap.add_argument("--output_name", default="")
    args = ap.parse_args()

    dst = Path(args.runtime_models_dir)
    dst.mkdir(parents=True, exist_ok=True)

    shutil.copy2(args.pill_cls_onnx, dst/"pill_cls.onnx")
    shutil.copy2(args.classes_json, dst/"classes.json")
    shutil.copy2(args.calibration_json, dst/"calibration.json")
    if args.imprint_db_json:
        shutil.copy2(args.imprint_db_json, dst/"imprint_db.json")
    if args.gallery_refs_npz:
        shutil.copy2(args.gallery_refs_npz, dst/"gallery_refs.npz")

    cal = json.loads(Path(args.calibration_json).read_text(encoding="utf-8"))
    img = int(cal.get("img", cal.get("img_size", 448)))
    mean = cal.get("mean", [0.485,0.456,0.406])
    std  = cal.get("std",  [0.229,0.224,0.225])

    man = {
        "schema_version": 1,
        "model_type": "pill_classifier",
        "onnx": {
            "file": "pill_cls.onnx",
            "input_mode": args.input_mode,
            "input_names": [s.strip() for s in args.input_names.split(",") if s.strip()],
            "output_name": args.output_name
        },
        "preprocess": {
            "img_size": img,
            "color_space": cal.get("color_space","RGB"),
            "normalize": {"mean": mean, "std": std},
            "crop_rotate": {"enabled": True}
        },
        "decision": {
            "min_conf_ok": float(cal.get("min_conf_ok", 85.0)),
            "min_margin_ok": float(cal.get("min_margin_ok", 12.0)),
            "ocr_min_quality": float(cal.get("ocr_min_quality", 0.75)),
            "ocr_min_conf": float(cal.get("ocr_min_conf", 0.85)),
            "verifier_trigger_margin": float(cal.get("verifier_trigger_margin", 15.0))
        }
    }
    (dst/"pack_manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    (dst/"version.json").write_text(json.dumps({"model_pack_version": args.version}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Exported model pack to:", dst)

if __name__ == "__main__":
    main()
