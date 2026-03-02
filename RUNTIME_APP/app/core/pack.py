from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from app.core.paths import models_dir

@dataclass
class PackONNX:
    file: Path
    input_mode: str = "auto"
    input_names: List[str] = None
    output_name: str = ""

@dataclass
class PackPreprocess:
    img_size: int
    color_space: str
    mean: List[float]
    std: List[float]
    crop_rotate: bool = True

@dataclass
class PackDecision:
    min_conf_ok: float
    min_margin_ok: float
    ocr_min_quality: float
    ocr_min_conf: float
    verifier_trigger_margin: float

@dataclass
class ModelPack:
    root: Path
    schema_version: int
    onnx: PackONNX
    preprocess: PackPreprocess
    decision: PackDecision
    classes: List[str]
    imprint_db: Dict[str, Any]

def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))

def load_pack() -> ModelPack:
    root = models_dir()
    manifest_path = root / "pack_manifest.json"
    if manifest_path.exists():
        man = _read_json(manifest_path)
        schema = int(man.get("schema_version", 1))
        onnx_info = man.get("onnx", {})
        pre = man.get("preprocess", {})
        dec = man.get("decision", {})
        onnx = PackONNX(
            file=root / str(onnx_info.get("file","pill_cls.onnx")),
            input_mode=str(onnx_info.get("input_mode","auto")),
            input_names=list(onnx_info.get("input_names") or []),
            output_name=str(onnx_info.get("output_name","")),
        )
        norm = pre.get("normalize", {})
        pp = PackPreprocess(
            img_size=int(pre.get("img_size", 448)),
            color_space=str(pre.get("color_space","RGB")),
            mean=list(norm.get("mean", [0.485,0.456,0.406])),
            std=list(norm.get("std", [0.229,0.224,0.225])),
            crop_rotate=bool(pre.get("crop_rotate", {}).get("enabled", True)),
        )
        dd = PackDecision(
            min_conf_ok=float(dec.get("min_conf_ok", 85.0)),
            min_margin_ok=float(dec.get("min_margin_ok", 12.0)),
            ocr_min_quality=float(dec.get("ocr_min_quality", 0.75)),
            ocr_min_conf=float(dec.get("ocr_min_conf", 0.85)),
            verifier_trigger_margin=float(dec.get("verifier_trigger_margin", 15.0)),
        )
    else:
        cal = _read_json(root/"calibration.json")
        schema = 0
        onnx = PackONNX(
            file=root/"pill_cls.onnx",
            input_mode=str(cal.get("input_mode","auto")),
            input_names=list(cal.get("input_names") or []),
            output_name=str(cal.get("output_name",""))
        )
        pp = PackPreprocess(
            img_size=int(cal.get("img", 448)),
            color_space=str(cal.get("color_space","RGB")),
            mean=list(cal.get("mean", [0.485,0.456,0.406])),
            std=list(cal.get("std", [0.229,0.224,0.225])),
            crop_rotate=True,
        )
        dd = PackDecision(
            min_conf_ok=float(cal.get("min_conf_ok", 85.0)),
            min_margin_ok=float(cal.get("min_margin_ok", 12.0)),
            ocr_min_quality=float(cal.get("ocr_min_quality", 0.75)),
            ocr_min_conf=float(cal.get("ocr_min_conf", 0.85)),
            verifier_trigger_margin=float(cal.get("verifier_trigger_margin", 15.0)),
        )
    classes = _read_json(root/"classes.json") if (root/"classes.json").exists() else []
    imprint = _read_json(root/"imprint_db.json") if (root/"imprint_db.json").exists() else {}
    return ModelPack(root=root, schema_version=schema, onnx=onnx, preprocess=pp, decision=dd, classes=classes, imprint_db=imprint)

def validate_pack(pack: ModelPack) -> Tuple[bool, List[str]]:
    errs = []
    if not pack.onnx.file.exists():
        errs.append(f"Missing ONNX file: {pack.onnx.file}")
    if not pack.classes:
        errs.append("classes.json is missing or empty")
    if pack.preprocess.img_size <= 0:
        errs.append("Invalid preprocess.img_size")
    if len(pack.preprocess.mean) != 3 or len(pack.preprocess.std) != 3:
        errs.append("Mean/Std must be length 3")
    return (len(errs)==0, errs)
