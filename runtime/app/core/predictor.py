from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import json
import logging

import numpy as np
import torch

from app.core.preprocess import read_image_unicode, to_model_input, extract_tiles_and_boxes, compute_quality_score
from app.core.reject import FrameEvidence, decide

LOGGER = logging.getLogger("pill_runtime")


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.clip(np.sum(e, axis=-1, keepdims=True), 1e-12, None)


@dataclass
class FramePrediction:
    path: str
    quality_score: float
    crop_ok: bool
    crop_ratio: float
    bbox: tuple[int, int, int, int]
    tile_boxes: list[tuple[int, int, int, int]]
    probs: np.ndarray
    top3: list[Dict[str, float]]
    embedding: Optional[np.ndarray]
    prototype_distance: Optional[float]
    prototype_threshold: Optional[float]


class RuntimeModelAdapter:
    def __init__(self, bundle_dir: str | Path):
        self.bundle_dir = Path(bundle_dir)
        labels_path = self.bundle_dir / "labels.json"
        model_path = self.bundle_dir / "model.ts"
        if not labels_path.exists():
            raise FileNotFoundError(f"labels.json not found: {labels_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"model.ts not found: {model_path}")

        self.labels = json.loads(labels_path.read_text(encoding="utf-8"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(str(model_path), map_location="cpu").to(self.device).eval()

        self.prototypes = None
        self.thresholds = None
        proto_path = self.bundle_dir / "prototypes.npz"
        if proto_path.exists():
            pack = np.load(proto_path)
            self.prototypes = pack["prototypes"].astype(np.float32)
            self.thresholds = pack["thresholds"].astype(np.float32)

    @torch.no_grad()
    def _run_model(self, global_chw: np.ndarray, tiles_tchw: np.ndarray):
        g = torch.from_numpy(global_chw[None, ...]).float().to(self.device)
        t = torch.from_numpy(tiles_tchw[None, ...]).float().to(self.device)
        out = self.model(g, t)

        if isinstance(out, dict):
            logits = out["logits"]
            emb = out.get("embedding")
        elif isinstance(out, (tuple, list)):
            logits = out[0]
            emb = out[1] if len(out) > 1 else None
        else:
            logits = out
            emb = None

        if torch.is_tensor(logits):
            logits = logits.detach().cpu().numpy()
        if emb is not None and torch.is_tensor(emb):
            emb = emb.detach().cpu().numpy()

        return logits[0], None if emb is None else emb[0]

    def prototype_info_for_class(self, emb: Optional[np.ndarray], class_idx: int):
        if emb is None or self.prototypes is None or self.thresholds is None:
            return None, None
        p = self.prototypes[class_idx]
        dist = float(np.linalg.norm(emb - p))
        th = float(self.thresholds[class_idx])
        return dist, th

    def top3_from_prob(self, prob: np.ndarray):
        idxs = np.argsort(prob)[::-1][:3]
        return [{"class_name": self.labels[i], "score": round(float(prob[i] * 100.0), 2)} for i in idxs]


class MultiImagePredictor:
    def __init__(self, bundle_dir: str | Path, runtime_cfg: Optional[dict] = None):
        self.runtime_cfg = runtime_cfg or {}
        self.adapter = RuntimeModelAdapter(bundle_dir)
        self.global_size = int(self.runtime_cfg.get("global_size", 1024))
        self.tile_size = int(self.runtime_cfg.get("tile_size", 512))
        self.num_tiles = int(self.runtime_cfg.get("num_tiles", 4))
        self.min_images = int(self.runtime_cfg.get("min_images", 2))
        self.max_images = int(self.runtime_cfg.get("max_images", 8))

    def _frame_predict(self, path: str) -> FramePrediction:
        bgr = read_image_unicode(path)
        q = compute_quality_score(bgr)
        global_chw = to_model_input(bgr, self.global_size)
        tiles_tchw, tile_boxes, loc = extract_tiles_and_boxes(bgr, self.tile_size, self.num_tiles)

        logits, emb = self.adapter._run_model(global_chw, tiles_tchw)
        probs = _softmax_np(logits[None, ...])[0]
        top3 = self.adapter.top3_from_prob(probs)
        winner_idx = int(np.argmax(probs))
        pdist, pth = self.adapter.prototype_info_for_class(emb, winner_idx)

        bbox = list(map(int, loc.bbox))
        LOGGER.info(
            "[FRAME] path=%s q=%.2f crop_ok=%s crop_ratio=%.3f bbox=(%d, %d, %d, %d) tiles=%s top3=%s proto_dist=%s proto_th=%s",
            path, q, loc.ok, loc.crop_ratio, bbox[0], bbox[1], bbox[2], bbox[3], tile_boxes, top3, pdist, pth
        )

        return FramePrediction(
            path=path,
            quality_score=float(q),
            crop_ok=bool(loc.ok),
            crop_ratio=float(loc.crop_ratio),
            bbox=tuple(bbox),
            tile_boxes=[tuple(map(int, b)) for b in tile_boxes],
            probs=probs,
            top3=top3,
            embedding=emb,
            prototype_distance=pdist,
            prototype_threshold=pth,
        )

    def _aggregate(self, frames: list[FramePrediction]):
        probs = np.stack([f.probs for f in frames], axis=0)
        fused_prob = probs.mean(axis=0)
        winner_idx = int(np.argmax(fused_prob))
        winner_name = self.adapter.labels[winner_idx]

        sorted_idx = np.argsort(fused_prob)[::-1]
        top1 = float(fused_prob[sorted_idx[0]] * 100.0)
        top2 = float(fused_prob[sorted_idx[1]] * 100.0) if len(sorted_idx) > 1 else 0.0
        margin = top1 - top2
        consensus = 100.0 * sum(int(np.argmax(f.probs)) == winner_idx for f in frames) / max(1, len(frames))
        avg_quality = float(np.mean([f.quality_score for f in frames])) if frames else 0.0

        emb_list = [f.embedding for f in frames if f.embedding is not None]
        fused_emb = None if not emb_list else np.mean(np.stack(emb_list, axis=0), axis=0)
        proto_dist, proto_th = self.adapter.prototype_info_for_class(fused_emb, winner_idx)

        evidences = []
        for f in frames:
            top1_name = f.top3[0]["class_name"]
            top1_conf = float(f.top3[0]["score"])
            top2_conf = float(f.top3[1]["score"]) if len(f.top3) > 1 else 0.0
            evidences.append(
                FrameEvidence(
                    path=f.path,
                    usable=True,
                    quality_score=f.quality_score,
                    top1_name=top1_name,
                    top1_conf=top1_conf,
                    margin=top1_conf - top2_conf,
                    prototype_distance=f.prototype_distance,
                    prototype_threshold=f.prototype_threshold,
                )
            )

        return {
            "winner_name": winner_name,
            "top1": round(top1, 2),
            "margin": round(margin, 2),
            "consensus": round(consensus, 2),
            "prototype_distance": proto_dist,
            "prototype_threshold": proto_th,
            "avg_quality": avg_quality,
            "frame_evidences": evidences,
        }

    def predict(self, request_id: str, image_paths: list[str]):
        if not image_paths:
            return {"request_id": request_id, "status": "ERROR", "class_name": None, "confidence_score": 0.0, "reason": "image_paths is empty", "reject_code": "EMPTY_INPUT", "debug": {}}

        image_paths = image_paths[: self.max_images]
        LOGGER.info("[REQ] count=%d images=%s", len(image_paths), image_paths)

        frames = [self._frame_predict(p) for p in image_paths]
        fused = self._aggregate(frames)

        decision = decide(
            class_name=fused["winner_name"],
            top1_prob=fused["top1"],
            margin=fused["margin"],
            consensus_score=fused["consensus"],
            prototype_distance=fused["prototype_distance"],
            prototype_threshold=fused["prototype_threshold"],
            usable_frames=len(frames),
            min_images=self.min_images,
            avg_quality=fused["avg_quality"],
            frame_evidences=fused["frame_evidences"],
            set_purity_cfg=self.runtime_cfg.get("set_purity"),
            decision_cfg=self.runtime_cfg.get("decision"),
        )

        LOGGER.info(
            "%s status=%s best=%s conf=%s top1=%.2f margin=%.2f consensus=%.2f proto_dist=%s proto_th=%s reject=%s",
            request_id, decision.status, decision.class_name, decision.confidence_score,
            fused["top1"], fused["margin"], fused["consensus"],
            fused["prototype_distance"], fused["prototype_threshold"], decision.reject_code
        )

        return {
            "request_id": request_id,
            "status": decision.status,
            "class_name": decision.class_name,
            "confidence_score": decision.confidence_score,
            "reason": decision.reason,
            "reject_code": decision.reject_code,
            "debug": {
                "winner_name": fused["winner_name"],
                "top1": fused["top1"],
                "margin": fused["margin"],
                "consensus": fused["consensus"],
                "prototype_distance": fused["prototype_distance"],
                "prototype_threshold": fused["prototype_threshold"],
                "frames": [
                    {
                        "path": f.path,
                        "quality_score": round(f.quality_score, 2),
                        "crop_ok": f.crop_ok,
                        "crop_ratio": round(f.crop_ratio, 3),
                        "bbox": list(f.bbox),
                        "tile_boxes": [list(tb) for tb in f.tile_boxes],
                        "top3": f.top3,
                        "prototype_distance": f.prototype_distance,
                        "prototype_threshold": f.prototype_threshold,
                    }
                    for f in frames
                ],
            },
        }
