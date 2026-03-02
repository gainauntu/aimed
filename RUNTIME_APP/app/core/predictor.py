from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2

from app.core.pack import load_pack, validate_pack
from app.core.providers.onnx_runtime import ORTModel
from app.core.providers.ocr_provider import OCRProvider
from app.core.pipelines.preprocess import preprocess
from app.core.pipelines.imprint import propose_imprint_roi, enhance_for_ocr
from app.core.pipelines.verifier import verify_against_gallery

def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    return e/(np.sum(e)+1e-12)

def _norm_text(s: str, rules: dict) -> str:
    if not s:
        return ""
    t = s
    if rules.get("upper", True):
        t = t.upper()
    if rules.get("strip_spaces", True):
        t = "".join(t.split())
    conf = rules.get("common_confusions", {})
    for k,v in conf.items():
        t = t.replace(k, v)
    return t

@dataclass
class PredictOutput:
    status: str
    best_name: Optional[str]
    conf_percent: float
    margin: float
    top3: List[Tuple[str, float]]
    ocr_text: str
    ocr_conf: float
    ocr_quality: float
    verifier_score: float
    reason: str

class UnifiedPredictor:
    def __init__(self, prefer: str = "auto"):
        self.pack = load_pack()
        ok, errs = validate_pack(self.pack)
        if not ok:
            raise RuntimeError("Invalid model pack:\n- " + "\n- ".join(errs))

        self.classes = list(self.pack.classes)
        self.imprint_db = dict(self.pack.imprint_db or {})
        self.rules = self.imprint_db.get("_normalize_rules", {"upper":True,"strip_spaces":True,"common_confusions":{}})

        self.img = int(self.pack.preprocess.img_size)
        self.mean = np.array(self.pack.preprocess.mean, dtype=np.float32)
        self.std  = np.array(self.pack.preprocess.std, dtype=np.float32)
        self.color_space = str(self.pack.preprocess.color_space).upper()

        self.min_conf_ok = float(self.pack.decision.min_conf_ok)
        self.min_margin_ok = float(self.pack.decision.min_margin_ok)
        self.ocr_min_quality = float(self.pack.decision.ocr_min_quality)
        self.ocr_min_conf = float(self.pack.decision.ocr_min_conf)
        self.verifier_trigger_margin = float(self.pack.decision.verifier_trigger_margin)

        self.cls = ORTModel(
            str(self.pack.onnx.file),
            prefer=prefer,
            input_names_override=(self.pack.onnx.input_names or None),
            output_name_override=self.pack.onnx.output_name or "",
        )
        self.input_mode = str(self.pack.onnx.input_mode or "auto").lower()

        self.ocr = OCRProvider(lang="korean", use_gpu=False)

        self.gallery: Dict[str, List[np.ndarray]] = {}
        gal_path = self.pack.root/"gallery_refs.npz"
        if gal_path.exists():
            data = np.load(str(gal_path), allow_pickle=True)
            for k in data.files:
                self.gallery[k] = [x for x in data[k]]

    def _infer_input_mode(self) -> str:
        if self.input_mode != "auto":
            return self.input_mode
        n_in = len(self.cls.input_names)
        if n_in == 2:
            return "two_inputs"
        if n_in == 1:
            shp = self.cls.input_shapes[0]
            if isinstance(shp, (list, tuple)) and len(shp) >= 2:
                if shp[1] == 6:
                    return "single_6ch"
                if shp[1] == 3:
                    return "single_3ch"
        return "single_6ch"

    def _prep_img(self, img_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(img_bgr, (self.img, self.img), interpolation=cv2.INTER_AREA)
        if self.color_space == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2,0,1))
        return img

    def _run_logits(self, a_bgr: np.ndarray, b_bgr: np.ndarray) -> np.ndarray:
        mode = self._infer_input_mode()
        names = self.cls.input_names

        if mode == "two_inputs":
            xa = self._prep_img(a_bgr)[None, ...].astype(np.float32)
            xb = self._prep_img(b_bgr)[None, ...].astype(np.float32)
            out = self.cls.run({names[0]: xa, names[1]: xb})
            return np.array(out, dtype=np.float32).reshape(-1)

        if mode == "single_3ch":
            xa = self._prep_img(a_bgr)[None, ...].astype(np.float32)
            out = self.cls.run({names[0]: xa})
            return np.array(out, dtype=np.float32).reshape(-1)

        xa = self._prep_img(a_bgr)
        xb = self._prep_img(b_bgr)
        pair = np.concatenate([xa, xb], axis=0)[None, ...].astype(np.float32)
        out = self.cls.run({names[0]: pair})
        return np.array(out, dtype=np.float32).reshape(-1)

    def _swap_ensemble_logits(self, a_bgr: np.ndarray, b_bgr: np.ndarray) -> np.ndarray:
        return (self._run_logits(a_bgr, b_bgr) + self._run_logits(b_bgr, a_bgr)) / 2.0

    def _ocr_match_class(self, ocr_text: str) -> Optional[str]:
        t = _norm_text(ocr_text, self.rules)
        if not t:
            return None
        for cls, variants in self.imprint_db.items():
            if cls.startswith("_"):
                continue
            for v in variants:
                if _norm_text(v, self.rules) == t:
                    return cls
        return None

    def predict(self, path_a: str, path_b: str) -> PredictOutput:
        try:
            imgA = cv2.imdecode(np.fromfile(path_a, dtype=np.uint8), cv2.IMREAD_COLOR)
            imgB = cv2.imdecode(np.fromfile(path_b, dtype=np.uint8), cv2.IMREAD_COLOR)
            if imgA is None or imgB is None:
                raise RuntimeError("Failed to read one of the images.")

            pa = preprocess(imgA)
            pb = preprocess(imgB)

            ocr_text=""; ocr_conf=0.0; ocr_quality=0.0
            if self.ocr.available():
                best = ("", 0.0, 0.0)
                for p in (pa, pb):
                    imp = propose_imprint_roi(p.roi_bgr, p.mask)
                    enh = enhance_for_ocr(imp.roi_bgr)
                    res = self.ocr.run(enh)
                    if (res.conf, res.quality) > (best[1], best[2]):
                        best = (res.text, res.conf, res.quality)
                ocr_text, ocr_conf, ocr_quality = best

            ocr_cls = None
            if ocr_conf >= self.ocr_min_conf and ocr_quality >= self.ocr_min_quality:
                ocr_cls = self._ocr_match_class(ocr_text)

            logits = self._swap_ensemble_logits(pa.roi_bgr, pb.roi_bgr)
            probs = _softmax(logits)*100.0
            order = np.argsort(-probs)
            top3 = [(self.classes[int(i)], float(probs[int(i)])) for i in order[:3]]
            best_name, best_conf = top3[0]
            second_conf = top3[1][1] if len(top3)>1 else 0.0
            margin = float(best_conf - second_conf)

            if ocr_cls is not None:
                if ocr_cls in [t[0] for t in top3] or margin >= (self.min_margin_ok/2):
                    return PredictOutput("OK", ocr_cls, 99.0, margin, top3, ocr_text, ocr_conf, ocr_quality, 0.0,
                                        "각인(OCR) 기반 확정 + 정규화(회전/중심)")

            verifier_score = 0.0
            if self.gallery and margin < self.verifier_trigger_margin:
                imp = propose_imprint_roi(pa.roi_bgr, pa.mask)
                query = enhance_for_ocr(imp.roi_bgr)
                candidates = [t[0] for t in top3]
                vr = verify_against_gallery(query, self.gallery, candidates)
                verifier_score = vr.score
                if vr.ok and vr.best_ref:
                    return PredictOutput("OK", vr.best_ref, max(best_conf, 90.0), margin, top3, ocr_text, ocr_conf, ocr_quality, verifier_score,
                                        "경계 사례: 패치 검증 통과")

            if best_conf >= self.min_conf_ok and margin >= self.min_margin_ok:
                return PredictOutput("OK", best_name, best_conf, margin, top3, ocr_text, ocr_conf, ocr_quality, verifier_score,
                                    "시각 분류 기준 충족")

            return PredictOutput("UNDECIDED", None, best_conf, margin, top3, ocr_text, ocr_conf, ocr_quality, verifier_score,
                                f"미확정: 신뢰도/구분도 부족(차이={margin:.1f}%). 재촬영 또는 면 교체 후 재시도.")
        except Exception as e:
            return PredictOutput("ERROR", None, 0.0, 0.0, [], "", 0.0, 0.0, 0.0, str(e))
