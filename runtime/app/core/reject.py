from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class FrameEvidence:
    path: str
    usable: bool
    quality_score: float
    top1_name: str
    top1_conf: float
    margin: float
    prototype_distance: Optional[float] = None
    prototype_threshold: Optional[float] = None


@dataclass
class DecisionResult:
    status: str
    class_name: Optional[str]
    confidence_score: float
    reason: str
    reject_code: Optional[str]


def default_set_purity_cfg():
    return {
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
    }


def default_decision_cfg():
    return {
        "min_avg_quality": 20.0,
        "default_top1_cut": 60.0,
        "default_margin_cut": 20.0,
        "default_consensus_cut": 60.0,
        "default_final_conf_cut": 70.0,
        "class_overrides": {
            "SHP 50": {"top1_cut": 57.0, "margin_cut": 18.0, "final_conf_cut": 65.0},
            "SPP RM": {"top1_cut": 57.0, "margin_cut": 18.0, "final_conf_cut": 65.0},
        },
    }


def _class_cfg(cfg: Dict[str, Any], class_name: str):
    out = dict(cfg)
    out.update((cfg.get("class_overrides") or {}).get(class_name, {}))
    return out


def _proto_ok(dist: Optional[float], th: Optional[float], mul: float = 1.0) -> bool:
    if dist is None or th is None:
        return True
    return float(dist) <= float(th) * float(mul)


def evaluate_set_purity(frames: List[FrameEvidence], winner_name: str, cfg: Dict[str, Any]):
    support = soft = contradiction = neutral = usable = 0
    for f in frames:
        if not f.usable:
            continue
        usable += 1

        if (
            f.top1_name != winner_name
            and f.quality_score >= float(cfg["min_quality"])
            and f.top1_conf >= float(cfg["min_contradict_conf"])
            and f.margin >= float(cfg["min_contradict_margin"])
            and _proto_ok(f.prototype_distance, f.prototype_threshold, 1.15)
        ):
            contradiction += 1
            continue

        if (
            f.top1_name == winner_name
            and f.quality_score >= float(cfg["min_quality"])
            and f.top1_conf >= float(cfg["min_support_conf"])
            and f.margin >= float(cfg["min_support_margin"])
            and _proto_ok(f.prototype_distance, f.prototype_threshold, 1.10)
        ):
            support += 1
            continue

        if (
            f.top1_name == winner_name
            and f.quality_score >= float(cfg["min_soft_quality"])
            and f.top1_conf >= float(cfg["min_soft_support_conf"])
            and f.margin >= float(cfg["min_soft_support_margin"])
            and _proto_ok(f.prototype_distance, f.prototype_threshold, 1.0)
        ):
            soft += 1
            continue

        neutral += 1

    effective_support = support + soft
    effective_support_ratio = effective_support / max(1, usable)
    support_ratio = support / max(1, usable)

    reject = None
    reason = "판정 기준 충족"
    if contradiction > int(cfg["max_contradictions"]):
        reject = "REJECT_SET_PURITY_CONTRADICTION"
        reason = "미확정: 이미지 간 판정 일관성이 부족함"
    elif usable > 0 and (
        effective_support < int(cfg["min_support_frames"])
        or effective_support_ratio < float(cfg["min_support_ratio"])
    ):
        reject = "REJECT_SET_PURITY_LOW_SUPPORT"
        reason = "미확정: 이미지 간 판정 일관성이 부족함"

    return {
        "reject_code": reject,
        "public_reason": reason,
        "support": support,
        "soft_support": soft,
        "effective_support": effective_support,
        "neutral": neutral,
        "contradiction": contradiction,
        "usable": usable,
        "support_ratio": support_ratio,
        "effective_support_ratio": effective_support_ratio,
    }


def decide(class_name: str, top1_prob: float, margin: float, consensus_score: float, prototype_distance: Optional[float], prototype_threshold: Optional[float], usable_frames: int, min_images: int, avg_quality: float, frame_evidences: List[FrameEvidence], set_purity_cfg: Dict[str, Any] | None = None, decision_cfg: Dict[str, Any] | None = None):
    spcfg = {**default_set_purity_cfg(), **(set_purity_cfg or {})}
    dcfg = {**default_decision_cfg(), **(decision_cfg or {})}
    ccfg = _class_cfg(dcfg, class_name)

    if usable_frames < min_images:
        return DecisionResult("UNDECIDED", None, 0.0, "미확정: 유효 이미지 수가 부족함", "REJECT_TOO_FEW_VALID_FRAMES")
    if avg_quality < float(dcfg["min_avg_quality"]):
        return DecisionResult("UNDECIDED", None, 0.0, "미확정: 입력 품질이 부족함", "REJECT_LOW_QUALITY")

    purity = evaluate_set_purity(frame_evidences, class_name, spcfg)
    if purity["reject_code"] is not None:
        return DecisionResult("UNDECIDED", None, 0.0, purity["public_reason"], purity["reject_code"])

    if top1_prob < float(ccfg.get("top1_cut", dcfg["default_top1_cut"])):
        return DecisionResult("UNDECIDED", None, 0.0, "미확정: 학습 클래스와의 일치도가 충분하지 않음", "REJECT_LOW_TOP1_CONFIDENCE")
    if margin < float(ccfg.get("margin_cut", dcfg["default_margin_cut"])):
        return DecisionResult("UNDECIDED", None, 0.0, "미확정: 상위 후보 간 차이가 충분하지 않음", "REJECT_LOW_MARGIN")
    if consensus_score < float(ccfg.get("consensus_cut", dcfg["default_consensus_cut"])):
        return DecisionResult("UNDECIDED", None, 0.0, "미확정: 이미지 간 판정 일관성이 부족함", "REJECT_FUSED_LOW_CONSENSUS")

    if prototype_distance is not None and prototype_threshold is not None and prototype_distance > prototype_threshold:
        return DecisionResult("UNDECIDED", None, 0.0, "미확정: 학습 클래스와의 일치도가 충분하지 않음", "REJECT_UNKNOWN_DISTANCE")

    prototype_score = 100.0
    if prototype_distance is not None and prototype_threshold is not None and prototype_threshold > 0:
        prototype_score = max(0.0, 100.0 * (1.0 - float(prototype_distance) / float(prototype_threshold)))

    conf = 0.38 * float(top1_prob) + 0.24 * min(100.0, max(0.0, float(margin) * 2.0)) + 0.20 * float(consensus_score) + 0.18 * float(prototype_score)
    conf = max(0.0, min(100.0, conf))

    if conf < float(ccfg.get("final_conf_cut", dcfg["default_final_conf_cut"])):
        return DecisionResult("UNDECIDED", None, 0.0, "미확정: 최종 신뢰도가 충분하지 않음", "REJECT_LOW_FINAL_CONFIDENCE")

    return DecisionResult("OK", class_name, round(conf, 2), "판정 기준 충족", None)
