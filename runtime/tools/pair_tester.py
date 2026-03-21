from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import json
import requests
import uuid

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def canonical_class_name(root: Path, path: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[0]


def subgroup_name(root: Path, path: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[1] if len(rel.parts) >= 3 else rel.parts[0]


def collect_dataset(root: Path):
    by_class = defaultdict(list)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            by_class[canonical_class_name(root, p)].append(p)
    return by_class


def send_predict(url: str, port: int, image_paths: list[str], timeout: int = 180) -> dict:
    endpoint = f"{url.rstrip('/')}:{port}/predict"
    payload = {"request_id": str(uuid.uuid4()), "image_paths": image_paths}
    try:
        r = requests.post(endpoint, json=payload, timeout=timeout)
        try:
            data = r.json()
        except Exception:
            data = {"status": "ERROR", "reason": r.text}
        data["_http_status"] = r.status_code
        data["_payload"] = payload
        return data
    except Exception as e:
        return {
            "_http_status": None,
            "_payload": payload,
            "status": "ERROR",
            "class_name": None,
            "confidence_score": 0.0,
            "reason": str(e),
            "reject_code": "REQUEST_EXCEPTION",
        }


def verdict(expected_same_class: bool, expected_class: str | None, resp: dict):
    status = resp.get("status")
    pred = resp.get("class_name")
    if status == "ERROR":
        return "ERROR"
    if expected_same_class:
        if status == "OK" and pred == expected_class:
            return "PASS"
        return "FAIL"
    if status == "UNDECIDED":
        return "PASS"
    if status == "OK":
        return "FALSE_ACCEPT"
    return "FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--url", default="http://127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--mode", choices=["within", "cross", "both"], default="both")
    ap.add_argument("--out", default="pair_test_results.json")
    ap.add_argument("--timeout", type=int, default=180)
    ns = ap.parse_args()

    root = Path(ns.data_root)
    by_class = collect_dataset(root)
    class_names = sorted(by_class.keys())
    results = []

    if ns.mode in ("within", "both"):
        for cls in class_names:
            imgs = sorted(by_class[cls])
            n = len(imgs)
            for i in range(n):
                for j in range(i + 1, n):
                    combo = [imgs[i], imgs[j]]
                    image_paths = [str(p) for p in combo]
                    resp = send_predict(ns.url, ns.port, image_paths, timeout=ns.timeout)
                    results.append({
                        "case_id": str(uuid.uuid4()),
                        "expected_class": cls,
                        "source_mode": "within_group",
                        "group_names": [subgroup_name(root, p) for p in combo],
                        "image_paths": image_paths,
                        "http_status": resp.get("_http_status"),
                        "server_status": resp.get("status"),
                        "predicted_class": resp.get("class_name"),
                        "confidence_score": float(resp.get("confidence_score") or 0.0),
                        "reason": resp.get("reason"),
                        "reject_code": resp.get("reject_code"),
                        "verdict": verdict(True, cls, resp),
                        "raw_response": resp,
                    })

    if ns.mode in ("cross", "both"):
        for i, cls_a in enumerate(class_names):
            for cls_b in class_names[i + 1:]:
                for pa in sorted(by_class[cls_a]):
                    for pb in sorted(by_class[cls_b]):
                        combo = [pa, pb]
                        image_paths = [str(p) for p in combo]
                        resp = send_predict(ns.url, ns.port, image_paths, timeout=ns.timeout)
                        results.append({
                            "case_id": str(uuid.uuid4()),
                            "expected_class": None,
                            "source_mode": "cross_group",
                            "group_names": [subgroup_name(root, pa), subgroup_name(root, pb)],
                            "image_paths": image_paths,
                            "http_status": resp.get("_http_status"),
                            "server_status": resp.get("status"),
                            "predicted_class": resp.get("class_name"),
                            "confidence_score": float(resp.get("confidence_score") or 0.0),
                            "reason": resp.get("reason"),
                            "reject_code": resp.get("reject_code"),
                            "verdict": verdict(False, None, resp),
                            "raw_response": resp,
                        })

    out_path = Path(ns.out)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = defaultdict(int)
    for r in results:
        summary[r["verdict"]] += 1

    print("saved:", out_path)
    print("total:", len(results))
    for k, v in sorted(summary.items()):
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
