from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.isotonic import IsotonicRegression
import joblib

from app.core.bootstrap import build_pipeline
from app.preprocess.background import BackgroundReference
from app.training.profiles import build_class_profiles


def cmd_predict(args: argparse.Namespace) -> int:
    pipeline = build_pipeline(args.config)
    image_a = cv2.imread(args.image_a, cv2.IMREAD_COLOR)
    image_b = cv2.imread(args.image_b, cv2.IMREAD_COLOR)
    if image_a is None or image_b is None:
        raise FileNotFoundError("Failed to read input images")
    result = pipeline.predict(image_a, image_b)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0


def cmd_build_background(args: argparse.Namespace) -> int:
    bg = BackgroundReference.build_from_folder(args.input_dir)
    bg.save(args.output)
    print(f"saved background reference to {args.output}")
    return 0


def cmd_build_profiles(args: argparse.Namespace) -> int:
    profiles = build_class_profiles(args.dataset_dir)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profiles, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved class profiles to {args.output}")
    return 0


def cmd_build_mock_calibrator(args: argparse.Namespace) -> int:
    weights = np.ones(12, dtype=np.float32) / 12.0
    raw_scores = np.linspace(0, 1, 20)
    targets = np.clip(raw_scores, 0, 1)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_scores, targets)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"weights": weights, "intercept": 0.0, "isotonic_model": iso}, args.output)
    print(f"saved mock calibrator to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pill-system")
    sub = parser.add_subparsers(dest="command", required=True)

    predict = sub.add_parser("predict")
    predict.add_argument("--config", default="configs/default.yaml")
    predict.add_argument("--image-a", required=True)
    predict.add_argument("--image-b", required=True)
    predict.set_defaults(func=cmd_predict)

    background = sub.add_parser("build-background")
    background.add_argument("--input-dir", required=True)
    background.add_argument("--output", required=True)
    background.set_defaults(func=cmd_build_background)

    profiles = sub.add_parser("build-profiles")
    profiles.add_argument("--dataset-dir", required=True)
    profiles.add_argument("--output", required=True)
    profiles.set_defaults(func=cmd_build_profiles)

    calibrator = sub.add_parser("build-mock-calibrator")
    calibrator.add_argument("--output", required=True)
    calibrator.set_defaults(func=cmd_build_mock_calibrator)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
