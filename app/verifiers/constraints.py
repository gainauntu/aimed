from __future__ import annotations

import json
from pathlib import Path

from app.domain.models import ConstraintCheck, ConstraintReport, PreprocessedImage


class ConstraintProfiles:
    def __init__(self, profiles: dict[str, dict]) -> None:
        self.profiles = profiles

    @classmethod
    def from_json(cls, path: str | Path) -> "ConstraintProfiles":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


class ConstraintGate:
    def __init__(self, profiles: ConstraintProfiles) -> None:
        self.profiles = profiles

    def verify(self, predicted_class: str, image_a: PreprocessedImage, image_b: PreprocessedImage) -> ConstraintReport:
        profile = self.profiles.profiles[predicted_class]
        checks_a = self._checks_for_image(profile, image_a)
        checks_b = self._checks_for_image(profile, image_b)
        all_checks = checks_a + checks_b
        failed = [c for c in all_checks if not c.passed]
        ratio = float(sum(c.passed for c in all_checks) / max(1, len(all_checks)))
        return ConstraintReport(
            passed=not failed,
            pass_ratio=ratio,
            checks_a=checks_a,
            checks_b=checks_b,
            reasons=[f"{c.name}: observed={c.observed}, expected={c.expected}" for c in failed],
        )

    @staticmethod
    def _within(observed: float, mean: float, std: float) -> tuple[bool, str]:
        low = mean - 3.0 * std
        high = mean + 3.0 * std
        return low <= observed <= high, f"[{low:.3f}, {high:.3f}]"

    def _checks_for_image(self, profile: dict, image: PreprocessedImage) -> list[ConstraintCheck]:
        s = image.stats
        items: list[ConstraintCheck] = []
        for key, observed in [
            ("hue", s.hue_mean),
            ("sat", s.sat_mean),
            ("val", s.val_mean),
            ("aspect", s.aspect_ratio),
            ("area", s.pixel_area_proxy),
            ("edge", s.edge_density),
        ]:
            mean = float(profile[f"{key}_mean"])
            std = float(profile[f"{key}_std"])
            passed, expected = self._within(float(observed), mean, max(std, 1e-6))
            items.append(ConstraintCheck(name=key, passed=passed, observed=float(observed), expected=expected))

        expected_shape = str(profile["expected_shape"])
        items.append(
            ConstraintCheck(
                name="shape",
                passed=s.shape_class.value == expected_shape,
                observed=s.shape_class.value,
                expected=expected_shape,
            )
        )
        return items
