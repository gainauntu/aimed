from __future__ import annotations

import numpy as np

from app.preprocess.tta import TTAEngine


def test_tta_produces_32_views() -> None:
    image = np.zeros((288, 288, 3), dtype=np.uint8)
    views = TTAEngine().generate(image, image)
    assert len(views) == 32
    assert {v.rotation_deg for v in views} == {0, 45, 90, 135, 180, 225, 270, 315}
