from __future__ import annotations

import numpy as np

from app.inference.fusion import symmetric_fusion


def test_symmetric_fusion_is_swap_invariant() -> None:
    a = np.random.default_rng(0).integers(0, 255, size=(288, 288, 3), dtype=np.uint8)
    b = np.random.default_rng(1).integers(0, 255, size=(288, 288, 3), dtype=np.uint8)
    fused_ab = symmetric_fusion(a, b)
    fused_ba = symmetric_fusion(b, a)
    assert np.allclose(fused_ab, fused_ba)
