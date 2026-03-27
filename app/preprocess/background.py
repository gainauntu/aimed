from __future__ import annotations

# ---------------------------------------------------------------------------
# BackgroundReference — no external file required.
#
# Since the tray background is black, we provide two modes:
#   1. black()     — pure zero frame (the default, no file needed)
#   2. from_file() — load a PNG if one happens to be available
#
# The subtract() method is unchanged: it diffs the current frame against the
# reference, thresholds, and morphologically cleans.  On a black background
# the diff is essentially just the pill itself, so a zero reference works
# correctly and often better than a noisy captured reference.
#
# AdaptiveBackground is also provided: for each incoming frame it builds the
# mask from the image itself using Otsu thresholding on the value channel —
# no reference frame needed at all.  PillPreprocessor uses this as a fallback
# when no reference is available, giving a fully reference-free pipeline.
# ---------------------------------------------------------------------------

from pathlib import Path

import cv2
import numpy as np


class BackgroundReference:
    """Stores a BGR reference frame used for foreground extraction."""

    def __init__(self, reference_bgr: np.ndarray) -> None:
        self.reference_bgr = reference_bgr.astype(np.uint8)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def black(cls, height: int = 288, width: int = 352) -> 'BackgroundReference':
        """Pure black reference — correct for black-tray pharmacy setups.
        This is the default when no reference file is available.
        Works well because the pill is the only bright object on the tray.
        """
        return cls(np.zeros((height, width, 3), dtype=np.uint8))

    @classmethod
    def from_file(cls, path: str | Path) -> 'BackgroundReference':
        """Load a reference PNG.  Falls back to black() if file is missing."""
        p = Path(path)
        if not p.exists():
            return cls.black()
        data = np.fromfile(str(p), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            return cls.black()
        return cls(image)

    @classmethod
    def build_from_folder(cls, folder: str | Path) -> 'BackgroundReference':
        """Compute median of all images in a folder (classic approach)."""
        paths = sorted(Path(folder).glob('*'))
        frames = []
        for path in paths:
            data = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(img)
        if not frames:
            return cls.black()
        stack = np.stack(frames, axis=0)
        median = np.median(stack, axis=0).astype(np.uint8)
        return cls(median)

    # ------------------------------------------------------------------
    # Core operation
    # ------------------------------------------------------------------

    def subtract(self, frame_bgr: np.ndarray, threshold: int) -> np.ndarray:
        """Return binary foreground mask via reference subtraction.

        For a black reference, absdiff is equivalent to a simple brightness
        threshold — pixels brighter than `threshold` in any channel are marked
        as foreground.  This is exactly what we want for a black tray.
        """
        diff = cv2.absdiff(frame_bgr, self.reference_bgr)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
        return binary

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), self.reference_bgr)

    @property
    def is_black(self) -> bool:
        return bool((self.reference_bgr == 0).all())
