"""User-adjustable frame preprocessing (crop, tone, clip)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

_GAMMA_LUT_CACHE: Dict[float, np.ndarray] = {}


def _gamma_lut(gamma: float) -> np.ndarray:
    key = round(gamma, 3)
    lut = _GAMMA_LUT_CACHE.get(key)
    if lut is None:
        inv_gamma = 1.0 / max(gamma, 0.01)
        lut = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
            dtype=np.uint8,
        )
        _GAMMA_LUT_CACHE[key] = lut
    return lut


@dataclass
class PreprocessSettings:
    """Settings applied to every frame before resize / segmentation."""

    brightness: int = 0  # -100 … 100
    contrast: float = 1.0  # 0.25 … 3.0
    gamma: float = 1.0  # 0.2 … 3.0
    crop_rect: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    clip_start: int = 0
    clip_end: int = -1  # inclusive; -1 = last frame

    def copy(self) -> "PreprocessSettings":
        return replace(self)

    def effective_clip_end(self, frame_count: int) -> int:
        if frame_count <= 0:
            return 0
        if self.clip_end < 0:
            return frame_count - 1
        return min(max(0, self.clip_end), frame_count - 1)

    def clipped_frame_count(self, full_count: int) -> int:
        if full_count <= 0:
            return 0
        start = max(0, min(self.clip_start, full_count - 1))
        end = self.effective_clip_end(full_count)
        return max(0, end - start + 1)

    def has_tone_adjustments(self) -> bool:
        return (
            self.brightness != 0
            or abs(self.contrast - 1.0) > 1e-6
            or abs(self.gamma - 1.0) > 1e-6
        )

    def is_default(self) -> bool:
        return (
            not self.has_tone_adjustments()
            and self.crop_rect is None
            and self.clip_start == 0
            and self.clip_end < 0
        )

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply crop and tone adjustments. Input/output RGB uint8."""
        if image is None or image.size == 0:
            return image

        if self.crop_rect is None and not self.has_tone_adjustments():
            return image

        out = image
        if self.crop_rect is not None:
            x, y, w, h = self.crop_rect
            ih, iw = out.shape[:2]
            x = max(0, min(x, iw - 1))
            y = max(0, min(y, ih - 1))
            w = max(1, min(w, iw - x))
            h = max(1, min(h, ih - y))
            out = out[y : y + h, x : x + w]

        if not self.has_tone_adjustments():
            return out.copy() if out is not image else image.copy()

        if out.size == 0:
            return out.copy()

        work = out.astype(np.float32)

        if self.brightness != 0:
            work = work + float(self.brightness)

        if abs(self.contrast - 1.0) > 1e-6:
            work = (work - 128.0) * self.contrast + 128.0

        np.clip(work, 0, 255, out=work)
        result = work.astype(np.uint8)

        if abs(self.gamma - 1.0) > 1e-6:
            result = cv2.LUT(result, _gamma_lut(self.gamma))

        return result
