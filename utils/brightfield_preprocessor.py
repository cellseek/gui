"""
Brightfield microscopy preprocessing for segmentation and tracking models.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BrightfieldPreprocessorConfig:
    enabled: bool = True
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    background_blur_sigma: float = 50.0
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    output_channels: str = "rgb"  # "rgb" or "gray"


class BrightfieldPreprocessor:
    """
    Edge-preserving denoise, flat-field correction, CLAHE, and 0–255 normalization.

    Input/output convention: RGB ``uint8`` with shape ``(H, W, 3)`` unless
    ``output_channels="gray"`` (then ``(H, W)``). Grayscale inputs are accepted.
    """

    def __init__(self, config: BrightfieldPreprocessorConfig | None = None):
        self.config = config or BrightfieldPreprocessorConfig()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self.config.enabled = bool(value)

    def transform(self, image: np.ndarray) -> np.ndarray:
        if not self.config.enabled:
            return self._ensure_output_format(image)

        gray_u8 = self._to_gray_uint8(image)
        cfg = self.config

        denoised = cv2.bilateralFilter(
            gray_u8,
            cfg.bilateral_d,
            cfg.bilateral_sigma_color,
            cfg.bilateral_sigma_space,
        )

        denoised_f = denoised.astype(np.float32)
        ksize = self._kernel_size_from_sigma(cfg.background_blur_sigma)
        background = cv2.GaussianBlur(
            denoised_f, ksize, cfg.background_blur_sigma
        )
        corrected = denoised_f / (background + 1e-6)
        corrected_u8 = self._minmax_uint8(corrected)

        clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip_limit,
            tileGridSize=cfg.clahe_tile_grid_size,
        )
        enhanced = clahe.apply(corrected_u8)
        normalized = self._minmax_uint8(enhanced.astype(np.float32))

        if cfg.output_channels == "gray":
            return normalized
        return np.stack([normalized, normalized, normalized], axis=-1)

    @staticmethod
    def _to_gray_uint8(image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim == 2:
            if arr.dtype == np.uint8:
                return arr
            return BrightfieldPreprocessor._minmax_uint8(arr.astype(np.float32))
        if arr.shape[2] == 1:
            channel = arr[..., 0]
            if channel.dtype == np.uint8:
                return channel
            return BrightfieldPreprocessor._minmax_uint8(channel.astype(np.float32))
        rgb = arr[..., :3]
        if rgb.dtype != np.uint8:
            rgb = BrightfieldPreprocessor._minmax_uint8(rgb.astype(np.float32))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    def _ensure_output_format(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim == 2:
            gray = self._minmax_uint8(arr.astype(np.float32))
            if self.config.output_channels == "gray":
                return gray
            return np.stack([gray, gray, gray], axis=-1)

        rgb = arr[..., :3]
        if rgb.dtype != np.uint8:
            rgb = self._minmax_uint8(rgb.astype(np.float32))
        if self.config.output_channels == "gray":
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return rgb

    @staticmethod
    def _minmax_uint8(image: np.ndarray) -> np.ndarray:
        arr = image.astype(np.float32, copy=False)
        vmin = float(np.min(arr))
        vmax = float(np.max(arr))
        if vmax <= vmin + 1e-6:
            return np.zeros(arr.shape, dtype=np.uint8)
        scaled = (arr - vmin) / (vmax - vmin) * 255.0
        return np.clip(scaled, 0, 255).astype(np.uint8)

    @staticmethod
    def _kernel_size_from_sigma(sigma: float) -> tuple[int, int]:
        k = int(max(3, round(sigma * 6)))
        if k % 2 == 0:
            k += 1
        return (k, k)
