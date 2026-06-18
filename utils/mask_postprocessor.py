"""
Morphological cleanup for segmentation / tracking label masks.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MaskPostprocessorConfig:
    enabled: bool = True
    opening_kernel_size: int = 3
    closing_kernel_size: int = 3
    min_area_pixels: int = 50


class MaskPostprocessor:
    """
    Remove small speckle artifacts, then fill pinholes.

    Supports binary masks and instance label masks (per-object cleanup).
    """

    def __init__(self, config: MaskPostprocessorConfig | None = None):
        self.config = config or MaskPostprocessorConfig()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self.config.enabled = bool(value)

    def process_binary_mask(self, binary_mask: np.ndarray) -> np.ndarray:
        """Apply opening, closing, and minimum-area filtering to a binary mask."""
        if not self.config.enabled:
            return binary_mask

        mask = (np.asarray(binary_mask) > 0).astype(np.uint8)
        if mask.max() == 0:
            return mask

        opened = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            self._kernel(self.config.opening_kernel_size),
        )
        closed = cv2.morphologyEx(
            opened,
            cv2.MORPH_CLOSE,
            self._kernel(self.config.closing_kernel_size),
        )
        return self._remove_small_components(closed)

    def process_label_mask(self, label_mask: np.ndarray) -> np.ndarray:
        """Clean each instance label independently; background stays 0."""
        labels = np.asarray(label_mask)
        if not self.config.enabled or labels.size == 0:
            return labels

        out = np.zeros(labels.shape, dtype=labels.dtype)
        for label_id in np.unique(labels):
            if label_id <= 0:
                continue
            binary = (labels == label_id).astype(np.uint8)
            cleaned = self.process_binary_mask(binary)
            out[cleaned > 0] = label_id
        return out

    @staticmethod
    def _kernel(size: int) -> np.ndarray:
        k = max(1, int(size))
        if k % 2 == 0:
            k += 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    def _remove_small_components(self, mask: np.ndarray) -> np.ndarray:
        min_area = max(0, int(self.config.min_area_pixels))
        if min_area <= 1:
            return mask

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        if num_labels <= 1:
            return mask

        out = np.zeros(mask.shape, dtype=np.uint8)
        for component_id in range(1, num_labels):
            if stats[component_id, cv2.CC_STAT_AREA] >= min_area:
                out[labels == component_id] = 1
        return out
