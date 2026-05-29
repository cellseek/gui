"""Mask / image alignment helpers."""

from __future__ import annotations

import cv2
import numpy as np


def align_mask_to_image(
    mask: np.ndarray, target_height: int, target_width: int
) -> np.ndarray:
    """Nearest-neighbor resize so a label mask matches an RGB frame."""
    if mask.shape[0] == target_height and mask.shape[1] == target_width:
        return mask
    work_dtype = np.uint16 if np.max(mask) > 255 else np.uint8
    resized = cv2.resize(
        mask.astype(work_dtype),
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(mask.dtype)
