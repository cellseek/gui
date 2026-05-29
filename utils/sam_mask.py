"""Helpers for SAM prediction masks."""

from __future__ import annotations

import numpy as np

from utils.mask_utils import align_mask_to_image


def normalize_sam_binary_mask(
    mask: np.ndarray, height: int, width: int
) -> np.ndarray:
    """Return a boolean (H, W) mask aligned to the display frame."""
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[0] if arr.shape[0] == 1 else np.max(arr, axis=0)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {arr.shape}")

    if arr.dtype == np.bool_:
        binary = arr
    else:
        binary = arr > 0.5 if np.issubdtype(arr.dtype, np.floating) else arr > 0

    if binary.shape[0] != height or binary.shape[1] != width:
        aligned = align_mask_to_image(binary.astype(np.uint8), height, width)
        binary = aligned > 0

    return binary
