"""Shared CellSAM model and single-frame segmentation."""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
from cellsam import CellSAM

from utils.image_preprocessor import ImagePreprocessor

SEGMENTATION_MAX_DIM = 512
INFERENCE_BATCH_SIZE = 16
_preprocessor = ImagePreprocessor(max_size=SEGMENTATION_MAX_DIM)

_model: Optional[CellSAM] = None
_model_lock = threading.Lock()


def get_cellsam_model() -> CellSAM:
    global _model
    with _model_lock:
        if _model is None:
            _model = CellSAM(gpu=True)
        return _model


def prepare_rgb_for_segmentation(image_rgb: np.ndarray) -> np.ndarray:
    """Resize like display frames (ImagePreprocessor) so masks align across frames."""
    img, _ = _preprocessor.resize_image(np.ascontiguousarray(image_rgb))
    return img


def segment_rgb(image_rgb: np.ndarray) -> np.ndarray:
    """Run CellSAM on an RGB image; returns a uint16 instance label mask at input size."""
    height, width = image_rgb.shape[:2]
    img = prepare_rgb_for_segmentation(image_rgb)
    model = get_cellsam_model()
    masks, _, _ = model.segment(
        img,
        diameter=None,
        normalize=True,
        tile_overlap=0.1,
        augment=False,
        batch_size=INFERENCE_BATCH_SIZE,
    )
    masks = masks.astype(np.uint16)
    if masks.shape[0] != height or masks.shape[1] != width:
        from utils.mask_utils import align_mask_to_image

        masks = align_mask_to_image(masks, height, width)
    return masks
