"""Cellpose-SAM (cpsam) model and single-frame segmentation."""

from __future__ import annotations

import contextlib
import io
import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from utils.image_preprocessor import ImagePreprocessor

if TYPE_CHECKING:
    from cellpose.models import CellposeModel

SEGMENTATION_MAX_DIM = 512
INFERENCE_BATCH_SIZE = 16
_preprocessor = ImagePreprocessor(max_size=SEGMENTATION_MAX_DIM)

_model: Optional[Any] = None
_model_lock = threading.Lock()


def _load_cellpose_models():
    """Import cellpose lazily; its package __init__ prints a startup banner."""
    with contextlib.redirect_stdout(io.StringIO()):
        from cellpose import models as cp_models

    logging.getLogger("cellpose").setLevel(logging.WARNING)
    return cp_models


def _pretrained_model_path() -> str:
    """Use bundled weights/cpsam if present, else cellpose downloads cpsam on first use."""
    local = Path(__file__).resolve().parent.parent / "weights" / "cpsam"
    if local.is_file():
        return os.fspath(local)
    return "cpsam"


def get_segmentation_model() -> CellposeModel:
    global _model
    with _model_lock:
        if _model is None:
            cp_models = _load_cellpose_models()
            _model = cp_models.CellposeModel(
                gpu=True,
                pretrained_model=_pretrained_model_path(),
            )
        return _model


# Backward-compatible alias
get_cellsam_model = get_segmentation_model


def prepare_rgb_for_segmentation(image_rgb: np.ndarray) -> np.ndarray:
    """Resize like display frames (ImagePreprocessor) so masks align across frames."""
    img, _ = _preprocessor.resize_image(np.ascontiguousarray(image_rgb))
    return img


def _ensure_three_channels(image: np.ndarray) -> np.ndarray:
    """Cellpose-SAM expects 3 channels (H, W, C)."""
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    channels = image.shape[-1]
    if channels == 1:
        return np.concatenate([image] * 3, axis=-1)
    if channels == 2:
        pad = np.zeros((*image.shape[:2], 1), dtype=image.dtype)
        return np.concatenate([image, pad], axis=-1)
    return image[..., :3]


def segment_rgb(image_rgb: np.ndarray) -> np.ndarray:
    """Run Cellpose-SAM on an RGB image; returns uint16 instance labels at input size."""
    height, width = image_rgb.shape[:2]
    img = _ensure_three_channels(prepare_rgb_for_segmentation(image_rgb))
    model = get_segmentation_model()
    masks, _, _ = model.eval(
        img,
        diameter=None,
        normalize=True,
        tile_overlap=0.1,
        augment=False,
        batch_size=INFERENCE_BATCH_SIZE,
        channel_axis=-1,
    )
    masks = np.asarray(masks, dtype=np.uint16)
    if masks.shape[0] != height or masks.shape[1] != width:
        from utils.mask_utils import align_mask_to_image

        masks = align_mask_to_image(masks, height, width)
    return masks
