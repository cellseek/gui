"""Trackastra-based identity linking between consecutive segmented frames."""

from __future__ import annotations

import threading
from typing import Dict, Optional

import cv2
import numpy as np

_TRACKASTRA_MODEL = None
_MODEL_LOCK = threading.Lock()
_PRETRAINED_MODEL = "general_2d"
_LINKING_MODE = "greedy_nodiv"


class _SilentProgbar:
    """Tqdm-compatible progress bar that does not print (for GUI background threads)."""

    def __init__(self, iterable=None, **_kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable) if self.iterable is not None else iter([])

    def update(self, _n: int = 1) -> None:
        pass

    def close(self) -> None:
        pass

    def set_description(self, *_args, **_kwargs) -> None:
        pass


def get_trackastra_model():
    """Load the pretrained Trackastra model once (thread-safe)."""
    global _TRACKASTRA_MODEL
    with _MODEL_LOCK:
        if _TRACKASTRA_MODEL is None:
            from trackastra.model import Trackastra

            _TRACKASTRA_MODEL = Trackastra.from_pretrained(
                _PRETRAINED_MODEL, device="automatic"
            )
        return _TRACKASTRA_MODEL


def rgb_to_trackastra_image(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 frame to grayscale uint16 (Trackastra time-lapse format)."""
    if rgb.ndim == 2:
        plane = rgb
    elif rgb.shape[2] == 1:
        plane = rgb[:, :, 0]
    else:
        plane = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    if plane.dtype == np.uint16:
        return plane
    if plane.max() <= 255:
        return (plane.astype(np.uint32) * 257).astype(np.uint16)
    return plane.astype(np.uint16)


def _label_map_from_graph(graph, frame_index: int = 1) -> Dict[int, int]:
    """
    Map segmentation labels on ``frame_index`` to labels on frame 0 using graph edges.
    """
    prev_time = frame_index - 1
    mapping: Dict[int, int] = {}

    for start, end in graph.edges:
        start_data = graph.nodes[start]
        end_data = graph.nodes[end]

        if int(start_data["time"]) == prev_time and int(end_data["time"]) == frame_index:
            mapping[int(end_data["label"])] = int(start_data["label"])
        elif int(end_data["time"]) == prev_time and int(start_data["time"]) == frame_index:
            mapping[int(start_data["label"])] = int(end_data["label"])

    return mapping


def link_masks_with_trackastra(
    previous_image: np.ndarray,
    previous_mask: np.ndarray,
    current_image: np.ndarray,
    current_mask: np.ndarray,
) -> np.ndarray:
    """
    Assign IDs on ``current_mask`` to match ``previous_mask`` using Trackastra.

    Cell region shapes come from ``current_mask``; only label IDs are changed.
    """
    if previous_mask.shape != current_mask.shape:
        raise ValueError(
            "Mask shapes must match: "
            f"{previous_mask.shape} vs {current_mask.shape}"
        )

    imgs = np.stack(
        [
            rgb_to_trackastra_image(previous_image),
            rgb_to_trackastra_image(current_image),
        ],
        axis=0,
    )
    masks = np.stack(
        [
            previous_mask.astype(np.uint16),
            current_mask.astype(np.uint16),
        ],
        axis=0,
    )

    model = get_trackastra_model()
    graph, _masks_tracked = model.track(
        imgs, masks, mode=_LINKING_MODE, progbar_class=_SilentProgbar
    )

    label_map = _label_map_from_graph(graph, frame_index=1)

    linked = np.zeros(current_mask.shape, dtype=np.uint16)
    next_id = int(previous_mask.max(initial=0)) + 1

    for new_label in np.unique(current_mask):
        if new_label == 0:
            continue
        new_label = int(new_label)
        if new_label in label_map:
            stable_id = label_map[new_label]
        else:
            stable_id = next_id
            next_id += 1
        linked[current_mask == new_label] = stable_id

    return linked
