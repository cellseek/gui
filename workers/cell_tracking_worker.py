"""Segment + link (next frame) or link-only (relink) via Trackastra."""

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from utils.cellsam_segment import segment_rgb
from utils.mask_utils import align_mask_to_image
from utils.trackastra_tracking import link_masks_with_trackastra


class CellTrackingWorker(QThread):
    result_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._mode = "track"
        self._previous_image: np.ndarray | None = None
        self._previous_mask: np.ndarray | None = None
        self._current_image: np.ndarray | None = None
        self._current_mask: np.ndarray | None = None

    def track_async(
        self,
        previous_image: np.ndarray,
        previous_mask: np.ndarray,
        current_image: np.ndarray,
    ) -> None:
        """CellSAM on current frame, then link IDs to the previous frame."""
        if self.isRunning():
            return

        self._mode = "track"
        self._previous_image = previous_image
        self._previous_mask = previous_mask
        self._current_image = current_image
        self._current_mask = None
        self.start()

    def relink_async(
        self,
        previous_image: np.ndarray,
        previous_mask: np.ndarray,
        current_image: np.ndarray,
        current_mask: np.ndarray,
    ) -> None:
        """Keep current mask regions; re-assign IDs from the previous frame."""
        if self.isRunning():
            return

        self._mode = "relink"
        self._previous_image = previous_image
        self._previous_mask = previous_mask
        self._current_image = current_image
        self._current_mask = current_mask
        self.start()

    def run(self) -> None:
        if (
            self._previous_image is None
            or self._previous_mask is None
            or self._current_image is None
        ):
            self.error_occurred.emit("Missing data for tracking")
            return

        try:
            height, width = self._current_image.shape[:2]
            previous_image = self._previous_image
            previous_mask = align_mask_to_image(self._previous_mask, height, width)

            if self._mode == "relink":
                if self._current_mask is None:
                    self.error_occurred.emit("Missing mask for relink")
                    return
                current_mask = align_mask_to_image(self._current_mask, height, width)
                if not np.any(current_mask):
                    self.error_occurred.emit("Current frame has no mask to relink")
                    return
                self.status_update.emit("Relinking cell identities (Trackastra)...")
            else:
                self.status_update.emit("Segmenting frame with CellSAM...")
                current_mask = segment_rgb(self._current_image)
                current_mask = align_mask_to_image(current_mask, height, width)
                self.status_update.emit("Linking cell identities (Trackastra)...")

            linked_mask = link_masks_with_trackastra(
                previous_image,
                previous_mask,
                self._current_image,
                current_mask,
            )
            self.result_ready.emit(linked_mask)
        except Exception as e:
            self.error_occurred.emit(str(e))
