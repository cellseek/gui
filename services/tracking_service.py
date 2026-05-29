"""Frame-to-frame tracking: CellSAM + Trackastra, or Trackastra link-only relink."""

from typing import Optional

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from workers.cell_tracking_worker import CellTrackingWorker


class TrackingService(QObject):
    """Segment/link on new frames; relink IDs on an existing mask after edits."""

    tracking_complete = pyqtSignal(np.ndarray)
    tracking_error = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._worker: Optional[CellTrackingWorker] = None

    @property
    def worker(self) -> Optional[CellTrackingWorker]:
        if self._worker is None:
            try:
                self._worker = CellTrackingWorker()
                self._worker.result_ready.connect(self.tracking_complete.emit)
                self._worker.error_occurred.connect(self.tracking_error.emit)
                self._worker.status_update.connect(self.status_update.emit)
            except Exception as e:
                print(f"Failed to initialize tracking worker: {e}")
                return None
        return self._worker

    def is_running(self) -> bool:
        worker = self.worker
        return worker is not None and worker.isRunning()

    def track_async(
        self,
        previous_image: np.ndarray,
        previous_mask: np.ndarray,
        current_image: np.ndarray,
    ) -> None:
        worker = self.worker
        if worker is None:
            self.tracking_error.emit("Tracking worker not initialized")
            return
        if worker.isRunning():
            self.tracking_error.emit("Tracking is already in progress")
            return
        try:
            worker.track_async(previous_image, previous_mask, current_image)
        except Exception as e:
            self.tracking_error.emit(f"Failed to start tracking: {e}")

    def relink_async(
        self,
        previous_image: np.ndarray,
        previous_mask: np.ndarray,
        current_image: np.ndarray,
        current_mask: np.ndarray,
    ) -> None:
        worker = self.worker
        if worker is None:
            self.tracking_error.emit("Tracking worker not initialized")
            return
        if worker.isRunning():
            self.tracking_error.emit("Tracking is already in progress")
            return
        try:
            worker.relink_async(
                previous_image, previous_mask, current_image, current_mask
            )
        except Exception as e:
            self.tracking_error.emit(f"Failed to start relink: {e}")
