"""
CUTIE tracking service for handling frame-by-frame tracking functionality
"""

from typing import Optional

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from workers.cutie_worker import CutieWorker


class CutieService(QObject):
    """Simple service for CUTIE frame-by-frame tracking"""

    tracking_complete = pyqtSignal(np.ndarray)
    tracking_error = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._cutie_worker: Optional[CutieWorker] = None

    @property
    def cutie_worker(self) -> Optional[CutieWorker]:
        """Create the worker thread on first track (model loads inside the worker)."""
        if self._cutie_worker is None:
            try:
                self._cutie_worker = CutieWorker()
                self._cutie_worker.result_ready.connect(self.tracking_complete.emit)
                self._cutie_worker.error_occurred.connect(self.tracking_error.emit)
                self._cutie_worker.status_update.connect(self.status_update.emit)
            except Exception as e:
                print(f"Failed to initialize CUTIE worker: {str(e)}")
                return None
        return self._cutie_worker

    def reset_tracker(self) -> None:
        """Invalidate CUTIE memory after mask edits or session reset."""
        if self._cutie_worker is not None:
            self._cutie_worker.reset_tracker()

    def track_async(
        self,
        previous_image: np.ndarray,
        previous_mask: np.ndarray,
        current_image: np.ndarray,
        previous_index: int,
        current_index: int,
    ) -> None:
        worker = self.cutie_worker
        if worker is None:
            self.tracking_error.emit("CUTIE worker not initialized")
            return

        if worker.isRunning():
            self.tracking_error.emit("CUTIE worker is busy")
            return

        try:
            worker.track_async(
                previous_image,
                previous_mask,
                current_image,
                previous_index,
                current_index,
            )
        except Exception as e:
            self.tracking_error.emit(f"Failed to start tracking: {str(e)}")

    def track(
        self,
        previous_image: np.ndarray,
        previous_mask: np.ndarray,
        current_image: np.ndarray,
        previous_index: int = 0,
        current_index: int = 1,
    ) -> Optional[np.ndarray]:
        worker = self.cutie_worker
        if worker is None:
            raise RuntimeError("CUTIE worker not initialized")

        return worker.track(
            previous_image,
            previous_mask,
            current_image,
            previous_index,
            current_index,
        )
