from typing import Optional

import numpy as np
from cutie.cutie_tracker import CutieTracker
from PyQt6.QtCore import QThread, pyqtSignal


class CutieWorker(QThread):
    result_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._tracker: Optional[CutieTracker] = None
        self._previous_image = None
        self._previous_mask = None
        self._current_image = None
        self._previous_index = 0
        self._current_index = 1

    def _ensure_tracker(self) -> CutieTracker:
        if self._tracker is None:
            self.status_update.emit("Loading CUTIE model...")
            self._tracker = CutieTracker()
        return self._tracker

    def reset_tracker(self) -> None:
        if self._tracker is not None:
            self._tracker.reset()

    def track_async(
        self,
        previous_image: np.ndarray,
        previous_mask: np.ndarray,
        current_image: np.ndarray,
        previous_index: int,
        current_index: int,
    ) -> None:
        if self.isRunning():
            return

        self._previous_image = previous_image
        self._previous_mask = previous_mask
        self._current_image = current_image
        self._previous_index = previous_index
        self._current_index = current_index
        self.start()

    def run(self):
        try:
            self.status_update.emit("Running CUTIE tracking...")
            tracker = self._ensure_tracker()
            predicted_mask = tracker.track(
                self._previous_image,
                self._previous_mask,
                self._current_image,
                previous_index=self._previous_index,
                current_index=self._current_index,
            )
            self.result_ready.emit(predicted_mask)
        except Exception as e:
            self.error_occurred.emit(str(e))

    def track(
        self,
        previous_image: np.ndarray,
        previous_mask: np.ndarray,
        current_image: np.ndarray,
        previous_index: int = 0,
        current_index: int = 1,
    ) -> np.ndarray:
        tracker = self._ensure_tracker()
        return tracker.track(
            previous_image,
            previous_mask,
            current_image,
            previous_index=previous_index,
            current_index=current_index,
        )
