import numpy as np
from cutie.cutie_tracker import CutieTracker
from PyQt6.QtCore import QThread, pyqtSignal


class CutieWorker(QThread):
    # Signals for async communication
    result_ready = pyqtSignal(np.ndarray)  # predicted mask
    error_occurred = pyqtSignal(str)  # error message
    status_update = pyqtSignal(str)  # status message

    def __init__(self):
        """
        Initialize CUTIE worker and model.
        """
        super().__init__()

        self._cancelled = False
        self.tracker = None

        # Task data for async operations
        self._previous_image = None
        self._previous_mask = None
        self._current_image = None

        # Initialize CUTIE tracker
        self.tracker = CutieTracker()

    def track_async(
        self,
        previous_image: np.ndarray,
        previous_mask: np.ndarray,
        current_image: np.ndarray,
    ) -> None:
        """
        Start tracking asynchronously.

        Args:
            previous_image: Previous frame image
            previous_mask: Previous frame mask
            current_image: Current frame image
        """
        if self.isRunning():
            return  # Already running

        # Store task data
        self._previous_image = previous_image.copy()
        self._previous_mask = previous_mask.copy()
        self._current_image = current_image.copy()

        self.start()

    def run(self):
        """Main thread execution method."""
        try:
            self.status_update.emit("Running CUTIE tracking...")

            # Use the new track method that handles both steps internally
            predicted_mask = self.tracker.track(
                self._previous_image, self._previous_mask, self._current_image
            )

            self.result_ready.emit(predicted_mask)

        except Exception as e:
            self.error_occurred.emit(str(e))

    def track(
        self,
        previous_image: np.ndarray,
        previous_mask: np.ndarray,
        current_image: np.ndarray,
    ) -> np.ndarray:
        """
        Perform a single tracking step with CUTIE (legacy synchronous method).

        Args:
            previous_image: Previous frame image
            previous_mask: Previous frame mask
            current_image: Current frame image

        Returns:
            Predicted mask for current frame
        """

        # Use the new track method that handles both steps internally
        predicted_mask = self.tracker.track(
            previous_image, previous_mask, current_image
        )

        return predicted_mask
