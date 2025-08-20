from typing import List, Optional

import numpy as np
import torch
from cutie.cutie_tracker import CutieTracker
from PyQt6.QtCore import QThread, pyqtSignal


class CutieWorker:
    """Worker thread for CUTIE tracking"""

    def __init__(self):
        """
        Initialize CUTIE worker and model.
        """
        super().__init__()

        self._cancelled = False
        self.tracker = None

        # Initialize CUTIE tracker
        try:
            self.tracker = CutieTracker()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CUTIE tracker: {e}")

    def cancel(self):
        """Cancel the tracking"""
        self._cancelled = True

    def step(
        self,
        previous_image: np.ndarray,
        previous_mask: np.ndarray,
        current_image: np.ndarray,
        frame_index: int = 0,
    ) -> np.ndarray:
        """
        Perform a single tracking step with CUTIE.

        Args:
            previous_image: Previous frame image
            previous_mask: Previous frame mask
            current_image: Current frame image
            frame_index: Current frame index (for progress reporting)

        Returns:
            Predicted mask for current frame
        """
        if self._cancelled:
            return None

        # Check if tracker is available
        if self.tracker is None:
            raise RuntimeError("CUTIE tracker not initialized")

        self.status_update.emit(f"Tracking frame {frame_index + 1}...")

        # Use the new track method that handles both steps internally
        try:
            predicted_mask = self.tracker.track(
                previous_image, previous_mask, current_image
            )
        except Exception as e:
            raise RuntimeError(f"Failed to track frame: {e}")

        self.status_update.emit("Frame tracking complete!")

        if not self._cancelled:
            self.frame_tracked.emit(frame_index, predicted_mask)

        return predicted_mask
