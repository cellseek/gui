from typing import List, Optional

import numpy as np
import torch
from PyQt6.QtCore import QThread, pyqtSignal


class CutieWorker(QThread):
    """Worker thread for CUTIE tracking"""

    progress_update = pyqtSignal(int, str)  # progress, status
    tracking_complete = pyqtSignal(dict)  # results: {frame_idx: masks}
    frame_tracked = pyqtSignal(
        int, np.ndarray
    )  # frame_idx, masks for single frame tracking
    error_occurred = pyqtSignal(str)  # error message

    def __init__(self):
        """
        Initialize CUTIE worker and model.
        """
        super().__init__()

        self._cancelled = False
        self.tracker = None

        # Initialize CUTIE tracker
        try:
            from cutie.cutie_tracker import CutieTracker

            self.tracker = CutieTracker()
            print("CUTIE tracker initialized successfully")
        except ImportError as e:
            print(f"Failed to import CutieTracker: {e}")
            raise RuntimeError(f"CUTIE not properly installed: {e}")
        except Exception as e:
            import traceback

            print(f"Failed to initialize CutieTracker: {e}")
            print(f"Traceback: {traceback.format_exc()}")
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

        # Validate inputs
        if previous_image is None:
            raise ValueError("Previous image is required for tracking")

        if previous_mask is None:
            raise ValueError("Previous mask is required for tracking")

        if current_image is None:
            raise ValueError("Current image is required for tracking")

        self.progress_update.emit(0, f"Tracking frame {frame_index + 1}...")

        self.progress_update.emit(25, "Processing previous frame...")

        # First step: process previous frame with its mask
        try:
            self.tracker.step(previous_image, previous_mask)
        except Exception as e:
            raise RuntimeError(f"Failed to process previous frame: {e}")

        self.progress_update.emit(75, "Tracking current frame...")

        # Second step: track current frame (no mask provided, get prediction)
        try:
            predicted_mask = self.tracker.step(current_image)
        except Exception as e:
            raise RuntimeError(f"Failed to track current frame: {e}")

        self.progress_update.emit(95, "Finalizing results...")

        self.progress_update.emit(100, "Frame tracking complete!")

        if not self._cancelled:
            self.frame_tracked.emit(frame_index, predicted_mask)

        return predicted_mask

    def cleanup(self):
        """Clean up tracker resources"""
        try:
            if hasattr(self, "tracker"):
                del self.tracker
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: Failed to clean up tracker: {e}")
