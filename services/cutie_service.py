"""
CUTIE tracking service for handling frame-by-frame tracking functionality
"""

from typing import Optional, Protocol

import numpy as np

from workers.cutie_worker import CutieWorker


class CutieServiceDelegate(Protocol):
    """Protocol for objects that can delegate CUTIE operations"""

    def emit_status_update(self, message: str) -> None: ...
    def show_error(self, title: str, message: str) -> None: ...


class CutieService:
    """Simple service for CUTIE frame-by-frame tracking"""

    def __init__(self):
        # Initialize CUTIE worker lazily
        self._cutie_worker: Optional[CutieWorker] = None

    @property
    def cutie_worker(self) -> Optional[CutieWorker]:
        """Initialize CUTIE worker on demand"""
        if self._cutie_worker is None:
            try:
                self._cutie_worker = CutieWorker()
                print("CUTIE worker initialized successfully")
            except Exception as e:
                print(f"Failed to initialize CUTIE worker: {str(e)}")
                return None
        return self._cutie_worker

    def track(
        self,
        previous_image: np.ndarray,
        previous_mask: np.ndarray,
        current_image: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Track the next frame using CUTIE.

        Args:
            previous_image: Previous frame image
            previous_mask: Mask from previous frame
            current_image: Current frame image to track

        Returns:
            Predicted mask for current frame, or None if tracking failed
        """
        try:
            # Check if CUTIE worker is available
            if self.cutie_worker is None:
                raise RuntimeError("CUTIE worker not initialized")

            # Use the step function to get prediction
            predicted_mask = self._cutie_worker.track(
                previous_image, previous_mask, current_image
            )

            return predicted_mask

        except Exception as e:
            raise RuntimeError(f"Tracking failed: {str(e)}")
