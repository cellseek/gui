import numpy as np
from cutie.cutie_tracker import CutieTracker


class CutieWorker:

    def __init__(self):
        """
        Initialize CUTIE worker and model.
        """
        super().__init__()

        self._cancelled = False
        self.tracker = None

        # Initialize CUTIE tracker
        self.tracker = CutieTracker()

    def track(
        self,
        previous_image: np.ndarray,
        previous_mask: np.ndarray,
        current_image: np.ndarray,
    ) -> np.ndarray:
        """
        Perform a single tracking step with CUTIE.

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
