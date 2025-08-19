import numpy as np
import torch
from PyQt6.QtCore import QThread, pyqtSignal
from segment_anything import SamPredictor, sam_model_registry


class SAMWorker(QThread):
    """Worker thread for SAM operations"""

    sam_complete = pyqtSignal(np.ndarray, float)  # mask, score
    error_occurred = pyqtSignal(str)  # error message

    def __init__(self):
        """Initialize SAM worker and model."""
        super().__init__()
        self._cancelled = False

        # Initialize SAM
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load SAM model
            sam = sam_model_registry["vit_h"](
                checkpoint="checkpoints/sam_vit_h_4b8939.pth"
            )
            sam.to(device)
            self.predictor = SamPredictor(sam)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SAM: {e}")

    def cancel(self):
        """Cancel the operation"""
        self._cancelled = True

    def set_image(self, image: np.ndarray):
        """Set the image for SAM processing."""
        try:
            self.predictor.set_image(image)
        except Exception as e:
            raise RuntimeError(f"Failed to set image: {e}")

    def predict_point(self, point: tuple) -> tuple:
        """Predict mask from point click.

        Args:
            point: (x, y) coordinates of the point

        Returns:
            tuple: (mask, score) - best mask and its score
        """
        if self._cancelled:
            return None, 0.0

        try:
            input_point = np.array([point])
            input_label = np.array([1])

            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )

            # Use the best mask (highest score)
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            score = scores[best_idx]

            return mask, score
        except Exception as e:
            raise RuntimeError(f"Point prediction failed: {e}")

    def predict_box(self, box: tuple) -> tuple:
        """Predict mask from bounding box.

        Args:
            box: (x1, y1, x2, y2) coordinates of the box

        Returns:
            tuple: (mask, score) - mask and its score
        """
        if self._cancelled:
            return None, 0.0

        try:
            input_box = np.array([box])  # [x1, y1, x2, y2]

            masks, scores, logits = self.predictor.predict(
                box=input_box,
                multimask_output=False,
            )

            mask = masks[0]
            score = scores[0]

            return mask, score
        except Exception as e:
            raise RuntimeError(f"Box prediction failed: {e}")

    def run(self):
        """Run method for QThread compatibility - placeholder"""
        # This method can be used for batch processing in the future
        # For now, individual predictions should use predict_point() or predict_box() methods directly
        pass

    def cleanup(self):
        """Clean up model resources"""
        try:
            if hasattr(self, "predictor"):
                del self.predictor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: Failed to clean up SAM model: {e}")
