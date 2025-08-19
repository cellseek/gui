import numpy as np
import torch
from PyQt6.QtCore import QThread, pyqtSignal
from segment_anything import SamPredictor, sam_model_registry


class SAMWorker(QThread):
    """Worker thread for SAM operations"""

    sam_complete = pyqtSignal(np.ndarray, float)  # mask, score
    error_occurred = pyqtSignal(str)  # error message

    def __init__(self, image: np.ndarray, operation: str, **kwargs):
        super().__init__()
        self.image = image
        self.operation = operation
        self.kwargs = kwargs
        self._cancelled = False

    def cancel(self):
        """Cancel the operation"""
        self._cancelled = True

    def run(self):
        """Run SAM operation in background thread"""
        try:

            # Initialize SAM
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load SAM model (you may need to adjust the model type and checkpoint path)
            sam = sam_model_registry["vit_h"](
                checkpoint="checkpoints/sam_vit_h_4b8939.pth"
            )
            sam.to(device)
            predictor = SamPredictor(sam)

            if self._cancelled:
                return

            # Set the image for the predictor
            predictor.set_image(self.image)

            if self.operation == "point":
                point = self.kwargs["point"]
                input_point = np.array([point])
                input_label = np.array([1])

                masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                )

                # Use the best mask (highest score)
                best_idx = np.argmax(scores)
                mask = masks[best_idx]
                score = scores[best_idx]

            elif self.operation == "box":
                box = self.kwargs["box"]
                input_box = np.array([box])  # [x1, y1, x2, y2]

                masks, scores, logits = predictor.predict(
                    box=input_box,
                    multimask_output=False,
                )

                mask = masks[0]
                score = scores[0]
            else:
                raise ValueError(f"Unknown operation: {self.operation}")

            if not self._cancelled:
                self.sam_complete.emit(mask, score)

        except Exception as e:
            self.error_occurred.emit(f"SAM operation failed: {str(e)}")
