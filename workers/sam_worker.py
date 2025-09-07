import numpy as np
import torch
from PyQt6.QtCore import QThread, pyqtSignal
from segment_anything import SamPredictor, sam_model_registry


class SamWorker(QThread):
    # Signals for async communication
    result_ready = pyqtSignal(np.ndarray, float)  # mask, score
    error_occurred = pyqtSignal(str)  # error message
    status_update = pyqtSignal(str)  # status message

    def __init__(self):
        """Initialize SAM worker and model."""
        super().__init__()

        # Initialize SAM
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = "weights/sam_vit_h_4b8939.pth"
        sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
        sam.to(device)

        # Set predictor
        self.predictor = SamPredictor(sam)

        # Task queue for async operations
        self._current_task = None
        self._current_image = None

    def set_image_async(self, image: np.ndarray):
        """Set the image for SAM processing asynchronously."""
        if self.isRunning():
            return  # Busy with another task

        self._current_image = image.copy()
        self._current_task = "set_image"
        self.start()

    def predict_point_async(self, point: tuple):
        """Predict mask from point click asynchronously."""
        if self.isRunning():
            return  # Busy with another task

        self._current_task = ("predict_point", point)
        self.start()

    def predict_box_async(self, box: tuple):
        """Predict mask from bounding box asynchronously."""
        if self.isRunning():
            return  # Busy with another task

        self._current_task = ("predict_box", box)
        self.start()

    def run(self):
        """Main thread execution method."""
        try:
            if self._current_task == "set_image":
                self.predictor.set_image(self._current_image)
                self.status_update.emit("Image loaded in SAM")

            elif isinstance(self._current_task, tuple):
                task_type, data = self._current_task

                if task_type == "predict_point":
                    point = data
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

                    self.result_ready.emit(mask, score)

                elif task_type == "predict_box":
                    box = data
                    input_box = np.array([box])  # [x1, y1, x2, y2]

                    masks, scores, logits = self.predictor.predict(
                        box=input_box,
                        multimask_output=False,
                    )

                    mask = masks[0]
                    score = scores[0]

                    self.result_ready.emit(mask, score)

        except Exception as e:
            self.error_occurred.emit(str(e))

        finally:
            self._current_task = None
