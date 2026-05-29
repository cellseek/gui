import numpy as np
import torch
from PyQt6.QtCore import QThread, pyqtSignal
from segment_anything import SamPredictor, sam_model_registry


class SamWorker(QThread):
    result_ready = pyqtSignal(np.ndarray, float)
    error_occurred = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._predictor = None
        self._current_task = None
        self._current_image = None

    def _ensure_predictor(self) -> SamPredictor:
        if self._predictor is None:
            self.status_update.emit("Loading SAM model...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = "weights/sam_vit_h_4b8939.pth"
            sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
            sam.to(device)
            self._predictor = SamPredictor(sam)
        return self._predictor

    def set_image_async(self, image: np.ndarray):
        if self.isRunning():
            return

        self._current_image = image.copy()
        self._current_task = "set_image"
        self.start()

    def predict_point_async(self, point: tuple):
        if self.isRunning():
            return

        self._current_task = ("predict_point", point)
        self.start()

    def predict_box_async(self, box: tuple):
        if self.isRunning():
            return

        self._current_task = ("predict_box", box)
        self.start()

    def run(self):
        try:
            predictor = self._ensure_predictor()

            if self._current_task == "set_image":
                predictor.set_image(self._current_image)
                self.status_update.emit("Image loaded in SAM")

            elif isinstance(self._current_task, tuple):
                task_type, data = self._current_task

                if task_type == "predict_point":
                    input_point = np.array([data])
                    input_label = np.array([1])

                    masks, scores, _ = predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=True,
                    )

                    best_idx = np.argmax(scores)
                    self.result_ready.emit(masks[best_idx], scores[best_idx])

                elif task_type == "predict_box":
                    input_box = np.array([data])

                    masks, scores, _ = predictor.predict(
                        box=input_box,
                        multimask_output=False,
                    )

                    self.result_ready.emit(masks[0], scores[0])

        except Exception as e:
            self.error_occurred.emit(str(e))

        finally:
            self._current_task = None
