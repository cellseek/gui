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
            checkpoint = "weights/sam_vit_b_01ec64.pth"
            sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
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

    def predict_paint_mask_async(self, paint_mask: np.ndarray):
        """Run SAM with a user-painted region as the mask prompt."""
        if self.isRunning():
            return

        self._current_task = ("predict_paint", paint_mask)
        self.start()

    @staticmethod
    def _predict_from_paint_stroke(
        predictor: SamPredictor, paint_mask: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Segment the painted cell using box + point prompts, then pick the mask
        that best overlaps the brush stroke (avoids whole-frame selections).
        """
        stroke = paint_mask.astype(bool)
        if not np.any(stroke):
            raise ValueError("Empty paint stroke")

        height, width = stroke.shape
        ys, xs = np.where(stroke)
        pad = max(4, int(min(height, width) * 0.01))
        x1 = max(0, int(xs.min()) - pad)
        y1 = max(0, int(ys.min()) - pad)
        x2 = min(width - 1, int(xs.max()) + pad)
        y2 = min(height - 1, int(ys.max()) + pad)

        cx = float(xs.mean())
        cy = float(ys.mean())
        input_box = np.array([x1, y1, x2, y2])
        input_point = np.array([[cx, cy]])
        input_label = np.array([1])

        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=True,
        )

        frame_area = height * width
        best_idx = 0
        best_iou = -1.0
        for i, candidate in enumerate(masks):
            coverage = float(np.count_nonzero(candidate)) / float(frame_area)
            if coverage > 0.85:
                continue
            intersection = np.count_nonzero(candidate & stroke)
            if intersection == 0:
                continue
            union = np.count_nonzero(candidate | stroke)
            iou = float(intersection) / float(union)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou < 0:
            # Fall back: smallest mask that still touches the stroke
            touching = [
                i
                for i, candidate in enumerate(masks)
                if np.any(candidate & stroke)
                and np.count_nonzero(candidate) / frame_area <= 0.85
            ]
            if touching:
                best_idx = min(
                    touching, key=lambda i: np.count_nonzero(masks[i])
                )
            else:
                best_idx = int(np.argmin([np.count_nonzero(m) for m in masks]))

        return masks[best_idx], float(scores[best_idx])

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

                elif task_type == "predict_paint":
                    mask, score = self._predict_from_paint_stroke(predictor, data)
                    self.result_ready.emit(mask, score)

        except Exception as e:
            self.error_occurred.emit(str(e))

        finally:
            self._current_task = None
