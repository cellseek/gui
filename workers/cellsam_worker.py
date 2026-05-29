import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from utils.cellsam_segment import segment_rgb


class CellSamWorker(QThread):
    result_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._current_frame_path = None
        self._current_image = None

    def run_async(self, first_frame_path: str = None, image_rgb: np.ndarray = None):
        if self.isRunning():
            return

        self._current_frame_path = first_frame_path
        self._current_image = image_rgb
        self.start()

    def run(self, first_frame_path=None, image_rgb=None):
        if first_frame_path is None:
            first_frame_path = self._current_frame_path
        if image_rgb is None:
            image_rgb = self._current_image

        if first_frame_path is None and image_rgb is None:
            self.error_occurred.emit("No frame provided for segmentation")
            return None

        try:
            self.status_update.emit("Loading first frame...")

            if image_rgb is not None:
                img = np.ascontiguousarray(image_rgb)
            else:
                loaded = cv2.imread(first_frame_path)
                if loaded is None:
                    raise RuntimeError(f"Failed to load first frame: {first_frame_path}")
                img = cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB)

            self.status_update.emit("Running CellSAM segmentation...")
            masks = segment_rgb(img)

            if hasattr(self, "result_ready"):
                self.result_ready.emit(masks)
                return None

            return {"frame_path": first_frame_path, "masks": masks}

        except Exception as e:
            if hasattr(self, "error_occurred"):
                self.error_occurred.emit(str(e))
            else:
                raise
