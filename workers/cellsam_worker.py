import cv2
import numpy as np
from cellsam import CellSAM
from PyQt6.QtCore import QThread, pyqtSignal


class CellSamWorker(QThread):
    # Signals for async communication
    result_ready = pyqtSignal(np.ndarray)  # masks
    error_occurred = pyqtSignal(str)  # error message
    status_update = pyqtSignal(str)  # status message

    def __init__(self):
        super().__init__()
        self._model = CellSAM(gpu=True)
        self._current_frame_path = None

    def run_async(self, first_frame_path: str):
        """Start CellSAM processing asynchronously"""
        if self.isRunning():
            return  # Already running

        self._current_frame_path = first_frame_path
        self.start()

    def run(self, first_frame_path=None):
        """Run CellSAM processing - can be called directly or as QThread.run()"""
        if first_frame_path is None:
            first_frame_path = self._current_frame_path

        if first_frame_path is None:
            self.error_occurred.emit("No frame path provided")
            return None

        try:
            self.status_update.emit("Loading first frame...")

            # Load and process first frame
            img = cv2.imread(first_frame_path)
            if img is None:
                raise RuntimeError(f"Failed to load first frame: {first_frame_path}")

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            self.status_update.emit("Running CellSAM segmentation...")

            # Run segmentation on first frame
            masks, flows, styles = self._model.segment(img, diameter=None)

            # If running in thread mode, emit signal
            if hasattr(self, "result_ready"):
                self.result_ready.emit(masks)
                return
            else:
                # Legacy mode for synchronous calls
                result = {
                    "frame_path": first_frame_path,
                    "masks": masks,
                    "flows": flows,
                    "styles": styles,
                    "original_image": img,
                }
                return result

        except Exception as e:
            if hasattr(self, "error_occurred"):
                self.error_occurred.emit(str(e))
            else:
                raise

    def cleanup(self):
        del self._model
