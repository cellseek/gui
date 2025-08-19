from typing import List

import cv2
from cellsam import CellSAM
from PyQt6.QtCore import QThread, pyqtSignal


class CellSAMWorker(QThread):
    """Worker thread for running CellSAM processing on first frame only"""

    progress_update = pyqtSignal(int, str)  # progress, status
    processing_complete = pyqtSignal(
        list, dict
    )  # frame_paths, first_frame_segmentation
    error_occurred = pyqtSignal(str)  # error message

    def __init__(self, frame_paths: List[str]):
        super().__init__()
        self.frame_paths = frame_paths
        self._cancelled = False

    def cancel(self):
        """Cancel the processing"""
        self._cancelled = True

    def run(self):
        """Run CellSAM processing on first frame only"""
        try:
            self.progress_update.emit(0, "Initializing CellSAM model...")

            # Initialize CellSAM model
            model = CellSAM(gpu=True)

            self.progress_update.emit(30, "Model loaded. Processing first frame...")

            if self._cancelled or not self.frame_paths:
                return

            # Process only the first frame
            first_frame_path = self.frame_paths[0]

            # Load and process first frame
            img = cv2.imread(first_frame_path)
            if img is None:
                self.error_occurred.emit(
                    f"Could not load first frame: {first_frame_path}"
                )
                return

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            self.progress_update.emit(60, "Running segmentation on first frame...")

            # Run segmentation on first frame
            masks, flows, styles = model.segment(img, diameter=None)

            # Store first frame results
            first_frame_result = {
                "frame_path": first_frame_path,
                "masks": masks,
                "flows": flows,
                "styles": styles,
                "original_image": img,
            }

            self.progress_update.emit(90, "Cleaning up model...")

            # Clean up model to free memory
            del model

            self.progress_update.emit(100, "First frame processing complete!")

            if not self._cancelled:
                self.processing_complete.emit(self.frame_paths, first_frame_result)

        except Exception as e:
            self.error_occurred.emit(f"CellSAM processing failed: {str(e)}")
