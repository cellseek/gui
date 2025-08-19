"""
SAM (Segment Anything Model) functionality mixin for frame-by-frame widget
"""

from typing import Tuple

import numpy as np
from PyQt6.QtWidgets import QMessageBox

from workers.sam_worker import SAMWorker


class SamMixin:
    """Mixin for SAM segmentation functionality"""

    def __init__(self):
        # Initialize SAM worker lazily
        self._sam_worker = None

    @property
    def sam_worker(self):
        """Lazy initialization of SAM worker"""
        if self._sam_worker is None:
            try:
                self._sam_worker = SAMWorker()
                print("SAM worker initialized successfully")
            except Exception as e:
                print(f"Failed to initialize SAM worker: {str(e)}")
                return None
        return self._sam_worker

    def cleanup_sam_worker(self):
        """Cleanup SAM worker"""
        try:
            if self._sam_worker is not None:
                self._sam_worker.cancel()
                self._sam_worker.cleanup()
                self._sam_worker = None
        except:
            pass  # Ignore cleanup errors during destruction

    def on_point_clicked(self, point: Tuple[int, int]):
        """Handle point click for SAM segmentation"""
        if not self.frames:
            return

        # Check if SAM worker is available
        if self.sam_worker is None:
            QMessageBox.warning(self, "SAM Error", "SAM worker not initialized")
            return

        if self.sam_worker.isRunning():
            return  # Worker is busy

        current_image = self.frames[self.current_frame_index]

        try:
            # Set image and predict
            self.sam_worker.set_image(current_image)
            mask, score = self.sam_worker.predict_point(point)

            # Emit the result directly
            self.on_sam_complete(mask, score)

        except Exception as e:
            self.on_sam_error(f"SAM point prediction failed: {str(e)}")

        self.status_update.emit(f"Running SAM on point {point}...")

    def on_box_drawn(self, box: Tuple[int, int, int, int]):
        """Handle box drawing for SAM segmentation"""
        if not self.frames:
            return

        # Check if SAM worker is available
        if self.sam_worker is None:
            QMessageBox.warning(self, "SAM Error", "SAM worker not initialized")
            return

        if self.sam_worker.isRunning():
            return  # Worker is busy

        current_image = self.frames[self.current_frame_index]

        try:
            # Set image and predict
            self.sam_worker.set_image(current_image)
            mask, score = self.sam_worker.predict_box(box)

            # Emit the result directly
            self.on_sam_complete(mask, score)

        except Exception as e:
            self.on_sam_error(f"SAM box prediction failed: {str(e)}")

        self.status_update.emit(f"Running SAM on box {box}...")

    def on_sam_complete(self, mask: np.ndarray, score: float):
        """Handle SAM completion"""
        # Note: No need to set worker to None since it's persistent now

        # Add mask to current frame
        current_masks = self.frame_masks.get(self.current_frame_index)
        if current_masks is None:
            # Create new mask array
            h, w = self.frames[self.current_frame_index].shape[:2]
            current_masks = np.zeros((h, w), dtype=np.uint16)

        # Find next available mask ID
        next_id = np.max(current_masks) + 1

        # Add new mask
        current_masks[mask > 0] = next_id
        self.frame_masks[self.current_frame_index] = current_masks
        self.curr_image_label.set_masks(current_masks)

        self.status_update.emit(f"Added mask {next_id} (score: {score:.3f})")

    def on_sam_error(self, error_message: str):
        """Handle SAM error"""
        self.status_update.emit("SAM operation failed")
        QMessageBox.warning(self, "SAM Error", error_message)
