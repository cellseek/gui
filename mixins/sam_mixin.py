"""
SAM (Segment Anything Model) functionality mixin for frame-by-frame widget
"""

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from PyQt6.QtWidgets import QMessageBox

from workers.sam_worker import SamWorker

if TYPE_CHECKING:
    from protocols.storage_protocol import StorageProtocol, UIProtocol


class SamMixin:
    """Mixin for SAM segmentation functionality

    Requires the implementing class to provide StorageProtocol and UIProtocol interfaces.
    """

    def __init__(self) -> None:
        # Initialize SAM worker lazily
        self._sam_worker: Optional[SamWorker] = None

    def _assert_protocols(self) -> None:
        """Assert that the implementing class provides required protocols"""
        if TYPE_CHECKING:
            # Type checker will verify these interfaces exist
            assert isinstance(self, StorageProtocol)
            assert isinstance(self, UIProtocol)

    @property
    def sam_worker(self) -> Optional[SamWorker]:
        """Lazy initialization of SAM worker"""
        if self._sam_worker is None:
            try:
                self._sam_worker = SamWorker()
                print("SAM worker initialized successfully")
            except Exception as e:
                print(f"Failed to initialize SAM worker: {str(e)}")
                return None
        return self._sam_worker

    def cleanup_sam_worker(self) -> None:
        """Cleanup SAM worker"""
        try:
            if self._sam_worker is not None:
                self._sam_worker.cancel()
                self._sam_worker.cleanup()
                self._sam_worker = None
        except:
            pass  # Ignore cleanup errors during destruction

    def on_point_clicked(self, point: Tuple[int, int]) -> None:
        """Handle point click for SAM segmentation"""
        if self.get_frame_count() == 0:
            return

        # Check if SAM worker is available
        if self.sam_worker is None:
            QMessageBox.warning(self, "SAM Error", "SAM worker not initialized")
            return

        if self.sam_worker.isRunning():
            return  # Worker is busy

        current_image = self.get_current_frame()

        try:
            # Set image and predict
            self.sam_worker.set_image(current_image)
            mask, score = self.sam_worker.predict_point(point)

            # Emit the result directly
            self.on_sam_complete(mask, score)

        except Exception as e:
            self.on_sam_error(f"SAM point prediction failed: {str(e)}")

        self.status_update.emit(f"Running SAM on point {point}...")

    def on_box_drawn(self, box: Tuple[int, int, int, int]) -> None:
        """Handle box drawing for SAM segmentation"""
        if self.get_frame_count() == 0:
            return

        # Check if SAM worker is available
        if self.sam_worker is None:
            QMessageBox.warning(self, "SAM Error", "SAM worker not initialized")
            return

        if self.sam_worker.isRunning():
            return  # Worker is busy

        current_image = self.get_current_frame()

        try:
            # Set image and predict
            self.sam_worker.set_image(current_image)
            mask, score = self.sam_worker.predict_box(box)

            # Emit the result directly
            self.on_sam_complete(mask, score)

        except Exception as e:
            self.on_sam_error(f"SAM box prediction failed: {str(e)}")

        self.status_update.emit(f"Running SAM on box {box}...")

    def on_sam_complete(self, mask: np.ndarray, score: float) -> None:
        """Handle SAM completion"""
        # Note: No need to set worker to None since it's persistent now

        # Add mask to current frame
        current_masks = self.get_current_frame_masks()
        if current_masks is None:
            # Create new mask array
            current_frame = self.get_current_frame()
            h, w = current_frame.shape[:2]
            current_masks = np.zeros((h, w), dtype=np.uint16)

        # Find next available mask ID
        next_id = np.max(current_masks) + 1

        # Add new mask
        current_masks[mask > 0] = next_id
        current_index = self.get_current_frame_index()
        self.set_mask_for_frame(current_index, current_masks)
        self.curr_image_label.set_masks(current_masks)

        self.status_update.emit(f"Added mask {next_id} (score: {score:.3f})")

    def on_sam_error(self, error_message: str) -> None:
        """Handle SAM error"""
        self.status_update.emit("SAM operation failed")
        QMessageBox.warning(self, "SAM Error", error_message)
