"""
SAM (Segment Anything Model) service for segmentation functionality
"""

from typing import Optional, Protocol, Tuple

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from workers.sam_worker import SamWorker


class SamServiceDelegate(Protocol):
    """Protocol for objects that can delegate SAM operations"""

    def get_frame_count(self) -> int: ...
    def get_current_frame(self) -> np.ndarray | None: ...
    def get_current_frame_masks(self) -> np.ndarray | None: ...
    def set_mask_for_frame(self, frame_index: int, masks: np.ndarray) -> None: ...
    def get_current_frame_index(self) -> int: ...
    def remove_masks_after_frame(self, frame_index: int) -> int: ...
    def emit_status_update(self, message: str) -> None: ...
    def show_warning(self, title: str, message: str) -> None: ...
    def update_current_display_masks(self, masks: np.ndarray) -> None: ...


class SamService(QObject):
    """Service for SAM segmentation functionality"""

    # Signals for async communication
    sam_result_ready = pyqtSignal(np.ndarray, float)  # mask, score
    sam_error = pyqtSignal(str)  # error message
    status_update = pyqtSignal(str)  # status message

    def __init__(self, delegate: SamServiceDelegate):
        super().__init__()
        self.delegate = delegate
        # Initialize SAM worker lazily
        self._sam_worker: Optional[SamWorker] = None
        # Track which frame index is currently loaded in SAM
        self._loaded_frame_index: Optional[int] = None
        # Track pending prediction requests
        self._pending_prediction: Optional[tuple] = None

        # Connect signals
        self.sam_result_ready.connect(self._on_sam_complete)
        self.sam_error.connect(self._on_sam_error)
        self.status_update.connect(self.delegate.emit_status_update)

    @property
    def sam_worker(self) -> Optional[SamWorker]:
        """Initialize SAM worker on demand"""
        if self._sam_worker is None:
            try:
                self.status_update.emit("Loading SAM model...")
                self._sam_worker = SamWorker()
                # Connect worker signals
                self._sam_worker.result_ready.connect(self.sam_result_ready.emit)
                self._sam_worker.error_occurred.connect(self.sam_error.emit)
                self._sam_worker.status_update.connect(self._on_worker_status_update)
                self.status_update.emit("SAM worker loaded")
            except Exception as e:
                print(f"Failed to initialize SAM worker: {str(e)}")
                self.status_update.emit(f"SAM initialization failed: {str(e)}")
                return None
        return self._sam_worker

    def _on_worker_status_update(self, message: str) -> None:
        """Handle status updates from SAM worker and execute pending predictions"""
        print(f"SAM Worker Status: {message}")  # Debug logging
        self.status_update.emit(message)

        # If image was just loaded and we have a pending prediction, execute it
        if message == "Image loaded in SAM" and self._pending_prediction is not None:
            prediction_type, prediction_data = self._pending_prediction
            self._pending_prediction = None

            print(
                f"Executing pending {prediction_type} prediction: {prediction_data}"
            )  # Debug logging

            try:
                if prediction_type == "point":
                    self.sam_worker.predict_point_async(prediction_data)
                    self.status_update.emit(
                        f"Running SAM on point {prediction_data}..."
                    )
                elif prediction_type == "box":
                    self.sam_worker.predict_box_async(prediction_data)
                    self.status_update.emit(f"Running SAM on box {prediction_data}...")
            except Exception as e:
                print(f"Error executing pending prediction: {e}")  # Debug logging
                self.sam_error.emit(f"SAM prediction failed: {str(e)}")
            except Exception as e:
                self.sam_error.emit(f"SAM prediction failed: {str(e)}")

    def _ensure_frame_loaded(self, for_prediction: bool = False) -> bool:
        """Ensure the current frame is loaded in SAM (one-time per frame)"""
        if self.sam_worker is None:
            return False

        current_frame_index = self.delegate.get_current_frame_index()

        # Check if we already have this frame loaded
        if self._loaded_frame_index == current_frame_index:
            return True

        # Load the current frame
        current_image = self.delegate.get_current_frame()
        if current_image is None:
            return False

        try:
            # Set image asynchronously
            self.sam_worker.set_image_async(current_image)
            print(f"SAM: Loading frame {current_frame_index} asynchronously")

            # Only mark as loaded if this is not for a prediction
            # If it's for a prediction, we'll wait for the "Image loaded in SAM" status
            if not for_prediction:
                self._loaded_frame_index = current_frame_index

            return True
        except Exception as e:
            print(f"SAM: Failed to load frame {current_frame_index}: {str(e)}")
            self._loaded_frame_index = None
            return False

    def on_point_clicked(self, point: Tuple[int, int]) -> None:
        """Handle point click for SAM segmentation"""
        print(f"SAM: Point clicked at {point}")  # Debug logging

        if self.delegate.get_frame_count() == 0:
            print("SAM: No frames loaded")  # Debug logging
            return

        # Check if SAM worker is available
        if self.sam_worker is None:
            print("SAM: Worker not initialized")  # Debug logging
            self.delegate.show_warning("SAM Error", "SAM worker not initialized")
            return

        if self.sam_worker.isRunning():
            print("SAM: Worker is busy")  # Debug logging
            return  # Worker is busy

        current_frame_index = self.delegate.get_current_frame_index()
        print(
            f"SAM: Current frame index: {current_frame_index}, loaded: {self._loaded_frame_index}"
        )  # Debug logging

        # Check if we already have this frame loaded
        if self._loaded_frame_index == current_frame_index:
            # Frame is already loaded, proceed with prediction
            print(
                "SAM: Frame already loaded, proceeding with prediction"
            )  # Debug logging
            try:
                self.sam_worker.predict_point_async(point)
                self.status_update.emit(f"Running SAM on point {point}...")
            except Exception as e:
                print(f"SAM: Error in direct prediction: {e}")  # Debug logging
                self.sam_error.emit(f"SAM point prediction failed: {str(e)}")
        else:
            # Need to load frame first, then predict
            print(
                "SAM: Need to load frame first, setting pending prediction"
            )  # Debug logging
            self._pending_prediction = ("point", point)
            if self._ensure_frame_loaded(for_prediction=True):
                self.status_update.emit("Loading image for SAM...")
            else:
                print("SAM: Failed to load frame")  # Debug logging
                self._pending_prediction = None
                self.delegate.show_warning("SAM Error", "Failed to load current frame")

    def on_box_drawn(self, box: Tuple[int, int, int, int]) -> None:
        """Handle box drawing for SAM segmentation"""
        if self.delegate.get_frame_count() == 0:
            return

        # Check if SAM worker is available
        if self.sam_worker is None:
            self.delegate.show_warning("SAM Error", "SAM worker not initialized")
            return

        if self.sam_worker.isRunning():
            return  # Worker is busy

        current_frame_index = self.delegate.get_current_frame_index()

        # Check if we already have this frame loaded
        if self._loaded_frame_index == current_frame_index:
            # Frame is already loaded, proceed with prediction
            try:
                self.sam_worker.predict_box_async(box)
                self.status_update.emit(f"Running SAM on box {box}...")
            except Exception as e:
                self.sam_error.emit(f"SAM box prediction failed: {str(e)}")
        else:
            # Need to load frame first, then predict
            self._pending_prediction = ("box", box)
            if self._ensure_frame_loaded(for_prediction=True):
                self.status_update.emit("Loading image for SAM...")
            else:
                self._pending_prediction = None
                self.delegate.show_warning("SAM Error", "Failed to load current frame")

    def _on_sam_complete(self, mask: np.ndarray, score: float) -> None:
        """Handle SAM completion"""
        # Mark frame as loaded if it wasn't already
        if self._loaded_frame_index != self.delegate.get_current_frame_index():
            self._loaded_frame_index = self.delegate.get_current_frame_index()

        # Add mask to current frame
        current_masks = self.delegate.get_current_frame_masks()
        if current_masks is None:
            # Create new mask array
            current_frame = self.delegate.get_current_frame()
            h, w = current_frame.shape[:2]
            current_masks = np.zeros((h, w), dtype=np.uint16)

        # Find next available mask ID
        next_id = np.max(current_masks) + 1

        # Add new mask
        current_masks[mask > 0] = next_id
        current_index = self.delegate.get_current_frame_index()
        self.delegate.set_mask_for_frame(current_index, current_masks)
        self.delegate.update_current_display_masks(current_masks)

        self.delegate.emit_status_update(f"Added mask {next_id} (score: {score:.3f})")

        # Handle consequences of mask modification - remove subsequent masks
        total_frames = self.delegate.get_frame_count()
        if current_index < total_frames - 1:
            removed_count = self.delegate.remove_masks_after_frame(current_index)
            if removed_count > 0:
                last_removed_frame = current_index + removed_count
                self.delegate.emit_status_update(
                    f"Added mask {next_id}: removed {removed_count} dependent masks "
                    f"(frames {current_index + 2}-{last_removed_frame + 1})"
                )

    def _on_sam_error(self, error_message: str) -> None:
        """Handle SAM error"""
        # Clear any pending prediction
        self._pending_prediction = None
        self.delegate.emit_status_update("SAM operation failed")
        self.delegate.show_warning("SAM Error", error_message)
