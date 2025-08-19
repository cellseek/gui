"""
CUTIE tracking functionality mixin for frame-by-frame widget
"""

from typing import List

import numpy as np
from PyQt6.QtWidgets import QMessageBox

from workers.cutie_worker import CutieWorker


class CutieMixin:
    """Mixin for CUTIE tracking functionality"""

    def __init__(self):
        # Initialize CUTIE worker lazily
        self._cutie_worker = None

    @property
    def cutie_worker(self):
        """Lazy initialization of CUTIE worker"""
        if self._cutie_worker is None:
            try:
                self._cutie_worker = CutieWorker()
                print("CUTIE worker initialized successfully")
            except Exception as e:
                print(f"Failed to initialize CUTIE worker: {str(e)}")
                return None
        return self._cutie_worker

    def cleanup_cutie_worker(self):
        """Cleanup CUTIE worker"""
        try:
            if self._cutie_worker is not None:
                self._cutie_worker.cancel()
                self._cutie_worker.cleanup()
                self._cutie_worker = None
        except:
            pass  # Ignore cleanup errors during destruction

    def load_frames_with_first_segmentation(
        self, image_paths: List[str], first_frame_result: dict
    ):
        """Load frames with first frame segmentation, then run CUTIE tracking"""
        self.frames = []
        self.frame_masks = {}
        self.cellsam_results = {}
        self.first_frame_segmented = True
        self._cutie_worker = None

        # Load all frames from paths
        for path in image_paths:
            try:
                import cv2

                image = cv2.imread(path)
                if image is not None:
                    # Convert BGR to RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self.frames.append(image_rgb)
            except Exception as e:
                QMessageBox.warning(
                    self, "Load Error", f"Failed to load {path}: {str(e)}"
                )

        if not self.frames:
            QMessageBox.warning(self, "Load Error", "No valid frames loaded")
            return

        # Store first frame segmentation results
        self.cellsam_results[0] = first_frame_result

        # Extract first frame masks
        first_frame_masks = first_frame_result["masks"]
        if first_frame_masks is not None and first_frame_masks.size > 0:
            self.frame_masks[0] = first_frame_masks.astype(np.uint16)
        else:
            QMessageBox.warning(
                self, "Load Error", "No masks found in first frame segmentation"
            )
            return

        self.current_frame_index = 0
        self.update_display()

        # Show status
        cell_count = np.max(first_frame_masks) if first_frame_masks.size > 0 else 0
        self.status_update.emit(
            f"Loaded {len(self.frames)} frames. Found {cell_count} cells in first frame. Starting tracking..."
        )

        # Start CUTIE tracking for remaining frames
        if len(self.frames) > 1:
            self.start_cutie_tracking()
        else:
            self.status_update.emit("Single frame loaded with segmentation")

    def start_cutie_tracking(self):
        """Start CUTIE tracking using first frame masks"""
        try:
            # Check if CUTIE worker is available
            if self._cutie_worker is None:
                QMessageBox.warning(self, "CUTIE Error", "CUTIE worker not initialized")
                return

            if self._cutie_worker.isRunning():
                QMessageBox.warning(self, "CUTIE Error", "CUTIE worker is busy")
                return

            # Get first frame masks
            first_frame_masks = self.frame_masks[0]

            # Show progress
            self.status_update.emit("Starting CUTIE batch tracking...")

            # For now, we'll process frames sequentially using the step function
            # This could be improved by implementing proper batch processing in the worker
            try:
                self._process_cutie_batch(first_frame_masks)
            except Exception as e:
                self.on_cutie_error(f"CUTIE batch tracking failed: {str(e)}")

        except Exception as e:
            QMessageBox.critical(
                self, "Tracking Error", f"Failed to start CUTIE tracking: {str(e)}"
            )

    def _track_next_frame(self, next_index: int):
        """Track the next frame using CutieWorker"""
        try:
            # Check if CUTIE worker is available
            if self._cutie_worker is None:
                QMessageBox.warning(self, "CUTIE Error", "CUTIE worker not initialized")
                return

            if self._cutie_worker.isRunning():
                return  # Worker is busy

            # Get previous frame and mask
            previous_image = self.frames[self.current_frame_index]
            previous_mask = self.frame_masks[self.current_frame_index]
            current_image = self.frames[next_index]

            # Use the step function directly
            try:
                predicted_mask = self._cutie_worker.step(
                    previous_image, previous_mask, current_image, next_index
                )

                # Handle the result directly
                self.on_single_frame_tracked(next_index, predicted_mask)

            except Exception as e:
                self.on_cutie_error(f"CUTIE tracking failed: {str(e)}")

        except Exception as e:
            QMessageBox.warning(
                self,
                "Tracking Error",
                f"Failed to start tracking for frame {next_index + 1}: {str(e)}",
            )

    def on_single_frame_tracked(self, frame_index: int, predicted_mask: np.ndarray):
        """Handle completion of single frame tracking"""
        try:
            # Store the predicted mask
            self.frame_masks[frame_index] = predicted_mask.astype(np.uint16)

            # Move to the tracked frame
            self.current_frame_index = frame_index
            self.update_display()

            self.status_update.emit(f"Frame {frame_index + 1} tracked successfully")

        except Exception as e:
            QMessageBox.critical(
                self, "Tracking Error", f"Failed to process tracking results: {str(e)}"
            )

    def on_cutie_error(self, error_message: str):
        """Handle CUTIE tracking error"""
        QMessageBox.critical(self, "CUTIE Error", error_message)
        self.status_update.emit("Cell tracking failed")

    def _process_cutie_batch(self, first_frame_masks):
        """Process CUTIE batch tracking - placeholder for batch implementation"""
        # This method would need to be implemented based on the actual
        # batch processing capabilities of the CUTIE worker
        # For now, this is a placeholder that would need the actual implementation
        pass
