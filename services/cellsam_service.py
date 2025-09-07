"""
CellSAM service for handling cell segmentation functionality
"""

from typing import List, Optional, Protocol

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from workers.cellsam_worker import CellSamWorker


class CellSamServiceDelegate(Protocol):
    """Protocol for objects that can delegate CellSAM operations"""

    def emit_status_update(self, message: str) -> None: ...
    def show_error(self, title: str, message: str) -> None: ...


class CellSamService(QObject):
    """Service for CellSAM segmentation functionality"""

    # Signals for async communication
    segmentation_complete = pyqtSignal(np.ndarray)  # masks
    segmentation_error = pyqtSignal(str)  # error message
    status_update = pyqtSignal(str)  # status message

    def __init__(self, delegate: CellSamServiceDelegate):
        super().__init__()
        self.delegate = delegate
        self._cellsam_worker = CellSamWorker()

        # Connect worker signals
        self._cellsam_worker.result_ready.connect(self._on_segmentation_complete)
        self._cellsam_worker.error_occurred.connect(self._on_error_occurred)
        self._cellsam_worker.status_update.connect(self._on_status_update)

        # Connect our signals to delegate
        self.status_update.connect(self.delegate.emit_status_update)
        self.segmentation_error.connect(
            lambda msg: self.delegate.show_error("CellSAM Error", msg)
        )

    def _on_status_update(self, status: str) -> None:
        """Handle status updates from CellSAM worker"""
        self.status_update.emit(status)

    def _on_error_occurred(self, error_message: str) -> None:
        """Handle CellSAM processing errors"""
        # Clean up worker
        self._cellsam_worker = None

        # Show error and update status
        self.segmentation_error.emit(error_message)
        self.status_update.emit("CellSAM processing failed")

    def _on_segmentation_complete(self, masks: np.ndarray) -> None:
        """Handle successful segmentation completion"""
        self.status_update.emit("CellSAM processing completed")
        self.segmentation_complete.emit(masks)

    def segment_first_frame_async(self, first_frame_path: str) -> None:
        """Start CellSAM segmentation asynchronously"""
        if self._cellsam_worker.isRunning():
            self.status_update.emit("CellSAM is already processing...")
            return

        try:
            self.status_update.emit("Starting CellSAM processing...")
            self._cellsam_worker.run_async(first_frame_path)
        except Exception as e:
            self._on_error_occurred(f"Failed to start segmentation: {str(e)}")

    def segment_first_frame(self, first_frame_path: str) -> Optional[np.ndarray]:
        """Legacy synchronous method - deprecated, use segment_first_frame_async instead"""
        try:
            self._on_status_update("Starting CellSAM processing...")
            result = self._cellsam_worker.run(first_frame_path)
            self._on_status_update("CellSAM processing completed")
            return result["masks"]
        except Exception as e:
            self._on_error_occurred(f"Failed to segment first frame: {str(e)}")
            return None
