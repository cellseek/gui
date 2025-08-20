"""
CellSAM service for handling cell segmentation functionality
"""

from typing import List, Optional, Protocol

import numpy as np

from workers.cellsam_worker import CellSamWorker


class CellSamServiceDelegate(Protocol):
    """Protocol for objects that can delegate CellSAM operations"""

    def emit_status_update(self, message: str) -> None: ...
    def show_error(self, title: str, message: str) -> None: ...


class CellSamService:
    """Service for CellSAM segmentation functionality"""

    def __init__(self, delegate: CellSamServiceDelegate):
        self.delegate = delegate
        self._cellsam_worker = CellSamWorker()

    def _on_status_update(self, status: str) -> None:
        """Handle status updates from CellSAM worker"""
        self.delegate.emit_status_update(status)

    def _on_error_occurred(self, error_message: str) -> None:
        """Handle CellSAM processing errors"""
        # Clean up worker
        self._cellsam_worker = None

        # Show error and update status
        self.delegate.show_error("CellSAM Error", error_message)
        self.delegate.emit_status_update("CellSAM processing failed")

    def segment_first_frame(self, first_frame_path: str) -> Optional[np.ndarray]:
        try:
            self._on_status_update("Starting CellSAM processing...")
            result = self._cellsam_worker.run(first_frame_path)
            self._on_status_update("CellSAM processing completed")
            return result["masks"]
        except Exception as e:
            self._on_error_occurred(str(e))
            return None
