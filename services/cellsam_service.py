"""
CellSAM service for handling cell segmentation functionality
"""

from typing import List, Optional, Protocol

import numpy as np

from workers.cellsam_worker import CellSamWorker


class CellSamServiceDelegate(Protocol):
    """Protocol for objects that can delegate CellSAM operations"""

    def emit_status_update(self, message: str) -> None: ...
    def emit_progress_update(self, progress: int, message: str) -> None: ...
    def show_error(self, title: str, message: str) -> None: ...
    def on_cellsam_processing_complete(
        self, frame_paths: List[str], first_frame_result: dict
    ) -> None: ...


class CellSamService:
    """Service for CellSAM segmentation functionality"""

    def __init__(self, delegate: CellSamServiceDelegate):
        self.delegate = delegate
        self._cellsam_worker: Optional[CellSamWorker] = None

    def process_frames(self, frame_paths: List[str]) -> bool:
        """
        Start CellSAM processing on the first frame of the provided frame paths.

        Args:
            frame_paths: List of frame file paths to process

        Returns:
            True if processing started successfully, False otherwise
        """
        if not frame_paths:
            self.delegate.show_error(
                "Error", "No frames provided for CellSAM processing"
            )
            return False

        if self._cellsam_worker is not None:
            self.delegate.show_error("Error", "CellSAM processing already in progress")
            return False

        try:
            # Create and configure worker
            self._cellsam_worker = CellSamWorker(frame_paths)
            self._cellsam_worker.progress_update.connect(self._on_progress_update)
            self._cellsam_worker.processing_complete.connect(
                self._on_processing_complete
            )
            self._cellsam_worker.error_occurred.connect(self._on_error_occurred)

            # Start processing
            self._cellsam_worker.start()
            self.delegate.emit_status_update("Initializing CellSAM segmentation...")

            return True

        except Exception as e:
            self.delegate.show_error(
                "Error", f"Failed to start CellSAM processing: {str(e)}"
            )
            self._cellsam_worker = None
            return False

    def cancel_processing(self) -> None:
        """Cancel ongoing CellSAM processing"""
        if self._cellsam_worker is not None:
            self._cellsam_worker.cancel()
            self._cellsam_worker = None
            self.delegate.emit_status_update("CellSAM processing cancelled")

    def is_processing(self) -> bool:
        """Check if CellSAM processing is currently running"""
        return self._cellsam_worker is not None

    def cleanup(self) -> None:
        """Cleanup CellSAM resources"""
        if self._cellsam_worker is not None:
            self._cellsam_worker.cancel()
            self._cellsam_worker = None

    def _on_progress_update(self, progress: int, status: str) -> None:
        """Handle progress updates from CellSAM worker"""
        self.delegate.emit_progress_update(progress, status)

    def _on_processing_complete(
        self, frame_paths: List[str], first_frame_result: dict
    ) -> None:
        """Handle CellSAM processing completion"""
        try:
            # Clean up worker
            self._cellsam_worker = None

            # Notify delegate of completion
            self.delegate.on_cellsam_processing_complete(
                frame_paths, first_frame_result
            )

            # Update status
            self.delegate.emit_status_update(
                f"First frame segmented - Loaded {len(frame_paths)} frames for tracking"
            )

        except Exception as e:
            self.delegate.show_error(
                "Error", f"Failed to handle CellSAM completion: {str(e)}"
            )

    def _on_error_occurred(self, error_message: str) -> None:
        """Handle CellSAM processing errors"""
        # Clean up worker
        self._cellsam_worker = None

        # Show error and update status
        self.delegate.show_error("CellSAM Error", error_message)
        self.delegate.emit_status_update("CellSAM processing failed")
