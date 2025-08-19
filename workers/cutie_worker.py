from typing import List

import numpy as np
import torch
from cutie.cutie_tracker import CutieTracker
from PyQt6.QtCore import QThread, pyqtSignal


class CutieWorker(QThread):
    """Worker thread for CUTIE tracking"""

    progress_update = pyqtSignal(int, str)  # progress, status
    tracking_complete = pyqtSignal(dict)  # results: {frame_idx: masks}
    error_occurred = pyqtSignal(str)  # error message

    def __init__(self, frames: List[np.ndarray], first_frame_masks: np.ndarray):
        super().__init__()
        self.frames = frames
        self.first_frame_masks = first_frame_masks
        self._cancelled = False

    def cancel(self):
        """Cancel the tracking"""
        self._cancelled = True

    def run(self):
        """Run CUTIE tracking on all frames"""
        try:
            self.progress_update.emit(0, "Initializing CUTIE tracker...")

            # Check if CutieTracker is available
            if CutieTracker is None:
                raise ImportError(
                    "CutieTracker could not be imported. Check CUTIE installation."
                )

            # Validate inputs
            if not self.frames:
                raise ValueError("No frames provided for tracking")

            if self.first_frame_masks is None:
                raise ValueError("No first frame masks provided")

            if len(self.frames) < 1:
                raise ValueError("At least one frame is required")

            # Initialize CUTIE tracker
            try:
                tracker = CutieTracker()
            except Exception as e:
                import traceback

                print(f"Failed to initialize CutieTracker: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                raise RuntimeError(f"Failed to initialize CUTIE tracker: {e}")

            self.progress_update.emit(10, "Setting up first frame...")

            # Convert first frame masks to CUTIE format
            first_frame = self.frames[0]

            # Validate first frame
            if first_frame is None or first_frame.size == 0:
                raise ValueError("First frame is empty or invalid")

            # Initialize tracker with first frame and masks
            try:
                tracker.step(first_frame, self.first_frame_masks)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize tracker with first frame: {e}"
                )

            tracking_results = {0: self.first_frame_masks}

            self.progress_update.emit(20, "Tracking subsequent frames...")

            # Track through all subsequent frames
            for i in range(1, len(self.frames)):
                if self._cancelled:
                    break

                frame = self.frames[i]

                # Validate frame
                if frame is None or frame.size == 0:
                    raise ValueError(f"Frame {i} is empty or invalid")

                try:
                    # Track frame using CUTIE
                    masks = tracker.step(frame)
                    tracking_results[i] = masks
                except Exception as e:
                    raise RuntimeError(f"Failed to track frame {i+1}: {e}")

                # Update progress
                progress = 20 + int((i / (len(self.frames) - 1)) * 70)
                self.progress_update.emit(
                    progress, f"Tracked frame {i+1}/{len(self.frames)}"
                )

            self.progress_update.emit(95, "Finalizing tracking results...")

            # Clean up tracker
            try:
                del tracker
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Warning: Failed to clean up tracker: {e}")

            self.progress_update.emit(100, "Tracking complete!")

            if not self._cancelled:
                self.tracking_complete.emit(tracking_results)

        except Exception as e:
            import traceback

            error_msg = f"CUTIE tracking failed: {type(e).__name__}: {str(e)}"
            if not str(e):
                error_msg = (
                    f"CUTIE tracking failed: {type(e).__name__} (no error message)"
                )
            traceback_str = traceback.format_exc()
            detailed_error = f"{error_msg}\n\nFull traceback:\n{traceback_str}"
            print(
                f"CUTIE Error Debug: {detailed_error}"
            )  # Also print to console for debugging
            self.error_occurred.emit(detailed_error)
