"""
Storage mixin for FrameByFrameWidget
Handles all data storage operations including frames, masks, and segmentation results.
"""

from typing import Dict, List, Optional

import numpy as np


class StorageMixin:
    """Mixin class to handle all storage operations for frame-by-frame analysis"""

    def __init__(self) -> None:
        """Initialize storage"""
        self._frames: List[np.ndarray] = []  # List of image arrays
        self._frame_masks: Dict[int, np.ndarray] = {}  # Dict[frame_index] = masks
        self._cellsam_results: Dict[int, dict] = {}  # Store CellSAM results
        self._current_frame_index: int = 0
        self._first_frame_segmented: bool = (
            False  # Track if CellSAM has run on first frame
        )
        self._image_paths: List[str] = []  # Store original image paths

    # Frame management
    def set_frames(self, frames: List[np.ndarray]) -> None:
        """Set the list of frames"""
        self._frames = frames.copy()
        self._current_frame_index = 0

    def get_frames(self) -> List[np.ndarray]:
        """Get all frames"""
        return self._frames.copy()

    def get_frame_count(self) -> int:
        """Get total number of frames"""
        return len(self._frames)

    def get_frame(self, index: int) -> Optional[np.ndarray]:
        """Get frame by index"""
        if 0 <= index < len(self._frames):
            return self._frames[index]
        return None

    def add_frame(self, frame: np.ndarray) -> None:
        """Add a frame to the collection"""
        self._frames.append(frame)

    def clear_frames(self) -> None:
        """Clear all frames"""
        self._frames.clear()
        self._current_frame_index = 0

    # Image paths management
    def set_image_paths(self, paths: List[str]) -> None:
        """Set image paths"""
        self._image_paths = paths.copy()

    def get_image_paths(self) -> List[str]:
        """Get image paths"""
        return self._image_paths.copy()

    # Current frame index management
    def set_current_frame_index(self, index: int) -> None:
        """Set current frame index"""
        if 0 <= index < len(self._frames):
            self._current_frame_index = index

    def get_current_frame_index(self) -> int:
        """Get current frame index"""
        return self._current_frame_index

    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get current frame"""
        return self.get_frame(self._current_frame_index)

    def has_previous_frame(self) -> bool:
        """Check if there's a previous frame"""
        return self._current_frame_index > 0

    def has_next_frame(self) -> bool:
        """Check if there's a next frame"""
        return self._current_frame_index < len(self._frames) - 1

    # Mask management
    def set_frame_masks(self, frame_masks: Dict[int, np.ndarray]) -> None:
        """Set all frame masks"""
        self._frame_masks = frame_masks.copy()

    def get_frame_masks(self) -> Dict[int, np.ndarray]:
        """Get all frame masks"""
        return self._frame_masks.copy()

    def set_mask_for_frame(self, frame_index: int, masks: np.ndarray) -> None:
        """Set masks for a specific frame"""
        self._frame_masks[frame_index] = masks

    def get_mask_for_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """Get masks for a specific frame"""
        return self._frame_masks.get(frame_index)

    def get_current_frame_masks(self) -> Optional[np.ndarray]:
        """Get masks for current frame"""
        return self.get_mask_for_frame(self._current_frame_index)

    def has_mask_for_frame(self, frame_index: int) -> bool:
        """Check if frame has masks"""
        return frame_index in self._frame_masks

    def remove_mask_for_frame(self, frame_index: int) -> None:
        """Remove masks for a specific frame"""
        if frame_index in self._frame_masks:
            del self._frame_masks[frame_index]

    def clear_all_masks(self) -> None:
        """Clear all masks"""
        self._frame_masks.clear()

    # CellSAM results management
    def set_cellsam_results(self, results: Dict[int, dict]) -> None:
        """Set CellSAM results"""
        self._cellsam_results = results.copy()

    def get_cellsam_results(self) -> Dict[int, dict]:
        """Get all CellSAM results"""
        return self._cellsam_results.copy()

    def set_cellsam_result_for_frame(self, frame_index: int, result: dict) -> None:
        """Set CellSAM result for a specific frame"""
        self._cellsam_results[frame_index] = result

    def get_cellsam_result_for_frame(self, frame_index: int) -> Optional[dict]:
        """Get CellSAM result for a specific frame"""
        return self._cellsam_results.get(frame_index)

    def clear_cellsam_results(self) -> None:
        """Clear all CellSAM results"""
        self._cellsam_results.clear()

    # First frame segmentation tracking
    def set_first_frame_segmented(self, segmented: bool) -> None:
        """Set whether first frame has been segmented"""
        self._first_frame_segmented = segmented

    def is_first_frame_segmented(self) -> bool:
        """Check if first frame has been segmented"""
        return self._first_frame_segmented

    # Utility methods
    def get_biggest_cell_id(self) -> int:
        """Get the biggest cell ID across all frames"""
        max_id = 0
        for masks in self._frame_masks.values():
            if masks is not None and masks.size > 0:
                frame_max = np.max(masks)
                max_id = max(max_id, frame_max)
        return int(max_id)

    def get_cell_ids_for_frame(self, frame_index: int) -> List[int]:
        """Get list of cell IDs in a specific frame"""
        masks = self.get_mask_for_frame(frame_index)
        if masks is not None and masks.size > 0:
            unique_ids = np.unique(masks)
            # Remove background (0)
            return [int(id_val) for id_val in unique_ids if id_val > 0]
        return []

    def clear_all_data(self) -> None:
        """Clear all stored data"""
        self.clear_frames()
        self.clear_all_masks()
        self.clear_cellsam_results()
        self._image_paths.clear()
        self._current_frame_index = 0
        self._first_frame_segmented = False
