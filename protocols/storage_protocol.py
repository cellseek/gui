"""
Protocol definition for storage interface used by mixins
"""

from typing import Dict, List, Optional, Protocol

import numpy as np


class StorageProtocol(Protocol):
    """Protocol defining the storage interface expected by mixins"""

    # Frame management
    def get_frame_count(self) -> int:
        """Get total number of frames"""
        ...

    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get current frame"""
        ...

    def get_frame(self, index: int) -> Optional[np.ndarray]:
        """Get frame by index"""
        ...

    def get_current_frame_index(self) -> int:
        """Get current frame index"""
        ...

    def set_current_frame_index(self, index: int) -> None:
        """Set current frame index"""
        ...

    def has_previous_frame(self) -> bool:
        """Check if there's a previous frame"""
        ...

    def has_next_frame(self) -> bool:
        """Check if there's a next frame"""
        ...

    # Mask management
    def get_current_frame_masks(self) -> Optional[np.ndarray]:
        """Get masks for current frame"""
        ...

    def get_mask_for_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """Get masks for a specific frame"""
        ...

    def set_mask_for_frame(self, frame_index: int, masks: np.ndarray) -> None:
        """Set masks for a specific frame"""
        ...

    def has_mask_for_frame(self, frame_index: int) -> bool:
        """Check if frame has masks"""
        ...

    # CellSAM results management
    def set_cellsam_result_for_frame(self, frame_index: int, result: dict) -> None:
        """Set CellSAM result for a specific frame"""
        ...

    def get_cellsam_result_for_frame(self, frame_index: int) -> Optional[dict]:
        """Get CellSAM result for a specific frame"""
        ...

    # Frame segmentation tracking
    def set_first_frame_segmented(self, segmented: bool) -> None:
        """Set whether first frame has been segmented"""
        ...

    def is_first_frame_segmented(self) -> bool:
        """Check if first frame has been segmented"""
        ...

    # Data management
    def clear_all_data(self) -> None:
        """Clear all stored data"""
        ...

    def set_frames(self, frames: List[np.ndarray]) -> None:
        """Set the list of frames"""
        ...

    def set_image_paths(self, paths: List[str]) -> None:
        """Set image paths"""
        ...

    def set_frame_masks(self, frame_masks: Dict[int, np.ndarray]) -> None:
        """Set all frame masks"""
        ...

    def set_cellsam_results(self, results: Dict[int, dict]) -> None:
        """Set CellSAM results"""
        ...
