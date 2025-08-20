"""
Storage service for handling frame data and masks
"""

from typing import Dict, List, Optional

import cv2
import numpy as np


class StorageService:
    """Service class to handle all storage operations for frame-by-frame analysis"""

    def __init__(self) -> None:
        """Initialize storage"""
        self._frame_masks: List[Optional[np.ndarray]] = (
            []
        )  # List of masks indexed by frame
        self._current_frame_index: int = 0
        self._image_paths: List[str] = []  # Store original image paths

    # Frame management
    def set_image_paths(self, paths: List[str]) -> None:
        """Set image paths for lazy loading"""
        self._image_paths = paths.copy()
        self._current_frame_index = 0
        # Resize masks list to match number of frames
        self._frame_masks = [None] * len(paths)

    def load_frame(self, index: int) -> Optional[np.ndarray]:
        """Load a frame from disk by index"""
        if not (0 <= index < len(self._image_paths)):
            return None

        try:
            image = cv2.imread(self._image_paths[index])
            if image is not None:
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image_rgb
        except Exception as e:
            print(f"Failed to load image {self._image_paths[index]}: {str(e)}")

        return None

    def get_frame_count(self) -> int:
        """Get total number of frames"""
        return len(self._image_paths)

    def get_frame(self, index: int) -> Optional[np.ndarray]:
        """Get frame by index (always loads from disk)"""
        return self.load_frame(index)

    def add_frame_path(self, path: str) -> None:
        """Add a frame path to the collection"""
        self._image_paths.append(path)
        # Add corresponding None mask entry
        self._frame_masks.append(None)

    def clear_frames(self) -> None:
        """Clear all frame paths"""
        self._image_paths.clear()
        self._frame_masks.clear()
        self._current_frame_index = 0

    # Image paths management
    def get_image_paths(self) -> List[str]:
        """Get image paths"""
        return self._image_paths.copy()

    # Current frame index management
    def set_current_frame_index(self, index: int) -> None:
        """Set current frame index"""
        max_index = self.get_frame_count() - 1
        if 0 <= index <= max_index:
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
        return self._current_frame_index < self.get_frame_count() - 1

    # Mask management
    def set_frame_masks(self, frame_masks: List[Optional[np.ndarray]]) -> None:
        """Set all frame masks"""
        self._frame_masks = frame_masks.copy()

    def get_frame_masks(self) -> List[Optional[np.ndarray]]:
        """Get all frame masks"""
        return self._frame_masks.copy()

    def set_mask_for_frame(self, frame_index: int, masks: np.ndarray) -> None:
        """Set masks for a specific frame"""
        if 0 <= frame_index < len(self._frame_masks):
            self._frame_masks[frame_index] = masks

    def get_mask_for_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """Get masks for a specific frame"""
        if 0 <= frame_index < len(self._frame_masks):
            return self._frame_masks[frame_index]
        return None

    def get_current_frame_masks(self) -> Optional[np.ndarray]:
        """Get masks for current frame"""
        return self.get_mask_for_frame(self._current_frame_index)

    def has_mask_for_frame(self, frame_index: int) -> bool:
        """Check if frame has masks"""
        if 0 <= frame_index < len(self._frame_masks):
            return self._frame_masks[frame_index] is not None
        return False

    def remove_mask_for_frame(self, frame_index: int) -> None:
        """Remove masks for a specific frame"""
        if 0 <= frame_index < len(self._frame_masks):
            self._frame_masks[frame_index] = None

    def remove_masks_after_frame(self, frame_index: int) -> int:
        """
        Remove all masks after the specified frame index.

        Args:
            frame_index: The frame index after which to remove masks

        Returns:
            Number of masks removed
        """
        count = 0
        for i in range(frame_index + 1, len(self._frame_masks)):
            if self._frame_masks[i] is not None:
                self._frame_masks[i] = None
                count += 1
        return count

    def clear_all_masks(self) -> None:
        """Clear all masks"""
        self._frame_masks = [None] * len(self._frame_masks)

    # Utility methods
    def get_biggest_cell_id(self) -> int:
        """Get the biggest cell ID across all frames"""
        max_id = 0
        for masks in self._frame_masks:
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
        self._current_frame_index = 0
