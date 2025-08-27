"""
Storage service for handling frame data and masks
"""

from typing import Dict, List, Optional

import cv2
import numpy as np

from utils.image_preprocessor import ImagePreprocessor


class StorageService:
    """Service class to handle all storage operations for frame-by-frame analysis"""

    def __init__(self) -> None:
        """Initialize storage"""
        self._frame_masks: List[Optional[np.ndarray]] = (
            []
        )  # List of masks indexed by frame
        self._current_frame_index: int = 0
        self._image_paths: List[str] = []  # Store original image paths
        self._processed_paths: List[str] = []  # Store processed image paths
        self._preprocessor = ImagePreprocessor(max_size=512)  # Max dimension 512px

    # Frame management
    def set_image_paths(self, paths: List[str]) -> None:
        """Set image paths and preprocess them for optimal performance"""
        # Check if we're setting the same paths - avoid reprocessing
        if hasattr(self, "_image_paths") and self._image_paths == paths:
            # Paths haven't changed, no need to reprocess
            return

        self._image_paths = paths.copy()
        self._current_frame_index = 0

        # Preprocess images for optimal performance
        try:
            self._processed_paths, scale_factors = (
                self._preprocessor.preprocess_image_list(paths)
            )
        except Exception as e:
            print(f"Warning: Image preprocessing failed: {e}")
            # Fallback to original paths
            self._processed_paths = paths.copy()

        # Resize masks list to match number of frames
        self._frame_masks = [None] * len(paths)

    def load_frame(self, index: int) -> Optional[np.ndarray]:
        """Load a frame from disk by index (uses processed/resized version)"""
        if not (0 <= index < len(self._processed_paths)):
            return None

        try:
            # Load from processed path for optimal performance
            image = cv2.imread(self._processed_paths[index])
            if image is not None:
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image_rgb
        except Exception as e:
            print(
                f"Failed to load processed image {self._processed_paths[index]}: {str(e)}"
            )

            # Fallback to original image
            try:
                image = cv2.imread(self._image_paths[index])
                if image is not None:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    return image_rgb
            except Exception as e2:
                print(
                    f"Failed to load original image {self._image_paths[index]}: {str(e2)}"
                )

        return None

    def load_original_frame(self, index: int) -> Optional[np.ndarray]:
        """Load a frame from the original (unprocessed) image"""
        if not (0 <= index < len(self._image_paths)):
            return None

        try:
            image = cv2.imread(self._image_paths[index])
            if image is not None:
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image_rgb
        except Exception as e:
            print(f"Failed to load original image {self._image_paths[index]}: {str(e)}")

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

        # Process the new image
        try:
            processed_path, _ = self._preprocessor.preprocess_image_from_path(path)
            self._processed_paths.append(processed_path)
        except Exception as e:
            print(f"Warning: Failed to preprocess {path}: {e}")
            self._processed_paths.append(path)  # Fallback to original

        # Add corresponding None mask entry
        self._frame_masks.append(None)

    def clear_frames(self) -> None:
        """Clear all frame paths and cleanup preprocessed data"""
        self._image_paths.clear()
        self._processed_paths.clear()
        self._frame_masks.clear()
        self._current_frame_index = 0

        # Cleanup temporary files
        self._preprocessor.cleanup_temp_directory()

    # Image paths management
    def get_image_paths(self) -> List[str]:
        """Get original image paths"""
        return self._image_paths.copy()

    def get_processed_paths(self) -> List[str]:
        """Get processed image paths (resized for optimal performance)"""
        return self._processed_paths.copy()

    def get_original_path(self, index: int) -> Optional[str]:
        """Get original path for a frame index"""
        if 0 <= index < len(self._image_paths):
            return self._image_paths[index]
        return None

    def get_processed_path(self, index: int) -> Optional[str]:
        """Get processed path for a frame index"""
        if 0 <= index < len(self._processed_paths):
            return self._processed_paths[index]
        return None

    def get_scale_factor(self) -> float:
        """Get the common scale factor used for image preprocessing"""
        return self._preprocessor.get_scale_factor()

    def scale_mask_to_original(self, mask: np.ndarray, frame_index: int) -> np.ndarray:
        """Scale a mask from processed size back to original image size"""
        original_path = self.get_original_path(frame_index)
        if original_path is None:
            return mask

        return self._preprocessor.scale_masks_to_original(mask, original_path)

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
        """Set masks for a specific frame (assumes masks are at processed resolution)"""
        if 0 <= frame_index < len(self._frame_masks):
            self._frame_masks[frame_index] = masks

    def get_mask_for_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """Get masks for a specific frame (at processed resolution)"""
        if 0 <= frame_index < len(self._frame_masks):
            return self._frame_masks[frame_index]
        return None

    def get_mask_for_frame_original_size(
        self, frame_index: int
    ) -> Optional[np.ndarray]:
        """Get masks for a specific frame scaled to original image resolution"""
        masks = self.get_mask_for_frame(frame_index)
        if masks is None:
            return None

        original_path = self.get_original_path(frame_index)
        if original_path is None:
            return masks

        # Scale masks back to original resolution
        return self._preprocessor.scale_masks_to_original(masks, original_path)

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
        """Clear all stored data and cleanup temporary files"""
        self.clear_frames()
        self._current_frame_index = 0
