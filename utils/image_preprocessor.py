"""
Image preprocessing utilities for optimal processing performance
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


class ImagePreprocessor:
    """
    Handles image preprocessing including resizing and temporary file management
    """

    def __init__(self, max_size: int = 512):
        """
        Initialize image preprocessor

        Args:
            max_size: Maximum dimension (width or height) for processed images
        """
        self.max_size = max_size
        self.temp_dir: Optional[Path] = None
        self.original_to_processed_paths = {}
        self.scale_factors = {}
        self._common_scale_factor: Optional[float] = None

    def create_temp_directory(self) -> Path:
        """Create temporary directory for processed images"""
        if self.temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="cellseek_processed_"))
        return self.temp_dir

    def cleanup_temp_directory(self):
        """Clean up temporary directory and all processed images"""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
        self.original_to_processed_paths.clear()
        self.scale_factors.clear()
        self._common_scale_factor = None

    def calculate_resize_dimensions(
        self, original_height: int, original_width: int
    ) -> Tuple[int, int, float]:
        """
        Calculate optimal resize dimensions maintaining aspect ratio

        Args:
            original_height: Original image height
            original_width: Original image width

        Returns:
            Tuple of (new_height, new_width, scale_factor)
        """
        # Find the larger dimension
        max_dim = max(original_height, original_width)

        # If image is already small enough, don't resize
        if max_dim <= self.max_size:
            return original_height, original_width, 1.0

        # Calculate scale factor based on the larger dimension
        scale_factor = self.max_size / max_dim

        # Calculate new dimensions
        new_height = int(original_height * scale_factor)
        new_width = int(original_width * scale_factor)

        return new_height, new_width, scale_factor

    def resize_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Resize image if needed

        Args:
            image: Input image in RGB format

        Returns:
            Tuple of (resized_image, scale_factor)
        """
        height, width = image.shape[:2]
        new_height, new_width, scale_factor = self.calculate_resize_dimensions(
            height, width
        )

        if scale_factor == 1.0:
            return image, scale_factor

        # Resize using high-quality interpolation
        resized = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        return resized, scale_factor

    def preprocess_image_from_path(self, image_path: str) -> Tuple[str, float]:
        """
        Preprocess image from file path and save to temp directory

        Args:
            image_path: Path to original image

        Returns:
            Tuple of (processed_image_path, scale_factor)
        """
        # Check if already processed
        if image_path in self.original_to_processed_paths:
            return (
                self.original_to_processed_paths[image_path],
                self.scale_factors[image_path],
            )

        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize if needed
        resized_image, scale_factor = self.resize_image(image_rgb)

        # Create temp directory if needed
        temp_dir = self.create_temp_directory()

        # Generate processed image path
        original_path = Path(image_path)
        processed_filename = f"{original_path.stem}_processed{original_path.suffix}"
        processed_path = temp_dir / processed_filename

        # Save processed image (convert back to BGR for OpenCV)
        processed_bgr = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(processed_path), processed_bgr)

        # Store mapping
        self.original_to_processed_paths[image_path] = str(processed_path)
        self.scale_factors[image_path] = scale_factor

        return str(processed_path), scale_factor

    def preprocess_image_list(self, image_paths: List[str]) -> List[str]:
        """
        Preprocess a list of images with validation that all images have the same size

        Args:
            image_paths: List of original image paths

        Returns:
            List of processed image paths

        Raises:
            ValueError: If images have different sizes
        """
        if not image_paths:
            return []

        # Check if all images are already processed
        all_already_processed = all(
            path in self.original_to_processed_paths for path in image_paths
        )

        if all_already_processed:
            # All images already processed, return cached results
            processed_paths = [
                self.original_to_processed_paths[path] for path in image_paths
            ]
            return processed_paths

        # First pass: validate all images have the same size
        reference_size = None
        for i, path in enumerate(image_paths):
            image = cv2.imread(path)
            if image is None:
                raise ValueError(f"Failed to load image: {path}")

            height, width = image.shape[:2]
            current_size = (height, width)

            if reference_size is None:
                reference_size = current_size
                # Calculate scale factor once based on first image
                _, _, self._common_scale_factor = self.calculate_resize_dimensions(
                    height, width
                )
                print(
                    f"Calculated common scale factor: {self._common_scale_factor} for size {reference_size}"
                )
            elif current_size != reference_size:
                raise ValueError(
                    f"Image {i+1} ({path}) has size {current_size}, "
                    f"but expected {reference_size}. All images must have the same dimensions."
                )

        # Second pass: process all images with the common scale factor
        processed_paths = []

        for path in image_paths:
            processed_path, scale_factor = self.preprocess_image_from_path(path)
            processed_paths.append(processed_path)

        print(
            f"Successfully preprocessed {len(image_paths)} images with scale factor {self._common_scale_factor}"
        )
        return processed_paths

    def scale_masks_to_original(
        self, masks: np.ndarray, original_path: str
    ) -> np.ndarray:
        """
        Scale masks back to original image dimensions

        Args:
            masks: Masks at processed resolution
            original_path: Path to original image

        Returns:
            Masks scaled to original resolution
        """
        if original_path not in self.scale_factors:
            return masks

        scale_factor = self.scale_factors[original_path]
        if scale_factor == 1.0:
            return masks

        # Load original image to get dimensions
        original_image = cv2.imread(original_path)
        if original_image is None:
            return masks

        original_height, original_width = original_image.shape[:2]

        # Resize masks using nearest neighbor to preserve mask values
        scaled_masks = cv2.resize(
            masks.astype(np.uint8),
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        )

        return scaled_masks.astype(masks.dtype)

    def get_scale_factor(self, image_path: str = None) -> float:
        """Get scale factor for a given image path or the common scale factor"""
        if image_path is None:
            # Return common scale factor
            return (
                self._common_scale_factor
                if self._common_scale_factor is not None
                else 1.0
            )
        return self.scale_factors.get(image_path, 1.0)

    def get_processed_path(self, original_path: str) -> Optional[str]:
        """Get processed path for an original image path"""
        return self.original_to_processed_paths.get(original_path)

    def needs_processing(self, image_path: str) -> bool:
        """Check if image needs processing (is larger than max_size)"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return False

            height, width = image.shape[:2]
            max_dim = max(height, width)
            return max_dim > self.max_size
        except:
            return False
