"""
Storage service for handling frame data and masks
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.frame_preprocess import PreprocessSettings
from utils.image_preprocessor import ImagePreprocessor
from utils.mask_postprocessor import MaskPostprocessor, MaskPostprocessorConfig
from utils.video_frame_source import VideoFrameSource

# Sample one frame every N video frames (~1 Hz at 30 fps).
DEFAULT_VIDEO_FRAME_INTERVAL = 30


class StorageService:
    """Frame and mask storage with lazy loading for images and video."""

    def __init__(self) -> None:
        self._frame_masks: List[Optional[np.ndarray]] = []
        self._current_frame_index: int = 0
        self._preprocessor = ImagePreprocessor(max_size=512)
        self._mask_postprocessor = MaskPostprocessor()
        self._preprocess_settings = PreprocessSettings()
        self._frame_cache: Dict[int, np.ndarray] = {}

        self._media_kind: Optional[str] = None  # "images" | "video"
        self._image_paths: List[str] = []
        self._all_image_paths: List[str] = []
        self._video_source: Optional[VideoFrameSource] = None
        self._frame_count: int = 0
        self._source_frame_count: int = 0
        self._index_offset: int = 0
        self._original_size: Optional[Tuple[int, int]] = None  # (height, width)
        self._processed_size: Optional[Tuple[int, int]] = None

    def set_brightfield_preprocessing_enabled(self, enabled: bool) -> None:
        """Enable/disable brightfield preprocessing for all loaded frames."""
        self._preprocessor.set_brightfield_enabled(enabled)
        self.invalidate_frame_cache()

    def set_mask_postprocessing_enabled(self, enabled: bool) -> None:
        """Enable/disable morphological mask cleanup on stored masks."""
        self._mask_postprocessor.enabled = enabled

    @staticmethod
    def default_mask_postprocessor_config() -> MaskPostprocessorConfig:
        return MaskPostprocessorConfig()

    @property
    def is_video(self) -> bool:
        return self._media_kind == "video"

    def get_session_key(self) -> tuple:
        """Identity for the currently opened media (used to avoid redundant reloads)."""
        if self._media_kind == "video" and self._video_source is not None:
            return (
                "video",
                self._video_source.video_path,
                self._video_source.frame_interval,
            )
        if self._media_kind == "images":
            return ("images", tuple(self._image_paths))
        return ("none",)

    def open_images(self, paths: List[str]) -> None:
        """Register an image sequence; frames are loaded and resized on demand."""
        if not paths:
            raise ValueError("No image paths provided")

        first = cv2.imread(paths[0])
        if first is None:
            raise ValueError(f"Failed to load image: {paths[0]}")

        self.clear_frames()
        self._media_kind = "images"
        self._image_paths = list(paths)
        self._all_image_paths = list(paths)
        self._image_paths = self._all_image_paths
        self._source_frame_count = len(paths)
        self._index_offset = 0
        self._frame_count = len(paths)
        self._frame_masks = [None] * self._frame_count

        rgb = self._read_rgb_full(0)
        if rgb is None:
            raise ValueError(f"Failed to load image: {paths[0]}")
        self._finalize_open_session(rgb)

    def open_video(
        self, video_path: str, frame_interval: int = DEFAULT_VIDEO_FRAME_INTERVAL
    ) -> None:
        """Register a video for lazy frame decoding (no bulk export to disk)."""
        self.clear_frames()
        self._video_source = VideoFrameSource(video_path, frame_interval)
        self._media_kind = "video"
        self._image_paths = []
        self._all_image_paths = []
        self._source_frame_count = self._video_source.logical_frame_count
        self._index_offset = 0
        self._frame_count = self._source_frame_count
        self._frame_masks = [None] * self._frame_count
        frame0 = self._read_rgb_full(0)
        if frame0 is None:
            raise RuntimeError("Failed to read first video frame")
        self._finalize_open_session(frame0)

    def set_image_paths(self, paths: List[str]) -> None:
        """Backward-compatible entry point for image sequences."""
        if self.get_session_key() == ("images", tuple(paths)):
            return
        self.open_images(paths)

    def get_preprocess_settings(self) -> PreprocessSettings:
        return self._preprocess_settings.copy()

    def get_source_frame_count(self) -> int:
        """Total frames in the opened media before clip."""
        return self._source_frame_count

    def get_full_frame_count(self) -> int:
        return self._source_frame_count

    def set_preprocess_settings(self, settings: PreprocessSettings) -> None:
        """Apply preprocessing + clip; clears cache and refreshes geometry."""
        self._preprocess_settings = settings.copy()
        self._apply_clip_from_settings()
        self.invalidate_frame_cache()
        rgb0 = self._read_rgb_full(0)
        if rgb0 is not None:
            self._finalize_open_session(rgb0)

    def _apply_clip_from_settings(self) -> None:
        """Trim exposed frame range to clip_start … clip_end."""
        if self._source_frame_count <= 0:
            self._frame_count = 0
            return

        start = max(
            0, min(self._preprocess_settings.clip_start, self._source_frame_count - 1)
        )
        end = self._preprocess_settings.effective_clip_end(self._source_frame_count)
        if end < start:
            end = start

        if self._media_kind == "images" and self._all_image_paths:
            self._image_paths = self._all_image_paths[start : end + 1]
            self._index_offset = 0
        else:
            self._index_offset = start

        self._frame_count = end - start + 1
        self._frame_masks = [None] * self._frame_count
        self._current_frame_index = 0

    def get_source_frame_size(self) -> Optional[Tuple[int, int]]:
        """Full-resolution frame size before crop (height, width)."""
        raw = self._read_rgb_raw(0)
        if raw is None:
            return None
        return (raw.shape[0], raw.shape[1])

    def _finalize_open_session(self, full_rgb: np.ndarray) -> None:
        processed, scale = self._preprocessor.resize_image(full_rgb)
        self._apply_display_geometry(full_rgb, processed, scale)
        self._frame_cache[0] = processed

    def _apply_display_geometry(
        self, full_rgb: np.ndarray, processed_rgb: np.ndarray, scale: float
    ) -> None:
        self._original_size = (full_rgb.shape[0], full_rgb.shape[1])
        self._processed_size = (processed_rgb.shape[0], processed_rgb.shape[1])
        self._preprocessor._common_scale_factor = scale

    def load_frame(self, index: int) -> Optional[np.ndarray]:
        """Load a display-resolution RGB frame (cached)."""
        if not (0 <= index < self._frame_count):
            return None

        cached = self._frame_cache.get(index)
        if cached is not None:
            return cached

        rgb = self._read_rgb_full(index)
        if rgb is None:
            return None

        processed, _ = self._preprocessor.resize_image(rgb)
        self._frame_cache[index] = processed
        return processed

    def _read_rgb_raw(self, index: int) -> Optional[np.ndarray]:
        """Decode frame without user preprocessing (clip-relative index)."""
        if not (0 <= index < self._frame_count):
            return None
        actual = index + self._index_offset
        if self._media_kind == "video" and self._video_source is not None:
            return self._video_source.read_rgb(actual)

        if self._media_kind == "images" and 0 <= index < len(self._image_paths):
            image = cv2.imread(self._image_paths[index])
            if image is not None:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return None

    def _read_rgb_full(self, index: int) -> Optional[np.ndarray]:
        rgb = self._read_rgb_raw(index)
        if rgb is None:
            return None
        rgb = self._preprocess_settings.apply(rgb)
        return self._preprocessor.apply_brightfield(rgb)

    def load_original_frame(self, index: int) -> Optional[np.ndarray]:
        """Load full-resolution RGB frame with preprocessing applied."""
        return self._read_rgb_full(index)

    def load_original_frame_raw(self, index: int) -> Optional[np.ndarray]:
        """Load full-resolution RGB frame before preprocessing (clip-relative)."""
        return self._read_rgb_raw(index)

    def load_original_frame_raw_full(self, full_index: int) -> Optional[np.ndarray]:
        """Load raw frame by absolute index (for preprocess UI / auto-adjust)."""
        if full_index < 0 or full_index >= self._source_frame_count:
            return None
        if self._media_kind == "video" and self._video_source is not None:
            return self._video_source.read_rgb(full_index)
        if self._media_kind == "images" and 0 <= full_index < len(self._all_image_paths):
            image = cv2.imread(self._all_image_paths[full_index])
            if image is not None:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return None

    def get_frame_count(self) -> int:
        return self._frame_count

    def get_frame(self, index: int) -> Optional[np.ndarray]:
        return self.load_frame(index)

    def clear_frames(self) -> None:
        if self._video_source is not None:
            self._video_source.close()
            self._video_source = None

        self._media_kind = None
        self._image_paths.clear()
        self._all_image_paths.clear()
        self._frame_masks.clear()
        self._frame_cache.clear()
        self._frame_count = 0
        self._source_frame_count = 0
        self._index_offset = 0
        self._original_size = None
        self._processed_size = None
        self._current_frame_index = 0
        self._preprocess_settings = PreprocessSettings()
        self._preprocessor.cleanup_temp_directory()

    def get_image_paths(self) -> List[str]:
        return self._image_paths.copy()

    def get_processed_paths(self) -> List[str]:
        """Legacy API; lazy mode has no pre-generated processed files."""
        return []

    def get_original_path(self, index: int) -> Optional[str]:
        if self._media_kind == "images" and 0 <= index < len(self._image_paths):
            return self._image_paths[index]
        if self._media_kind == "video" and self._video_source is not None:
            return self._video_source.video_path
        return None

    def get_processed_path(self, index: int) -> Optional[str]:
        return None

    def get_scale_factor(self) -> float:
        return self._preprocessor.get_scale_factor()

    def scale_mask_to_original(self, mask: np.ndarray, frame_index: int) -> np.ndarray:
        if self._original_size is None or self._processed_size is None:
            return mask

        orig_h, orig_w = self._original_size
        proc_h, proc_w = self._processed_size
        if mask.shape[0] == orig_h and mask.shape[1] == orig_w:
            return mask
        if mask.shape[0] != proc_h or mask.shape[1] != proc_w:
            mask = self._preprocessor._resize_masks(mask, proc_w, proc_h)

        return self._preprocessor._resize_masks(mask, orig_w, orig_h)

    def scale_mask_to_processed(self, mask: np.ndarray, frame_index: int) -> np.ndarray:
        if self._processed_size is None:
            return mask
        proc_h, proc_w = self._processed_size
        return self._preprocessor._resize_masks(mask, proc_w, proc_h)

    def set_current_frame_index(self, index: int) -> None:
        max_index = self.get_frame_count() - 1
        if 0 <= index <= max_index:
            self._current_frame_index = index

    def get_current_frame_index(self) -> int:
        return self._current_frame_index

    def get_current_frame(self) -> Optional[np.ndarray]:
        return self.get_frame(self._current_frame_index)

    def has_previous_frame(self) -> bool:
        return self._current_frame_index > 0

    def has_next_frame(self) -> bool:
        return self._current_frame_index < self.get_frame_count() - 1

    def set_frame_masks(self, frame_masks: List[Optional[np.ndarray]]) -> None:
        self._frame_masks = frame_masks.copy()

    def get_frame_masks(self) -> List[Optional[np.ndarray]]:
        return self._frame_masks.copy()

    def set_mask_for_frame(self, frame_index: int, masks: np.ndarray) -> None:
        if 0 <= frame_index < len(self._frame_masks):
            cleaned = self._mask_postprocessor.process_label_mask(masks)
            self._frame_masks[frame_index] = cleaned

    def invalidate_frame_cache(self, frame_index: Optional[int] = None) -> None:
        if frame_index is None:
            self._frame_cache.clear()
        else:
            self._frame_cache.pop(frame_index, None)

    def get_mask_for_frame(self, frame_index: int) -> Optional[np.ndarray]:
        if 0 <= frame_index < len(self._frame_masks):
            return self._frame_masks[frame_index]
        return None

    def get_mask_for_frame_original_size(
        self, frame_index: int
    ) -> Optional[np.ndarray]:
        masks = self.get_mask_for_frame(frame_index)
        if masks is None:
            return None
        return self.scale_mask_to_original(masks, frame_index)

    def get_current_frame_masks(self) -> Optional[np.ndarray]:
        return self.get_mask_for_frame(self._current_frame_index)

    def has_mask_for_frame(self, frame_index: int) -> bool:
        if 0 <= frame_index < len(self._frame_masks):
            return self._frame_masks[frame_index] is not None
        return False

    def remove_mask_for_frame(self, frame_index: int) -> None:
        if 0 <= frame_index < len(self._frame_masks):
            self._frame_masks[frame_index] = None

    def remove_masks_after_frame(self, frame_index: int) -> int:
        count = 0
        for i in range(frame_index + 1, len(self._frame_masks)):
            if self._frame_masks[i] is not None:
                self._frame_masks[i] = None
                count += 1
        return count

    def clear_all_masks(self) -> None:
        self._frame_masks = [None] * len(self._frame_masks)

    def get_biggest_cell_id(self) -> int:
        max_id = 0
        for masks in self._frame_masks:
            if masks is not None and masks.size > 0:
                max_id = max(max_id, int(np.max(masks)))
        return max_id

    def get_cell_ids_for_frame(self, frame_index: int) -> List[int]:
        masks = self.get_mask_for_frame(frame_index)
        if masks is not None and masks.size > 0:
            return [int(v) for v in np.unique(masks) if v > 0]
        return []

    def clear_all_data(self) -> None:
        self.clear_frames()
