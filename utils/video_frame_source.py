"""Lazy frame access for video files without extracting every frame to disk."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class VideoFrameSource:
    """Read sampled frames from a video on demand."""

    def __init__(self, video_path: str, frame_interval: int = 30):
        self.video_path = str(video_path)
        self.frame_interval = max(1, int(frame_interval))
        self._cap: Optional[cv2.VideoCapture] = None
        self.width, self.height = self._read_dimensions()
        self.logical_frame_count = self._count_logical_frames()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _ensure_cap(self) -> cv2.VideoCapture:
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.video_path)
            if not self._cap.isOpened():
                raise RuntimeError(f"Failed to open video: {self.video_path}")
        return self._cap

    def _read_dimensions(self) -> Tuple[int, int]:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid video dimensions: {self.video_path}")
        return width, height

    def _count_logical_frames(self) -> int:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return 0

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0:
            cap.release()
            return (total + self.frame_interval - 1) // self.frame_interval

        logical = 0
        index = 0
        while True:
            if not cap.grab():
                break
            if index % self.frame_interval == 0:
                logical += 1
            index += 1
        cap.release()
        return logical

    def video_frame_index(self, logical_index: int) -> int:
        return logical_index * self.frame_interval

    def read_bgr(self, logical_index: int) -> Optional[np.ndarray]:
        if logical_index < 0 or logical_index >= self.logical_frame_count:
            return None

        cap = self._ensure_cap()
        target = self.video_frame_index(logical_index)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            self._cap = None
            cap = self._ensure_cap()
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ret, frame = cap.read()
        if not ret or frame is None:
            return None
        return frame

    def read_rgb(self, logical_index: int) -> Optional[np.ndarray]:
        frame = self.read_bgr(logical_index)
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
