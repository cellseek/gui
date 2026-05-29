"""
Image import widget with drag and drop support
"""

from pathlib import Path
from typing import List, Optional

import cv2
from natsort import natsorted
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QVBoxLayout, QWidget

from services.storage_service import DEFAULT_VIDEO_FRAME_INTERVAL
from utils.media_formats import VIDEO_EXTENSIONS, media_files_dialog_filter
from widgets.dropzone_widget import DropZoneWidget
from workers.video_import_worker import VideoImportWorker


class MediaImportWidget(QWidget):
    """Widget for importing image files and videos (lazy video loading)."""

    frames_ready = pyqtSignal(list)
    video_ready = pyqtSignal(str, int, int)  # path, frame_interval, logical_frame_count
    status_update = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._video_import_worker: Optional[VideoImportWorker] = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(8)

        self.drop_zone = DropZoneWidget()
        self.drop_zone.files_dropped.connect(self.handle_dropped_files)
        self.drop_zone.upload_clicked.connect(self.handle_upload_button)
        layout.addWidget(self.drop_zone)

    def handle_upload_button(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Media Files",
            "",
            media_files_dialog_filter(),
        )
        if file_paths:
            self.process_files(file_paths)

    def handle_dropped_files(self, file_paths: List[str]):
        if file_paths:
            self.process_files(file_paths)

    def process_files(self, file_paths: List[str]):
        videos = []
        image_paths = []

        for file_path in file_paths:
            ext = Path(file_path).suffix.lower()
            if ext in VIDEO_EXTENSIONS:
                videos.append(file_path)
            else:
                image_paths.append(file_path)

        if len(videos) > 1:
            QMessageBox.warning(
                self,
                "Multiple Videos",
                "Please drop one video at a time.",
            )
            videos = videos[:1]

        if videos:
            self._start_video_import(videos[0])

        if image_paths:
            self._finish_image_import(image_paths)

        if not videos and not image_paths:
            QMessageBox.warning(
                self, "No Valid Media", "No valid image or video files found"
            )

    def _start_video_import(self, video_path: str):
        if self._video_import_worker is not None and self._video_import_worker.isRunning():
            return

        self.status_update.emit("Opening video...")
        self.drop_zone.setEnabled(False)

        self._video_import_worker = VideoImportWorker(
            video_path, DEFAULT_VIDEO_FRAME_INTERVAL
        )
        self._video_import_worker.finished.connect(self._on_video_probe_finished)
        self._video_import_worker.error_occurred.connect(self._on_video_probe_error)
        self._video_import_worker.start()

    def _on_video_probe_finished(self, video_path: str, frame_interval: int, frame_count: int):
        self.drop_zone.setEnabled(True)
        self.video_ready.emit(video_path, frame_interval, frame_count)
        self.status_update.emit(
            f"Video ready: {frame_count} frames (every {frame_interval} video frames)"
        )

    def _on_video_probe_error(self, message: str):
        self.drop_zone.setEnabled(True)
        QMessageBox.warning(
            self,
            "Video Error",
            f"Could not open video.\n{message}\n\n"
            "Try re-encoding to H.264 MP4 or MJPEG AVI.",
        )
        self.status_update.emit("Ready - Import video or images to begin")

    def _finish_image_import(self, file_paths: List[str]):
        valid_paths = []

        for file_path in file_paths:
            try:
                img = cv2.imread(file_path)
                if img is not None:
                    valid_paths.append(file_path)
                else:
                    QMessageBox.warning(
                        self,
                        "Invalid Image",
                        f"Could not read image: {Path(file_path).name}",
                    )
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Image Error",
                    f"Error reading {Path(file_path).name}: {str(e)}",
                )

        if valid_paths:
            sorted_paths = natsorted(valid_paths)
            self.frames_ready.emit(sorted_paths)
            self.status_update.emit(f"Loaded {len(sorted_paths)} images")

    def reset_state(self):
        self.status_update.emit("Ready - Import video or images to begin")
        self.drop_zone.setEnabled(True)
