"""
Image import widget with drag and drop support
"""

from pathlib import Path
from typing import List

import cv2
from natsort import natsorted
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from widgets.dropzone_widget import DropZoneWidget
from widgets.video_trim_widget import VideoTrimWidget


class MediaImportWidget(QWidget):
    """Widget for importing image files"""

    # Signals
    frames_ready = pyqtSignal(list)  # list of frame paths
    status_update = pyqtSignal(str)  # status message
    auto_segment_toggled = pyqtSignal(bool)  # auto-segmentation enabled/disabled

    def __init__(self):
        super().__init__()

        # Setup UI
        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Stacked widget to switch between import and trim views
        self.stacked_widget = QStackedWidget()
        layout.addWidget(self.stacked_widget)

        # Page 0: Import page (dropzone)
        import_page = QWidget()
        import_layout = QVBoxLayout(import_page)
        import_layout.setContentsMargins(32, 32, 32, 32)
        import_layout.setSpacing(8)

        # Auto-segmentation toggle
        self.auto_segment_checkbox = QCheckBox("Auto-segment first frame with CellSAM")
        self.auto_segment_checkbox.setChecked(True)  # Default to enabled
        self.auto_segment_checkbox.toggled.connect(self.auto_segment_toggled.emit)
        import_layout.addWidget(self.auto_segment_checkbox)

        # Drop zone with upload button
        self.drop_zone = DropZoneWidget()
        self.drop_zone.files_dropped.connect(self.handle_dropped_files)
        self.drop_zone.upload_clicked.connect(self.handle_upload_button)
        import_layout.addWidget(self.drop_zone)

        self.stacked_widget.addWidget(import_page)

        # Page 1: Video trim page (will be populated when needed)
        self.video_trim_widget = None

    def handle_upload_button(self):
        """Handle upload button click - show file dialog"""
        # Show file dialog that accepts image files
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Image Files",
            "",
            "Media Files (*.png *.jpg *.jpeg *.tiff *.tif *.bmp *.gif *.mp4);;All Files (*)",
        )

        if file_paths:
            self.process_files(file_paths)

    def select_image_files(self):
        """Open file dialog to select multiple image files"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Image Files",
            "",
            "Image Files (*.png *.jpg *.jpeg *.tiff *.tif *.bmp *.gif);;All Files (*)",
        )

        if file_paths:
            self.process_files(file_paths)

    def extract_frames_from_video(
        self, video_path, frame_interval=1, start_sec=None, end_sec=None
    ):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = int(start_sec * fps) if start_sec is not None else 0
        end_frame = int(end_sec * fps) if end_sec is not None else total_frames

        temp_dir = Path("temp_frames")
        temp_dir.mkdir(exist_ok=True)
        frame_paths = []
        index = 0

        while True:
            ret, frame = cap.read()
            if not ret or index > end_frame:
                break

            if index >= start_frame and index % frame_interval == 0:
                frame_path = temp_dir / f"frame_{index:04d}.png"
                cv2.imwrite(str(frame_path), frame)
                frame_paths.append(str(frame_path))

            index += 1

        cap.release()
        return frame_paths

    def handle_dropped_files(self, file_paths: List[str]):
        """Handle files dropped on the drop zone"""
        if file_paths:
            self.process_files(file_paths)

    def handle_video_file(self, file_path: str):
        """Handle .mp4 video file with optional trimming - shows embedded trim UI"""
        # Store the video path for later processing
        self.pending_video_path = file_path

        # Create and show video trim widget
        if self.video_trim_widget is not None:
            self.video_trim_widget.cleanup()
            self.stacked_widget.removeWidget(self.video_trim_widget)
            self.video_trim_widget.deleteLater()

        self.video_trim_widget = VideoTrimWidget(file_path, self)
        self.video_trim_widget.trim_confirmed.connect(self.on_trim_confirmed)
        self.video_trim_widget.trim_cancelled.connect(self.on_trim_cancelled)

        # Add to stacked widget if not already there
        if self.stacked_widget.count() == 1:
            self.stacked_widget.addWidget(self.video_trim_widget)
        else:
            self.stacked_widget.removeWidget(self.stacked_widget.widget(1))
            self.stacked_widget.insertWidget(1, self.video_trim_widget)

        # Switch to trim view
        self.stacked_widget.setCurrentIndex(1)

    def on_trim_confirmed(self, start_sec: float, end_sec: float):
        """Called when user confirms trim selection"""
        # Extract frames with the selected range
        video_frames = self.extract_frames_from_video(
            self.pending_video_path,
            frame_interval=1,
            start_sec=start_sec,
            end_sec=end_sec,
        )

        # Switch back to import view
        self.stacked_widget.setCurrentIndex(0)

        # Process the frames
        if video_frames:
            sorted_paths = natsorted(video_frames)
            self.frames_ready.emit(sorted_paths)
            self.status_update.emit(f"Loaded {len(sorted_paths)} frames")
        else:
            QMessageBox.warning(
                self, "No Frames", "No frames were extracted from the video"
            )

    def on_trim_cancelled(self):
        """Called when user cancels trim"""
        # Switch back to import view
        self.stacked_widget.setCurrentIndex(0)
        self.status_update.emit("Video import cancelled")

    def process_files(self, file_paths):
        """Process image and video files"""
        valid_paths = []

        for file_path in file_paths:
            ext = Path(file_path).suffix.lower()

            if ext == ".mp4":
                # Show trim UI - processing happens after user confirms
                self.handle_video_file(file_path)
                return  # Exit early, will continue after trim confirmation
            else:
                image = cv2.imread(file_path)
                if image is not None:
                    valid_paths.append(file_path)
                else:
                    QMessageBox.warning(
                        self,
                        "Invalid Image",
                        f"Could not read image: {Path(file_path).name}",
                    )

        if valid_paths:
            sorted_paths = natsorted(valid_paths)
            self.frames_ready.emit(sorted_paths)
            self.status_update.emit(f"Loaded {len(sorted_paths)} frames")
        else:
            QMessageBox.warning(
                self, "No Valid Media", "No valid image or video frames found"
            )

    def is_auto_segment_enabled(self) -> bool:
        """Get current state of auto-segmentation toggle"""
        return self.auto_segment_checkbox.isChecked()

    def reset_state(self):
        """Reset internal state after restart"""
        # Clear any cached paths or data
        self.status_update.emit("Ready - Import video or images to begin")

        # Optionally clear temp_frames folder
        temp_dir = Path("temp_frames")
        if temp_dir.exists():
            for file in temp_dir.glob("*.png"):
                file.unlink()
