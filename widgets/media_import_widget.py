"""
Image import widget with drag and drop support
"""

from pathlib import Path
from typing import List

import cv2
from natsort import natsorted
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QFileDialog, QMessageBox, QVBoxLayout, QWidget

from widgets.dropzone_widget import DropZoneWidget


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
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(8)

        # Auto-segmentation toggle
        self.auto_segment_checkbox = QCheckBox("Auto-segment first frame with CellSAM")
        self.auto_segment_checkbox.setChecked(True)  # Default to enabled
        self.auto_segment_checkbox.toggled.connect(self.auto_segment_toggled.emit)
        layout.addWidget(self.auto_segment_checkbox)

        # Drop zone with upload button
        self.drop_zone = DropZoneWidget()
        self.drop_zone.files_dropped.connect(self.handle_dropped_files)
        self.drop_zone.upload_clicked.connect(self.handle_upload_button)
        layout.addWidget(self.drop_zone)

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

    def extract_frames_from_video(self, video_path, frame_interval=50):
        """Extract every Nth frame from MP4 video and save as images"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        temp_dir = Path("temp_frames")
        temp_dir.mkdir(exist_ok=True)
        frame_paths = []
        index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if index % frame_interval == 0:
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

    def process_files(self, file_paths):
        """Process image and video files"""
        valid_paths = []

        for file_path in file_paths:
            ext = Path(file_path).suffix.lower()

            if ext == ".mp4":
                # Extract video frames
                video_frames = self.extract_frames_from_video(
                    file_path, frame_interval=1
                )
                if video_frames:
                    valid_paths.extend(video_frames)
                else:
                    QMessageBox.warning(
                        self,
                        "Video Error",
                        f"Failed to extract frames from: {Path(file_path).name}",
                    )
            else:
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
            # Emit frames ready with current auto-segment setting
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
