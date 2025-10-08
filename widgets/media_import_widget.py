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
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton

class VideoTrimDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Trim Video")
        self.setMinimumWidth(300)

        layout = QVBoxLayout(self)

        # Start time
        start_layout = QHBoxLayout()
        start_layout.addWidget(QLabel("Start (sec):"))
        self.start_input = QLineEdit()
        self.start_input.setPlaceholderText("Leave empty for full video")
        start_layout.addWidget(self.start_input)
        layout.addLayout(start_layout)

        # End time
        end_layout = QHBoxLayout()
        end_layout.addWidget(QLabel("End (sec):"))
        self.end_input = QLineEdit()
        self.end_input.setPlaceholderText("Leave empty for full video")
        end_layout.addWidget(self.end_input)
        layout.addLayout(end_layout)

        # Buttons
        button_layout = QHBoxLayout()
        confirm_btn = QPushButton("Confirm")
        cancel_btn = QPushButton("Cancel")
        confirm_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(confirm_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

    def get_trim_range(self):
        start_text = self.start_input.text().strip()
        end_text = self.end_input.text().strip()

        try:
            start = float(start_text) if start_text else None
            end = float(end_text) if end_text else None
            return start, end
        except ValueError:
            return "invalid", "invalid"




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

    def extract_frames_from_video(self, video_path, frame_interval=1, start_sec=None, end_sec=None):
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
            
    def handle_video_file(self, file_path: str) -> List[str]:
        """Handle .mp4 video file with optional trimming"""
        while True:
            dialog = VideoTrimDialog(self)
            if dialog.exec():
                start_sec, end_sec = dialog.get_trim_range()

                if start_sec == "invalid" or end_sec == "invalid":
                    QMessageBox.warning(self, "Invalid Input", "Start and End must be valid numbers.")
                    continue

                cap = cv2.VideoCapture(file_path)
                if not cap.isOpened():
                    QMessageBox.warning(self, "Video Error", "Unable to open video.")
                    return []

                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration_sec = total_frames / fps if fps > 0 else 0

                # Correction out of bounds
                if start_sec is not None and start_sec < 0:
                    start_sec = 0.0
                if end_sec is not None and end_sec > duration_sec:
                    end_sec = duration_sec

                # Default logic
                if start_sec is not None and end_sec is None:
                    end_sec = duration_sec
                if end_sec is not None and start_sec is None:
                    start_sec = 0.0

                return self.extract_frames_from_video(
                    file_path,
                    frame_interval=1,
                    start_sec=start_sec,
                    end_sec=end_sec,
                )
            else:
                return []

    def process_files(self, file_paths):
        """Process image and video files"""
        valid_paths = []

        for file_path in file_paths:
            ext = Path(file_path).suffix.lower()

            if ext == ".mp4":
                video_frames = self.handle_video_file(file_path)
                if video_frames:
                    valid_paths.extend(video_frames)
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
