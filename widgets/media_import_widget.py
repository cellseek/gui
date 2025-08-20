"""
Image import widget with drag and drop support
"""

from pathlib import Path
from typing import List

import cv2
from natsort import natsorted
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QVBoxLayout, QWidget

from widgets.dropzone_widget import DropZoneWidget


class MediaImportWidget(QWidget):
    """Widget for importing image files"""

    # Signals
    frames_ready = pyqtSignal(list)  # list of frame paths
    status_update = pyqtSignal(str)  # status message

    def __init__(self):
        super().__init__()

        # Setup UI
        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(8)

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
            "Image Files (*.png *.jpg *.jpeg *.tiff *.tif *.bmp *.gif);;All Files (*)",
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

    def handle_dropped_files(self, file_paths: List[str]):
        """Handle files dropped on the drop zone"""
        if file_paths:
            self.process_files(file_paths)

    def process_files(self, file_paths: List[str]):
        """Process image files"""

        # Validate all files are readable images
        valid_paths = []

        for file_path in file_paths:
            try:
                # Try to read the image to verify it's valid
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
            # Sort paths naturally (frame_001.png, frame_002.png, etc.)
            sorted_paths = natsorted(valid_paths)
            self.frames_ready.emit(sorted_paths)
            self.status_update.emit(f"Loaded {len(sorted_paths)} image files")
        else:
            QMessageBox.warning(self, "No Valid Images", "No valid image files found")
