"""
Frame manager widget for loading and managing image frames
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from natsort import natsorted
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class FrameLoadWorker(QThread):
    """Worker thread for loading frames"""

    progress = pyqtSignal(int, str)  # progress, status
    frame_loaded = pyqtSignal(int, dict)  # index, frame_data
    finished = pyqtSignal(int)  # total_frames
    error = pyqtSignal(str)  # error_message

    def __init__(self, file_paths: List[str]):
        super().__init__()
        self.file_paths = file_paths
        self._cancelled = False

    def cancel(self):
        """Cancel the loading process"""
        self._cancelled = True

    def run(self):
        """Load frames in background thread"""
        try:
            total_files = len(self.file_paths)
            loaded_frames = 0

            for i, file_path in enumerate(self.file_paths):
                if self._cancelled:
                    break

                self.progress.emit(
                    int((i / total_files) * 100), f"Loading {Path(file_path).name}"
                )

                try:
                    # Load image
                    image = cv2.imread(file_path)
                    if image is None:
                        self.error.emit(f"Failed to load image: {file_path}")
                        continue

                    # Convert BGR to RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Create frame data
                    frame_data = {
                        "path": file_path,
                        "name": Path(file_path).name,
                        "image": image_rgb,
                        "shape": image_rgb.shape,
                        "index": loaded_frames,
                    }

                    self.frame_loaded.emit(loaded_frames, frame_data)
                    loaded_frames += 1

                except Exception as e:
                    self.error.emit(f"Error loading {file_path}: {str(e)}")

            if not self._cancelled:
                self.finished.emit(loaded_frames)

        except Exception as e:
            self.error.emit(f"Frame loading failed: {str(e)}")


class FramePreviewWidget(QWidget):
    """Widget for displaying frame previews as list items"""

    def __init__(self, frame_index: int, frame_data: Dict[str, Any]):
        super().__init__()
        self.frame_index = frame_index
        self.frame_data = frame_data

        # Set up widget
        self.setFixedHeight(80)
        self.setStyleSheet(
            """
            FramePreviewWidget {
                border: 1px solid #606060;
                border-radius: 4px;
                background-color: #404040;
                margin: 2px;
            }
            FramePreviewWidget:hover {
                border-color: #0078d4;
                background-color: #454545;
            }
        """
        )

        # Create layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Create thumbnail label
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setFixedSize(64, 64)
        self.thumbnail_label.setStyleSheet(
            """
            QLabel {
                border: 1px solid #505050;
                border-radius: 3px;
                background-color: #353535;
            }
        """
        )
        self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail_label.setScaledContents(True)
        layout.addWidget(self.thumbnail_label)

        # Create info layout
        info_layout = QVBoxLayout()
        info_layout.setSpacing(4)

        # Filename label
        self.filename_label = QLabel(frame_data["name"])
        self.filename_label.setStyleSheet(
            """
            QLabel {
                color: #ffffff;
                font-weight: bold;
                font-size: 14px;
                border: none;
                background: transparent;
            }
        """
        )
        info_layout.addWidget(self.filename_label)

        # Details label
        h, w, c = frame_data["shape"]
        details_text = f"Size: {w} × {h} | Index: {frame_index}"
        self.details_label = QLabel(details_text)
        self.details_label.setStyleSheet(
            """
            QLabel {
                color: #b0b0b0;
                font-size: 12px;
                border: none;
                background: transparent;
            }
        """
        )
        info_layout.addWidget(self.details_label)

        info_layout.addStretch()
        layout.addLayout(info_layout)

        layout.addStretch()

        # Create thumbnail
        self.update_thumbnail()

        # Set tooltip
        self.setToolTip(f"{frame_data['name']}\nSize: {w} × {h}\nIndex: {frame_index}")

    def update_thumbnail(self):
        """Update the thumbnail image"""
        try:
            image = self.frame_data["image"]

            # Create thumbnail (square crop from center)
            h, w = image.shape[:2]
            size = 60  # Slightly smaller than label size for padding

            # Crop to square from center
            if w > h:
                start_x = (w - h) // 2
                square_image = image[:, start_x : start_x + h]
            else:
                start_y = (h - w) // 2
                square_image = image[start_y : start_y + w, :]

            # Resize to thumbnail size
            thumbnail = cv2.resize(square_image, (size, size))

            # Convert to QPixmap
            h, w, ch = thumbnail.shape
            bytes_per_line = ch * w
            qt_image = QPixmap.fromImage(
                QImage(
                    thumbnail.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
                )
            )

            self.thumbnail_label.setPixmap(qt_image)

        except Exception as e:
            self.thumbnail_label.setText("Error")
            self.thumbnail_label.setStyleSheet(
                """
                QLabel {
                    border: 1px solid #505050;
                    border-radius: 3px;
                    background-color: #353535;
                    color: #ff6b6b;
                    font-size: 10px;
                }
            """
            )

    def mousePressEvent(self, event):
        """Handle mouse press events - disabled"""
        # No click handling needed
        super().mousePressEvent(event)


class FrameListWidget(QScrollArea):
    """List widget for displaying frame thumbnails"""

    def __init__(self):
        super().__init__()
        self.frames: List[Dict[str, Any]] = []
        self.preview_widgets: List[FramePreviewWidget] = []

        # Set up scroll area
        self.setWidgetResizable(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Create content widget
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setSpacing(4)
        self.content_layout.setContentsMargins(8, 8, 8, 8)

        self.setWidget(self.content_widget)

        # Empty state label
        self.create_empty_label()

    def create_empty_label(self):
        """Create or recreate the empty state label"""
        self.empty_label = QLabel("Drop image frames here or click 'Load Frames'")
        self.empty_label.setWordWrap(True)
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet(
            """
            QLabel {
                color: #808080; 
                font-size: 12pt;
                padding: 40px;
                border: 2px dashed #606060;
                border-radius: 8px;
                background-color: #353535;
            }
        """
        )
        self.content_layout.addWidget(self.empty_label)

    def add_frame(self, frame_data: Dict[str, Any]):
        """Add a frame to the list"""
        frame_index = len(self.frames)
        self.frames.append(frame_data)

        # Remove empty label if this is the first frame
        if frame_index == 0:
            if hasattr(self, "empty_label") and self.empty_label is not None:
                self.empty_label.setVisible(False)
                # Remove empty label from layout
                self.content_layout.removeWidget(self.empty_label)
                self.empty_label.deleteLater()
                self.empty_label = None

        # Remove existing stretch if any
        stretch_item = None
        for i in range(self.content_layout.count()):
            item = self.content_layout.itemAt(i)
            if item.spacerItem():
                stretch_item = item
                break

        if stretch_item:
            self.content_layout.removeItem(stretch_item)

        # Create preview widget
        preview = FramePreviewWidget(frame_index, frame_data)
        self.preview_widgets.append(preview)

        # Add to layout
        self.content_layout.addWidget(preview)

        # Add stretch at the end to keep items at the top
        self.content_layout.addStretch()

    def clear_frames(self):
        """Clear all frames"""
        # Clear widgets
        for preview in self.preview_widgets:
            preview.deleteLater()

        self.frames.clear()
        self.preview_widgets.clear()

        # Clear layout
        while self.content_layout.count():
            child = self.content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Show empty label
        self.create_empty_label()

    def get_frame(self, index: int) -> Optional[Dict[str, Any]]:
        """Get frame data by index"""
        if 0 <= index < len(self.frames):
            return self.frames[index]
        return None

    def get_frame_count(self) -> int:
        """Get total frame count"""
        return len(self.frames)


class FrameManagerWidget(QWidget):
    """Main widget for managing image frames"""

    frames_loaded = pyqtSignal(int)  # frame_count
    clear_all_requested = pyqtSignal()  # Signal for clearing all data in all tabs

    def __init__(self):
        super().__init__()
        self.load_worker: Optional[FrameLoadWorker] = None

        self.setup_ui()
        self.setup_drag_drop()

    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header
        header_layout = QHBoxLayout()

        self.title_label = QLabel("Frames")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        header_layout.addWidget(self.title_label)

        header_layout.addStretch()

        self.frame_count_label = QLabel("0 frames")
        header_layout.addWidget(self.frame_count_label)

        layout.addLayout(header_layout)

        # Controls
        controls_layout = QHBoxLayout()

        self.load_button = QPushButton("Load Frames")
        self.load_button.clicked.connect(self.load_frames_dialog)
        controls_layout.addWidget(self.load_button)

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_all_data)
        self.clear_button.setEnabled(False)
        controls_layout.addWidget(self.clear_button)

        controls_layout.addStretch()

        layout.addLayout(controls_layout)

        # Progress bar (initially hidden)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.progress_label = QLabel()
        self.progress_label.setVisible(False)
        layout.addWidget(self.progress_label)

        # Frame list
        self.frame_list = FrameListWidget()
        layout.addWidget(self.frame_list)

    def setup_drag_drop(self):
        """Setup drag and drop functionality"""
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        """Handle drag enter events"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        """Handle drop events"""
        urls = event.mimeData().urls()
        file_paths = []

        for url in urls:
            file_path = url.toLocalFile()
            if self.is_image_file(file_path):
                file_paths.append(file_path)

        if file_paths:
            self.load_frames(file_paths)

    def is_image_file(self, file_path: str) -> bool:
        """Check if file is a supported image format"""
        supported_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
        return Path(file_path).suffix.lower() in supported_extensions

    def load_frames_dialog(self):
        """Show file dialog to load frames"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Load Image Frames",
            "",
            "Image Files (*.png *.jpg *.jpeg *.tiff *.tif *.bmp);;All Files (*)",
        )

        if file_paths:
            self.load_frames(file_paths)

    def load_frames(self, file_paths: List[str]):
        """Load frames from file paths"""
        if self.load_worker and self.load_worker.isRunning():
            self.load_worker.cancel()
            self.load_worker.wait()

        # Sort file paths naturally
        file_paths = natsorted(file_paths)

        # Clear existing frames
        self.clear_frames()

        # Show progress
        self.show_progress(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        # Start loading worker
        self.load_worker = FrameLoadWorker(file_paths)
        self.load_worker.progress.connect(self.on_load_progress)
        self.load_worker.frame_loaded.connect(self.on_frame_loaded)
        self.load_worker.finished.connect(self.on_load_finished)
        self.load_worker.error.connect(self.on_load_error)
        self.load_worker.start()

        # Disable controls
        self.load_button.setEnabled(False)
        self.clear_button.setEnabled(False)

    def clear_frames(self):
        """Clear all loaded frames"""
        self.frame_list.clear_frames()
        self.update_frame_count(0)
        self.clear_button.setEnabled(False)
        self.frames_loaded.emit(0)

    def clear_all_data(self):
        """Clear all data in all tabs and reset the application state"""
        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            "Clear All Data",
            "This will clear all data in all tabs (frames, segmentation, tracking, analysis, and export). Are you sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Emit signal to main window to reset all panels
            self.clear_all_requested.emit()

            # Also clear frames in this panel
            self.clear_frames()

    def show_progress(self, show: bool):
        """Show or hide progress indicators"""
        self.progress_bar.setVisible(show)
        self.progress_label.setVisible(show)

    def update_frame_count(self, count: int):
        """Update frame count display"""
        self.frame_count_label.setText(f"{count} frames")

    # Worker event handlers
    def on_load_progress(self, progress: int, status: str):
        """Handle load progress updates"""
        self.progress_bar.setValue(progress)
        self.progress_label.setText(status)

    def on_frame_loaded(self, index: int, frame_data: Dict[str, Any]):
        """Handle frame loaded event"""
        self.frame_list.add_frame(frame_data)
        self.update_frame_count(index + 1)

    def on_load_finished(self, total_frames: int):
        """Handle load finished event"""
        self.show_progress(False)
        self.load_button.setEnabled(True)
        self.clear_button.setEnabled(total_frames > 0)

        self.frames_loaded.emit(total_frames)

    def on_load_error(self, error_message: str):
        """Handle load error event"""
        QMessageBox.warning(self, "Loading Error", error_message)

    # Public interface
    def get_frame(self, index: int) -> Optional[Dict[str, Any]]:
        """Get frame data by index"""
        return self.frame_list.get_frame(index)

    def get_frame_count(self) -> int:
        """Get total frame count"""
        return self.frame_list.get_frame_count()

    def get_all_frames(self) -> List[Dict[str, Any]]:
        """Get all frame data"""
        frames = []
        for i in range(self.get_frame_count()):
            frame_data = self.get_frame(i)
            if frame_data:
                frames.append(frame_data)
        return frames

    def get_frame_data(self) -> Dict[str, Any]:
        """Get frame data for project saving"""
        return {
            "frames": [
                {
                    "path": frame["path"],
                    "name": frame["name"],
                    "shape": frame["shape"],
                    "index": frame["index"],
                }
                for frame in self.get_all_frames()
            ]
        }

    def set_frame_data(self, data: Dict[str, Any]):
        """Set frame data from project loading"""
        if "frames" in data:
            frame_paths = [
                frame["path"]
                for frame in data["frames"]
                if Path(frame["path"]).exists()
            ]
            if frame_paths:
                self.load_frames(frame_paths)
