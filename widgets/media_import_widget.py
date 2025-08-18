"""
Video and image import widget with drag and drop support
"""

import os
from pathlib import Path
from typing import List, Optional

import cv2
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QFont, QPalette
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class VideoExtractorWorker(QThread):
    """Worker thread for extracting frames from video"""

    progress_update = pyqtSignal(int, str)  # progress, status
    frame_extracted = pyqtSignal(str)  # frame path
    extraction_complete = pyqtSignal(list)  # list of frame paths
    error_occurred = pyqtSignal(str)  # error message

    def __init__(self, video_path: str, output_dir: str, frame_step: int = 1):
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir
        self.frame_step = frame_step
        self._cancelled = False

    def cancel(self):
        """Cancel the extraction"""
        self._cancelled = True

    def run(self):
        """Extract frames from video"""
        try:
            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)

            # Open video
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error_occurred.emit(f"Could not open video: {self.video_path}")
                return

            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            self.progress_update.emit(
                0, f"Extracting frames from video (FPS: {fps:.1f})..."
            )

            frame_paths = []
            frame_count = 0
            extracted_count = 0

            while True:
                if self._cancelled:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                # Extract every nth frame based on frame_step
                if frame_count % self.frame_step == 0:
                    # Save frame
                    frame_filename = f"frame_{extracted_count:06d}.png"
                    frame_path = os.path.join(self.output_dir, frame_filename)

                    if cv2.imwrite(frame_path, frame):
                        frame_paths.append(frame_path)
                        self.frame_extracted.emit(frame_path)
                        extracted_count += 1

                frame_count += 1

                # Update progress
                progress = int((frame_count / total_frames) * 100)
                status = (
                    f"Extracted {extracted_count} frames ({frame_count}/{total_frames})"
                )
                self.progress_update.emit(progress, status)

            cap.release()

            if not self._cancelled:
                self.extraction_complete.emit(frame_paths)

        except Exception as e:
            self.error_occurred.emit(f"Frame extraction failed: {str(e)}")


class DropZoneWidget(QWidget):
    """Drag and drop zone for files"""

    files_dropped = pyqtSignal(list)  # list of file paths

    def __init__(self, accept_text: str = "Drop files here"):
        super().__init__()

        self.setAcceptDrops(True)
        self.setMinimumHeight(200)

        # Setup UI
        layout = QVBoxLayout(self)

        # Drop zone label
        self.drop_label = QLabel(accept_text)
        self.drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_label.setWordWrap(True)

        # Style the drop zone
        font = QFont()
        font.setPointSize(12)
        self.drop_label.setFont(font)

        layout.addWidget(self.drop_label)

        # Set initial style
        self.setStyleSheet(
            """
            DropZoneWidget {
                border: 3px dashed #606060;
                border-radius: 8px;
                background-color: #404040;
            }
            QLabel {
                color: #b0b0b0;
                background: transparent;
                border: none;
            }
        """
        )

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event"""
        if event.mimeData().hasUrls():
            # Check if any dropped files are valid
            valid_files = []
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    if self._is_valid_file(file_path):
                        valid_files.append(file_path)

            if valid_files:
                event.acceptProposedAction()
                self.setStyleSheet(
                    """
                    DropZoneWidget {
                        border: 3px dashed #0078d4;
                        border-radius: 8px;
                        background-color: #454545;
                    }
                    QLabel {
                        color: #ffffff;
                        background: transparent;
                        border: none;
                    }
                """
                )
            else:
                event.ignore()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        """Handle drag leave event"""
        self.setStyleSheet(
            """
            DropZoneWidget {
                border: 3px dashed #606060;
                border-radius: 8px;
                background-color: #404040;
            }
            QLabel {
                color: #b0b0b0;
                background: transparent;
                border: none;
            }
        """
        )

    def dropEvent(self, event: QDropEvent):
        """Handle drop event"""
        file_paths = []

        for url in event.mimeData().urls():
            if url.isLocalFile():
                file_path = url.toLocalFile()
                if self._is_valid_file(file_path):
                    file_paths.append(file_path)

        if file_paths:
            self.files_dropped.emit(file_paths)
            event.acceptProposedAction()

        # Reset style
        self.dragLeaveEvent(event)

    def _is_valid_file(self, file_path: str) -> bool:
        """Check if file is valid image or video"""
        ext = Path(file_path).suffix.lower()

        # Image extensions
        image_exts = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif"}
        # Video extensions
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

        return ext in image_exts or ext in video_exts


class MediaImportWidget(QWidget):
    """Widget for importing video or image files"""

    # Signals
    frames_ready = pyqtSignal(list)  # list of frame paths
    status_update = pyqtSignal(str)  # status message
    progress_update = pyqtSignal(int, str)  # progress, message

    def __init__(self):
        super().__init__()

        # State
        self.video_extractor_worker = None
        self.temp_frame_dir = None

        # Setup UI
        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Title
        title_label = QLabel("Import Video or Images")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #ffffff;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # Instructions
        instructions = QLabel(
            "Drag and drop a video file or multiple image files, or use the buttons below to browse."
        )
        instructions.setWordWrap(True)
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instructions.setStyleSheet("color: #b0b0b0; margin: 10px;")
        layout.addWidget(instructions)

        # Drop zone
        self.drop_zone = DropZoneWidget(
            "Drop video file (.mp4, .avi, .mov, etc.) or\n"
            "multiple image files (.png, .jpg, .tiff, etc.)"
        )
        self.drop_zone.files_dropped.connect(self.handle_dropped_files)
        layout.addWidget(self.drop_zone)

        # Manual selection buttons
        button_layout = QHBoxLayout()

        self.select_video_button = QPushButton("Select Video File")
        self.select_video_button.clicked.connect(self.select_video_file)
        button_layout.addWidget(self.select_video_button)

        self.select_images_button = QPushButton("Select Image Files")
        self.select_images_button.clicked.connect(self.select_image_files)
        button_layout.addWidget(self.select_images_button)

        layout.addLayout(button_layout)

        # File list
        self.setup_file_list(layout)

        # Progress area (initially hidden)
        self.setup_progress_area(layout)

        # Action buttons
        self.setup_action_buttons(layout)

    def setup_file_list(self, parent_layout):
        """Setup file list display"""
        file_group = QGroupBox("Selected Files")
        layout = QVBoxLayout(file_group)

        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(150)
        layout.addWidget(self.file_list)

        # Clear button
        clear_layout = QHBoxLayout()
        clear_layout.addStretch()

        self.clear_button = QPushButton("Clear All")
        self.clear_button.clicked.connect(self.clear_files)
        self.clear_button.setEnabled(False)
        clear_layout.addWidget(self.clear_button)

        layout.addLayout(clear_layout)

        parent_layout.addWidget(file_group)

    def setup_progress_area(self, parent_layout):
        """Setup progress display area"""
        self.progress_group = QGroupBox("Processing")
        layout = QVBoxLayout(self.progress_group)

        from PyQt6.QtWidgets import QProgressBar

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("")
        layout.addWidget(self.progress_label)

        # Cancel button
        cancel_layout = QHBoxLayout()
        cancel_layout.addStretch()

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_processing)
        cancel_layout.addWidget(self.cancel_button)

        layout.addLayout(cancel_layout)

        self.progress_group.setVisible(False)
        parent_layout.addWidget(self.progress_group)

    def setup_action_buttons(self, parent_layout):
        """Setup action buttons"""
        action_layout = QHBoxLayout()
        action_layout.addStretch()

        self.process_button = QPushButton("Process and Continue")
        self.process_button.clicked.connect(self.process_files)
        self.process_button.setEnabled(False)
        self.process_button.setStyleSheet(
            """
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:disabled {
                background-color: #404040;
                color: #808080;
            }
        """
        )
        action_layout.addWidget(self.process_button)

        parent_layout.addLayout(action_layout)

    def select_video_file(self):
        """Open file dialog to select video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;All Files (*)",
        )

        if file_path:
            self.clear_files()
            self.add_files([file_path])

    def select_image_files(self):
        """Open file dialog to select multiple image files"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Image Files",
            "",
            "Image Files (*.png *.jpg *.jpeg *.tiff *.tif *.bmp *.gif);;All Files (*)",
        )

        if file_paths:
            self.clear_files()
            self.add_files(file_paths)

    def handle_dropped_files(self, file_paths: List[str]):
        """Handle files dropped on the drop zone"""
        if file_paths:
            self.clear_files()
            self.add_files(file_paths)

    def add_files(self, file_paths: List[str]):
        """Add files to the list"""
        for file_path in file_paths:
            item = QListWidgetItem(Path(file_path).name)
            item.setData(Qt.ItemDataRole.UserRole, file_path)
            item.setToolTip(file_path)
            self.file_list.addItem(item)

        self.clear_button.setEnabled(self.file_list.count() > 0)
        self.process_button.setEnabled(self.file_list.count() > 0)

        self.status_update.emit(f"Added {len(file_paths)} file(s)")

    def clear_files(self):
        """Clear all files from the list"""
        self.file_list.clear()
        self.clear_button.setEnabled(False)
        self.process_button.setEnabled(False)
        self.status_update.emit("Cleared file list")

    def get_file_paths(self) -> List[str]:
        """Get all file paths from the list"""
        file_paths = []
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            file_path = item.data(Qt.ItemDataRole.UserRole)
            file_paths.append(file_path)
        return file_paths

    def process_files(self):
        """Process the selected files"""
        file_paths = self.get_file_paths()
        if not file_paths:
            return

        # Check if we have a single video file or multiple image files
        if len(file_paths) == 1:
            file_path = file_paths[0]
            ext = Path(file_path).suffix.lower()
            video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

            if ext in video_exts:
                self.extract_video_frames(file_path)
                return

        # Treat as image files
        self.process_image_files(file_paths)

    def extract_video_frames(self, video_path: str):
        """Extract frames from video file"""
        if self.video_extractor_worker is not None:
            return

        # Create temporary directory for frames
        import tempfile

        self.temp_frame_dir = tempfile.mkdtemp(prefix="cellseek_frames_")

        # Ask user for frame extraction settings
        frame_step, ok = self._get_frame_step()
        if not ok:
            return

        # Start extraction
        self.video_extractor_worker = VideoExtractorWorker(
            video_path, self.temp_frame_dir, frame_step
        )
        self.video_extractor_worker.progress_update.connect(self.on_extraction_progress)
        self.video_extractor_worker.extraction_complete.connect(
            self.on_extraction_complete
        )
        self.video_extractor_worker.error_occurred.connect(self.on_extraction_error)
        self.video_extractor_worker.start()

        # Show progress
        self.progress_group.setVisible(True)
        self.process_button.setEnabled(False)

    def _get_frame_step(self) -> tuple:
        """Get frame extraction step from user"""
        from PyQt6.QtWidgets import QInputDialog

        step, ok = QInputDialog.getInt(
            self,
            "Frame Extraction Settings",
            "Extract every Nth frame (1 = all frames, 2 = every other frame, etc.):",
            value=1,
            min=1,
            max=100,
        )

        return step, ok

    def process_image_files(self, file_paths: List[str]):
        """Process image files directly"""
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
            from natsort import natsorted

            sorted_paths = natsorted(valid_paths)
            self.frames_ready.emit(sorted_paths)
            self.status_update.emit(f"Loaded {len(sorted_paths)} image files")
        else:
            QMessageBox.warning(self, "No Valid Images", "No valid image files found")

    def on_extraction_progress(self, progress: int, status: str):
        """Handle video extraction progress"""
        self.progress_bar.setValue(progress)
        self.progress_label.setText(status)
        self.progress_update.emit(progress, status)

    def on_extraction_complete(self, frame_paths: List[str]):
        """Handle video extraction completion"""
        self.video_extractor_worker = None
        self.progress_group.setVisible(False)
        self.process_button.setEnabled(True)

        if frame_paths:
            self.frames_ready.emit(frame_paths)
            self.status_update.emit(f"Extracted {len(frame_paths)} frames from video")
        else:
            QMessageBox.warning(
                self, "Extraction Failed", "No frames were extracted from the video"
            )

    def on_extraction_error(self, error_message: str):
        """Handle video extraction error"""
        self.video_extractor_worker = None
        self.progress_group.setVisible(False)
        self.process_button.setEnabled(True)

        QMessageBox.critical(self, "Extraction Error", error_message)
        self.status_update.emit("Video extraction failed")

    def cancel_processing(self):
        """Cancel current processing"""
        if self.video_extractor_worker is not None:
            self.video_extractor_worker.cancel()
            self.video_extractor_worker.wait()
            self.video_extractor_worker = None

        self.progress_group.setVisible(False)
        self.process_button.setEnabled(True)
        self.status_update.emit("Processing cancelled")

    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_frame_dir and os.path.exists(self.temp_frame_dir):
            import shutil

            try:
                shutil.rmtree(self.temp_frame_dir)
            except Exception:
                pass  # Ignore cleanup errors
