"""
Video and image import widget with drag and drop support
"""

import os
from pathlib import Path
from typing import List

import cv2
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QFont, QPalette
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
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


class EnhancedDropZoneWidget(QWidget):
    """Enhanced drop zone with integrated upload button"""

    files_dropped = pyqtSignal(list)  # list of file paths
    upload_clicked = pyqtSignal()  # upload button clicked

    def __init__(self):
        super().__init__()

        self.setAcceptDrops(True)
        self.setMinimumHeight(300)

        # Create a frame for the border
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        from PyQt6.QtWidgets import QFrame

        self.border_frame = QFrame()
        self.border_frame.setFrameStyle(QFrame.Shape.Box)
        self.border_frame.setLineWidth(4)

        # Setup UI inside the frame
        layout = QVBoxLayout(self.border_frame)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Drop zone icon and text
        self.drop_label = QLabel(
            "Drop video or image files here\n\nSupported formats: MP4, AVI, MOV, PNG, JPG, TIFF"
        )
        self.drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_label.setWordWrap(True)

        # Style the drop zone label
        font = QFont()
        self.drop_label.setFont(font)

        layout.addWidget(self.drop_label)

        # Upload button
        self.upload_button = QPushButton("Browse Files")
        self.upload_button.clicked.connect(self.upload_clicked.emit)
        self.upload_button.setStyleSheet(
            """
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 12px 32px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
        """
        )
        layout.addWidget(self.upload_button)

        main_layout.addWidget(self.border_frame)

        # Set initial style with dotted border
        self.border_frame.setStyleSheet(
            """
            QFrame {
                border: 4px dashed #008080;
                border-radius: 12px;
                background-color: rgba(64, 64, 64, 0.8);
                margin: 10px;
            }
            QLabel {
                color: #b0b0b0;
                border: none;
                font-size: 14px;
                background-color: transparent;
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
                self.border_frame.setStyleSheet(
                    """
                    QFrame {
                        border: 4px dashed #0078d4;
                        border-radius: 12px;
                        background-color: rgba(69, 69, 69, 0.9);
                        margin: 10px;
                    }
                    QLabel {
                        color: #ffffff;
                        border: none;
                        font-size: 14px;
                        background-color: transparent;
                    }
                """
                )
            else:
                event.ignore()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        """Handle drag leave event"""
        self.border_frame.setStyleSheet(
            """
            QFrame {
                border: 4px dashed #008080;
                border-radius: 12px;
                background-color: rgba(64, 64, 64, 0.8);
                margin: 10px;
            }
            QLabel {
                color: #b0b0b0;
                border: none;
                font-size: 14px;
                background-color: transparent;
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


class DropZoneWidget(QWidget):
    """Drag and drop zone for files"""

    files_dropped = pyqtSignal(list)  # list of file paths

    def __init__(self, accept_text: str = "Drop files here"):
        super().__init__()

        self.setAcceptDrops(True)
        self.setMinimumHeight(300)

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
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(8)

        # Drop zone with upload button - make it take 80% of the space
        self.drop_zone = EnhancedDropZoneWidget()
        self.drop_zone.files_dropped.connect(self.handle_dropped_files)
        self.drop_zone.upload_clicked.connect(self.handle_upload_button)
        layout.addWidget(self.drop_zone, 8)

        # Progress area (initially hidden) - give it minimal space when visible
        self.setup_progress_area(layout)

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
        parent_layout.addWidget(
            self.progress_group, 2
        )  # Give progress area 20% when visible

    def handle_upload_button(self):
        """Handle upload button click - show file dialog"""
        # Show file dialog that accepts both video and image files
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video or Image Files",
            "",
            "Media Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.png *.jpg *.jpeg *.tiff *.tif *.bmp *.gif);;Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;Image Files (*.png *.jpg *.jpeg *.tiff *.tif *.bmp *.gif);;All Files (*)",
        )

        if file_paths:
            self.process_files(file_paths)

    def select_video_file(self):
        """Open file dialog to select video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;All Files (*)",
        )

        if file_path:
            self.process_files([file_path])

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
        """Process the selected files directly"""
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

        QMessageBox.critical(self, "Extraction Error", error_message)
        self.status_update.emit("Video extraction failed")

    def cancel_processing(self):
        """Cancel current processing"""
        if self.video_extractor_worker is not None:
            self.video_extractor_worker.cancel()
            self.video_extractor_worker.wait()
            self.video_extractor_worker = None

        self.progress_group.setVisible(False)
        self.status_update.emit("Processing cancelled")

    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_frame_dir and os.path.exists(self.temp_frame_dir):
            import shutil

            try:
                shutil.rmtree(self.temp_frame_dir)
            except Exception:
                pass  # Ignore cleanup errors
