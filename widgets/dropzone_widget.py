from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QFont
from PyQt6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


class DropZoneWidget(QWidget):
    """Drop zone with integrated upload button"""

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
