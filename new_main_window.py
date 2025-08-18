"""
New main window for frame-by-frame cell tracking workflow
"""

from pathlib import Path
from typing import List

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from widgets.frame_by_frame_widget import FrameByFrameWidget
from widgets.media_import_widget import MediaImportWidget


class NewMainWindow(QMainWindow):
    """New main window for frame-by-frame cell tracking workflow"""

    def __init__(self):
        super().__init__()

        # Setup UI
        self.setup_ui()
        self.setup_status_bar()
        self.setup_connections()

        # Set window properties
        self.setWindowTitle("CellSeek - Frame-by-Frame Cell Tracking")
        self.setMinimumSize(800, 600)

        # Set window size to 90% of screen size for better visibility
        screen = self.screen().availableGeometry()
        window_width = int(screen.width() * 0.9)
        window_height = int(screen.height() * 0.9)
        self.resize(window_width, window_height)

        # Center window on screen
        self.center_on_screen()

        # Start with import screen
        self.stacked_widget.setCurrentIndex(0)

    def setup_ui(self):
        """Setup the main user interface"""
        # Create central widget with stacked layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create stacked widget for different screens
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)

        # Screen 1: Media Import
        self.media_import_widget = MediaImportWidget()
        self.stacked_widget.addWidget(self.media_import_widget)

        # Screen 2: Frame-by-Frame Processing
        self.frame_by_frame_widget = FrameByFrameWidget()
        self.stacked_widget.addWidget(self.frame_by_frame_widget)

    def setup_status_bar(self):
        """Setup the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Status label
        self.status_label = QLabel("Ready - Import video or images to begin")
        self.status_bar.addWidget(self.status_label)

        # Memory usage label
        self.memory_label = QLabel("Memory: 0 MB")
        self.status_bar.addPermanentWidget(self.memory_label)

        # Timer for memory updates
        self.memory_timer = QTimer()
        self.memory_timer.timeout.connect(self.update_memory_usage)
        self.memory_timer.start(2000)  # Update every 2 seconds

    def setup_connections(self):
        """Setup signal connections between components"""
        # Media import connections
        self.media_import_widget.frames_ready.connect(self.on_frames_ready)
        self.media_import_widget.status_update.connect(self.on_status_update)
        self.media_import_widget.progress_update.connect(self.on_progress_update)

        # Frame-by-frame connections
        self.frame_by_frame_widget.status_update.connect(self.on_status_update)
        self.frame_by_frame_widget.progress_update.connect(self.on_progress_update)

    def center_on_screen(self):
        """Position the window at the center of the screen"""
        screen = self.screen().availableGeometry()
        window = self.frameGeometry()
        # Center both horizontally and vertically
        x = screen.center().x() - window.width() // 2
        y = screen.center().y() - window.height() // 2
        self.move(x, y)

    def on_frames_ready(self, frame_paths: List[str]):
        """Handle frames ready from media import"""
        try:
            # Load frames into frame-by-frame widget
            self.frame_by_frame_widget.load_frames(frame_paths)

            # Switch to frame-by-frame screen
            self.stacked_widget.setCurrentIndex(1)

            # Update status
            self.status_label.setText(
                f"Loaded {len(frame_paths)} frames - Ready for segmentation"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load frames: {str(e)}")

    def on_status_update(self, message: str):
        """Handle status updates"""
        self.status_label.setText(message)

    def on_progress_update(self, progress: int, message: str):
        """Handle progress updates"""
        # For now, just update status
        # Could add a progress bar to status bar if needed
        self.status_label.setText(f"{message} ({progress}%)")

    def update_memory_usage(self):
        """Update memory usage display"""
        try:
            import psutil

            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_label.setText(f"Memory: {memory_mb:.0f} MB")
        except ImportError:
            self.memory_label.setText("Memory: N/A")

    def go_back_to_import(self):
        """Go back to import screen (for future use)"""
        reply = QMessageBox.question(
            self,
            "Go Back",
            "Are you sure you want to go back to import? All current work will be lost.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Clean up current work
            self.frame_by_frame_widget = FrameByFrameWidget()
            self.stacked_widget.removeWidget(self.stacked_widget.widget(1))
            self.stacked_widget.addWidget(self.frame_by_frame_widget)

            # Reconnect signals
            self.frame_by_frame_widget.status_update.connect(self.on_status_update)
            self.frame_by_frame_widget.progress_update.connect(self.on_progress_update)

            # Switch to import screen
            self.stacked_widget.setCurrentIndex(0)
            self.status_label.setText("Ready - Import video or images to begin")

    def closeEvent(self, event):
        """Handle window close event"""
        # Clean up temporary files
        if hasattr(self.media_import_widget, "cleanup"):
            self.media_import_widget.cleanup()

        event.accept()

    def keyPressEvent(self, event):
        """Handle global key press events"""
        # Allow Escape key to go back from frame-by-frame to import
        if event.key() == Qt.Key.Key_Escape and self.stacked_widget.currentIndex() == 1:
            self.go_back_to_import()
        else:
            super().keyPressEvent(event)
