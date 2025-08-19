"""
New main window for frame-by-frame cell tracking workflow
"""

import os

# Import CellSAM
import sys
import tempfile
from typing import List

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
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

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "cellsam"))
from cellsam.model import CellSAM


class CellSAMWorker(QThread):
    """Worker thread for running CellSAM processing on first frame only"""

    progress_update = pyqtSignal(int, str)  # progress, status
    processing_complete = pyqtSignal(
        list, dict
    )  # frame_paths, first_frame_segmentation
    error_occurred = pyqtSignal(str)  # error message

    def __init__(self, frame_paths: List[str]):
        super().__init__()
        self.frame_paths = frame_paths
        self._cancelled = False

    def cancel(self):
        """Cancel the processing"""
        self._cancelled = True

    def run(self):
        """Run CellSAM processing on first frame only"""
        try:
            self.progress_update.emit(0, "Initializing CellSAM model...")

            # Initialize CellSAM model
            model = CellSAM(gpu=True)

            self.progress_update.emit(30, "Model loaded. Processing first frame...")

            if self._cancelled or not self.frame_paths:
                return

            # Process only the first frame
            first_frame_path = self.frame_paths[0]

            # Load and process first frame
            img = cv2.imread(first_frame_path)
            if img is None:
                self.error_occurred.emit(
                    f"Could not load first frame: {first_frame_path}"
                )
                return

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            self.progress_update.emit(60, "Running segmentation on first frame...")

            # Run segmentation on first frame
            masks, flows, styles = model.segment(img, diameter=None)

            # Store first frame results
            first_frame_result = {
                "frame_path": first_frame_path,
                "masks": masks,
                "flows": flows,
                "styles": styles,
                "original_image": img,
            }

            self.progress_update.emit(90, "Cleaning up model...")

            # Clean up model to free memory
            del model
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.progress_update.emit(100, "First frame processing complete!")

            if not self._cancelled:
                self.processing_complete.emit(self.frame_paths, first_frame_result)

        except Exception as e:
            self.error_occurred.emit(f"CellSAM processing failed: {str(e)}")


class MainWindow(QMainWindow):
    """New main window for frame-by-frame cell tracking workflow"""

    def __init__(self):
        super().__init__()

        # State
        self.cellsam_worker = None

        # Setup UI
        self.setup_ui()
        self.setup_status_bar()
        self.setup_connections()

        # Set window properties
        self.setWindowTitle("CellSeek")
        self.setMinimumSize(800, 600)

        # Set screen size
        screen = self.screen().availableGeometry()
        window_width = int(screen.width() * 0.75)
        window_height = int(screen.height() * 0.75)
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
        """Handle frames ready from media import - run CellSAM processing"""
        try:
            # Start CellSAM processing
            self.cellsam_worker = CellSAMWorker(frame_paths)
            self.cellsam_worker.progress_update.connect(self.on_cellsam_progress)
            self.cellsam_worker.processing_complete.connect(self.on_cellsam_complete)
            self.cellsam_worker.error_occurred.connect(self.on_cellsam_error)

            self.cellsam_worker.start()

            self.status_label.setText("Initializing CellSAM segmentation...")

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to start CellSAM processing: {str(e)}"
            )

    def on_cellsam_progress(self, progress: int, status: str):
        """Handle CellSAM processing progress"""
        self.status_label.setText(f"{status} ({progress}%)")

    def on_cellsam_complete(self, frame_paths: List[str], first_frame_result: dict):
        """Handle CellSAM processing completion"""
        try:
            self.cellsam_worker = None

            # Load frames and first frame segmentation into frame-by-frame widget
            self.frame_by_frame_widget.load_frames_with_first_segmentation(
                frame_paths, first_frame_result
            )

            # Switch to frame-by-frame screen
            self.stacked_widget.setCurrentIndex(1)

            # Update status
            self.status_label.setText(
                f"First frame segmented - Loaded {len(frame_paths)} frames for tracking"
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to load processed frames: {str(e)}"
            )

    def on_cellsam_error(self, error_message: str):
        """Handle CellSAM processing error"""
        self.cellsam_worker = None

        QMessageBox.critical(self, "CellSAM Error", error_message)
        self.status_label.setText("CellSAM processing failed")

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
