"""
New main window for frame-by-frame cell tracking workflow
"""

from typing import List

import numpy as np
import psutil
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QLabel,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from services.cellsam_service import CellSamService
from widgets.frame_by_frame_widget import FrameByFrameWidget
from widgets.media_import_widget import MediaImportWidget


class MainWindow(QMainWindow):
    """New main window for frame-by-frame cell tracking workflow"""

    # ------------------------------------------------------------------------ #
    # ---------------------------- Initialization ---------------------------- #
    # ------------------------------------------------------------------------ #

    def __init__(self):
        super().__init__()

        # Setup services
        self.cellsam_service = CellSamService(self)

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

        # Screen 3: Export Processing
        from widgets.export_widget import ExportWidget

        self.export_widget = ExportWidget(self.frame_by_frame_widget.storage_service)
        self.stacked_widget.addWidget(self.export_widget)

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

        # Frame-by-frame connections
        self.frame_by_frame_widget.status_update.connect(self.on_status_update)
        self.frame_by_frame_widget.export_requested.connect(self.show_export_view)
        self.frame_by_frame_widget.restart_requested.connect(self.on_restart_requested)

        # Export widget connections
        self.export_widget.back_to_tracking.connect(self.on_back_to_tracking)

        # CellSAM async connections
        self.cellsam_service.segmentation_complete.connect(self._on_cellsam_complete)
        self.cellsam_service.segmentation_error.connect(self._on_cellsam_error)
        self.cellsam_service.status_update.connect(self.on_status_update)

    # ------------------------------------------------------------------------ #
    # ----------------------------- UI Management ---------------------------- #
    # ------------------------------------------------------------------------ #

    def center_on_screen(self):
        """Position the window at the center of the screen"""
        screen = self.screen().availableGeometry()
        window = self.frameGeometry()
        # Center both horizontally and vertically
        x = screen.center().x() - window.width() // 2
        y = screen.center().y() - window.height() // 2
        self.move(x, y)

    # CellSamServiceDelegate methods
    def emit_status_update(self, message: str) -> None:
        """Handle status updates"""
        self.status_label.setText(message)

    def show_error(self, title: str, message: str) -> None:
        """Show error message box"""
        QMessageBox.critical(self, title, message)

    def update_memory_usage(self):
        """Update memory usage display"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_label.setText(f"Memory: {memory_mb:.0f} MB")

    # ------------------------------------------------------------------------ #
    # ---------------------------- Event Handlers ---------------------------- #
    # ------------------------------------------------------------------------ #

    def on_status_update(self, message: str):
        """Handle status updates"""
        self.status_label.setText(message)

    def on_frames_ready(self, frame_paths: List[str]):
        """Handle frames ready from media import - run CellSAM processing"""

        # First, preprocess the frames by setting them in storage service
        # This will trigger the image preprocessing (resizing) internally
        self.frame_by_frame_widget.storage_service.set_image_paths(frame_paths)

        # Get the processed path for the first frame
        first_frame_processed_path = (
            self.frame_by_frame_widget.storage_service.get_processed_path(0)
        )
        if first_frame_processed_path is None:
            # Fallback to original path if preprocessing failed
            first_frame_processed_path = frame_paths[0]

        # Store frame paths for later use
        self._pending_frame_paths = frame_paths

        # Run CellSAM on the processed first frame asynchronously
        self.cellsam_service.segment_first_frame_async(first_frame_processed_path)

    def _on_cellsam_complete(self, first_frame_masks: np.ndarray):
        """Handle successful CellSAM completion"""
        if hasattr(self, "_pending_frame_paths"):
            # Initialize with CellSAM results
            # Note: masks are kept at processed size to match displayed images
            self.frame_by_frame_widget.initialize(
                self._pending_frame_paths, first_frame_masks
            )
            self.status_label.setText("Frames loaded with CellSAM segmentation")

            # Switch to frame-by-frame view
            self.stacked_widget.setCurrentIndex(1)

            # Initialize models
            self.frame_by_frame_widget.initialize_models()

            # Clean up
            delattr(self, "_pending_frame_paths")

    def _on_cellsam_error(self, error_message: str):
        """Handle CellSAM error"""
        if hasattr(self, "_pending_frame_paths"):
            # CellSAM failed, show error but still allow manual annotation
            self.show_error(
                "CellSAM Error",
                f"Failed to segment first frame: {error_message}. You can manually annotate the first frame.",
            )
            # Create a dummy mask for initialization
            dummy_mask = np.zeros(
                (512, 512), dtype=np.uint16
            )  # Will be resized by initialize
            self.frame_by_frame_widget.initialize(self._pending_frame_paths, dummy_mask)

            # Switch to frame-by-frame view
            self.stacked_widget.setCurrentIndex(1)

            # Initialize models
            self.frame_by_frame_widget.initialize_models()

            # Clean up
            delattr(self, "_pending_frame_paths")

    def on_back_to_tracking(self):
        """Handle back to tracking navigation"""
        self.stacked_widget.setCurrentIndex(1)

    def on_restart_requested(self):
        """Handle restart navigation - go back to media import"""
        self.stacked_widget.setCurrentIndex(0)
        self.status_label.setText("Ready - Import video or images to begin")

    def show_export_view(self):
        """Show the export view"""
        self.export_widget.initialize()
        self.stacked_widget.setCurrentIndex(2)
