"""
New main window for frame-by-frame cell tracking workflow
"""

from typing import List

import numpy as np
import psutil
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QApplication,
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
        self.media_import_widget.frame_widget = self.frame_by_frame_widget

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
        self.media_import_widget.video_ready.connect(self.on_video_ready)
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

    def _set_segmentation_wait_cursor(self, waiting: bool) -> None:
        if waiting:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()

    def on_frames_ready(self, frame_paths: List[str]):
        """Handle image sequence import (lazy per-frame loading)."""
        storage = self.frame_by_frame_widget.storage_service
        try:
            storage.open_images(frame_paths)
        except Exception as e:
            self.show_error("Import Error", str(e))
            return

        self._begin_tracking_session(auto_segment=self.media_import_widget.is_auto_segment_enabled())

    def on_video_ready(self, video_path: str, frame_interval: int, frame_count: int):
        """Handle video import after metadata probe (frames decoded on demand)."""
        storage = self.frame_by_frame_widget.storage_service
        try:
            storage.open_video(video_path, frame_interval)
        except Exception as e:
            self.show_error("Import Error", str(e))
            return

        self.status_label.setText(
            f"Video loaded: {frame_count} frames — decoding on demand"
        )
        self._begin_tracking_session(auto_segment=self.media_import_widget.is_auto_segment_enabled())

    def _begin_tracking_session(self, auto_segment: bool):
        if auto_segment:
            storage = self.frame_by_frame_widget.storage_service
            first_frame = storage.get_frame(0)
            if first_frame is None:
                self.show_error("Import Error", "Failed to load first frame")
                return

            self.status_label.setText("Segmenting first frame...")
            self._set_segmentation_wait_cursor(True)
            self.cellsam_service.segment_first_frame_async(image_rgb=first_frame)
        else:
            self._initialize_without_segmentation()

    def _initialize_without_segmentation(self):
        """Initialize frame-by-frame widget without auto-segmentation"""
        first_frame = self.frame_by_frame_widget.storage_service.get_frame(0)
        if first_frame is None:
            self.show_error("Import Error", "Failed to load first frame")
            return

        h, w = first_frame.shape[:2]
        empty_masks = np.zeros((h, w), dtype=np.uint16)
        self.frame_by_frame_widget.initialize(empty_masks)
        self.status_label.setText("Frames loaded without auto-segmentation")
        self.stacked_widget.setCurrentIndex(1)

    def _on_cellsam_complete(self, first_frame_masks: np.ndarray):
        """Handle successful CellSAM completion"""
        self._set_segmentation_wait_cursor(False)
        self.frame_by_frame_widget.initialize(first_frame_masks)
        self.status_label.setText("Frames loaded with CellSAM segmentation")
        self.stacked_widget.setCurrentIndex(1)

    def _on_cellsam_error(self, error_message: str):
        """Handle CellSAM error"""
        self._set_segmentation_wait_cursor(False)
        self.show_error(
            "CellSAM Error",
            f"Failed to segment first frame: {error_message}. "
            "You can manually annotate the first frame.",
        )
        first_frame = self.frame_by_frame_widget.storage_service.get_frame(0)
        if first_frame is not None:
            h, w = first_frame.shape[:2]
        else:
            h, w = 512, 512
        dummy_mask = np.zeros((h, w), dtype=np.uint16)
        self.frame_by_frame_widget.initialize(dummy_mask)
        self.stacked_widget.setCurrentIndex(1)

    def on_back_to_tracking(self):
        """Handle back to tracking navigation"""
        self.stacked_widget.setCurrentIndex(1)

    def on_restart_requested(self):
        """Handle restart navigation - go back to media import"""
        # Clear all state in frame by frame widget
        self.frame_by_frame_widget.clear_all_data()

        # Clear any pending frame paths
        if hasattr(self, "_pending_frame_paths"):
            delattr(self, "_pending_frame_paths")

        # Reset media import widget state
        self.media_import_widget.reset_state()

        # Switch back to import view
        self.stacked_widget.setCurrentIndex(0)
        self.status_label.setText("Ready - Import video or images to begin")

    def show_export_view(self):
        """Show the export view"""
        self.export_widget.initialize()
        self.stacked_widget.setCurrentIndex(2)
