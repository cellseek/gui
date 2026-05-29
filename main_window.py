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
from utils.frame_preprocess import PreprocessSettings
from widgets.frame_by_frame_widget import FrameByFrameWidget
from widgets.media_import_widget import MediaImportWidget
from widgets.preprocess_widget import PreprocessWidget

# Stacked widget screen indices
IDX_IMPORT = 0
IDX_PREPROCESS = 1
IDX_TRACKING = 2
IDX_EXPORT = 3


class MainWindow(QMainWindow):
    """New main window for frame-by-frame cell tracking workflow"""

    # ------------------------------------------------------------------------ #
    # ---------------------------- Initialization ---------------------------- #
    # ------------------------------------------------------------------------ #

    def __init__(self):
        super().__init__()

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

        # Screen 0: Media Import
        self.media_import_widget = MediaImportWidget()
        self.stacked_widget.addWidget(self.media_import_widget)

        # Screen 1: Preprocess (crop / tone) before tracking
        self.preprocess_widget = PreprocessWidget()
        self.stacked_widget.addWidget(self.preprocess_widget)

        # Screen 2: Frame-by-Frame Processing
        self.frame_by_frame_widget = FrameByFrameWidget()
        self.stacked_widget.addWidget(self.frame_by_frame_widget)
        self.media_import_widget.frame_widget = self.frame_by_frame_widget

        # Screen 3: Export
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

        self.preprocess_widget.preprocess_confirmed.connect(
            self.on_preprocess_confirmed
        )
        self.preprocess_widget.status_update.connect(self.on_status_update)

        # Frame-by-frame connections
        self.frame_by_frame_widget.status_update.connect(self.on_status_update)
        self.frame_by_frame_widget.export_requested.connect(self.show_export_view)
        self.frame_by_frame_widget.restart_requested.connect(self.on_restart_requested)

        # Export widget connections
        self.export_widget.back_to_tracking.connect(self.on_back_to_tracking)

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
        """Handle image sequence import (lazy per-frame loading)."""
        storage = self.frame_by_frame_widget.storage_service
        try:
            storage.open_images(frame_paths)
        except Exception as e:
            self.show_error("Import Error", str(e))
            return

        self._show_preprocess_screen()

    def on_video_ready(self, video_path: str, frame_interval: int, frame_count: int):
        """Handle video import after metadata probe (frames decoded on demand)."""
        storage = self.frame_by_frame_widget.storage_service
        try:
            storage.open_video(video_path, frame_interval)
        except Exception as e:
            self.show_error("Import Error", str(e))
            return

        self.status_label.setText(
            f"Video loaded: {frame_count} frames — scrub to preview & preprocess"
        )
        self._show_preprocess_screen()

    def _show_preprocess_screen(self) -> None:
        storage = self.frame_by_frame_widget.storage_service
        self.preprocess_widget.load_storage(storage)
        self.stacked_widget.setCurrentIndex(IDX_PREPROCESS)

    def on_preprocess_confirmed(self, settings: PreprocessSettings) -> None:
        storage = self.frame_by_frame_widget.storage_service
        storage.set_preprocess_settings(settings)
        self._begin_tracking_session()

    def _begin_tracking_session(self) -> None:
        """Open tracking and auto-segment frame 0 with CellSAM."""
        storage = self.frame_by_frame_widget.storage_service
        first_frame = storage.get_frame(0)
        if first_frame is None:
            self.show_error("Import Error", "Failed to load first frame")
            return

        h, w = first_frame.shape[:2]
        self.frame_by_frame_widget.initialize(np.zeros((h, w), dtype=np.uint16))
        self.stacked_widget.setCurrentIndex(IDX_TRACKING)

        self.status_label.setText("Segmenting first frame with CellSAM...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.cellsam_service.segment_first_frame_async(image_rgb=first_frame)

    def _on_cellsam_complete(self, first_frame_masks: np.ndarray) -> None:
        QApplication.restoreOverrideCursor()
        self.frame_by_frame_widget.initialize(first_frame_masks)
        self.status_label.setText(
            "First frame segmented — refine with SAM / Paint, then press Next"
        )

    def _on_cellsam_error(self, error_message: str) -> None:
        QApplication.restoreOverrideCursor()
        self.show_error(
            "CellSAM Error",
            f"Could not segment the first frame: {error_message}\n"
            "You can annotate manually with SAM or Paint.",
        )
        first_frame = self.frame_by_frame_widget.storage_service.get_frame(0)
        if first_frame is not None:
            h, w = first_frame.shape[:2]
            self.frame_by_frame_widget.initialize(np.zeros((h, w), dtype=np.uint16))

    def on_back_to_tracking(self):
        """Handle back to tracking navigation"""
        self.stacked_widget.setCurrentIndex(IDX_TRACKING)

    def on_restart_requested(self):
        """Return to preprocess to adjust crop, trim, and tone; then re-enter tracking."""
        storage = self.frame_by_frame_widget.storage_service
        if storage.get_source_frame_count() <= 0:
            return

        settings = storage.get_preprocess_settings()
        self.frame_by_frame_widget.clear_tracking_state()
        self.preprocess_widget.load_storage(storage, settings=settings)
        self.stacked_widget.setCurrentIndex(IDX_PREPROCESS)
        self.status_label.setText(
            "Adjust crop and image settings, then continue to tracking"
        )

    def show_export_view(self):
        """Show the export view"""
        self.export_widget.initialize()
        self.stacked_widget.setCurrentIndex(IDX_EXPORT)
