"""
Frame-by-frame segmentation and tracking widget
"""

from typing import Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QKeySequence, QResizeEvent, QShortcut, QShowEvent
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from services.annotation_service import AnnotationService
from services.cell_track_service import CellTrackService
from services.tracking_service import TrackingService
from services.mask_history_service import MaskHistoryService, MaskSnapshot
from utils.mask_utils import align_mask_to_image
from services.sam_service import SamService
from services.storage_service import StorageService
from widgets.cell_sidebar_widget import CellSidebarWidget
from widgets.interactive_frame_widget import AnnotationMode, InteractiveFrameWidget


class FrameByFrameWidget(QWidget):
    """Main widget for frame-by-frame segmentation and tracking"""

    # Signals
    status_update = pyqtSignal(str)
    export_requested = pyqtSignal()
    restart_requested = pyqtSignal()

    def __init__(self):
        super().__init__()

        # Initialize services with dependency injection
        self.storage_service = StorageService()
        self.annotation_service = AnnotationService(self)
        self.sam_service = SamService(self)
        self.tracking_service = TrackingService()
        self.cell_track_service = CellTrackService()
        self._mask_history = MaskHistoryService()

        # Track pending async operations
        self._pending_frame_index: Optional[int] = None
        self._pending_tracking_mode: Optional[Literal["relink", "advance"]] = None
        self._selected_cell_id: Optional[int] = None
        self._syncing_view = False
        self.frames_splitter: Optional[QSplitter] = None

        # Setup UI
        self.setup_ui()
        self.setup_shortcuts()
        self.setup_async_connections()

    # ------------------------------------------------------------------------ #
    # ------------------------------- UI setup ------------------------------- #
    # ------------------------------------------------------------------------ #

    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Control panel - minimum height
        self.setup_control_panel(layout)

        # Image panel - takes remaining height
        self.setup_image_panel(layout)

    def setup_control_panel(self, parent_layout):
        """Setup merged frame navigation and segmentation tools panel"""
        merged_group = QGroupBox("Control Panel")
        main_layout = QVBoxLayout(merged_group)
        main_layout.setSpacing(12)

        # Set size policy to prefer minimum height
        merged_group.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        # Top row: Frame navigation
        nav_layout = QHBoxLayout()

        # Frame info
        self.frame_info_label = QLabel("No frames loaded")
        nav_layout.addWidget(self.frame_info_label)

        nav_layout.addStretch()

        # Navigation buttons
        self.prev_button = QPushButton("Previous (A)")
        self.prev_button.setStyleSheet(
            """
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 6px 12px;
                font-weight: bold;
                border-radius: 4px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004085;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """
        )
        self.prev_button.clicked.connect(self.previous_frame)
        self.prev_button.setEnabled(False)
        nav_layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next (D)")
        self.next_button.setStyleSheet(
            """
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 6px 12px;
                font-weight: bold;
                border-radius: 4px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004085;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """
        )
        self.next_button.clicked.connect(self.next_frame)
        self.next_button.setEnabled(False)
        nav_layout.addWidget(self.next_button)

        nav_layout.addSpacing(12)

        self.undo_button = QPushButton("Undo")
        self.undo_button.setToolTip("Undo last mask edit (Ctrl+Z)")
        self.undo_button.clicked.connect(self.on_undo)
        self.undo_button.setEnabled(False)
        nav_layout.addWidget(self.undo_button)

        self.redo_button = QPushButton("Redo")
        self.redo_button.setToolTip("Redo (Ctrl+Y)")
        self.redo_button.clicked.connect(self.on_redo)
        self.redo_button.setEnabled(False)
        nav_layout.addWidget(self.redo_button)

        self.relink_button = QPushButton("Relink")
        self.relink_button.setToolTip(
            "Re-assign cell IDs on this frame to match the previous frame "
            "(keeps your current mask shapes)"
        )
        self.relink_button.clicked.connect(self.on_relink)
        self.relink_button.setEnabled(False)
        nav_layout.addWidget(self.relink_button)

        nav_layout.addSpacing(20)

        # Export button
        self.export_button = QPushButton("Export Data")
        self.export_button.setStyleSheet(
            """
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 8px 16px;
                font-weight: bold;
                border-radius: 4px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
        """
        )
        self.export_button.clicked.connect(self.on_export_requested)
        self.export_button.setVisible(False)  # Initially hidden
        nav_layout.addWidget(self.export_button)

        # Restart button
        self.restart_button = QPushButton("Restart")
        self.restart_button.setStyleSheet(
            """
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 8px 16px;
                font-weight: bold;
                border-radius: 4px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
            QPushButton:pressed {
                background-color: #495057;
            }
        """
        )
        self.restart_button.clicked.connect(self.on_restart_requested)
        self.restart_button.setVisible(False)  # Initially hidden
        nav_layout.addWidget(self.restart_button)

        main_layout.addLayout(nav_layout)

        # Bottom row: Segmentation tools
        tools_layout = QHBoxLayout()

        # Tool mode label
        tools_label = QLabel("Tools:")
        tools_label.setStyleSheet("font-weight: bold;")
        tools_layout.addWidget(tools_label)

        # Mode selection buttons in a compact layout
        self.mode_group = QButtonGroup()

        self.view_radio = QRadioButton("View (1)")
        self.view_radio.setChecked(True)
        self.view_radio.toggled.connect(
            lambda: self.annotation_service.set_annotation_mode(
                self.curr_image_label, AnnotationMode.VIEW
            )
        )
        self.mode_group.addButton(self.view_radio)
        tools_layout.addWidget(self.view_radio)

        self.click_radio = QRadioButton("Click Add (2)")
        self.click_radio.toggled.connect(
            lambda: self.annotation_service.set_annotation_mode(
                self.curr_image_label, AnnotationMode.CLICK_ADD
            )
        )
        self.mode_group.addButton(self.click_radio)
        tools_layout.addWidget(self.click_radio)

        self.box_radio = QRadioButton("Box Add (3)")
        self.box_radio.toggled.connect(
            lambda: self.annotation_service.set_annotation_mode(
                self.curr_image_label, AnnotationMode.BOX_ADD
            )
        )
        self.mode_group.addButton(self.box_radio)
        tools_layout.addWidget(self.box_radio)

        self.paint_radio = QRadioButton("Paint Add (6)")
        self.paint_radio.setToolTip(
            "Paint over a cell, release to segment with SAM. "
            "Right-click or Shift = erase stroke."
        )
        self.paint_radio.toggled.connect(
            lambda: self.annotation_service.set_annotation_mode(
                self.curr_image_label, AnnotationMode.PAINT_ADD
            )
        )
        self.mode_group.addButton(self.paint_radio)
        tools_layout.addWidget(self.paint_radio)

        brush_label = QLabel("Brush:")
        tools_layout.addWidget(brush_label)

        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(2, 32)
        self.brush_slider.setValue(6)
        self.brush_slider.setFixedWidth(80)
        self.brush_slider.setToolTip("Brush size in pixels")
        self.brush_slider.valueChanged.connect(self.on_brush_size_changed)
        tools_layout.addWidget(self.brush_slider)

        self.brush_value_label = QLabel("6")
        self.brush_value_label.setFixedWidth(24)
        tools_layout.addWidget(self.brush_value_label)

        self.remove_radio = QRadioButton("Remove (4)")
        self.remove_radio.toggled.connect(
            lambda: self.annotation_service.set_annotation_mode(
                self.curr_image_label, AnnotationMode.MASK_REMOVE
            )
        )
        self.mode_group.addButton(self.remove_radio)
        tools_layout.addWidget(self.remove_radio)

        self.edit_id_radio = QRadioButton("Edit ID (5)")
        self.edit_id_radio.toggled.connect(
            lambda: self.annotation_service.set_annotation_mode(
                self.curr_image_label, AnnotationMode.EDIT_CELL_ID
            )
        )
        self.mode_group.addButton(self.edit_id_radio)
        tools_layout.addWidget(self.edit_id_radio)

        tools_layout.addStretch()

        main_layout.addLayout(tools_layout)

        # Additional controls row: Transparency and cell ID toggle
        controls_layout = QHBoxLayout()

        # Mask transparency control
        transparency_label = QLabel("Mask Opacity:")
        controls_layout.addWidget(transparency_label)

        self.transparency_slider = QSlider(Qt.Orientation.Horizontal)
        self.transparency_slider.setRange(0, 100)
        self.transparency_slider.setValue(30)  # Default 30% opacity
        self.transparency_slider.setFixedWidth(100)
        self.transparency_slider.valueChanged.connect(self.on_transparency_changed)
        controls_layout.addWidget(self.transparency_slider)

        self.transparency_value_label = QLabel("30%")
        self.transparency_value_label.setFixedWidth(35)
        controls_layout.addWidget(self.transparency_value_label)

        controls_layout.addSpacing(20)

        # Cell ID toggle
        self.show_cell_ids_checkbox = QCheckBox("Show Cell IDs")
        self.show_cell_ids_checkbox.setChecked(True)  # Default to showing cell IDs
        self.show_cell_ids_checkbox.toggled.connect(self.on_cell_id_toggle_changed)
        controls_layout.addWidget(self.show_cell_ids_checkbox)

        controls_layout.addSpacing(16)

        label_size_label = QLabel("Label Size:")
        controls_layout.addWidget(label_size_label)

        self.label_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.label_size_slider.setRange(50, 200)
        self.label_size_slider.setValue(100)
        self.label_size_slider.setFixedWidth(100)
        self.label_size_slider.valueChanged.connect(self.on_label_size_changed)
        controls_layout.addWidget(self.label_size_slider)

        self.label_size_value_label = QLabel("100%")
        self.label_size_value_label.setFixedWidth(40)
        controls_layout.addWidget(self.label_size_value_label)

        controls_layout.addStretch()

        main_layout.addLayout(controls_layout)

        parent_layout.addWidget(merged_group)

    def setup_image_panel(self, parent_layout):
        """Setup dual image display with cell sidebar."""
        image_group = QGroupBox("Frames & Cells")
        layout = QHBoxLayout(image_group)

        image_group.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        outer_splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(outer_splitter)

        self.frames_splitter = QSplitter(Qt.Orientation.Horizontal)

        self.prev_frame_panel = QWidget()
        prev_layout = QVBoxLayout(self.prev_frame_panel)
        prev_layout.setContentsMargins(0, 0, 0, 0)
        prev_layout.addWidget(QLabel("Previous Frame:"))
        self.prev_image_label = InteractiveFrameWidget()
        self.prev_image_label.set_annotation_mode(AnnotationMode.VIEW)

        size_policy = QSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.prev_image_label.setSizePolicy(size_policy)
        prev_layout.addWidget(self.prev_image_label, stretch=1)
        self.frames_splitter.addWidget(self.prev_frame_panel)

        curr_panel = QWidget()
        curr_layout = QVBoxLayout(curr_panel)
        curr_layout.setContentsMargins(0, 0, 0, 0)
        curr_layout.addWidget(
            QLabel("Current Frame (View: drag to pan, click to select cell):")
        )
        self.curr_image_label = InteractiveFrameWidget()
        self.curr_image_label.point_clicked.connect(self.sam_service.on_point_clicked)
        self.curr_image_label.box_drawn.connect(self.sam_service.on_box_drawn)
        self.curr_image_label.paint_mask_ready.connect(self.sam_service.on_paint_mask)
        self.curr_image_label.mask_clicked.connect(
            self.annotation_service.on_mask_clicked
        )
        self.curr_image_label.cell_id_edit_requested.connect(
            self.annotation_service.on_cell_id_edit_requested
        )
        self.curr_image_label.cell_picked.connect(self.on_cell_picked_on_canvas)
        self.curr_image_label.setSizePolicy(size_policy)
        curr_layout.addWidget(self.curr_image_label, stretch=1)
        self.frames_splitter.addWidget(curr_panel)

        self.frames_splitter.setStretchFactor(0, 1)
        self.frames_splitter.setStretchFactor(1, 1)
        self.frames_splitter.setChildrenCollapsible(False)
        self.prev_image_label.view_changed.connect(self._on_prev_view_changed)
        self.curr_image_label.view_changed.connect(self._on_curr_view_changed)

        outer_splitter.addWidget(self.frames_splitter)

        self.cell_sidebar = CellSidebarWidget()
        self.cell_sidebar.jump_to_frame.connect(self.on_sidebar_jump_to_frame)
        self.cell_sidebar.solo_mode_changed.connect(self._apply_cell_highlighting)
        outer_splitter.addWidget(self.cell_sidebar)

        outer_splitter.setStretchFactor(0, 4)
        outer_splitter.setStretchFactor(1, 1)
        outer_splitter.setSizes([900, 280])

        parent_layout.addWidget(image_group, 1)

    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # Frame navigation shortcuts
        self.prev_shortcut = QShortcut(QKeySequence("A"), self)
        self.prev_shortcut.activated.connect(self.previous_frame)

        self.next_shortcut = QShortcut(QKeySequence("D"), self)
        self.next_shortcut.activated.connect(self.next_frame)

        # Toggle between SAM and Track modes
        self.mode_shortcut = QShortcut(QKeySequence("Tab"), self)
        self.mode_shortcut.activated.connect(self.toggle_annotation_mode)

        self.undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self.undo_shortcut.activated.connect(self.on_undo)

        self.redo_shortcut = QShortcut(QKeySequence.StandardKey.Redo, self)
        self.redo_shortcut.activated.connect(self.on_redo)

    def setup_async_connections(self):
        """Setup connections for async operations"""
        self.tracking_service.tracking_complete.connect(self._on_tracking_complete)
        self.tracking_service.tracking_error.connect(self._on_tracking_error)
        self.tracking_service.status_update.connect(self.status_update.emit)

    # ------------------------------------------------------------------------ #
    # ------------------------------- Callbacks ------------------------------ #
    # ------------------------------------------------------------------------ #

    def on_transparency_changed(self, value):
        """Handle transparency slider change"""
        transparency = value / 100.0  # Convert to 0-1 range
        self.transparency_value_label.setText(f"{value}%")
        self.curr_image_label.set_mask_transparency(transparency)
        self.prev_image_label.set_mask_transparency(transparency)

    def on_cell_id_toggle_changed(self, checked):
        """Handle cell ID toggle change"""
        self.curr_image_label.set_show_cell_ids(checked)
        self.prev_image_label.set_show_cell_ids(checked)

    def on_label_size_changed(self, value: int):
        """Handle label size slider change."""
        self.label_size_value_label.setText(f"{value}%")
        scale = value / 100.0
        self.curr_image_label.set_cell_id_scale(scale)
        self.prev_image_label.set_cell_id_scale(scale)

    def on_brush_size_changed(self, value: int) -> None:
        self.brush_value_label.setText(str(value))
        self.curr_image_label.set_brush_radius(value)

    def on_cell_picked_on_canvas(self, cell_id: int):
        self._selected_cell_id = cell_id
        self.refresh_cell_inspector()
        self._apply_cell_highlighting()

    def on_sidebar_jump_to_frame(self, frame_index: int):
        if 0 <= frame_index < self.storage_service.get_frame_count():
            self.storage_service.set_current_frame_index(frame_index)
            self.update_display()

    def refresh_cell_inspector(self):
        """Update inspector for the selected cell, if any."""
        if (
            self.storage_service.get_frame_count() == 0
            or self._selected_cell_id is None
        ):
            self.cell_sidebar.clear()
            return

        current = self.storage_service.get_current_frame_index()
        detail = self.cell_track_service.analyze_cell(
            self.storage_service.get_mask_for_frame,
            self.storage_service.get_frame_count(),
            current,
            self._selected_cell_id,
        )
        if detail is None:
            self._selected_cell_id = None
            self.cell_sidebar.clear()
            self._apply_cell_highlighting()
            return
        self.cell_sidebar.show_cell(detail)

    def _on_prev_view_changed(self, zoom: float, pan_x: float, pan_y: float):
        self._sync_view_to(self.curr_image_label, zoom, pan_x, pan_y)

    def _on_curr_view_changed(self, zoom: float, pan_x: float, pan_y: float):
        self._sync_view_to(self.prev_image_label, zoom, pan_x, pan_y)

    def _sync_view_to(
        self,
        target: InteractiveFrameWidget,
        zoom: float,
        pan_x: float,
        pan_y: float,
    ) -> None:
        if self._syncing_view or target.image is None:
            return
        self._syncing_view = True
        target.apply_view_state(zoom, (pan_x, pan_y))
        self._syncing_view = False

    def _equalize_frame_panels(self) -> None:
        if self.frames_splitter is None:
            return
        total = self.frames_splitter.width()
        if total <= 0:
            return
        if not self.storage_service.has_previous_frame():
            self.frames_splitter.setSizes([0, total])
        else:
            half = max(total // 2, 1)
            self.frames_splitter.setSizes([half, half])

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._equalize_frame_panels)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._equalize_frame_panels()

    def _apply_cell_highlighting(self):
        highlight = self._selected_cell_id
        solo = (
            self._selected_cell_id
            if self.cell_sidebar.solo_checkbox.isChecked()
            else None
        )
        self.curr_image_label.set_highlight_cell_id(highlight)
        self.curr_image_label.set_solo_cell_id(solo)

    def on_export_requested(self):
        """Handle export button click"""
        self.export_requested.emit()

    def on_restart_requested(self):
        """Handle restart button click"""
        self.restart_requested.emit()

    def clear_tracking_state(self) -> None:
        """Clear masks and tracking UI but keep loaded media and preprocess settings."""
        self._mask_history.clear()
        self._update_undo_redo_buttons()
        self.storage_service.clear_all_masks()
        self.storage_service.invalidate_frame_cache()
        self.storage_service.set_current_frame_index(0)
        self._selected_cell_id = None
        self._pending_frame_index = None
        self.curr_image_label.clear_all_state()
        self.prev_image_label.clear_all_state()
        self.cell_sidebar.clear()

    def clear_all_data(self):
        """Clear all data and reset widget state"""
        self._mask_history.clear()
        self.storage_service.clear_all_data()

        self.curr_image_label.clear_all_state()
        self.prev_image_label.clear_all_state()
        self.cell_sidebar.clear()
        self._selected_cell_id = None

        self.frame_info_label.setText("No frames loaded")
        self.export_button.setVisible(False)
        self.restart_button.setVisible(False)
        self._update_undo_redo_buttons()

    # ------------------------------------------------------------------------ #
    # ---------------------------- Initialization ---------------------------- #
    # ------------------------------------------------------------------------ #

    def initialize(self, first_frame_mask: np.ndarray):
        """Apply first-frame masks; media must already be opened in storage."""
        self._mask_history.clear()
        frame = self.storage_service.get_frame(0)
        if frame is not None:
            h, w = frame.shape[:2]
            first_frame_mask = align_mask_to_image(
                first_frame_mask.astype(np.uint16), h, w
            )
        else:
            first_frame_mask = first_frame_mask.astype(np.uint16)
        self.storage_service.set_mask_for_frame(0, first_frame_mask)
        self.storage_service.set_current_frame_index(0)
        self.update_display()
        self.refresh_cell_inspector()

    # ------------------------------------------------------------------------ #
    # -------------------------------- Display ------------------------------- #
    # ------------------------------------------------------------------------ #

    def update_display(self):
        """Update the image display"""
        if self.storage_service.get_frame_count() == 0:
            return

        # Update frame info
        total_frames = self.storage_service.get_frame_count()
        current_index = self.storage_service.get_current_frame_index()
        self.frame_info_label.setText(f"Frame {current_index + 1} / {total_frames}")

        # Update navigation buttons
        tracking = self.is_tracking_running()
        self.prev_button.setEnabled(
            self.storage_service.has_previous_frame() and not tracking
        )
        self.next_button.setEnabled(
            self.storage_service.has_next_frame() and not tracking
        )

        # Update export button - only show when at last frame
        self.export_button.setVisible(self.is_at_last_frame())
        # Show restart button when frames are loaded (any frame)
        self.restart_button.setVisible(total_frames > 0)
        # Note: auto_segment_button is hidden and not enabled for manual use

        # Update current frame display (right side - editable, no cell IDs for clarity)
        current_image = self.storage_service.get_current_frame()
        self.curr_image_label.set_image(current_image)

        # Set current frame masks if available
        current_masks = self.storage_service.get_current_frame_masks()
        self.curr_image_label.set_masks(current_masks)

        # Previous frame panel (hidden on frame 0)
        if self.storage_service.has_previous_frame():
            self.prev_frame_panel.setVisible(True)
            prev_image = self.storage_service.get_frame(current_index - 1)
            self.prev_image_label.set_image(prev_image)
            prev_masks = self.storage_service.get_mask_for_frame(current_index - 1)
            self.prev_image_label.set_masks(prev_masks)
        else:
            self.prev_frame_panel.setVisible(False)
            self.prev_image_label.clear_all_state()

        self.refresh_cell_inspector()
        self._apply_cell_highlighting()
        self._equalize_frame_panels()
        self._update_undo_redo_buttons()
        self._update_relink_button()

    def _update_relink_button(self) -> None:
        if not hasattr(self, "relink_button"):
            return
        current_index = self.storage_service.get_current_frame_index()
        prev_mask = (
            self.storage_service.get_mask_for_frame(current_index - 1)
            if current_index > 0
            else None
        )
        curr_mask = self.storage_service.get_current_frame_masks()
        can_relink = (
            current_index > 0
            and prev_mask is not None
            and np.any(prev_mask)
            and curr_mask is not None
            and np.any(curr_mask)
            and not self.is_tracking_running()
        )
        self.relink_button.setEnabled(can_relink)

    def previous_frame(self):
        """Go to previous frame"""
        if self.is_tracking_running():
            return

        if self.storage_service.has_previous_frame():
            current_index = self.storage_service.get_current_frame_index()
            self.storage_service.set_current_frame_index(current_index - 1)
            self.update_display()

    def next_frame(self):
        """Go to next frame; auto-segment and link if that frame has no mask yet."""
        if self.is_tracking_running():
            return

        if not self.storage_service.has_next_frame():
            return

        current_index = self.storage_service.get_current_frame_index()
        next_index = current_index + 1

        if (
            not self.storage_service.has_mask_for_frame(next_index)
            and next_index > 0
            and self.storage_service.has_mask_for_frame(current_index)
        ):
            previous_image = self.storage_service.get_frame(current_index)
            previous_mask = self.storage_service.get_mask_for_frame(current_index)
            current_image = self.storage_service.get_frame(next_index)
            if previous_image is None or current_image is None:
                return

            self._pending_frame_index = next_index
            self._pending_tracking_mode = "advance"
            self.tracking_service.track_async(
                previous_image, previous_mask, current_image
            )
            return

        self.storage_service.set_current_frame_index(next_index)
        self.update_display()

    def on_relink(self) -> None:
        """Re-assign IDs on the current mask using the previous frame."""
        if self.is_tracking_running():
            return

        current_index = self.storage_service.get_current_frame_index()
        if current_index <= 0:
            self.show_warning(
                "Relink",
                "The first frame has no previous frame to link from.",
            )
            return

        prev_index = current_index - 1
        previous_mask = self.storage_service.get_mask_for_frame(prev_index)
        if previous_mask is None or not np.any(previous_mask):
            self.show_warning(
                "Relink",
                "The previous frame has no masks to link from.",
            )
            return

        current_mask = self.storage_service.get_mask_for_frame(current_index)
        if current_mask is None or not np.any(current_mask):
            self.show_warning(
                "Relink",
                "The current frame has no mask. Segment it first (e.g. SAM / Paint), "
                "or use Next on an empty frame to auto-segment.",
            )
            return

        previous_image = self.storage_service.get_frame(prev_index)
        current_image = self.storage_service.get_frame(current_index)
        if previous_image is None or current_image is None:
            self.show_warning("Relink", "Failed to load frames for linking.")
            return

        self.begin_mask_edit()
        self._pending_frame_index = current_index
        self._pending_tracking_mode = "relink"
        self.tracking_service.relink_async(
            previous_image,
            previous_mask,
            current_image,
            current_mask,
        )

    def _on_tracking_complete(self, predicted_mask: np.ndarray):
        """Handle successful relink / tracking completion."""
        if self._pending_frame_index is None:
            return

        frame_index = self._pending_frame_index
        self.storage_service.set_mask_for_frame(
            frame_index, predicted_mask.astype(np.uint16)
        )

        removed_count = 0
        if frame_index < self.storage_service.get_frame_count() - 1:
            removed_count = self.storage_service.remove_masks_after_frame(frame_index)

        mode = self._pending_tracking_mode
        if mode == "advance":
            self.storage_service.set_current_frame_index(frame_index)

        self.update_display()

        if mode == "relink":
            msg = f"Frame {frame_index + 1} relinked"
            if removed_count > 0:
                msg += f" (cleared {removed_count} later frame masks)"
            self.status_update.emit(msg)
        elif mode == "advance":
            self.status_update.emit(f"Frame {frame_index + 1} segmented and linked")
        else:
            self.status_update.emit(f"Frame {frame_index + 1} updated")

        self._pending_frame_index = None
        self._pending_tracking_mode = None

    def _on_tracking_error(self, error_message: str):
        """Handle tracking error"""
        if self._pending_frame_index is None:
            return

        mode = self._pending_tracking_mode
        self.show_warning("Linking Error", f"Linking failed: {error_message}")
        if mode == "advance":
            self.storage_service.set_current_frame_index(self._pending_frame_index)
        self._pending_frame_index = None
        self._pending_tracking_mode = None
        self.update_display()

    def begin_mask_edit(self) -> None:
        """Snapshot masks from the current frame onward for undo."""
        self._mask_history.push(self._capture_mask_snapshot_from_current())
        self._update_undo_redo_buttons()

    def _capture_mask_snapshot_from_current(self) -> MaskSnapshot:
        return self._capture_mask_snapshot_from(
            self.storage_service.get_current_frame_index()
        )

    def _capture_mask_snapshot_from(self, start_frame: int) -> MaskSnapshot:
        snapshot: MaskSnapshot = []
        for i in range(start_frame, self.storage_service.get_frame_count()):
            masks = self.storage_service.get_mask_for_frame(i)
            snapshot.append((i, masks.copy() if masks is not None else None))
        return snapshot

    def _restore_mask_snapshot(self, snapshot: MaskSnapshot) -> None:
        for frame_index, masks in snapshot:
            if masks is None:
                self.storage_service.remove_mask_for_frame(frame_index)
            else:
                self.storage_service.set_mask_for_frame(frame_index, masks)
        self.update_display()

    def on_undo(self) -> None:
        if self.is_tracking_running():
            return
        snapshot = self._mask_history.undo(self._capture_mask_snapshot_from_current)
        if snapshot is None:
            return
        self._restore_mask_snapshot(snapshot)
        self.status_update.emit("Undid mask edit")
        self._update_undo_redo_buttons()

    def on_redo(self) -> None:
        if self.is_tracking_running():
            return
        snapshot = self._mask_history.redo(self._capture_mask_snapshot_from_current)
        if snapshot is None:
            return
        self._restore_mask_snapshot(snapshot)
        self.status_update.emit("Redid mask edit")
        self._update_undo_redo_buttons()

    def _update_undo_redo_buttons(self) -> None:
        can_edit = not self.is_tracking_running()
        self.undo_button.setEnabled(can_edit and self._mask_history.can_undo())
        self.redo_button.setEnabled(can_edit and self._mask_history.can_redo())

    def toggle_annotation_mode(self):
        """Toggle between annotation modes using Tab key"""
        # Get current checked radio button
        if self.view_radio.isChecked():
            self.click_radio.setChecked(True)
        elif self.click_radio.isChecked():
            self.box_radio.setChecked(True)
        elif self.box_radio.isChecked():
            self.paint_radio.setChecked(True)
        elif self.paint_radio.isChecked():
            self.remove_radio.setChecked(True)
        elif self.remove_radio.isChecked():
            self.edit_id_radio.setChecked(True)
        elif self.edit_id_radio.isChecked():
            self.view_radio.setChecked(True)

    # ------------------------------------------------------------------------ #
    # --------------------------- Delegate Methods --------------------------- #
    # ------------------------------------------------------------------------ #

    def get_current_frame_masks(self) -> Optional[np.ndarray]:
        """Delegate for annotation service"""
        return self.storage_service.get_current_frame_masks()

    def set_mask_for_frame(self, frame_index: int, masks: np.ndarray) -> None:
        """Delegate for annotation service"""
        self.storage_service.set_mask_for_frame(frame_index, masks)

    def get_current_frame_index(self) -> int:
        """Delegate for annotation service"""
        return self.storage_service.get_current_frame_index()

    def get_frame_count(self) -> int:
        """Delegate for annotation and SAM services"""
        return self.storage_service.get_frame_count()

    def remove_masks_after_frame(self, frame_index: int) -> int:
        """Delegate for annotation service"""
        return self.storage_service.remove_masks_after_frame(frame_index)

    def emit_status_update(self, message: str) -> None:
        """Delegate for annotation service"""
        self.status_update.emit(message)

    def show_message_box(
        self, title: str, message: str, box_type: str = "information"
    ) -> None:
        """Delegate for annotation service"""
        if box_type == "information":
            QMessageBox.information(self, title, message)
        elif box_type == "warning":
            QMessageBox.warning(self, title, message)
        elif box_type == "critical":
            QMessageBox.critical(self, title, message)

    def show_question_box(self, title: str, message: str) -> bool:
        """Delegate for annotation service"""
        reply = QMessageBox.question(
            self,
            title,
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    def update_current_display_masks(self, masks: np.ndarray) -> None:
        """Delegate for annotation service"""
        self.curr_image_label.set_masks(masks)
        self.refresh_cell_inspector()
        self._apply_cell_highlighting()

    def get_current_frame(self) -> Optional[np.ndarray]:
        """Delegate for SAM service"""
        return self.storage_service.get_current_frame()

    def show_warning(self, title: str, message: str) -> None:
        """Delegate for SAM service"""
        QMessageBox.warning(self, title, message)

    def is_tracking_running(self) -> bool:
        return self.tracking_service.is_running()

    def is_at_last_frame(self) -> bool:
        """Check if we're currently at the last frame"""
        current_index = self.storage_service.get_current_frame_index()
        total_frames = self.storage_service.get_frame_count()
        return current_index == total_frames - 1
