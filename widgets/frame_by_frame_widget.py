"""
Frame-by-frame segmentation and tracking widget
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QButtonGroup,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from mixins.annotation_mixin import AnnotationMixin
from mixins.cutie_mixin import CutieMixin
from mixins.sam_mixin import SamMixin
from mixins.storage_mixin import StorageMixin
from widgets.interactive_frame_widget import AnnotationMode, InteractiveFrameWidget


class FrameByFrameWidget(QWidget, SamMixin, CutieMixin, AnnotationMixin, StorageMixin):
    """Main widget for frame-by-frame segmentation and tracking"""

    # Signals
    status_update = pyqtSignal(str)
    progress_update = pyqtSignal(int, str)  # progress, message

    def __init__(self):
        super().__init__()

        # Initialize mixins
        SamMixin.__init__(self)
        CutieMixin.__init__(self)
        AnnotationMixin.__init__(self)
        StorageMixin.__init__(self)

        # Setup UI
        self.setup_ui()
        self.setup_shortcuts()

    def __del__(self):
        """Cleanup when widget is destroyed."""
        try:
            # Cancel and cleanup workers via mixins
            self.cleanup_cutie_worker()
            self.cleanup_sam_worker()
        except:
            pass  # Ignore cleanup errors during destruction

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
        self.prev_button = QPushButton("◀ Previous (A)")
        self.prev_button.clicked.connect(self.previous_frame)
        self.prev_button.setEnabled(False)
        nav_layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next (D) ▶")
        self.next_button.clicked.connect(self.next_frame)
        self.next_button.setEnabled(False)
        nav_layout.addWidget(self.next_button)

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
            lambda: self.set_annotation_mode(AnnotationMode.VIEW)
        )
        self.mode_group.addButton(self.view_radio)
        tools_layout.addWidget(self.view_radio)

        self.click_radio = QRadioButton("Click Add (2)")
        self.click_radio.toggled.connect(
            lambda: self.set_annotation_mode(AnnotationMode.CLICK_ADD)
        )
        self.mode_group.addButton(self.click_radio)
        tools_layout.addWidget(self.click_radio)

        self.box_radio = QRadioButton("Box Add (3)")
        self.box_radio.toggled.connect(
            lambda: self.set_annotation_mode(AnnotationMode.BOX_ADD)
        )
        self.mode_group.addButton(self.box_radio)
        tools_layout.addWidget(self.box_radio)

        self.remove_radio = QRadioButton("Remove (4)")
        self.remove_radio.toggled.connect(
            lambda: self.set_annotation_mode(AnnotationMode.MASK_REMOVE)
        )
        self.mode_group.addButton(self.remove_radio)
        tools_layout.addWidget(self.remove_radio)

        self.edit_id_radio = QRadioButton("Edit ID (5)")
        self.edit_id_radio.toggled.connect(
            lambda: self.set_annotation_mode(AnnotationMode.EDIT_CELL_ID)
        )
        self.mode_group.addButton(self.edit_id_radio)
        tools_layout.addWidget(self.edit_id_radio)

        tools_layout.addStretch()

        main_layout.addLayout(tools_layout)

        parent_layout.addWidget(merged_group)

    def setup_image_panel(self, parent_layout):
        """Setup dual image display"""
        image_group = QGroupBox("Current Frame vs Previous Frame")
        layout = QHBoxLayout(image_group)

        # Set size policy to expand and take all remaining vertical space
        image_group.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Previous frame panel
        prev_panel = QWidget()
        prev_layout = QVBoxLayout(prev_panel)
        prev_layout.addWidget(QLabel("Previous Frame:"))
        self.prev_image_label = InteractiveFrameWidget()
        self.prev_image_label.setEnabled(False)  # Non-interactive

        size_policy = QSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.prev_image_label.setSizePolicy(size_policy)
        prev_layout.addWidget(self.prev_image_label)
        splitter.addWidget(prev_panel)

        # Current frame panel
        curr_panel = QWidget()
        curr_layout = QVBoxLayout(curr_panel)
        curr_layout.addWidget(QLabel("Current Frame:"))
        self.curr_image_label = InteractiveFrameWidget()
        self.curr_image_label.point_clicked.connect(self.on_point_clicked)
        self.curr_image_label.box_drawn.connect(self.on_box_drawn)
        self.curr_image_label.mask_clicked.connect(self.on_mask_clicked)
        self.curr_image_label.cell_id_edit_requested.connect(
            self.on_cell_id_edit_requested
        )
        # Make both image displays expand to fill available space equally
        self.curr_image_label.setSizePolicy(size_policy)
        curr_layout.addWidget(self.curr_image_label)
        splitter.addWidget(curr_panel)

        # Set equal proportional sizes (50% each)
        splitter.setStretchFactor(0, 1)  # Previous panel
        splitter.setStretchFactor(1, 1)  # Current panel

        # Ensure both panels start with equal sizes
        splitter.setSizes([1, 1])  # Equal proportions

        # Add with stretch factor to make it expand in the parent layout
        parent_layout.addWidget(image_group, 1)  # stretch factor of 1

    def setup_tool_panel(self, parent_layout):
        """Setup annotation tools"""
        tool_group = QGroupBox("Segmentation Tools")
        layout = QVBoxLayout(tool_group)

        # Mode selection
        mode_layout = QHBoxLayout()

        self.mode_group = QButtonGroup()

        self.view_radio = QRadioButton("View Mode (1)")
        self.view_radio.setChecked(True)
        self.view_radio.toggled.connect(
            lambda: self.set_annotation_mode(AnnotationMode.VIEW)
        )
        self.mode_group.addButton(self.view_radio)
        mode_layout.addWidget(self.view_radio)

        self.click_radio = QRadioButton("Add by Click (2)")
        self.click_radio.toggled.connect(
            lambda: self.set_annotation_mode(AnnotationMode.CLICK_ADD)
        )
        self.mode_group.addButton(self.click_radio)
        mode_layout.addWidget(self.click_radio)

        self.box_radio = QRadioButton("Add by Box (3)")
        self.box_radio.toggled.connect(
            lambda: self.set_annotation_mode(AnnotationMode.BOX_ADD)
        )
        self.mode_group.addButton(self.box_radio)
        mode_layout.addWidget(self.box_radio)

        self.remove_radio = QRadioButton("Remove Mask (4)")
        self.remove_radio.toggled.connect(
            lambda: self.set_annotation_mode(AnnotationMode.MASK_REMOVE)
        )
        self.mode_group.addButton(self.remove_radio)
        mode_layout.addWidget(self.remove_radio)

        self.edit_id_radio = QRadioButton("Edit Cell ID (5)")
        self.edit_id_radio.toggled.connect(
            lambda: self.set_annotation_mode(AnnotationMode.EDIT_CELL_ID)
        )
        self.mode_group.addButton(self.edit_id_radio)
        mode_layout.addWidget(self.edit_id_radio)

        layout.addLayout(mode_layout)

        parent_layout.addWidget(tool_group)

    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # Navigation
        QShortcut(QKeySequence("A"), self, self.previous_frame)
        QShortcut(QKeySequence("D"), self, self.next_frame)

        # Tools
        QShortcut(QKeySequence("1"), self, lambda: self.view_radio.setChecked(True))
        QShortcut(QKeySequence("2"), self, lambda: self.click_radio.setChecked(True))
        QShortcut(QKeySequence("3"), self, lambda: self.box_radio.setChecked(True))
        QShortcut(QKeySequence("4"), self, lambda: self.remove_radio.setChecked(True))
        QShortcut(QKeySequence("5"), self, lambda: self.edit_id_radio.setChecked(True))

        # Note: Removed 'S' shortcut for CellSAM as it now only runs automatically on first frame

    def load_frames(self, image_paths: List[str]):
        """Load frames from image paths"""
        self.clear_all_data()

        # Cancel any running workers
        if self._cutie_worker is not None:
            self._cutie_worker.cancel()
            self._cutie_worker = None

        frames = []
        for path in image_paths:
            try:
                image = cv2.imread(path)
                if image is not None:
                    # Convert BGR to RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    frames.append(image_rgb)
            except Exception as e:
                QMessageBox.warning(
                    self, "Load Error", f"Failed to load {path}: {str(e)}"
                )

        if frames:
            self.set_frames(frames)
            self.set_image_paths(image_paths)
            self.set_current_frame_index(0)
            self.update_display()
            self.status_update.emit(f"Loaded {len(frames)} frames")

        else:
            QMessageBox.warning(self, "Load Error", "No valid frames loaded")

    def load_frames_with_segmentation(
        self, image_paths: List[str], segmentation_results: List[dict]
    ):
        """Load frames with pre-computed segmentation results"""
        self.clear_all_data()
        self.set_first_frame_segmented(True)  # Mark as already segmented

        # Cancel any running workers
        if self._cutie_worker is not None:
            self._cutie_worker.cancel()
            self._cutie_worker = None

        frames = []
        frame_masks = {}
        cellsam_results = {}

        # Load frames from segmentation results (they include the original images)
        for i, result in enumerate(segmentation_results):
            try:
                # Use the original image from CellSAM results
                image_rgb = result["original_image"]
                frames.append(image_rgb)

                # Store the masks
                masks = result["masks"]
                if masks is not None and masks.size > 0:
                    frame_masks[i] = masks.astype(np.uint16)

                # Store full CellSAM results for reference
                cellsam_results[i] = result

            except Exception as e:
                QMessageBox.warning(
                    self, "Load Error", f"Failed to load frame {i+1}: {str(e)}"
                )

        if frames:
            self.set_frames(frames)
            self.set_image_paths(image_paths)
            self.set_frame_masks(frame_masks)
            self.set_cellsam_results(cellsam_results)
            self.set_current_frame_index(0)
            self.update_display()
            self.status_update.emit(f"Loaded {len(frames)} frames with segmentation")
        else:
            QMessageBox.warning(self, "Load Error", "No valid frames loaded")

    def update_display(self):
        """Update the image display"""
        if self.get_frame_count() == 0:
            return

        # Update frame info
        total_frames = self.get_frame_count()
        current_index = self.get_current_frame_index()
        self.frame_info_label.setText(f"Frame {current_index + 1} / {total_frames}")

        # Update navigation buttons
        self.prev_button.setEnabled(self.has_previous_frame())
        self.next_button.setEnabled(self.has_next_frame())
        # Note: auto_segment_button is hidden and not enabled for manual use

        # Update current frame display (right side - editable, no cell IDs for clarity)
        current_image = self.get_current_frame()
        self.curr_image_label.set_image(current_image)

        # Set current frame masks if available
        current_masks = self.get_current_frame_masks()
        self.curr_image_label.set_masks(current_masks)

        # Update previous frame display (left side - shows cell IDs)
        if self.has_previous_frame():
            prev_image = self.get_frame(current_index - 1)
            self.prev_image_label.set_image(prev_image)

            # Set previous frame masks if available
            prev_masks = self.get_mask_for_frame(current_index - 1)
            self.prev_image_label.set_masks(prev_masks)
        else:
            # First frame - no previous frame
            self.prev_image_label.clear()
            self.prev_image_label.setText("First Frame")

    def previous_frame(self):
        """Go to previous frame"""
        if self.has_previous_frame():
            current_index = self.get_current_frame_index()
            self.set_current_frame_index(current_index - 1)
            self.update_display()

    def next_frame(self):
        """Go to next frame and run tracking if needed"""
        if self.has_next_frame():
            current_index = self.get_current_frame_index()
            next_index = current_index + 1

            # If this is a new frame that needs tracking, run CUTIE
            if (
                not self.has_mask_for_frame(next_index)
                and next_index > 0
                and self.has_mask_for_frame(current_index)
            ):
                # Use worker for frame-by-frame tracking
                self._track_next_frame(next_index)
            else:
                # Just move to next frame (already has mask or is first frame)
                self.set_current_frame_index(next_index)
                self.update_display()

    def get_current_masks(self) -> Optional[np.ndarray]:
        """Get masks for current frame"""
        return self.get_current_frame_masks()

    def get_all_masks(self) -> Dict[int, np.ndarray]:
        """Get all frame masks"""
        return self.get_frame_masks()
