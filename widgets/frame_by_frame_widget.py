"""
Frame-by-frame segmentation and tracking widget
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
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

from services.annotation_service import AnnotationService
from services.cutie_service import CutieService
from services.sam_service import SamService
from services.storage_service import StorageService
from widgets.interactive_frame_widget import AnnotationMode, InteractiveFrameWidget


class FrameByFrameWidget(QWidget):
    """Main widget for frame-by-frame segmentation and tracking"""

    # Signals
    status_update = pyqtSignal(str)
    progress_update = pyqtSignal(int, str)  # progress, message

    def __init__(self):
        super().__init__()

        # Initialize services with dependency injection
        self.storage_service = StorageService()
        self.annotation_service = AnnotationService(self)
        self.sam_service = SamService(self)
        self.cutie_service = CutieService()

        # Setup UI
        self.setup_ui()
        self.setup_shortcuts()

    def __del__(self):
        """Cleanup when widget is destroyed."""
        try:
            # Cancel and cleanup workers via services
            self.sam_service.cleanup_sam_worker()
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
        self.curr_image_label.point_clicked.connect(self.sam_service.on_point_clicked)
        self.curr_image_label.box_drawn.connect(self.sam_service.on_box_drawn)
        self.curr_image_label.mask_clicked.connect(
            self.annotation_service.on_mask_clicked
        )
        self.curr_image_label.cell_id_edit_requested.connect(
            self.annotation_service.on_cell_id_edit_requested
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
        """Load frames using lazy loading (don't load all images into memory)"""
        self.storage_service.clear_all_data()

        # Set image paths for lazy loading
        self.storage_service.set_image_paths(image_paths)
        self.storage_service.set_current_frame_index(0)
        self.update_display()
        self.status_update.emit(f"Loaded {len(image_paths)} frames (lazy loading)")

    def load_frames_with_first_segmentation(
        self, image_paths: List[str], first_frame_result: dict
    ):
        """Load frames with first frame segmentation for tracking"""
        self.storage_service.clear_all_data()

        # Set image paths for lazy loading
        self.storage_service.set_image_paths(image_paths)

        # Store first frame segmentation results
        self.storage_service.set_cellsam_result_for_frame(0, first_frame_result)

        # Extract first frame masks
        first_frame_masks = first_frame_result["masks"]
        if first_frame_masks is not None and first_frame_masks.size > 0:
            self.storage_service.set_mask_for_frame(
                0, first_frame_masks.astype(np.uint16)
            )
        else:
            QMessageBox.warning(
                self, "Load Error", "No masks found in first frame segmentation"
            )
            return

        self.storage_service.set_current_frame_index(0)
        self.update_display()

        # Show status
        cell_count = np.max(first_frame_masks) if first_frame_masks.size > 0 else 0
        self.status_update.emit(
            f"Loaded {len(image_paths)} frames. Found {cell_count} cells in first frame. Ready for tracking."
        )

    def load_frames_with_segmentation(
        self, image_paths: List[str], segmentation_results: List[dict]
    ):
        """Load frames with pre-computed segmentation results"""
        self.storage_service.clear_all_data()

        # Set image paths for lazy loading
        self.storage_service.set_image_paths(image_paths)

        frame_masks = {}
        cellsam_results = {}

        # Store the segmentation data (don't load actual images yet)
        for i, result in enumerate(segmentation_results):
            try:
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

        self.storage_service.set_frame_masks(frame_masks)
        self.storage_service.set_cellsam_results(cellsam_results)
        self.storage_service.set_current_frame_index(0)
        self.update_display()
        self.status_update.emit(
            f"Loaded {len(image_paths)} frames with segmentation (lazy loading)"
        )

    def update_display(self):
        """Update the image display"""
        if self.storage_service.get_frame_count() == 0:
            return

        # Update frame info
        total_frames = self.storage_service.get_frame_count()
        current_index = self.storage_service.get_current_frame_index()
        self.frame_info_label.setText(f"Frame {current_index + 1} / {total_frames}")

        # Update navigation buttons
        self.prev_button.setEnabled(self.storage_service.has_previous_frame())
        self.next_button.setEnabled(self.storage_service.has_next_frame())
        # Note: auto_segment_button is hidden and not enabled for manual use

        # Update current frame display (right side - editable, no cell IDs for clarity)
        current_image = self.storage_service.get_current_frame()
        self.curr_image_label.set_image(current_image)

        # Set current frame masks if available
        current_masks = self.storage_service.get_current_frame_masks()
        self.curr_image_label.set_masks(current_masks)

        # Update previous frame display (left side - shows cell IDs)
        if self.storage_service.has_previous_frame():
            prev_image = self.storage_service.get_frame(current_index - 1)
            self.prev_image_label.set_image(prev_image)

            # Set previous frame masks if available
            prev_masks = self.storage_service.get_mask_for_frame(current_index - 1)
            self.prev_image_label.set_masks(prev_masks)
        else:
            # First frame - no previous frame
            self.prev_image_label.clear()
            self.prev_image_label.setText("First Frame")

    def previous_frame(self):
        """Go to previous frame"""
        if self.storage_service.has_previous_frame():
            current_index = self.storage_service.get_current_frame_index()
            self.storage_service.set_current_frame_index(current_index - 1)
            self.update_display()

    def next_frame(self):
        """Go to next frame and run tracking if needed"""
        if self.storage_service.has_next_frame():
            current_index = self.storage_service.get_current_frame_index()
            next_index = current_index + 1

            # If this is a new frame that needs tracking, run CUTIE
            if (
                not self.storage_service.has_mask_for_frame(next_index)
                and next_index > 0
                and self.storage_service.has_mask_for_frame(current_index)
            ):
                # Get data for tracking
                previous_image = self.storage_service.get_frame(current_index)
                previous_mask = self.storage_service.get_mask_for_frame(current_index)
                current_image = self.storage_service.get_frame(next_index)

                # Run CUTIE tracking
                predicted_mask = self.cutie_service.track(
                    previous_image, previous_mask, current_image, next_index
                )

                if predicted_mask is not None:
                    # Store the predicted mask and move to next frame
                    self.storage_service.set_mask_for_frame(
                        next_index, predicted_mask.astype(np.uint16)
                    )
                    self.storage_service.set_current_frame_index(next_index)
                    self.update_display()
                    self.status_update.emit(
                        f"Frame {next_index + 1} tracked successfully"
                    )
                else:
                    # Tracking failed, show error but still allow navigation
                    self.show_warning(
                        "Tracking Error", "CUTIE tracking failed for this frame"
                    )
                    self.storage_service.set_current_frame_index(next_index)
                    self.update_display()
            else:
                # Just move to next frame (already has mask or is first frame)
                self.storage_service.set_current_frame_index(next_index)
                self.update_display()

    def get_current_masks(self) -> Optional[np.ndarray]:
        """Get masks for current frame"""
        return self.storage_service.get_current_frame_masks()

    def get_all_masks(self) -> Dict[int, np.ndarray]:
        """Get all frame masks"""
        return self.storage_service.get_frame_masks()

    # Delegate methods for services

    # StorageService delegates (already implemented via storage_service calls)

    # AnnotationService delegates
    def get_current_frame_masks(self) -> Optional[np.ndarray]:
        """Delegate for annotation service"""
        return self.storage_service.get_current_frame_masks()

    def set_mask_for_frame(self, frame_index: int, masks: np.ndarray) -> None:
        """Delegate for annotation service"""
        self.storage_service.set_mask_for_frame(frame_index, masks)

    def get_current_frame_index(self) -> int:
        """Delegate for annotation service"""
        return self.storage_service.get_current_frame_index()

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

    # SamService delegates
    def get_frame_count(self) -> int:
        """Delegate for SAM service"""
        return self.storage_service.get_frame_count()

    def get_current_frame(self) -> Optional[np.ndarray]:
        """Delegate for SAM service"""
        return self.storage_service.get_current_frame()

    def show_warning(self, title: str, message: str) -> None:
        """Delegate for SAM service"""
        QMessageBox.warning(self, title, message)
