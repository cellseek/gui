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

from widgets.interactive_frame_widget import AnnotationMode, InteractiveFrameWidget
from workers.cutie_worker import CutieWorker
from workers.sam_worker import SAMWorker


class FrameByFrameWidget(QWidget):
    """Main widget for frame-by-frame segmentation and tracking"""

    # Signals
    status_update = pyqtSignal(str)
    progress_update = pyqtSignal(int, str)  # progress, message

    def __init__(self):
        super().__init__()

        # State
        self.frames = []  # List of image arrays
        self.current_frame_index = 0
        self.frame_masks = {}  # Dict[frame_index] = masks
        self.cellsam_results = {}  # Store CellSAM results
        self.first_frame_segmented = False  # Track if CellSAM has run on first frame

        # CUTIE tracker for real-time tracking
        self.cutie_tracker = None

        # Workers
        self.sam_worker = None
        self.cutie_worker = None

        # Setup UI
        self.setup_ui()
        self.setup_shortcuts()

    def __del__(self):
        """Cleanup when widget is destroyed."""
        try:
            if self.cutie_tracker is not None:
                # Proper cleanup of tracker resources
                self.cutie_tracker.reset()
                self.cutie_tracker = None
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
        self.frames = []
        self.frame_masks = {}
        self.cellsam_results = {}
        self.first_frame_segmented = False

        # Reset tracker
        if self.cutie_tracker is not None:
            self.cutie_tracker.reset()
            self.cutie_tracker = None

        for path in image_paths:
            try:
                image = cv2.imread(path)
                if image is not None:
                    # Convert BGR to RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self.frames.append(image_rgb)
            except Exception as e:
                QMessageBox.warning(
                    self, "Load Error", f"Failed to load {path}: {str(e)}"
                )

        if self.frames:
            self.current_frame_index = 0
            self.update_display()
            self.status_update.emit(f"Loaded {len(self.frames)} frames")

        else:
            QMessageBox.warning(self, "Load Error", "No valid frames loaded")

    def load_frames_with_segmentation(
        self, image_paths: List[str], segmentation_results: List[dict]
    ):
        """Load frames with pre-computed segmentation results"""
        self.frames = []
        self.frame_masks = {}
        self.cellsam_results = {}
        self.first_frame_segmented = True  # Mark as already segmented

        # Reset tracker
        if self.cutie_tracker is not None:
            self.cutie_tracker.reset()
            self.cutie_tracker = None

        # Load frames from segmentation results (they include the original images)
        for i, result in enumerate(segmentation_results):
            try:
                # Use the original image from CellSAM results
                image_rgb = result["original_image"]
                self.frames.append(image_rgb)

                # Store the masks
                masks = result["masks"]
                if masks is not None and masks.size > 0:
                    self.frame_masks[i] = masks.astype(np.uint16)

                # Store full CellSAM results for reference
                self.cellsam_results[i] = result

            except Exception as e:
                QMessageBox.warning(
                    self, "Load Error", f"Failed to load frame {i+1}: {str(e)}"
                )

        if self.frames:
            self.current_frame_index = 0
            self.update_display()
            self.status_update.emit(
                f"Loaded {len(self.frames)} frames with segmentation"
            )
        else:
            QMessageBox.warning(self, "Load Error", "No valid frames loaded")

    def load_frames_with_first_segmentation(
        self, image_paths: List[str], first_frame_result: dict
    ):
        """Load frames with first frame segmentation, then run CUTIE tracking"""
        self.frames = []
        self.frame_masks = {}
        self.cellsam_results = {}
        self.first_frame_segmented = True
        self.cutie_worker = None

        # Reset tracker
        if self.cutie_tracker is not None:
            self.cutie_tracker.reset()
            self.cutie_tracker = None

        # Load all frames from paths
        for path in image_paths:
            try:
                image = cv2.imread(path)
                if image is not None:
                    # Convert BGR to RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self.frames.append(image_rgb)
            except Exception as e:
                QMessageBox.warning(
                    self, "Load Error", f"Failed to load {path}: {str(e)}"
                )

        if not self.frames:
            QMessageBox.warning(self, "Load Error", "No valid frames loaded")
            return

        # Store first frame segmentation results
        self.cellsam_results[0] = first_frame_result

        # Extract first frame masks
        first_frame_masks = first_frame_result["masks"]
        if first_frame_masks is not None and first_frame_masks.size > 0:
            self.frame_masks[0] = first_frame_masks.astype(np.uint16)
        else:
            QMessageBox.warning(
                self, "Load Error", "No masks found in first frame segmentation"
            )
            return

        self.current_frame_index = 0
        self.update_display()

        # Show status
        cell_count = np.max(first_frame_masks) if first_frame_masks.size > 0 else 0
        self.status_update.emit(
            f"Loaded {len(self.frames)} frames. Found {cell_count} cells in first frame. Starting tracking..."
        )

        # Start CUTIE tracking for remaining frames
        if len(self.frames) > 1:
            self.start_cutie_tracking()
        else:
            self.status_update.emit("Single frame loaded with segmentation")

    def start_cutie_tracking(self):
        """Start CUTIE tracking using first frame masks"""
        try:
            # Get first frame masks
            first_frame_masks = self.frame_masks[0]

            # Show progress
            self.status_update.emit("Initializing CUTIE tracking...")

            # Start CUTIE tracking worker
            self.cutie_worker = CutieWorker(self.frames, first_frame_masks)
            self.cutie_worker.progress_update.connect(self.on_cutie_progress)
            self.cutie_worker.tracking_complete.connect(self.on_cutie_complete)
            self.cutie_worker.error_occurred.connect(self.on_cutie_error)

            self.cutie_worker.start()

        except Exception as e:
            QMessageBox.critical(
                self, "Tracking Error", f"Failed to start CUTIE tracking: {str(e)}"
            )

    def on_cutie_progress(self, progress: int, status: str):
        """Handle CUTIE tracking progress"""
        self.status_update.emit(f"{status} ({progress}%)")

    def on_cutie_complete(self, tracking_results: dict):
        """Handle CUTIE tracking completion"""
        try:
            self.cutie_worker = None

            # Store tracking results
            for frame_idx, masks in tracking_results.items():
                if masks is not None and masks.size > 0:
                    self.frame_masks[frame_idx] = masks.astype(np.uint16)

            # Update display
            self.update_display()

            # Show completion status
            total_cells = (
                np.max(list(tracking_results.values())[0]) if tracking_results else 0
            )
            self.status_update.emit(
                f"Tracking complete! {total_cells} cells tracked across {len(self.frames)} frames"
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Tracking Error", f"Failed to process tracking results: {str(e)}"
            )

    def on_cutie_error(self, error_message: str):
        """Handle CUTIE tracking error"""
        self.cutie_worker = None
        QMessageBox.critical(self, "CUTIE Error", error_message)
        self.status_update.emit("Cell tracking failed")

    def update_display(self):
        """Update the image display"""
        if not self.frames:
            return

        # Update frame info
        total_frames = len(self.frames)
        self.frame_info_label.setText(
            f"Frame {self.current_frame_index + 1} / {total_frames}"
        )

        # Update navigation buttons
        self.prev_button.setEnabled(self.current_frame_index > 0)
        self.next_button.setEnabled(self.current_frame_index < total_frames - 1)
        # Note: auto_segment_button is hidden and not enabled for manual use

        # Update current frame display (right side - editable, no cell IDs for clarity)
        current_image = self.frames[self.current_frame_index]
        self.curr_image_label.set_image(current_image)

        # Set current frame masks if available
        current_masks = self.frame_masks.get(self.current_frame_index)
        self.curr_image_label.set_masks(current_masks)

        # Update previous frame display (left side - shows cell IDs)
        if self.current_frame_index > 0:
            prev_image = self.frames[self.current_frame_index - 1]
            self.prev_image_label.set_image(prev_image)

            # Set previous frame masks if available
            prev_masks = self.frame_masks.get(self.current_frame_index - 1)
            self.prev_image_label.set_masks(prev_masks)
        else:
            # First frame - no previous frame
            self.prev_image_label.clear()
            self.prev_image_label.setText("First Frame")

    def set_annotation_mode(self, mode: AnnotationMode):
        """Set the annotation mode"""
        self.curr_image_label.set_annotation_mode(mode)

    def previous_frame(self):
        """Go to previous frame"""
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
            self.update_display()

    def next_frame(self):
        """Go to next frame and run tracking if needed"""
        if self.current_frame_index < len(self.frames) - 1:
            next_index = self.current_frame_index + 1

            # If we have a tracker and this is a new frame, run tracking
            if (
                self.cutie_tracker is not None
                and next_index not in self.frame_masks
                and next_index > 0
            ):

                try:
                    # Get the next frame
                    next_frame = self.frames[next_index]

                    # Use current frame mask as reference if available
                    current_mask = self.frame_masks.get(self.current_frame_index)

                    # Run tracking step
                    self.status_update.emit(f"Tracking frame {next_index + 1}...")

                    if current_mask is not None:
                        # Step with reference mask from current frame
                        predicted_mask = self.cutie_tracker.step(
                            next_frame, current_mask
                        )

                    else:
                        # No current mask available - cannot track without reference
                        raise RuntimeError(
                            f"Cannot track frame {next_index + 1}: no reference mask available from frame {self.current_frame_index + 1}"
                        )

                    # Store the prediction
                    self.frame_masks[next_index] = predicted_mask

                    self.status_update.emit(f"Tracked frame {next_index + 1}")

                except Exception as e:
                    QMessageBox.warning(
                        self,
                        "Tracking Error",
                        f"Failed to track frame {next_index + 1}: {str(e)}",
                    )

            # Move to next frame
            self.current_frame_index = next_index
            self.update_display()

    def on_point_clicked(self, point: Tuple[int, int]):
        """Handle point click for SAM segmentation"""
        if not self.frames:
            return

        current_image = self.frames[self.current_frame_index]

        # Start SAM worker
        if self.sam_worker is not None:
            return

        self.sam_worker = SAMWorker(current_image, "point", point=point)
        self.sam_worker.sam_complete.connect(self.on_sam_complete)
        self.sam_worker.error_occurred.connect(self.on_sam_error)
        self.sam_worker.start()

        self.status_update.emit(f"Running SAM on point {point}...")

    def on_box_drawn(self, box: Tuple[int, int, int, int]):
        """Handle box drawing for SAM segmentation"""
        if not self.frames:
            return

        current_image = self.frames[self.current_frame_index]

        # Start SAM worker
        if self.sam_worker is not None:
            return

        self.sam_worker = SAMWorker(current_image, "box", box=box)
        self.sam_worker.sam_complete.connect(self.on_sam_complete)
        self.sam_worker.error_occurred.connect(self.on_sam_error)
        self.sam_worker.start()

        self.status_update.emit(f"Running SAM on box {box}...")

    def on_mask_clicked(self, point: Tuple[int, int]):
        """Handle mask removal"""
        current_masks = self.frame_masks.get(self.current_frame_index)
        if current_masks is None:
            return

        x, y = point
        if 0 <= y < current_masks.shape[0] and 0 <= x < current_masks.shape[1]:
            mask_id = current_masks[y, x]
            if mask_id > 0:
                # Remove this mask
                current_masks[current_masks == mask_id] = 0
                self.frame_masks[self.current_frame_index] = current_masks
                self.curr_image_label.set_masks(current_masks)

                self.status_update.emit(f"Removed mask {mask_id}")

    def on_sam_complete(self, mask: np.ndarray, score: float):
        """Handle SAM completion"""
        self.sam_worker = None

        # Add mask to current frame
        current_masks = self.frame_masks.get(self.current_frame_index)
        if current_masks is None:
            # Create new mask array
            h, w = self.frames[self.current_frame_index].shape[:2]
            current_masks = np.zeros((h, w), dtype=np.uint16)

        # Find next available mask ID
        next_id = np.max(current_masks) + 1

        # Add new mask
        current_masks[mask > 0] = next_id
        self.frame_masks[self.current_frame_index] = current_masks
        self.curr_image_label.set_masks(current_masks)

        self.status_update.emit(f"Added mask {next_id} (score: {score:.3f})")

    def on_sam_error(self, error_message: str):
        """Handle SAM error"""
        self.sam_worker = None
        self.status_update.emit("SAM operation failed")
        QMessageBox.warning(self, "SAM Error", error_message)

    def on_cell_id_edit_requested(self, point: Tuple[int, int], current_cell_id: int):
        """Handle cell ID editing request"""
        from PyQt6.QtWidgets import QInputDialog

        x, y = point

        if current_cell_id == 0:
            QMessageBox.information(
                self, "Edit Cell ID", "No cell at this location to edit."
            )
            return

        # Get new cell ID from user
        new_id, ok = QInputDialog.getInt(
            self,
            "Edit Cell ID",
            f"Enter new ID for cell {current_cell_id}:",
            value=current_cell_id,
            min=1,
            max=9999,
        )

        if ok and new_id != current_cell_id:
            current_masks = self.frame_masks.get(self.current_frame_index)
            if current_masks is not None:
                # Check if new ID already exists
                if new_id in current_masks:
                    reply = QMessageBox.question(
                        self,
                        "ID Conflict",
                        f"Cell ID {new_id} already exists. Do you want to merge the cells?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    )
                    if reply == QMessageBox.StandardButton.No:
                        return

                # Update the cell ID
                current_masks[current_masks == current_cell_id] = new_id
                self.frame_masks[self.current_frame_index] = current_masks
                self.curr_image_label.set_masks(current_masks)

                self.status_update.emit(
                    f"Changed cell ID from {current_cell_id} to {new_id}"
                )

    def get_current_masks(self) -> Optional[np.ndarray]:
        """Get masks for current frame"""
        return self.frame_masks.get(self.current_frame_index)

    def get_all_masks(self) -> Dict[int, np.ndarray]:
        """Get all frame masks"""
        return self.frame_masks.copy()
