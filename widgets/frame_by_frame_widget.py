"""
Frame-by-frame segmentation and tracking widget
"""

import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

# Add project paths
current_dir = Path(__file__).parent.parent.parent
sam_path = current_dir / "sam"
cutie_path = current_dir / "cutie"
sys.path.insert(0, str(sam_path.resolve()))
sys.path.insert(0, str(cutie_path.resolve()))


class AnnotationMode(Enum):
    """Annotation mode enumeration"""

    VIEW = "view"
    CLICK_ADD = "click_add"
    BOX_ADD = "box_add"
    MASK_REMOVE = "mask_remove"


class CellSAMWorker(QThread):
    """Worker thread for CellSAM segmentation"""

    progress_update = pyqtSignal(str)  # status message
    segmentation_complete = pyqtSignal(dict)  # results
    error_occurred = pyqtSignal(str)  # error message

    def __init__(self, image: np.ndarray, parameters: Dict[str, Any]):
        super().__init__()
        self.image = image
        self.parameters = parameters
        self._cancelled = False

    def cancel(self):
        """Cancel the segmentation"""
        self._cancelled = True

    def run(self):
        """Run CellSAM segmentation in background thread"""
        try:
            self.progress_update.emit("Initializing CellSAM...")

            # Import CellSAM
            from sam import CellSAM

            # Initialize model
            device = self.parameters.get("device", "auto")
            if device == "auto":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = CellSAM(device=device)

            if self._cancelled:
                return

            self.progress_update.emit("Running segmentation...")

            # Run segmentation
            masks, flows, styles = model.segment(
                self.image,
                diameter=self.parameters.get("diameter", 30.0),
                flow_threshold=self.parameters.get("flow_threshold", 0.4),
                cellprob_threshold=self.parameters.get("cellprob_threshold", 0.0),
                normalize=True,
            )

            if self._cancelled:
                return

            # Create results
            results = {
                "masks": masks,
                "flows": flows,
                "styles": styles,
                "cell_count": np.max(masks) if masks.size > 0 else 0,
                "parameters": self.parameters.copy(),
            }

            self.progress_update.emit("Segmentation completed")
            self.segmentation_complete.emit(results)

        except Exception as e:
            self.error_occurred.emit(f"Segmentation failed: {str(e)}")


class SAMWorker(QThread):
    """Worker thread for SAM operations"""

    sam_complete = pyqtSignal(np.ndarray, float)  # mask, score
    error_occurred = pyqtSignal(str)  # error message

    def __init__(self, image: np.ndarray, operation: str, **kwargs):
        super().__init__()
        self.image = image
        self.operation = operation
        self.kwargs = kwargs
        self._cancelled = False

    def cancel(self):
        """Cancel the operation"""
        self._cancelled = True

    def run(self):
        """Run SAM operation in background thread"""
        try:
            # Import SAM
            from sam import SAM

            # Initialize SAM
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            sam = SAM(device=device)

            if self._cancelled:
                return

            if self.operation == "point":
                point = self.kwargs["point"]
                mask, score, _ = sam.segment_point(self.image, [point])
            elif self.operation == "box":
                box = self.kwargs["box"]
                mask, score, _ = sam.segment_box(self.image, box)
            else:
                raise ValueError(f"Unknown operation: {self.operation}")

            if not self._cancelled:
                self.sam_complete.emit(mask, score)

        except Exception as e:
            self.error_occurred.emit(f"SAM operation failed: {str(e)}")


class InteractiveImageLabel(QLabel):
    """Interactive image label for annotation"""

    point_clicked = pyqtSignal(tuple)  # (x, y)
    box_drawn = pyqtSignal(tuple)  # (x1, y1, x2, y2)
    mask_clicked = pyqtSignal(tuple)  # (x, y) for mask removal

    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 400)
        self.setStyleSheet("border: 1px solid #606060; background-color: #353535;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setScaledContents(False)

        # State
        self.annotation_mode = AnnotationMode.VIEW
        self.image = None
        self.masks = None
        self.overlay_image = None
        self.scale_factor = 1.0
        self.image_offset = (0, 0)

        # Box drawing
        self.drawing_box = False
        self.box_start = None
        self.box_end = None

    def set_annotation_mode(self, mode: AnnotationMode):
        """Set the annotation mode"""
        self.annotation_mode = mode

        if mode == AnnotationMode.VIEW:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        elif mode in [AnnotationMode.CLICK_ADD, AnnotationMode.MASK_REMOVE]:
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif mode == AnnotationMode.BOX_ADD:
            self.setCursor(Qt.CursorShape.CrossCursor)

    def set_image(self, image: np.ndarray):
        """Set the base image"""
        self.image = image.copy()
        self.update_display()

    def set_masks(self, masks: np.ndarray):
        """Set the segmentation masks"""
        self.masks = masks.copy() if masks is not None else None
        self.update_display()

    def update_display(self):
        """Update the display with image and masks"""
        if self.image is None:
            self.clear()
            return

        # Create display image
        display_image = self.image.copy()

        # Overlay masks if available
        if self.masks is not None:
            display_image = self._overlay_masks(display_image, self.masks)

        # Convert to QPixmap and display
        h, w = display_image.shape[:2]
        bytes_per_line = 3 * w
        q_image = QImage(
            display_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )

        # Scale to fit widget
        widget_size = self.size()
        pixmap = QPixmap.fromImage(q_image)

        # Calculate scale factor to fit in widget while maintaining aspect ratio
        scale_x = widget_size.width() / w
        scale_y = widget_size.height() / h
        self.scale_factor = min(scale_x, scale_y, 1.0)  # Don't scale up

        if self.scale_factor < 1.0:
            scaled_pixmap = pixmap.scaled(
                int(w * self.scale_factor),
                int(h * self.scale_factor),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        else:
            scaled_pixmap = pixmap

        # Calculate offset to center the image
        self.image_offset = (
            (widget_size.width() - scaled_pixmap.width()) // 2,
            (widget_size.height() - scaled_pixmap.height()) // 2,
        )

        self.setPixmap(scaled_pixmap)

    def _overlay_masks(self, image: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """Overlay masks on image with transparency"""
        if masks is None or np.max(masks) == 0:
            return image

        overlay = image.copy()

        # Generate colors for each mask
        num_objects = int(np.max(masks))
        colors = self._generate_colors(num_objects)

        for obj_id in range(1, num_objects + 1):
            mask = masks == obj_id
            if np.any(mask):
                color = np.array(colors[obj_id - 1], dtype=np.uint8)
                overlay[mask] = (0.7 * overlay[mask] + 0.3 * color).astype(np.uint8)

        return overlay

    def _generate_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for masks"""
        colors = []
        for i in range(num_colors):
            hue = (i * 137.508) % 360  # Golden angle approximation
            # Convert HSV to RGB (simplified)
            c = 1.0
            x = c * (1 - abs((hue / 60) % 2 - 1))
            m = 0

            if 0 <= hue < 60:
                r, g, b = c, x, 0
            elif 60 <= hue < 120:
                r, g, b = x, c, 0
            elif 120 <= hue < 180:
                r, g, b = 0, c, x
            elif 180 <= hue < 240:
                r, g, b = 0, x, c
            elif 240 <= hue < 300:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x

            colors.append((int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)))

        return colors

    def _widget_to_image_coords(self, widget_x: int, widget_y: int) -> Tuple[int, int]:
        """Convert widget coordinates to image coordinates"""
        if self.image is None:
            return (0, 0)

        # Account for image offset and scaling
        image_x = int((widget_x - self.image_offset[0]) / self.scale_factor)
        image_y = int((widget_y - self.image_offset[1]) / self.scale_factor)

        # Clamp to image bounds
        image_x = max(0, min(image_x, self.image.shape[1] - 1))
        image_y = max(0, min(image_y, self.image.shape[0] - 1))

        return (image_x, image_y)

    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if self.annotation_mode == AnnotationMode.VIEW:
            return

        pos = event.position().toPoint()
        image_coords = self._widget_to_image_coords(pos.x(), pos.y())

        if self.annotation_mode == AnnotationMode.CLICK_ADD:
            self.point_clicked.emit(image_coords)

        elif self.annotation_mode == AnnotationMode.BOX_ADD:
            if event.button() == Qt.MouseButton.LeftButton:
                self.drawing_box = True
                self.box_start = image_coords
                self.box_end = image_coords

        elif self.annotation_mode == AnnotationMode.MASK_REMOVE:
            self.mask_clicked.emit(image_coords)

    def mouseMoveEvent(self, event):
        """Handle mouse move events"""
        if self.annotation_mode == AnnotationMode.BOX_ADD and self.drawing_box:
            pos = event.position().toPoint()
            self.box_end = self._widget_to_image_coords(pos.x(), pos.y())
            self.update()  # Trigger repaint to show box

    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if self.annotation_mode == AnnotationMode.BOX_ADD and self.drawing_box:
            self.drawing_box = False
            if self.box_start and self.box_end:
                x1, y1 = self.box_start
                x2, y2 = self.box_end
                # Ensure proper box format (x1 < x2, y1 < y2)
                box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                self.box_drawn.emit(box)
            self.box_start = None
            self.box_end = None
            self.update()

    def paintEvent(self, event):
        """Custom paint event to draw box during drawing"""
        super().paintEvent(event)

        if (
            self.annotation_mode == AnnotationMode.BOX_ADD
            and self.drawing_box
            and self.box_start
            and self.box_end
        ):

            painter = QPainter(self)
            painter.setPen(QPen(QColor(0, 120, 212), 2, Qt.PenStyle.DashLine))

            # Convert image coordinates back to widget coordinates
            x1 = int(self.box_start[0] * self.scale_factor + self.image_offset[0])
            y1 = int(self.box_start[1] * self.scale_factor + self.image_offset[1])
            x2 = int(self.box_end[0] * self.scale_factor + self.image_offset[0])
            y2 = int(self.box_end[1] * self.scale_factor + self.image_offset[1])

            painter.drawRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))


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

        # Workers
        self.cellsam_worker = None
        self.sam_worker = None

        # Setup UI
        self.setup_ui()
        self.setup_shortcuts()

    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Control panel
        self.setup_control_panel(layout)

        # Image panel
        self.setup_image_panel(layout)

        # Tool panel
        self.setup_tool_panel(layout)

        # Progress panel
        self.setup_progress_panel(layout)

    def setup_control_panel(self, parent_layout):
        """Setup frame navigation controls"""
        control_group = QGroupBox("Frame Navigation")
        layout = QHBoxLayout(control_group)

        # Frame info
        self.frame_info_label = QLabel("No frames loaded")
        layout.addWidget(self.frame_info_label)

        layout.addStretch()

        # Navigation buttons
        self.prev_button = QPushButton("◀ Previous (A)")
        self.prev_button.clicked.connect(self.previous_frame)
        self.prev_button.setEnabled(False)
        layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next (D) ▶")
        self.next_button.clicked.connect(self.next_frame)
        self.next_button.setEnabled(False)
        layout.addWidget(self.next_button)

        # Auto-segment button (hidden from user - only runs automatically on first frame)
        self.auto_segment_button = QPushButton("Auto Segment (S)")
        self.auto_segment_button.clicked.connect(self.run_cellsam)
        self.auto_segment_button.setEnabled(False)
        self.auto_segment_button.setVisible(False)  # Hide from user interface
        layout.addWidget(self.auto_segment_button)

        parent_layout.addWidget(control_group)

    def setup_image_panel(self, parent_layout):
        """Setup dual image display"""
        image_group = QGroupBox("Current Frame vs Previous Frame")
        layout = QHBoxLayout(image_group)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Previous frame panel
        prev_panel = QWidget()
        prev_layout = QVBoxLayout(prev_panel)
        prev_layout.addWidget(QLabel("Previous Frame:"))
        self.prev_image_label = InteractiveImageLabel()
        self.prev_image_label.setEnabled(False)  # Non-interactive
        prev_layout.addWidget(self.prev_image_label)
        splitter.addWidget(prev_panel)

        # Current frame panel
        curr_panel = QWidget()
        curr_layout = QVBoxLayout(curr_panel)
        curr_layout.addWidget(QLabel("Current Frame:"))
        self.curr_image_label = InteractiveImageLabel()
        self.curr_image_label.point_clicked.connect(self.on_point_clicked)
        self.curr_image_label.box_drawn.connect(self.on_box_drawn)
        self.curr_image_label.mask_clicked.connect(self.on_mask_clicked)
        curr_layout.addWidget(self.curr_image_label)
        splitter.addWidget(curr_panel)

        # Set equal sizes
        splitter.setSizes([400, 400])

        parent_layout.addWidget(image_group)

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

        layout.addLayout(mode_layout)

        # CellSAM parameters
        param_layout = QHBoxLayout()

        param_layout.addWidget(QLabel("Diameter:"))
        self.diameter_spin = QDoubleSpinBox()
        self.diameter_spin.setRange(10, 100)
        self.diameter_spin.setValue(30)
        self.diameter_spin.setSuffix(" px")
        param_layout.addWidget(self.diameter_spin)

        param_layout.addWidget(QLabel("Flow Threshold:"))
        self.flow_threshold_spin = QDoubleSpinBox()
        self.flow_threshold_spin.setRange(0.1, 1.0)
        self.flow_threshold_spin.setValue(0.4)
        self.flow_threshold_spin.setSingleStep(0.1)
        param_layout.addWidget(self.flow_threshold_spin)

        param_layout.addStretch()
        layout.addLayout(param_layout)

        parent_layout.addWidget(tool_group)

    def setup_progress_panel(self, parent_layout):
        """Setup progress display"""
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        parent_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        parent_layout.addWidget(self.status_label)

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

        # Note: Removed 'S' shortcut for CellSAM as it now only runs automatically on first frame

    def load_frames(self, image_paths: List[str]):
        """Load frames from image paths"""
        self.frames = []
        self.frame_masks = {}
        self.cellsam_results = {}
        self.first_frame_segmented = False

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

            # Automatically run CellSAM on the first frame
            self.auto_run_cellsam_on_first_frame()
        else:
            QMessageBox.warning(self, "Load Error", "No valid frames loaded")

    def auto_run_cellsam_on_first_frame(self):
        """Automatically run CellSAM on the first frame"""
        if not self.first_frame_segmented and self.current_frame_index == 0:
            self.status_update.emit("Running automatic segmentation on first frame...")
            # Set flag to prevent running again
            self.first_frame_segmented = True
            # Run CellSAM on the first frame
            self.run_cellsam()

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

        # Update current frame display
        current_image = self.frames[self.current_frame_index]
        self.curr_image_label.set_image(current_image)

        # Set current frame masks if available
        current_masks = self.frame_masks.get(self.current_frame_index)
        self.curr_image_label.set_masks(current_masks)

        # Update previous frame display
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
        """Go to next frame"""
        if self.current_frame_index < len(self.frames) - 1:
            self.current_frame_index += 1
            self.update_display()

    def run_cellsam(self):
        """Run CellSAM segmentation on current frame (only allowed on first frame)"""
        if not self.frames or self.cellsam_worker is not None:
            return

        # Only allow CellSAM on the first frame
        if self.current_frame_index != 0:
            QMessageBox.information(
                self,
                "CellSAM Restriction",
                "CellSAM automatic segmentation only runs on the first frame.\n"
                "Use manual SAM tools for corrections on subsequent frames.",
            )
            return

        current_image = self.frames[self.current_frame_index]

        # Get parameters
        parameters = {
            "diameter": self.diameter_spin.value(),
            "flow_threshold": self.flow_threshold_spin.value(),
            "cellprob_threshold": 0.0,
            "device": "auto",
        }

        # Start worker
        self.cellsam_worker = CellSAMWorker(current_image, parameters)
        self.cellsam_worker.progress_update.connect(self.on_cellsam_progress)
        self.cellsam_worker.segmentation_complete.connect(self.on_cellsam_complete)
        self.cellsam_worker.error_occurred.connect(self.on_cellsam_error)
        self.cellsam_worker.start()

        # Update UI
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        # Note: auto_segment_button is already disabled and hidden

    def on_cellsam_progress(self, message: str):
        """Handle CellSAM progress updates"""
        self.status_label.setText(message)

    def on_cellsam_complete(self, results: dict):
        """Handle CellSAM completion"""
        self.cellsam_worker = None
        self.progress_bar.setVisible(False)
        # Note: auto_segment_button remains disabled and hidden

        # Store results
        self.cellsam_results[self.current_frame_index] = results
        self.frame_masks[self.current_frame_index] = results["masks"]

        # Update display
        self.curr_image_label.set_masks(results["masks"])

        cell_count = results["cell_count"]
        self.status_label.setText(f"CellSAM completed: {cell_count} cells detected")

    def on_cellsam_error(self, error_message: str):
        """Handle CellSAM error"""
        self.cellsam_worker = None
        self.progress_bar.setVisible(False)
        # Note: auto_segment_button remains disabled and hidden

        self.status_label.setText("CellSAM failed")
        QMessageBox.critical(self, "CellSAM Error", error_message)

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

        self.status_label.setText(f"Running SAM on point {point}...")

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

        self.status_label.setText(f"Running SAM on box {box}...")

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

                self.status_label.setText(f"Removed mask {mask_id}")

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

        self.status_label.setText(f"Added mask {next_id} (score: {score:.3f})")

    def on_sam_error(self, error_message: str):
        """Handle SAM error"""
        self.sam_worker = None
        self.status_label.setText("SAM operation failed")
        QMessageBox.warning(self, "SAM Error", error_message)

    def get_current_masks(self) -> Optional[np.ndarray]:
        """Get masks for current frame"""
        return self.frame_masks.get(self.current_frame_index)

    def get_all_masks(self) -> Dict[int, np.ndarray]:
        """Get all frame masks"""
        return self.frame_masks.copy()
