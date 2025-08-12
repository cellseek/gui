"""
Segmentation panel for CellSAM and SAM integration
Enhanced with manual annotation capabilities
"""

import colorsys
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PyQt6.QtCore import QPoint, QRect, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QCursor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

# Add project paths
current_dir = Path(__file__).parent.parent.parent
sam_path = current_dir / "sam"
sys.path.insert(0, str(sam_path.resolve()))


class AnnotationMode(Enum):
    """Annotation mode enumeration"""

    VIEW = "view"
    CLICK_ADD = "click_add"
    BOX_ADD = "box_add"
    MASK_REMOVE = "mask_remove"


class SegmentationWorker(QThread):
    """Worker thread for running CellSAM segmentation"""

    progress_update = pyqtSignal(str)  # status message
    segmentation_complete = pyqtSignal(dict)  # results
    error_occurred = pyqtSignal(str)  # error message

    def __init__(
        self, image: np.ndarray, parameters: Dict[str, Any], preloaded_model=None
    ):
        super().__init__()
        self.image = image
        self.parameters = parameters
        self.preloaded_model = preloaded_model  # Use preloaded CellSAM if available
        self._cancelled = False

    def cancel(self):
        """Cancel the segmentation"""
        self._cancelled = True

    def run(self):
        """Run CellSAM segmentation in background thread"""
        try:
            self.progress_update.emit("Initializing CellSAM...")

            # Use preloaded model if available, otherwise create new one
            if self.preloaded_model is not None:
                model = self.preloaded_model
                self.progress_update.emit("Using preloaded CellSAM model...")
            else:
                # Import CellSAM
                from sam import CellSAM

                # Initialize model
                device = self.parameters.get("device", "auto")
                if device == "auto":
                    device = torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )

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


class SAMPreviewWorker(QThread):
    """Worker thread for quick SAM preview on hover"""

    preview_ready = pyqtSignal(np.ndarray, tuple, float)  # preview mask, point, score
    error_occurred = pyqtSignal(str)  # error message

    def __init__(
        self,
        image: np.ndarray,
        point: Tuple[int, int],
        preloaded_sam=None,
        cache_dict=None,  # Reference to cache dictionary
    ):
        super().__init__()
        self.image = image
        self.point = point
        self.preloaded_sam = preloaded_sam
        self.cache_dict = cache_dict  # Cache to store results
        self._cancelled = False

    def cancel(self):
        """Cancel the preview"""
        self._cancelled = True

    def run(self):
        """Run quick SAM preview"""
        try:
            if self._cancelled or self.preloaded_sam is None:
                return

            # Check cache first
            if self.cache_dict is not None and self.point in self.cache_dict:
                cached_mask, cached_score, _ = self.cache_dict[self.point]
                if not self._cancelled:
                    self.preview_ready.emit(cached_mask, self.point, cached_score)
                return

            predictor = self.preloaded_sam

            # Quick prediction with single point
            input_point = np.array([self.point])
            input_label = np.array([1])  # Positive point

            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,  # Single mask for speed
            )

            if self._cancelled:
                return

            # Use the best mask
            if len(masks) > 0:
                best_mask = masks[0]
                best_score = scores[0]

                # Cache the result for reuse
                if self.cache_dict is not None:
                    self.cache_dict[self.point] = (best_mask, best_score, logits[0])

                self.preview_ready.emit(best_mask, self.point, best_score)

        except Exception as e:
            if not self._cancelled:
                self.error_occurred.emit(f"Preview failed: {str(e)}")


class SAMWorker(QThread):
    """Worker thread for running SAM annotation"""

    progress_update = pyqtSignal(str)  # status message
    annotation_complete = pyqtSignal(np.ndarray)  # updated masks
    error_occurred = pyqtSignal(str)  # error message

    def __init__(
        self,
        image: np.ndarray,
        points: List[Tuple[int, int]],
        point_labels: List[int],
        boxes: List[Tuple[int, int, int, int]],
        existing_masks: Optional[np.ndarray] = None,
        sam_checkpoint: Optional[str] = None,
        model_type: str = "vit_h",
        preloaded_sam=None,  # Preloaded SAM predictor
        cache_dict=None,  # Cache to check for existing predictions
    ):
        super().__init__()
        self.image = image
        self.points = points
        self.point_labels = point_labels  # 1 for positive, 0 for negative
        self.boxes = boxes
        self.existing_masks = existing_masks
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type
        self.preloaded_sam = preloaded_sam  # Use preloaded SAM if available
        self.cache_dict = cache_dict  # Cache to check for existing predictions
        self._cancelled = False

    def cancel(self):
        """Cancel the annotation"""
        self._cancelled = True

    def run(self):
        """Run SAM annotation in background thread"""
        try:
            self.progress_update.emit("Initializing SAM...")

            # Use preloaded SAM if available
            if self.preloaded_sam is not None:
                predictor = self.preloaded_sam
                self.progress_update.emit("Using preloaded SAM model...")
            else:
                # Import SAM
                try:
                    from segment_anything import SamPredictor, sam_model_registry
                except ImportError:
                    raise ImportError(
                        "segment-anything library not installed. Please install with: pip install segment-anything"
                    )

                # Load SAM model
                if self.sam_checkpoint and Path(self.sam_checkpoint).exists():
                    sam_checkpoint = self.sam_checkpoint
                else:
                    # Look for SAM checkpoint in checkpoints folder
                    possible_paths = [
                        "checkpoints/sam_vit_h_4b8939.pth",
                        "checkpoints/sam_vit_l_0b3195.pth",
                        "checkpoints/sam_vit_b_01ec64.pth",
                        "sam_vit_h_4b8939.pth",
                        "sam_vit_l_0b3195.pth",
                        "sam_vit_b_01ec64.pth",
                    ]

                    sam_checkpoint = None
                    for path in possible_paths:
                        if Path(path).exists():
                            sam_checkpoint = path
                            # Adjust model type based on checkpoint
                            if "vit_l" in path:
                                self.model_type = "vit_l"
                            elif "vit_b" in path:
                                self.model_type = "vit_b"
                            elif "vit_h" in path:
                                self.model_type = "vit_h"
                            break

                    if not sam_checkpoint:
                        raise FileNotFoundError(
                            "SAM checkpoint not found in checkpoints folder. Please download a SAM model checkpoint:\n"
                            "- vit_h: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth\n"
                            "- vit_l: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth\n"
                            "- vit_b: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth\n"
                            "and place it in the 'checkpoints' folder."
                        )

                device = "cuda" if torch.cuda.is_available() else "cpu"

                sam = sam_model_registry[self.model_type](checkpoint=sam_checkpoint)
                sam.to(device=device)
                predictor = SamPredictor(sam)

            self.progress_update.emit("Setting image...")

            # Set the image for SAM
            predictor.set_image(self.image)

            if self._cancelled:
                return

            self.progress_update.emit("Running SAM annotation...")

            # Run SAM annotation
            updated_masks = self._run_sam_annotation(predictor)

            self.progress_update.emit("Annotation completed")
            self.annotation_complete.emit(updated_masks)

        except Exception as e:
            self.error_occurred.emit(f"SAM annotation failed: {str(e)}")

    def _run_sam_annotation(self, predictor) -> np.ndarray:
        """Run SAM annotation using the real SAM model - one prediction at a time"""
        if self.existing_masks is not None:
            masks = self.existing_masks.copy()
        else:
            masks = np.zeros(self.image.shape[:2], dtype=np.uint16)

        # Get next mask ID
        next_id = np.max(masks) + 1 if masks.size > 0 and np.max(masks) > 0 else 1

        # Process each point individually
        for point, label in zip(self.points, self.point_labels):
            # Check cache first to avoid recomputation
            if self.cache_dict and point in self.cache_dict:
                cached_mask, cached_score, _ = self.cache_dict[point]
                # Use cached result if it's good quality
                if cached_score > 0.5:
                    new_mask = cached_mask

                    # Simple overlap check - only add if it's mostly in empty space
                    existing_overlap = np.sum(new_mask & (masks > 0))
                    new_mask_size = np.sum(new_mask)

                    if (
                        new_mask_size > 0 and existing_overlap / new_mask_size < 0.5
                    ):  # Less than 50% overlap
                        # Remove any overlapping pixels and add the rest
                        clean_mask = new_mask & (masks == 0)
                        if np.sum(clean_mask) > 10:  # Minimum size threshold
                            masks[clean_mask] = next_id
                            next_id += 1
                    continue  # Skip SAM prediction since we used cache

            # Run SAM prediction for single point (only if not in cache)
            sam_masks, scores, logits = predictor.predict(
                point_coords=np.array([point]),
                point_labels=np.array([label]),
                multimask_output=False,  # Single mask output - no overlaps!
            )

            # Add the mask if it's good quality and doesn't completely overlap existing masks
            if len(sam_masks) > 0 and scores[0] > 0.5:
                new_mask = sam_masks[0]

                # Cache the result for future use
                if self.cache_dict is not None:
                    self.cache_dict[point] = (new_mask, scores[0], logits[0])

                # Simple overlap check - only add if it's mostly in empty space
                existing_overlap = np.sum(new_mask & (masks > 0))
                new_mask_size = np.sum(new_mask)

                if existing_overlap / new_mask_size < 0.5:  # Less than 50% overlap
                    # Remove any overlapping pixels and add the rest
                    clean_mask = new_mask & (masks == 0)
                    if np.sum(clean_mask) > 10:  # Minimum size threshold
                        masks[clean_mask] = next_id
                        next_id += 1

        # Process each box individually
        for box in self.boxes:
            # Run SAM prediction for single box
            sam_masks, scores, logits = predictor.predict(
                box=np.array(box),
                multimask_output=False,  # Single mask output - no overlaps!
            )

            # Add the mask if it's good quality and doesn't completely overlap existing masks
            if len(sam_masks) > 0 and scores[0] > 0.5:
                new_mask = sam_masks[0]

                # Simple overlap check - only add if it's mostly in empty space
                existing_overlap = np.sum(new_mask & (masks > 0))
                new_mask_size = np.sum(new_mask)

                if existing_overlap / new_mask_size < 0.5:  # Less than 50% overlap
                    # Remove any overlapping pixels and add the rest
                    clean_mask = new_mask & (masks == 0)
                    if np.sum(clean_mask) > 10:  # Minimum size threshold
                        masks[clean_mask] = next_id
                        next_id += 1

        return masks


class ImageDisplayWidget(QLabel):
    """Enhanced widget for displaying images with interactive annotation"""

    clicked = pyqtSignal(int, int)  # x, y coordinates
    box_selected = pyqtSignal(int, int, int, int)  # x1, y1, x2, y2
    annotation_changed = pyqtSignal()  # annotation state changed
    hover_preview_requested = pyqtSignal(int, int)  # x, y coordinates for preview

    def __init__(self):
        super().__init__()
        self.original_image: Optional[np.ndarray] = None
        self.masks: Optional[np.ndarray] = None
        self.show_masks = True
        self.mask_alpha = 0.5
        self.selected_cells = set()

        # Annotation state
        self.annotation_mode = AnnotationMode.VIEW
        self.click_points = []  # List of (x, y) tuples for clicking
        self.annotation_boxes = []  # List of (x1, y1, x2, y2) tuples

        # Color mapping for consistent mask colors
        self.cell_colors = {}  # Map from cell_id to color

        # Box selection state
        self.drawing_box = False
        self.box_start_point = None
        self.current_box = None

        # Hover preview state
        self.show_hover_preview = False
        self.hover_preview_mask: Optional[np.ndarray] = None
        self.hover_timer = None

        # Set up widget
        self.setMinimumSize(400, 400)
        self.setStyleSheet(
            """
            QLabel {
                border: 1px solid #606060;
                background-color: #353535;
            }
        """
        )
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setScaledContents(False)

        # Default text
        self.setText("No image loaded")

    def set_annotation_mode(self, mode: AnnotationMode):
        """Set the current annotation mode"""
        self.annotation_mode = mode

        # Update cursor based on mode
        if mode == AnnotationMode.CLICK_ADD:
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
            self.show_hover_preview = True
            self.setMouseTracking(True)  # Enable mouse tracking for hover
        elif mode == AnnotationMode.MASK_REMOVE:
            self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            self.show_hover_preview = False
            self.setMouseTracking(False)
        elif mode == AnnotationMode.BOX_ADD:
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
            self.show_hover_preview = False
            self.setMouseTracking(False)
        else:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            self.show_hover_preview = False
            self.setMouseTracking(False)

        # Clear any existing hover preview
        self.hover_preview_mask = None
        self.update_display()

    def set_hover_preview_mask(self, mask: Optional[np.ndarray]):
        """Set the hover preview mask"""
        self.hover_preview_mask = mask
        self.update_display()

    def clear_hover_preview(self):
        """Clear the hover preview mask"""
        self.hover_preview_mask = None
        self.update_display()

    def clear_annotations(self):
        """Clear all annotation points and boxes"""
        self.click_points.clear()
        self.annotation_boxes.clear()
        self.current_box = None
        self.drawing_box = False
        self.update_display()
        self.annotation_changed.emit()

    def get_annotation_data(self) -> Dict[str, Any]:
        """Get current annotation data"""
        return {
            "click_points": self.click_points.copy(),
            "boxes": self.annotation_boxes.copy(),
        }

    def set_image(self, image: np.ndarray):
        """Set the image to display"""
        if image is None:
            return

        if image.size == 0:
            return

        self.original_image = image.copy()
        self.masks = None
        self.selected_cells.clear()
        self.cell_colors.clear()  # Reset color mapping for new image
        self.clear_annotations()
        self.update_display()

    def set_masks(self, masks: np.ndarray):
        """Set segmentation masks"""
        self.masks = masks

        # Generate colors for new cell IDs
        if masks is not None:
            unique_ids = np.unique(masks[masks > 0])
            for cell_id in unique_ids:
                if cell_id not in self.cell_colors:
                    # Generate a consistent color based on cell_id
                    self.cell_colors[cell_id] = self._generate_color_for_id(cell_id)

        self.update_display()

    def toggle_mask_visibility(self, visible: bool):
        """Toggle mask overlay visibility"""
        self.show_masks = visible
        self.update_display()

    def set_mask_alpha(self, alpha: float):
        """Set mask overlay transparency"""
        self.mask_alpha = alpha
        self.update_display()

    def toggle_cell_selection(self, cell_id: int):
        """Toggle cell selection"""
        if cell_id in self.selected_cells:
            self.selected_cells.remove(cell_id)
        else:
            self.selected_cells.add(cell_id)
        self.update_display()

    def select_all_cells(self):
        """Select all cells"""
        if self.masks is not None:
            unique_ids = np.unique(self.masks[self.masks > 0])
            self.selected_cells = set(unique_ids)
            self.update_display()

    def clear_selection(self):
        """Clear cell selection"""
        self.selected_cells.clear()
        self.update_display()

    def update_display(self):
        """Update the display with current image and overlays"""
        if self.original_image is None:
            return

        # Start with original image
        display_image = self.original_image.copy()

        # Add mask overlay if available and enabled
        if self.masks is not None and self.show_masks:
            display_image = self._add_mask_overlay(display_image)

        # Convert to QPixmap and display
        pixmap = self._numpy_to_pixmap(display_image)

        # Scale to fit widget while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        # Add annotation overlays
        if self.click_points or self.annotation_boxes or self.current_box:
            scaled_pixmap = self._add_annotation_overlay(scaled_pixmap)

        self.setPixmap(scaled_pixmap)

    def _add_annotation_overlay(self, pixmap: QPixmap) -> QPixmap:
        """Add annotation overlay to pixmap"""
        painter = QPainter(pixmap)

        # Calculate scaling factors
        image_h, image_w = self.original_image.shape[:2]
        pixmap_w, pixmap_h = pixmap.width(), pixmap.height()
        scale_x = pixmap_w / image_w
        scale_y = pixmap_h / image_h

        # Draw click points (green)
        painter.setPen(QPen(QColor(0, 255, 0), 3))
        painter.setBrush(QBrush(QColor(0, 255, 0)))
        for x, y in self.click_points:
            scaled_x = int(x * scale_x)
            scaled_y = int(y * scale_y)
            painter.drawEllipse(scaled_x - 5, scaled_y - 5, 10, 10)

        # Draw annotation boxes (blue)
        painter.setPen(QPen(QColor(0, 100, 255), 2))
        painter.setBrush(QBrush())  # No fill
        for x1, y1, x2, y2 in self.annotation_boxes:
            scaled_x1 = int(x1 * scale_x)
            scaled_y1 = int(y1 * scale_y)
            scaled_x2 = int(x2 * scale_x)
            scaled_y2 = int(y2 * scale_y)
            painter.drawRect(
                scaled_x1, scaled_y1, scaled_x2 - scaled_x1, scaled_y2 - scaled_y1
            )

        # Draw current box being drawn (yellow)
        if self.current_box:
            painter.setPen(QPen(QColor(255, 255, 0), 2))
            x1, y1, x2, y2 = self.current_box
            scaled_x1 = int(x1 * scale_x)
            scaled_y1 = int(y1 * scale_y)
            scaled_x2 = int(x2 * scale_x)
            scaled_y2 = int(y2 * scale_y)
            painter.drawRect(
                scaled_x1, scaled_y1, scaled_x2 - scaled_x1, scaled_y2 - scaled_y1
            )

        painter.end()
        return pixmap

    def _add_mask_overlay(self, image: np.ndarray) -> np.ndarray:
        """Add mask overlay to image"""
        overlay = image.copy()

        # Add regular masks if available and enabled
        if self.masks is not None:
            # Get unique cell IDs from current masks
            unique_ids = np.unique(self.masks[self.masks > 0])

            for cell_id in unique_ids:
                mask = self.masks == cell_id

                # Use consistent color from mapping
                if cell_id in self.cell_colors:
                    color = self.cell_colors[cell_id]
                else:
                    # Fallback color generation if not in mapping
                    color = self._generate_color_for_id(cell_id)
                    self.cell_colors[cell_id] = color

                # Highlight selected cells
                if cell_id in self.selected_cells:
                    color = (255, 255, 0)  # Yellow for selected

                # Apply color to mask area
                overlay[mask] = (1 - self.mask_alpha) * overlay[
                    mask
                ] + self.mask_alpha * np.array(color)

        # Add hover preview mask if available
        if self.hover_preview_mask is not None and self.show_hover_preview:
            preview_color = (0, 255, 255)  # Cyan for preview
            preview_alpha = 0.4  # More transparent for preview

            overlay[self.hover_preview_mask] = (1 - preview_alpha) * overlay[
                self.hover_preview_mask
            ] + preview_alpha * np.array(preview_color)

        return overlay.astype(np.uint8)

    def _generate_color_for_id(self, cell_id: int) -> Tuple[int, int, int]:
        """Generate a consistent color for a specific cell ID"""
        import colorsys

        # Use cell_id to generate a consistent hue
        hue = ((cell_id - 1) * 137.508) % 360  # Golden angle for good distribution
        saturation = 70 + ((cell_id - 1) % 3) * 10
        value = 80 + ((cell_id - 1) % 2) * 20

        # Convert HSV to RGB
        r, g, b = colorsys.hsv_to_rgb(hue / 360, saturation / 100, value / 100)
        return (int(r * 255), int(g * 255), int(b * 255))

    def _generate_colors(self, n_colors: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for cells"""
        colors = []
        for i in range(n_colors):
            hue = (i * 137.508) % 360  # Golden angle
            saturation = 70 + (i % 3) * 10
            value = 80 + (i % 2) * 20

            # Convert HSV to RGB
            import colorsys

            r, g, b = colorsys.hsv_to_rgb(hue / 360, saturation / 100, value / 100)
            colors.append((int(r * 255), int(g * 255), int(b * 255)))

        return colors

    def _numpy_to_pixmap(self, image: np.ndarray) -> QPixmap:
        """Convert numpy array to QPixmap"""
        try:
            # Ensure image is uint8
            if image.dtype != np.uint8:
                if image.dtype == np.float32 or image.dtype == np.float64:
                    # Assume values are in range [0, 1] and scale to [0, 255]
                    image = (image * 255).astype(np.uint8)
                else:
                    # Convert to uint8 directly
                    image = image.astype(np.uint8)

            # Handle grayscale images (2D)
            if len(image.shape) == 2:
                h, w = image.shape
                # Convert grayscale to RGB
                image_rgb = np.stack([image, image, image], axis=-1)
                h, w, ch = image_rgb.shape
                bytes_per_line = ch * w
                qt_image = QImage(
                    image_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
                )
            else:
                # Handle RGB images (3D)
                h, w, ch = image.shape
                if ch != 3:
                    raise ValueError(f"Expected 3 channels for RGB image, got {ch}")
                bytes_per_line = ch * w
                qt_image = QImage(
                    image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
                )

            pixmap = QPixmap.fromImage(qt_image)
            return pixmap

        except Exception as e:
            print(f"Error in _numpy_to_pixmap: {e}")
            # Return a fallback pixmap
            fallback_image = np.zeros((100, 100, 3), dtype=np.uint8)
            fallback_image[:, :] = [128, 128, 128]  # Gray color
            h, w, ch = fallback_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(
                fallback_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
            )
            return QPixmap.fromImage(qt_image)

    def _pixel_to_image_coords(self, pixel_x: int, pixel_y: int) -> Tuple[int, int]:
        """Convert pixel coordinates to image coordinates"""
        pixmap = self.pixmap()
        if not pixmap or self.original_image is None:
            return 0, 0

        # Calculate position in original image coordinates
        widget_size = self.size()
        pixmap_size = pixmap.size()

        # Calculate offset for centered image
        x_offset = (widget_size.width() - pixmap_size.width()) // 2
        y_offset = (widget_size.height() - pixmap_size.height()) // 2

        # Get click position relative to pixmap
        click_x = pixel_x - x_offset
        click_y = pixel_y - y_offset

        if 0 <= click_x < pixmap_size.width() and 0 <= click_y < pixmap_size.height():
            # Scale to original image coordinates
            image_h, image_w = self.original_image.shape[:2]
            orig_x = int(click_x * image_w / pixmap_size.width())
            orig_y = int(click_y * image_h / pixmap_size.height())
            return orig_x, orig_y

        return 0, 0

    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.original_image is not None
        ):
            orig_x, orig_y = self._pixel_to_image_coords(
                event.pos().x(), event.pos().y()
            )

            if self.annotation_mode == AnnotationMode.CLICK_ADD:
                self.click_points.append((orig_x, orig_y))
                self.update_display()
                self.annotation_changed.emit()

            elif self.annotation_mode == AnnotationMode.BOX_ADD:
                self.drawing_box = True
                self.box_start_point = (orig_x, orig_y)
                self.current_box = (orig_x, orig_y, orig_x, orig_y)

            elif self.annotation_mode == AnnotationMode.VIEW and self.masks is not None:
                # View mode - no longer allow cell selection since we track all cells
                pass

            elif (
                self.annotation_mode == AnnotationMode.MASK_REMOVE
                and self.masks is not None
            ):
                # Mask removal mode - remove the clicked mask
                image_h, image_w = self.original_image.shape[:2]
                if 0 <= orig_x < image_w and 0 <= orig_y < image_h:
                    cell_id = self.masks[orig_y, orig_x]
                    if cell_id > 0:
                        # Remove the entire mask for this cell
                        self.masks[self.masks == cell_id] = 0
                        # Remove from selected cells if it was selected
                        if cell_id in self.selected_cells:
                            self.selected_cells.remove(cell_id)
                        # Note: We keep the color in self.cell_colors for consistency
                        # in case the same cell_id appears again
                        self.update_display()
                        self.annotation_changed.emit()

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move events"""
        if self.drawing_box and self.annotation_mode == AnnotationMode.BOX_ADD:
            orig_x, orig_y = self._pixel_to_image_coords(
                event.pos().x(), event.pos().y()
            )
            if self.box_start_point:
                self.current_box = (
                    self.box_start_point[0],
                    self.box_start_point[1],
                    orig_x,
                    orig_y,
                )
                self.update_display()

        # Handle hover preview for click add mode
        elif (
            self.annotation_mode == AnnotationMode.CLICK_ADD
            and self.show_hover_preview
            and self.original_image is not None
        ):
            orig_x, orig_y = self._pixel_to_image_coords(
                event.pos().x(), event.pos().y()
            )

            # Debounce hover requests using a timer
            if self.hover_timer is not None:
                self.hover_timer.stop()

            self.hover_timer = QTimer()
            self.hover_timer.setSingleShot(True)
            self.hover_timer.timeout.connect(
                lambda: self.hover_preview_requested.emit(orig_x, orig_y)
            )
            self.hover_timer.start(100)  # 100ms delay

        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        """Handle mouse leave events"""
        # Clear hover preview when mouse leaves the widget
        if self.hover_preview_mask is not None:
            self.clear_hover_preview()

        # Stop any pending hover timer
        if self.hover_timer is not None:
            self.hover_timer.stop()

        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.drawing_box
            and self.current_box
        ):
            # Finalize box
            x1, y1, x2, y2 = self.current_box
            # Ensure proper ordering
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            # Only add box if it has reasonable size
            if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                self.annotation_boxes.append((x1, y1, x2, y2))
                self.annotation_changed.emit()

            self.drawing_box = False
            self.box_start_point = None
            self.current_box = None
            self.update_display()

        super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        """Handle resize events"""
        super().resizeEvent(event)
        self.update_display()


class SegmentationPanel(QWidget):
    """Enhanced panel for cell segmentation using CellSAM and SAM annotation"""

    segmentation_started = pyqtSignal()
    segmentation_completed = pyqtSignal(dict)  # results
    segmentation_error = pyqtSignal(str)  # error message
    annotation_completed = pyqtSignal(np.ndarray)  # updated masks

    def __init__(self):
        super().__init__()
        self.current_frame: Optional[Dict[str, Any]] = None
        self.segmentation_results: Optional[Dict[str, Any]] = None
        self.segmentation_worker: Optional[SegmentationWorker] = None
        self.sam_worker: Optional[SAMWorker] = None
        self.sam_preview_worker: Optional[SAMPreviewWorker] = None
        self.sam_auto_timer: Optional[QTimer] = None  # Timer for automatic SAM runs
        self.main_window = None  # Reference to main window for preloaded models
        self.preloaded_models = {}  # Cache for preloaded models

        # Cache for SAM predictions to avoid recomputation
        self.sam_prediction_cache = {}  # Key: (x, y), Value: (mask, score, logits)

        self.setup_ui()
        self.setEnabled(False)  # Disabled until frames are loaded

    def set_main_window(self, main_window):
        """Set reference to main window for accessing preloaded models"""
        self.main_window = main_window

    def set_preloaded_models(self, models):
        """Set preloaded models for faster access"""
        self.preloaded_models = models

    def get_preloaded_model(self, model_name):
        """Get a preloaded model if available"""
        if model_name in self.preloaded_models:
            return self.preloaded_models[model_name]
        elif self.main_window:
            models = self.main_window.get_preloaded_models()
            return models.get(model_name)
        return None

    def enable_panel(self):
        """Enable the segmentation panel"""
        self.setEnabled(True)

    def disable_panel(self):
        """Disable the segmentation panel"""
        self.setEnabled(False)

    def setup_ui(self):
        """Setup the enhanced user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Create splitter for image and controls
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left side - Image display
        self.image_widget = ImageDisplayWidget()
        self.image_widget.annotation_changed.connect(self.on_annotation_changed)
        self.image_widget.hover_preview_requested.connect(
            self.on_hover_preview_requested
        )
        splitter.addWidget(self.image_widget)

        # Right side - Controls with tabs
        controls_widget = self.create_controls_widget()
        splitter.addWidget(controls_widget)

        # Set splitter proportions (70% image, 30% controls)
        splitter.setSizes([700, 300])

        # Progress bar (initially hidden)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel()
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)

    def create_controls_widget(self) -> QWidget:
        """Create the enhanced controls widget with tabs"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Create tab widget
        tab_widget = QTabWidget()

        # Tab 1: CellSAM Segmentation
        cellsam_tab = self.create_cellsam_tab()
        tab_widget.addTab(cellsam_tab, "CellSAM")

        # Tab 2: SAM Annotation
        sam_tab = self.create_sam_annotation_tab()
        tab_widget.addTab(sam_tab, "SAM Annotation")

        # Tab 3: Results & Display
        results_tab = self.create_results_tab()
        tab_widget.addTab(results_tab, "Results")

        layout.addWidget(tab_widget)

        return widget

    def create_cellsam_tab(self) -> QWidget:
        """Create CellSAM segmentation tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)

        # CellSAM Parameters
        cellsam_group = QGroupBox("CellSAM Parameters")
        cellsam_layout = QVBoxLayout(cellsam_group)

        # Diameter
        diameter_layout = QHBoxLayout()
        diameter_layout.addWidget(QLabel("Cell Diameter:"))
        self.diameter_spinbox = QDoubleSpinBox()
        self.diameter_spinbox.setRange(5.0, 200.0)
        self.diameter_spinbox.setValue(30.0)
        self.diameter_spinbox.setSuffix(" px")
        diameter_layout.addWidget(self.diameter_spinbox)
        cellsam_layout.addLayout(diameter_layout)

        # Flow threshold
        flow_layout = QHBoxLayout()
        flow_layout.addWidget(QLabel("Flow Threshold:"))
        self.flow_threshold_spinbox = QDoubleSpinBox()
        self.flow_threshold_spinbox.setRange(0.0, 2.0)
        self.flow_threshold_spinbox.setValue(0.4)
        self.flow_threshold_spinbox.setDecimals(2)
        self.flow_threshold_spinbox.setSingleStep(0.05)
        flow_layout.addWidget(self.flow_threshold_spinbox)
        cellsam_layout.addLayout(flow_layout)

        # Cell probability threshold
        cellprob_layout = QHBoxLayout()
        cellprob_layout.addWidget(QLabel("Cell Prob Threshold:"))
        self.cellprob_threshold_spinbox = QDoubleSpinBox()
        self.cellprob_threshold_spinbox.setRange(-10.0, 10.0)
        self.cellprob_threshold_spinbox.setValue(0.0)
        self.cellprob_threshold_spinbox.setDecimals(1)
        cellprob_layout.addWidget(self.cellprob_threshold_spinbox)
        cellsam_layout.addLayout(cellprob_layout)

        # Device selection
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu"])
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                self.device_combo.addItem(f"cuda:{i}")
        device_layout.addWidget(self.device_combo)
        cellsam_layout.addLayout(device_layout)

        layout.addWidget(cellsam_group)

        # Segmentation controls
        segment_group = QGroupBox("Automatic Segmentation")
        segment_layout = QVBoxLayout(segment_group)

        self.segment_button = QPushButton("🔬 Run CellSAM")
        self.segment_button.clicked.connect(self.run_segmentation)
        self.segment_button.setEnabled(False)
        self.segment_button.setStyleSheet(
            """
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:disabled {
                background-color: #404040;
                color: #808080;
            }
        """
        )
        segment_layout.addWidget(self.segment_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_segmentation)
        self.cancel_button.setEnabled(False)
        segment_layout.addWidget(self.cancel_button)

        layout.addWidget(segment_group)

        layout.addStretch()
        return widget

    def create_sam_annotation_tab(self) -> QWidget:
        """Create SAM annotation tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)

        # Annotation mode selection
        mode_group = QGroupBox("Annotation Mode")
        mode_layout = QVBoxLayout(mode_group)

        self.annotation_mode_group = QButtonGroup()

        self.view_mode_radio = QRadioButton("👁️ View Cells")
        self.view_mode_radio.setChecked(True)
        self.annotation_mode_group.addButton(self.view_mode_radio, 0)
        mode_layout.addWidget(self.view_mode_radio)

        self.click_add_radio = QRadioButton("🖱️ Click to Add")
        self.annotation_mode_group.addButton(self.click_add_radio, 1)
        mode_layout.addWidget(self.click_add_radio)

        self.box_add_radio = QRadioButton("📦 Add Box")
        self.annotation_mode_group.addButton(self.box_add_radio, 2)
        mode_layout.addWidget(self.box_add_radio)

        self.mask_remove_radio = QRadioButton("🗑️ Remove Mask")
        self.annotation_mode_group.addButton(self.mask_remove_radio, 3)
        mode_layout.addWidget(self.mask_remove_radio)

        self.annotation_mode_group.buttonClicked.connect(
            self.on_annotation_mode_changed
        )

        layout.addWidget(mode_group)

        # SAM Settings (simplified)
        sam_settings_group = QGroupBox("SAM Settings")
        sam_settings_layout = QVBoxLayout(sam_settings_group)

        # Checkpoint path display (read-only)
        checkpoint_layout = QHBoxLayout()
        checkpoint_layout.addWidget(QLabel("Model:"))
        self.sam_model_info = QLabel("ViT-H (checkpoints/sam_vit_h_4b8939.pth)")
        self.sam_model_info.setStyleSheet(
            """
            QLabel {
                background-color: #2d2d2d;
                border: 1px solid #555;
                padding: 2px 4px;
                border-radius: 2px;
                font-family: monospace;
                font-size: 9px;
            }
        """
        )
        checkpoint_layout.addWidget(self.sam_model_info)
        sam_settings_layout.addLayout(checkpoint_layout)

        # Hover preview toggle
        self.hover_preview_checkbox = QCheckBox("🔍 Show Hover Preview")
        self.hover_preview_checkbox.setChecked(True)
        self.hover_preview_checkbox.setToolTip(
            "Show preview of SAM segmentation when hovering in Click mode"
        )
        sam_settings_layout.addWidget(self.hover_preview_checkbox)

        layout.addWidget(sam_settings_group)

        # Annotation controls
        annotation_group = QGroupBox("Annotation Controls")
        annotation_layout = QVBoxLayout(annotation_group)

        # Current annotation info
        self.annotation_info = QLabel("No annotations")
        self.annotation_info.setWordWrap(True)
        self.annotation_info.setStyleSheet(
            """
            QLabel {
                background-color: #2d2d2d;
                border: 1px solid #555;
                padding: 4px;
                border-radius: 4px;
            }
        """
        )
        annotation_layout.addWidget(self.annotation_info)

        # Control buttons
        button_layout = QHBoxLayout()

        self.clear_annotations_button = QPushButton("Clear All")
        self.clear_annotations_button.clicked.connect(self.clear_annotations)
        button_layout.addWidget(self.clear_annotations_button)

        self.clear_cache_button = QPushButton("Clear Cache")
        self.clear_cache_button.clicked.connect(self.clear_sam_cache)
        self.clear_cache_button.setToolTip(
            "Clear SAM prediction cache to force recomputation"
        )
        button_layout.addWidget(self.clear_cache_button)

        annotation_layout.addLayout(button_layout)

        layout.addWidget(annotation_group)

        layout.addStretch()
        return widget

    def create_results_tab(self) -> QWidget:
        """Create results and display tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)

        # Results info
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)

        self.cell_count_label = QLabel("Cells detected: 0")
        results_layout.addWidget(self.cell_count_label)

        layout.addWidget(results_group)

        # Display controls
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)

        self.show_masks_checkbox = QCheckBox("Show Masks")
        self.show_masks_checkbox.setChecked(True)
        self.show_masks_checkbox.toggled.connect(self.toggle_mask_visibility)
        display_layout.addWidget(self.show_masks_checkbox)

        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("Mask Alpha:"))
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(50)
        self.alpha_slider.valueChanged.connect(self.update_mask_alpha)
        alpha_layout.addWidget(self.alpha_slider)
        display_layout.addLayout(alpha_layout)

        layout.addWidget(display_group)

        # Export section
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)

        self.export_masks_button = QPushButton("Export Masks")
        self.export_masks_button.clicked.connect(self.export_masks)
        self.export_masks_button.setEnabled(False)
        export_layout.addWidget(self.export_masks_button)

        self.export_overlay_button = QPushButton("Export Overlay")
        self.export_overlay_button.clicked.connect(self.export_overlay)
        self.export_overlay_button.setEnabled(False)
        export_layout.addWidget(self.export_overlay_button)

        layout.addWidget(export_group)

        layout.addStretch()
        return widget

    def on_annotation_mode_changed(self, button):
        """Handle annotation mode change"""
        button_id = self.annotation_mode_group.id(button)

        if button_id == 0:  # View mode
            self.image_widget.set_annotation_mode(AnnotationMode.VIEW)
        elif button_id == 1:  # Click add
            self.image_widget.set_annotation_mode(AnnotationMode.CLICK_ADD)
        elif button_id == 2:  # Box add
            self.image_widget.set_annotation_mode(AnnotationMode.BOX_ADD)
        elif button_id == 3:  # Mask remove
            self.image_widget.set_annotation_mode(AnnotationMode.MASK_REMOVE)

    def on_annotation_changed(self):
        """Handle annotation changes"""
        annotation_data = self.image_widget.get_annotation_data()

        # Update annotation info
        click_count = len(annotation_data["click_points"])
        box_count = len(annotation_data["boxes"])

        info_text = f"Click points: {click_count}\nBoxes: {box_count}"
        self.annotation_info.setText(info_text)

        # Check if we can use cached results for immediate application
        has_annotations = click_count > 0 or box_count > 0
        if has_annotations and self.segmentation_results is not None:
            # Check if all click points are cached - if so, apply immediately
            all_points_cached = all(
                point in self.sam_prediction_cache
                for point in annotation_data["click_points"]
            )

            if all_points_cached and box_count == 0:  # Only cached points, no boxes
                # Apply cached results immediately without SAM worker
                self._apply_cached_sam_results(annotation_data["click_points"])
            else:
                # Auto-run SAM with a small delay to allow for multiple rapid clicks/boxes
                if self.sam_auto_timer is not None:
                    self.sam_auto_timer.stop()

                self.sam_auto_timer = QTimer()
                self.sam_auto_timer.setSingleShot(True)
                self.sam_auto_timer.timeout.connect(self.run_sam_annotation)
                self.sam_auto_timer.start(500)  # 500ms delay for debouncing

    def browse_sam_checkpoint(self):
        """Browse for SAM checkpoint file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select SAM Checkpoint", "", "PyTorch Files (*.pth);;All Files (*)"
        )

        if file_path:
            self.sam_checkpoint_path.setText(file_path)

    def get_sam_model_type(self) -> str:
        """Get the SAM model type - always vit_h"""
        return "vit_h"

    def get_sam_checkpoint_path(self) -> Optional[str]:
        """Get the SAM checkpoint path - always None for auto-detect"""
        return None

    def clear_annotations(self):
        """Clear all annotations"""
        self.image_widget.clear_annotations()

    def clear_sam_cache(self):
        """Clear SAM prediction cache"""
        self.sam_prediction_cache.clear()

    def _apply_cached_sam_results(self, points: List[Tuple[int, int]]):
        """Apply SAM results directly from cache without running SAM worker"""
        if not self.segmentation_results or not points:
            return

        # Start with existing masks
        masks = self.segmentation_results["masks"].copy()

        # Get next mask ID
        next_id = np.max(masks) + 1 if masks.size > 0 and np.max(masks) > 0 else 1

        # Apply each cached point
        for point in points:
            if point in self.sam_prediction_cache:
                cached_mask, cached_score, _ = self.sam_prediction_cache[point]

                # Use cached result if it's good quality
                if cached_score > 0.5:
                    new_mask = cached_mask

                    # Simple overlap check - only add if it's mostly in empty space
                    existing_overlap = np.sum(new_mask & (masks > 0))
                    new_mask_size = np.sum(new_mask)

                    if (
                        new_mask_size > 0 and existing_overlap / new_mask_size < 0.5
                    ):  # Less than 50% overlap
                        # Remove any overlapping pixels and add the rest
                        clean_mask = new_mask & (masks == 0)
                        if np.sum(clean_mask) > 10:  # Minimum size threshold
                            masks[clean_mask] = next_id
                            next_id += 1

        # Update segmentation results
        self.segmentation_results["masks"] = masks
        self.segmentation_results["cell_count"] = np.max(masks) if masks.size > 0 else 0

        # Update image display
        self.image_widget.set_masks(masks)

        # Clear annotations after applying cached results
        self.image_widget.clear_annotations()

        # Update results display
        self.update_results_display()

        # Emit signal (cached results applied instantly)
        self.annotation_completed.emit(masks)

    def run_sam_annotation(self):
        """Run SAM annotation based on current annotations"""
        if not self.current_frame or not self.segmentation_results:
            return

        annotation_data = self.image_widget.get_annotation_data()

        # Prepare points and labels for SAM (all click points are positive)
        points = annotation_data["click_points"]
        point_labels = [1] * len(points)  # All points are positive

        # Start SAM worker
        preloaded_sam = self.get_preloaded_model("sam")
        self.sam_worker = SAMWorker(
            self.current_frame["image"],
            points,
            point_labels,
            annotation_data["boxes"],
            self.segmentation_results["masks"],
            sam_checkpoint=self.get_sam_checkpoint_path(),
            model_type=self.get_sam_model_type(),
            preloaded_sam=preloaded_sam,
            cache_dict=self.sam_prediction_cache,  # Pass cache for efficiency
        )

        self.sam_worker.progress_update.connect(self.on_progress_update)
        self.sam_worker.annotation_complete.connect(self.on_sam_annotation_complete)
        self.sam_worker.error_occurred.connect(self.on_sam_annotation_error)
        self.sam_worker.start()

        # Update UI state - show progress for automatic SAM runs
        self.show_progress(True)

    def on_sam_annotation_complete(self, updated_masks: np.ndarray):
        """Handle SAM annotation completion"""
        # Update segmentation results
        if self.segmentation_results:
            self.segmentation_results["masks"] = updated_masks
            self.segmentation_results["cell_count"] = (
                np.max(updated_masks) if updated_masks.size > 0 else 0
            )

        # Update image display
        self.image_widget.set_masks(updated_masks)

        # Clear annotations after SAM completion
        self.image_widget.clear_annotations()

        # Update results
        self.update_results_display()

        # Reset UI state
        self.reset_ui_state()

        # Emit signal
        self.annotation_completed.emit(updated_masks)

    def on_sam_annotation_error(self, error_message: str):
        """Handle SAM annotation errors"""
        self.reset_ui_state()
        QMessageBox.critical(self, "SAM Annotation Error", error_message)

    def set_current_frame(self, frame_data: Dict[str, Any]):
        """Set the current frame for segmentation"""
        self.current_frame = frame_data
        self.image_widget.set_image(frame_data["image"])

        # Enable the panel and segmentation button
        self.setEnabled(True)
        self.segment_button.setEnabled(True)

        # Clear previous results and cache
        self.segmentation_results = None
        self.sam_prediction_cache.clear()  # Clear cache for new image
        self.update_results_display()

        # Initialize SAM model for hover previews if available
        self._initialize_sam_for_image()

    def _initialize_sam_for_image(self):
        """Initialize SAM model with the current image for faster hover previews"""
        if not self.current_frame or self.current_frame.get("image") is None:
            return

        preloaded_sam = self.get_preloaded_model("sam")
        if preloaded_sam and hasattr(preloaded_sam, "set_image"):
            try:
                # Set the image for the SAM predictor to enable fast hover previews
                preloaded_sam.set_image(self.current_frame["image"])
            except Exception as e:
                # Silently handle any errors in SAM initialization
                print(f"Warning: Could not initialize SAM for hover previews: {e}")

    def run_segmentation(self):
        """Run CellSAM segmentation on current frame"""
        if not self.current_frame:
            return

        # Get parameters
        parameters = {
            "diameter": self.diameter_spinbox.value(),
            "flow_threshold": self.flow_threshold_spinbox.value(),
            "cellprob_threshold": self.cellprob_threshold_spinbox.value(),
            "device": self.device_combo.currentText(),
        }

        # Start segmentation worker
        preloaded_cellsam = self.get_preloaded_model("cellsam")
        self.segmentation_worker = SegmentationWorker(
            self.current_frame["image"], parameters, preloaded_cellsam
        )
        self.segmentation_worker.progress_update.connect(self.on_progress_update)
        self.segmentation_worker.segmentation_complete.connect(
            self.on_segmentation_complete
        )
        self.segmentation_worker.error_occurred.connect(self.on_segmentation_error)
        self.segmentation_worker.start()

        # Update UI state
        self.segment_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.show_progress(True)

        # Emit signal
        self.segmentation_started.emit()

    def cancel_segmentation(self):
        """Cancel the current segmentation"""
        if self.segmentation_worker and self.segmentation_worker.isRunning():
            self.segmentation_worker.cancel()
            self.segmentation_worker.wait()

        self.reset_ui_state()

    def on_progress_update(self, status: str):
        """Handle progress updates"""
        self.status_label.setText(status)

    def on_segmentation_complete(self, results: Dict[str, Any]):
        """Handle segmentation completion"""
        self.segmentation_results = results

        # Update image display
        self.image_widget.set_masks(results["masks"])

        # Update results
        self.update_results_display()

        # Reset UI state
        self.reset_ui_state()

        # Enable export buttons
        self.export_masks_button.setEnabled(True)
        self.export_overlay_button.setEnabled(True)

        # Add selected_cells field for compatibility (but empty since we don't use selection anymore)
        results_with_selection = results.copy()
        results_with_selection["selected_cells"] = []

        # Emit signal
        self.segmentation_completed.emit(results_with_selection)

    def on_segmentation_error(self, error_message: str):
        """Handle segmentation errors"""
        self.reset_ui_state()
        self.segmentation_error.emit(error_message)

    def show_progress(self, show: bool):
        """Show or hide progress indicators"""
        self.progress_bar.setVisible(show)
        self.status_label.setVisible(show)

        if show:
            self.progress_bar.setRange(0, 0)  # Indeterminate

    def reset_ui_state(self):
        """Reset UI to normal state"""
        self.segment_button.setEnabled(self.current_frame is not None)
        self.cancel_button.setEnabled(False)
        self.show_progress(False)

    def update_results_display(self):
        """Update the results display"""
        if self.segmentation_results:
            cell_count = self.segmentation_results["cell_count"]
            self.cell_count_label.setText(f"Cells detected: {cell_count}")

            # Enable export buttons
            self.export_masks_button.setEnabled(True)
            self.export_overlay_button.setEnabled(True)
        else:
            self.cell_count_label.setText("Cells detected: 0")
            self.export_masks_button.setEnabled(False)
            self.export_overlay_button.setEnabled(False)

    def toggle_mask_visibility(self, visible: bool):
        """Toggle mask overlay visibility"""
        self.image_widget.toggle_mask_visibility(visible)

    def update_mask_alpha(self, value: int):
        """Update mask transparency"""
        alpha = value / 100.0
        self.image_widget.set_mask_alpha(alpha)

    def export_masks(self):
        """Export segmentation masks"""
        if not self.segmentation_results:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Masks",
            "",
            "NumPy Files (*.npy);;TIFF Files (*.tif);;All Files (*)",
        )

        if file_path:
            try:
                masks = self.segmentation_results["masks"]

                if file_path.endswith(".npy"):
                    np.save(file_path, masks)
                elif file_path.endswith(".tif") or file_path.endswith(".tiff"):
                    cv2.imwrite(file_path, masks.astype(np.uint16))
                else:
                    np.save(file_path, masks)

                QMessageBox.information(
                    self, "Export", f"Masks exported to:\n{file_path}"
                )

            except Exception as e:
                QMessageBox.critical(
                    self, "Export Error", f"Failed to export masks:\n{str(e)}"
                )

    def export_overlay(self):
        """Export image with mask overlay"""
        if not self.segmentation_results or not self.current_frame:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Overlay",
            "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)",
        )

        if file_path:
            try:
                # Create overlay image
                image = self.current_frame["image"]
                masks = self.segmentation_results["masks"]

                overlay = self.image_widget._add_mask_overlay(image)

                # Convert RGB to BGR for OpenCV
                overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, overlay_bgr)

                QMessageBox.information(
                    self, "Export", f"Overlay exported to:\n{file_path}"
                )

            except Exception as e:
                QMessageBox.critical(
                    self, "Export Error", f"Failed to export overlay:\n{str(e)}"
                )

    # Public interface
    def get_data(self) -> Dict[str, Any]:
        """Get segmentation data for project saving"""
        data = {
            "parameters": {
                "diameter": self.diameter_spinbox.value(),
                "flow_threshold": self.flow_threshold_spinbox.value(),
                "cellprob_threshold": self.cellprob_threshold_spinbox.value(),
                "device": self.device_combo.currentText(),
            },
            "display": {
                "show_masks": self.show_masks_checkbox.isChecked(),
                "mask_alpha": self.alpha_slider.value() / 100.0,
            },
        }

        if self.segmentation_results:
            data["results"] = {
                "cell_count": self.segmentation_results["cell_count"],
            }

        # Save annotation data
        annotation_data = self.image_widget.get_annotation_data()
        data["annotations"] = annotation_data

        return data

    def set_data(self, data: Dict[str, Any]):
        """Set segmentation data from project loading"""
        if "parameters" in data:
            params = data["parameters"]
            self.diameter_spinbox.setValue(params.get("diameter", 30.0))
            self.flow_threshold_spinbox.setValue(params.get("flow_threshold", 0.4))
            self.cellprob_threshold_spinbox.setValue(
                params.get("cellprob_threshold", 0.0)
            )

            device = params.get("device", "auto")
            index = self.device_combo.findText(device)
            if index >= 0:
                self.device_combo.setCurrentIndex(index)

        if "display" in data:
            display = data["display"]
            self.show_masks_checkbox.setChecked(display.get("show_masks", True))
            alpha = int(display.get("mask_alpha", 0.5) * 100)
            self.alpha_slider.setValue(alpha)

    def on_hover_preview_requested(self, x: int, y: int):
        """Handle hover preview request"""
        # Check if hover preview is enabled
        if not self.hover_preview_checkbox.isChecked():
            self.image_widget.clear_hover_preview()
            return

        # Only generate preview if we have a SAM model loaded and image is available
        preloaded_sam = self.get_preloaded_model("sam")
        if (
            not preloaded_sam
            or not self.current_frame
            or self.current_frame.get("image") is None
        ):
            self.image_widget.clear_hover_preview()
            return

        # Cancel any existing preview worker
        if self.sam_preview_worker and self.sam_preview_worker.isRunning():
            self.sam_preview_worker.cancel()
            self.sam_preview_worker.wait()

        # Start new preview worker
        self.sam_preview_worker = SAMPreviewWorker(
            self.current_frame["image"],
            (x, y),
            preloaded_sam,
            self.sam_prediction_cache,
        )

        self.sam_preview_worker.preview_ready.connect(self.on_hover_preview_ready)
        self.sam_preview_worker.error_occurred.connect(self.on_hover_preview_error)
        self.sam_preview_worker.start()

    def on_hover_preview_ready(self, mask: np.ndarray, point: tuple, score: float):
        """Handle hover preview result"""
        self.image_widget.set_hover_preview_mask(mask)

    def on_hover_preview_error(self, error_message: str):
        """Handle hover preview error (silently)"""
        # Don't show error messages for preview failures as they're non-critical
        self.image_widget.clear_hover_preview()

    def reset(self):
        """Reset the panel to initial state"""
        self.current_frame = None
        self.segmentation_results = None

        # Cancel any running workers
        if self.sam_preview_worker and self.sam_preview_worker.isRunning():
            self.sam_preview_worker.cancel()
            self.sam_preview_worker.wait()

        # Reset UI
        self.image_widget.set_image(np.zeros((400, 400, 3), dtype=np.uint8))
        self.image_widget.clear_hover_preview()
        self.segment_button.setEnabled(False)
        self.export_masks_button.setEnabled(False)
        self.export_overlay_button.setEnabled(False)

        # Reset parameters to defaults
        self.diameter_spinbox.setValue(30.0)
        self.flow_threshold_spinbox.setValue(0.4)
        self.cellprob_threshold_spinbox.setValue(0.0)
        self.device_combo.setCurrentIndex(0)
        self.show_masks_checkbox.setChecked(True)
        self.alpha_slider.setValue(50)
        self.hover_preview_checkbox.setChecked(True)

        # Reset annotation mode
        self.view_mode_radio.setChecked(True)
        self.image_widget.set_annotation_mode(AnnotationMode.VIEW)

        self.update_results_display()
