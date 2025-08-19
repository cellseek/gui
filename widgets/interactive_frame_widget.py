from enum import Enum
from typing import List, Tuple

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QLabel


class AnnotationMode(Enum):
    """Annotation mode enumeration"""

    VIEW = "view"
    CLICK_ADD = "click_add"
    BOX_ADD = "box_add"
    MASK_REMOVE = "mask_remove"
    EDIT_CELL_ID = "edit_cell_id"


class InteractiveFrameWidget(QLabel):
    """Interactive frame widget for annotation correction"""

    point_clicked = pyqtSignal(tuple)  # (x, y)
    box_drawn = pyqtSignal(tuple)  # (x1, y1, x2, y2)
    mask_clicked = pyqtSignal(tuple)  # (x, y) for mask removal
    cell_id_edit_requested = pyqtSignal(tuple, int)  # (x, y), current_cell_id
    mouse_hover = pyqtSignal(tuple)  # (x, y) for live preview

    def __init__(self):
        super().__init__()
        self.setStyleSheet("border: 1px solid #606060; background-color: #353535;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setScaledContents(False)

        # State
        self.annotation_mode = AnnotationMode.VIEW
        self.image = None
        self.masks = None
        self.preview_mask = None
        self.preview_score = 0.0
        self.overlay_image = None
        self.scale_factor = 1.0
        self.image_offset = (0, 0)

        # Box drawing
        self.drawing_box = False
        self.box_start = None
        self.box_end = None
        
        # Enable mouse tracking for hover events
        self.setMouseTracking(True)

    def set_annotation_mode(self, mode: AnnotationMode):
        """Set the annotation mode"""
        self.annotation_mode = mode

        # Clear preview when not in click mode
        if mode != AnnotationMode.CLICK_ADD:
            self.preview_mask = None
            self.update_display()

        if mode == AnnotationMode.VIEW:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        elif mode in [
            AnnotationMode.CLICK_ADD,
            AnnotationMode.MASK_REMOVE,
            AnnotationMode.EDIT_CELL_ID,
        ]:
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

    def set_preview_mask(self, mask: np.ndarray, score: float):
        """Set preview mask for live preview"""
        self.preview_mask = mask.copy() if mask is not None else None
        self.preview_score = score
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
            
        # Overlay preview mask if available and in click mode
        if (self.preview_mask is not None and 
            self.annotation_mode == AnnotationMode.CLICK_ADD):
            display_image = self._overlay_preview_mask(display_image, self.preview_mask)

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
        """Overlay masks on image with transparency and optional cell IDs"""
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

        overlay = self._add_cell_id_text(overlay, masks, colors)

        return overlay

    def _add_cell_id_text(
        self, overlay: np.ndarray, masks: np.ndarray, colors: List[Tuple[int, int, int]]
    ) -> np.ndarray:
        """Add cell ID text to the overlay"""
        import cv2

        num_objects = int(np.max(masks))

        for obj_id in range(1, num_objects + 1):
            mask = masks == obj_id
            if np.any(mask):
                # Find centroid of the mask
                y_coords, x_coords = np.where(mask)
                if len(y_coords) > 0:
                    center_y = int(np.mean(y_coords))
                    center_x = int(np.mean(x_coords))

                    # Choose text color (white or black based on background)
                    text_color = (255, 255, 255)  # White text

                    # Add text
                    cv2.putText(
                        overlay,
                        str(obj_id),
                        (center_x - 10, center_y + 5),  # Slight offset for centering
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,  # Font scale
                        text_color,
                        2,  # Thickness
                        cv2.LINE_AA,
                    )

                    # Add black outline for better visibility
                    cv2.putText(
                        overlay,
                        str(obj_id),
                        (center_x - 10, center_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),  # Black outline
                        4,  # Thicker for outline
                        cv2.LINE_AA,
                    )

                    # Add white text on top
                    cv2.putText(
                        overlay,
                        str(obj_id),
                        (center_x - 10, center_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        text_color,
                        2,
                        cv2.LINE_AA,
                    )

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

    def _overlay_preview_mask(self, image: np.ndarray, preview_mask: np.ndarray) -> np.ndarray:
        """Overlay preview mask with a semi-transparent cyan color"""
        if preview_mask is None:
            return image
            
        overlay = image.copy()
        # Use cyan color for preview with high transparency
        preview_color = np.array([0, 255, 255], dtype=np.uint8)  # Cyan
        mask = preview_mask > 0
        if np.any(mask):
            # More transparent preview (0.8 original + 0.2 preview)
            overlay[mask] = (0.8 * overlay[mask] + 0.2 * preview_color).astype(np.uint8)
        
        return overlay

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

        elif self.annotation_mode == AnnotationMode.EDIT_CELL_ID:
            # Get current cell ID at the clicked location
            if self.masks is not None:
                x, y = image_coords
                if 0 <= y < self.masks.shape[0] and 0 <= x < self.masks.shape[1]:
                    current_cell_id = int(self.masks[y, x])
                    self.cell_id_edit_requested.emit(image_coords, current_cell_id)

    def mouseMoveEvent(self, event):
        """Handle mouse move events"""
        if self.annotation_mode == AnnotationMode.BOX_ADD and self.drawing_box:
            pos = event.position().toPoint()
            self.box_end = self._widget_to_image_coords(pos.x(), pos.y())
            self.update()  # Trigger repaint to show box
        elif self.annotation_mode == AnnotationMode.CLICK_ADD:
            # Emit hover signal for live preview
            pos = event.position().toPoint()
            image_coords = self._widget_to_image_coords(pos.x(), pos.y())
            self.mouse_hover.emit(image_coords)

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
