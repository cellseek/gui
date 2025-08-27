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
    BRUSH_ADD = "brush_add"
    MASK_REMOVE = "mask_remove"
    EDIT_CELL_ID = "edit_cell_id"


class InteractiveFrameWidget(QLabel):
    """Interactive frame widget for annotation correction"""

    point_clicked = pyqtSignal(tuple)  # (x, y)
    box_drawn = pyqtSignal(tuple)  # (x1, y1, x2, y2)
    brush_drawn = pyqtSignal(np.ndarray)  # brush mask
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
        self.mask_transparency = 0.3  # Default transparency for mask overlay

        # Pan and zoom
        self.pan_offset = (0, 0)
        self.zoom_factor = 1.0
        self.panning = False
        self.last_pan_point = None

        # Box drawing
        self.drawing_box = False
        self.box_start = None
        self.box_end = None

        # Brush drawing
        self.drawing_brush = False
        self.brush_mask = None
        self.brush_size = 10
        self.brush_points = []
        self._last_mouse_pos = None

        # Enable mouse tracking for hover events
        self.setMouseTracking(True)

    def set_annotation_mode(self, mode: AnnotationMode):
        """Set the annotation mode"""
        self.annotation_mode = mode

        # Clear preview when not in click mode
        if mode != AnnotationMode.CLICK_ADD:
            self.preview_mask = None
            self.update_display()

        # Clear brush when not in brush mode
        if mode != AnnotationMode.BRUSH_ADD:
            self.brush_mask = None
            self.brush_points = []
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
        elif mode == AnnotationMode.BRUSH_ADD:
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

    def set_mask_transparency(self, transparency: float):
        """Set the transparency for mask overlay (0.0 = transparent, 1.0 = opaque)"""
        self.mask_transparency = max(0.0, min(1.0, transparency))
        self.update_display()

    def set_brush_size(self, size: int):
        """Set the brush size for brush annotation mode"""
        self.brush_size = max(1, size)

    def set_zoom_factor(self, zoom: float):
        """Set the zoom factor"""
        self.zoom_factor = max(0.1, min(10.0, zoom))  # Limit zoom range
        self.update_display()

    def set_pan_offset(self, offset: tuple):
        """Set the pan offset"""
        self.pan_offset = offset
        self.update_display()

    def reset_view(self):
        """Reset zoom and pan to default"""
        self.zoom_factor = 1.0
        self.pan_offset = (0, 0)
        self.update_display()

    def zoom_to_fit(self):
        """Reset zoom to fit the image in the widget"""
        self.zoom_factor = 1.0
        self.pan_offset = (0, 0)
        self.update_display()

    def update_display(self):
        """Update the display with current image and overlays"""
        if self.image is None:
            return

        # Create display image
        display_image = self.image.copy()

        # Overlay masks if available
        if self.masks is not None:
            display_image = self._overlay_masks(display_image, self.masks)

        # Overlay preview mask if available and in click mode
        if (
            self.preview_mask is not None
            and self.annotation_mode == AnnotationMode.CLICK_ADD
        ):
            display_image = self._overlay_preview_mask(display_image, self.preview_mask)

        # Overlay brush mask if available and in brush mode
        if (
            self.brush_mask is not None
            and self.annotation_mode == AnnotationMode.BRUSH_ADD
        ):
            display_image = self._overlay_brush_mask(display_image, self.brush_mask)

        # Convert to QPixmap and store for painting
        h, w = display_image.shape[:2]
        bytes_per_line = 3 * w
        q_image = QImage(
            display_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )

        # Store the original pixmap without scaling
        self.overlay_image = QPixmap.fromImage(q_image)

        # Calculate transformation parameters but don't apply them to the pixmap
        widget_size = self.size()
        available_width = max(widget_size.width() - 4, 1)
        available_height = max(widget_size.height() - 4, 1)

        base_scale_x = available_width / w
        base_scale_y = available_height / h
        base_scale_factor = min(base_scale_x, base_scale_y)

        # Apply zoom on top of base scale
        self.scale_factor = base_scale_factor * self.zoom_factor

        # Calculate the size of the scaled image
        scaled_width = int(w * self.scale_factor)
        scaled_height = int(h * self.scale_factor)

        # Calculate offset with pan
        base_offset_x = (widget_size.width() - scaled_width) // 2
        base_offset_y = (widget_size.height() - scaled_height) // 2

        self.image_offset = (
            base_offset_x + self.pan_offset[0],
            base_offset_y + self.pan_offset[1],
        )

        # Clear the pixmap and trigger a repaint
        self.setPixmap(QPixmap())
        self.update()

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
                # Use the configurable transparency
                alpha = self.mask_transparency
                overlay[mask] = ((1 - alpha) * overlay[mask] + alpha * color).astype(
                    np.uint8
                )

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

    def resizeEvent(self, event):
        """Handle widget resize events by updating the display"""
        super().resizeEvent(event)
        # Update display when widget is resized to rescale the image
        if self.image is not None:
            self.update_display()

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

    def _overlay_preview_mask(
        self, image: np.ndarray, preview_mask: np.ndarray
    ) -> np.ndarray:
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

    def _overlay_brush_mask(
        self, image: np.ndarray, brush_mask: np.ndarray
    ) -> np.ndarray:
        """Overlay brush mask with a semi-transparent green color"""
        if brush_mask is None:
            return image

        overlay = image.copy()
        # Use green color for brush with medium transparency
        brush_color = np.array([0, 255, 0], dtype=np.uint8)  # Green
        mask = brush_mask > 0
        if np.any(mask):
            # Medium transparency for brush strokes
            overlay[mask] = (0.7 * overlay[mask] + 0.3 * brush_color).astype(np.uint8)

        return overlay

    def _widget_to_image_coords(self, widget_x: int, widget_y: int) -> Tuple[int, int]:
        """Convert widget coordinates to image coordinates accounting for pan and zoom"""
        if self.image is None:
            return (0, 0)

        # Get the image size
        h, w = self.image.shape[:2]

        # Calculate the widget size
        widget_size = self.size()

        # Calculate base scale factor (before zoom)
        available_width = max(widget_size.width() - 4, 1)
        available_height = max(widget_size.height() - 4, 1)
        base_scale_x = available_width / w
        base_scale_y = available_height / h
        base_scale_factor = min(base_scale_x, base_scale_y)

        # Total scale factor includes zoom
        total_scale = base_scale_factor * self.zoom_factor

        # Calculate base offset (centering) before pan
        scaled_width = w * total_scale
        scaled_height = h * total_scale
        base_offset_x = (widget_size.width() - scaled_width) // 2
        base_offset_y = (widget_size.height() - scaled_height) // 2

        # Total offset includes pan
        total_offset_x = base_offset_x + self.pan_offset[0]
        total_offset_y = base_offset_y + self.pan_offset[1]

        # Convert widget coordinates to image coordinates
        image_x = (widget_x - total_offset_x) / total_scale
        image_y = (widget_y - total_offset_y) / total_scale

        # Round and clamp to image bounds
        image_x = max(0, min(int(round(image_x)), w - 1))
        image_y = max(0, min(int(round(image_y)), h - 1))

        return (image_x, image_y)

    def _image_to_widget_coords(self, image_x: int, image_y: int) -> Tuple[int, int]:
        """Convert image coordinates to widget coordinates accounting for pan and zoom"""
        if self.image is None:
            return (0, 0)

        # Get the image size
        h, w = self.image.shape[:2]

        # Calculate the widget size
        widget_size = self.size()

        # Calculate base scale factor (before zoom)
        available_width = max(widget_size.width() - 4, 1)
        available_height = max(widget_size.height() - 4, 1)
        base_scale_x = available_width / w
        base_scale_y = available_height / h
        base_scale_factor = min(base_scale_x, base_scale_y)

        # Total scale factor includes zoom
        total_scale = base_scale_factor * self.zoom_factor

        # Calculate base offset (centering) before pan
        scaled_width = w * total_scale
        scaled_height = h * total_scale
        base_offset_x = (widget_size.width() - scaled_width) // 2
        base_offset_y = (widget_size.height() - scaled_height) // 2

        # Total offset includes pan
        total_offset_x = base_offset_x + self.pan_offset[0]
        total_offset_y = base_offset_y + self.pan_offset[1]

        # Convert image coordinates to widget coordinates
        widget_x = int(image_x * total_scale + total_offset_x)
        widget_y = int(image_y * total_scale + total_offset_y)

        return (widget_x, widget_y)

    def mousePressEvent(self, event):
        """Handle mouse press events"""
        pos = event.position().toPoint()

        # Handle pan mode (middle mouse button or Ctrl+left click)
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self.panning = True
            self.last_pan_point = (pos.x(), pos.y())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if self.annotation_mode == AnnotationMode.VIEW:
            return

        image_coords = self._widget_to_image_coords(pos.x(), pos.y())

        if self.annotation_mode == AnnotationMode.CLICK_ADD:
            self.point_clicked.emit(image_coords)

        elif self.annotation_mode == AnnotationMode.BOX_ADD:
            if event.button() == Qt.MouseButton.LeftButton:
                self.drawing_box = True
                self.box_start = image_coords
                self.box_end = image_coords

        elif self.annotation_mode == AnnotationMode.BRUSH_ADD:
            if event.button() == Qt.MouseButton.LeftButton:
                self.drawing_brush = True
                # Initialize brush mask if needed
                if self.brush_mask is None and self.image is not None:
                    h, w = self.image.shape[:2]
                    self.brush_mask = np.zeros((h, w), dtype=np.uint8)
                self.brush_points = [image_coords]
                self._add_brush_stroke(image_coords)

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
        pos = event.position().toPoint()

        # Store last mouse position for brush preview
        self._last_mouse_pos = pos

        # Handle panning
        if self.panning and self.last_pan_point:
            dx = pos.x() - self.last_pan_point[0]
            dy = pos.y() - self.last_pan_point[1]

            # Calculate new pan offset
            new_pan_x = self.pan_offset[0] + dx
            new_pan_y = self.pan_offset[1] + dy

            # Add some reasonable limits to prevent panning too far
            if self.image is not None:
                widget_size = self.size()
                h, w = self.image.shape[:2]

                # Calculate scaled image size
                available_width = max(widget_size.width() - 4, 1)
                available_height = max(widget_size.height() - 4, 1)
                base_scale_x = available_width / w
                base_scale_y = available_height / h
                base_scale_factor = min(base_scale_x, base_scale_y)
                total_scale = base_scale_factor * self.zoom_factor

                scaled_width = w * total_scale
                scaled_height = h * total_scale

                # Limit panning to keep at least part of the image visible
                max_pan_x = scaled_width * 0.8
                max_pan_y = scaled_height * 0.8

                new_pan_x = max(-max_pan_x, min(max_pan_x, new_pan_x))
                new_pan_y = max(-max_pan_y, min(max_pan_y, new_pan_y))

            self.pan_offset = (new_pan_x, new_pan_y)
            self.last_pan_point = (pos.x(), pos.y())
            self.update_display()
            return

        if self.annotation_mode == AnnotationMode.BOX_ADD and self.drawing_box:
            self.box_end = self._widget_to_image_coords(pos.x(), pos.y())
            self.update()  # Trigger repaint to show box
        elif self.annotation_mode == AnnotationMode.BRUSH_ADD and self.drawing_brush:
            image_coords = self._widget_to_image_coords(pos.x(), pos.y())
            self.brush_points.append(image_coords)
            self._add_brush_stroke(image_coords)
            # Don't call SAM here - wait for mouse release
            # Update to show current brush stroke
            self.update()
        elif self.annotation_mode == AnnotationMode.CLICK_ADD:
            # Emit hover signal for live preview
            image_coords = self._widget_to_image_coords(pos.x(), pos.y())
            self.mouse_hover.emit(image_coords)

        # Always update brush preview when in brush mode
        if self.annotation_mode == AnnotationMode.BRUSH_ADD:
            self.update()

    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        # Handle pan release
        if self.panning:
            self.panning = False
            self.last_pan_point = None
            # Reset cursor based on current mode
            if self.annotation_mode == AnnotationMode.VIEW:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            else:
                self.setCursor(Qt.CursorShape.CrossCursor)
            return

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
        elif self.annotation_mode == AnnotationMode.BRUSH_ADD and self.drawing_brush:
            self.drawing_brush = False
            if self.brush_mask is not None:
                # Emit the brush mask for SAM processing
                self.brush_drawn.emit(self.brush_mask.copy())
                # Clear the brush mask for next stroke
                self.brush_mask = None
                self.brush_points = []
                self.update_display()

    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming"""
        # Only zoom if Ctrl is held or if we're in view mode
        if (
            event.modifiers() & Qt.KeyboardModifier.ControlModifier
            or self.annotation_mode == AnnotationMode.VIEW
        ):
            # Get wheel delta
            delta = event.angleDelta().y()

            # Calculate zoom factor (positive delta = zoom in, negative = zoom out)
            zoom_in = delta > 0
            zoom_factor = 1.1 if zoom_in else 1.0 / 1.1

            # Apply zoom
            new_zoom = self.zoom_factor * zoom_factor
            new_zoom = max(0.1, min(10.0, new_zoom))  # Limit zoom range

            if new_zoom != self.zoom_factor:
                self.zoom_factor = new_zoom
                self.update_display()

            event.accept()
        else:
            super().wheelEvent(event)

    def _add_brush_stroke(self, center_point):
        """Add a brush stroke at the given center point"""
        if self.brush_mask is None or self.image is None:
            return

        x, y = center_point
        h, w = self.image.shape[:2]

        # Create circular brush
        radius = self.brush_size // 2

        # Get bounds for the brush stroke
        y_min = max(0, y - radius)
        y_max = min(h, y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(w, x + radius + 1)

        # Create circular mask for this stroke
        for py in range(y_min, y_max):
            for px in range(x_min, x_max):
                distance = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
                if distance <= radius:
                    self.brush_mask[py, px] = 255

        # Update display to show live brush stroke
        self.update_display()

    def paintEvent(self, event):
        """Custom paint event to draw image with transformations and overlays"""
        painter = QPainter(self)

        # Fill background
        painter.fillRect(self.rect(), QColor(53, 53, 53))

        # Draw the image if available
        if self.overlay_image is not None and self.image is not None:
            h, w = self.image.shape[:2]

            # Calculate the scaled size
            scaled_width = int(w * self.scale_factor)
            scaled_height = int(h * self.scale_factor)

            # Draw the scaled image at the calculated offset
            target_rect = (
                self.image_offset[0],
                self.image_offset[1],
                scaled_width,
                scaled_height,
            )

            painter.drawPixmap(
                target_rect[0],
                target_rect[1],
                target_rect[2],
                target_rect[3],
                self.overlay_image,
            )

        # Draw bounding box during box drawing
        if (
            self.annotation_mode == AnnotationMode.BOX_ADD
            and self.drawing_box
            and self.box_start
            and self.box_end
        ):
            painter.setPen(QPen(QColor(0, 120, 212), 2, Qt.PenStyle.DashLine))

            # Convert image coordinates to widget coordinates
            x1, y1 = self._image_to_widget_coords(self.box_start[0], self.box_start[1])
            x2, y2 = self._image_to_widget_coords(self.box_end[0], self.box_end[1])

            painter.drawRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))

        # Draw current brush stroke during drawing
        if (
            self.annotation_mode == AnnotationMode.BRUSH_ADD
            and self.drawing_brush
            and len(self.brush_points) > 1
        ):
            painter.setPen(QPen(QColor(255, 255, 0), 3, Qt.PenStyle.SolidLine))

            # Draw lines connecting brush points
            for i in range(1, len(self.brush_points)):
                x1, y1 = self._image_to_widget_coords(
                    self.brush_points[i - 1][0], self.brush_points[i - 1][1]
                )
                x2, y2 = self._image_to_widget_coords(
                    self.brush_points[i][0], self.brush_points[i][1]
                )
                painter.drawLine(x1, y1, x2, y2)

        # Draw brush preview circle for brush mode
        if (
            self.annotation_mode == AnnotationMode.BRUSH_ADD
            and hasattr(self, "_last_mouse_pos")
            and self._last_mouse_pos
        ):
            painter.setPen(QPen(QColor(0, 255, 0), 2, Qt.PenStyle.SolidLine))

            # Calculate the brush radius in widget coordinates
            if self.image is not None:
                brush_radius_widget = max(
                    1, int(self.brush_size * self.scale_factor / 2)
                )

                # Draw brush preview circle at last mouse position
                painter.drawEllipse(
                    self._last_mouse_pos.x() - brush_radius_widget,
                    self._last_mouse_pos.y() - brush_radius_widget,
                    brush_radius_widget * 2,
                    brush_radius_widget * 2,
                )
