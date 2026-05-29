from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import QEvent, Qt, QRect, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QLabel


class AnnotationMode(Enum):
    """Annotation mode enumeration"""

    VIEW = "view"
    CLICK_ADD = "click_add"
    BOX_ADD = "box_add"
    PAINT_ADD = "paint_add"
    MASK_REMOVE = "mask_remove"
    EDIT_CELL_ID = "edit_cell_id"


class InteractiveFrameWidget(QLabel):
    """Interactive frame widget for annotation correction"""

    point_clicked = pyqtSignal(tuple)  # (x, y)
    box_drawn = pyqtSignal(tuple)  # (x1, y1, x2, y2)
    paint_mask_ready = pyqtSignal(object)  # bool ndarray (H, W)
    mask_clicked = pyqtSignal(tuple)  # (x, y) for mask removal
    cell_id_edit_requested = pyqtSignal(tuple, int)  # (x, y), current_cell_id
    cell_picked = pyqtSignal(int)  # cell_id at click (view mode)
    mouse_hover = pyqtSignal(tuple)  # (x, y) for live preview
    view_changed = pyqtSignal(float, float, float)  # zoom, pan_x, pan_y

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
        self._suppress_view_signal = False
        self._click_pending = False
        self._press_widget_pos: Optional[Tuple[int, int]] = None
        self._drag_pan_threshold = 4

        # Box drawing
        self.drawing_box = False
        self.box_start = None
        self.box_end = None

        # Brush painting (SAM mask prompt)
        self.brush_radius = 6
        self.painting = False
        self.paint_erase = False
        self.paint_mask: Optional[np.ndarray] = None
        self._last_paint_point: Optional[Tuple[int, int]] = None

        # Enable mouse tracking for hover events
        self.setMouseTracking(True)

        # Cell ID overlay toggle
        self.show_cell_ids = True
        self.cell_id_scale = 1.0
        self.highlight_cell_id: Optional[int] = None
        self.solo_cell_id: Optional[int] = None

    def set_annotation_mode(self, mode: AnnotationMode):
        """Set the annotation mode"""
        self.annotation_mode = mode
        self._reset_pointer_state()

        # Clear preview when not in click mode
        if mode != AnnotationMode.CLICK_ADD:
            self.preview_mask = None

        if mode != AnnotationMode.PAINT_ADD:
            self._clear_paint_mask()

        self._update_cursor_for_mode()
        self.update_display()

    def _update_cursor_for_mode(self) -> None:
        if self.annotation_mode == AnnotationMode.VIEW:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)

    def _reset_pointer_state(self) -> None:
        """Clear transient mouse state (avoids stuck pan/draw after SAM or dialogs)."""
        self.panning = False
        self.last_pan_point = None
        self._click_pending = False
        self._press_widget_pos = None
        self.drawing_box = False
        self.box_start = None
        self.box_end = None
        if self.painting:
            self._clear_paint_mask()

    def leaveEvent(self, event: QEvent) -> None:
        self._reset_pointer_state()
        self._update_cursor_for_mode()
        super().leaveEvent(event)

    def focusOutEvent(self, event) -> None:
        self._reset_pointer_state()
        self._update_cursor_for_mode()
        super().focusOutEvent(event)

    def set_brush_radius(self, radius: int) -> None:
        self.brush_radius = max(2, min(32, int(radius)))

    def _clear_paint_mask(self) -> None:
        self.painting = False
        self.paint_erase = False
        self.paint_mask = None
        self._last_paint_point = None

    def _ensure_paint_buffer(self) -> None:
        if self.image is None:
            return
        h, w = self.image.shape[:2]
        if self.paint_mask is None or self.paint_mask.shape != (h, w):
            self.paint_mask = np.zeros((h, w), dtype=np.uint8)

    def _paint_stroke_point(self, x: int, y: int, erase: bool) -> None:
        self._ensure_paint_buffer()
        if self.paint_mask is None:
            return
        value = 0 if erase else 255
        cv2.circle(
            self.paint_mask, (x, y), self.brush_radius, int(value), thickness=-1
        )

    def _paint_stroke_line(
        self, x0: int, y0: int, x1: int, y1: int, erase: bool
    ) -> None:
        self._ensure_paint_buffer()
        if self.paint_mask is None:
            return
        value = 0 if erase else 255
        thickness = max(2, self.brush_radius * 2)
        cv2.line(
            self.paint_mask,
            (x0, y0),
            (x1, y1),
            int(value),
            thickness=thickness,
        )

    def set_show_cell_ids(self, show: bool):
        """Toggle cell ID overlay visibility"""
        self.show_cell_ids = show
        self.update()

    def set_cell_id_scale(self, scale: float):
        """Scale factor for ID label size (1.0 = default)."""
        self.cell_id_scale = max(0.5, min(2.5, scale))
        self.update()

    def set_highlight_cell_id(self, cell_id: Optional[int]) -> None:
        self.highlight_cell_id = cell_id
        self.update_display()

    def set_solo_cell_id(self, cell_id: Optional[int]) -> None:
        self.solo_cell_id = cell_id
        self.update_display()

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

    def clear_all_state(self):
        """Clear all internal state including images and masks"""
        self.image = None
        self.masks = None
        self.preview_mask = None
        self.preview_score = 0.0
        self.overlay_image = None
        self.scale_factor = 1.0
        self.image_offset = (0, 0)
        self.pan_offset = (0, 0)
        self.zoom_factor = 1.0
        self.panning = False
        self.last_pan_point = None
        self._clear_paint_mask()

        # Clear display
        self.clear()
        self.setText("No image loaded")

    def set_mask_transparency(self, transparency: float):
        """Set the transparency for mask overlay (0.0 = transparent, 1.0 = opaque)"""
        self.mask_transparency = max(0.0, min(1.0, transparency))
        self.update_display()

    def set_zoom_factor(self, zoom: float):
        """Set the zoom factor"""
        self.zoom_factor = max(0.1, min(10.0, zoom))  # Limit zoom range
        self._update_view_transform()
        self._emit_view_changed()

    def set_pan_offset(self, offset: tuple):
        """Set the pan offset"""
        self.pan_offset = self._normalize_pan(offset)
        self._update_view_transform()
        self._emit_view_changed()

    def apply_view_state(self, zoom: float, pan: tuple) -> None:
        """Apply zoom/pan without emitting view_changed (for linked panes)."""
        self._suppress_view_signal = True
        self.zoom_factor = max(0.1, min(10.0, zoom))
        self.pan_offset = self._normalize_pan(pan)
        self._update_view_transform()
        self._suppress_view_signal = False

    @staticmethod
    def _normalize_pan(pan: tuple) -> Tuple[int, int]:
        return int(round(pan[0])), int(round(pan[1]))

    def _emit_view_changed(self) -> None:
        if not self._suppress_view_signal:
            self.view_changed.emit(
                self.zoom_factor,
                float(self.pan_offset[0]),
                float(self.pan_offset[1]),
            )

    def reset_view(self):
        """Reset zoom and pan to default"""
        self.zoom_factor = 1.0
        self.pan_offset = (0, 0)
        self._update_view_transform()
        self._emit_view_changed()

    def zoom_to_fit(self):
        """Reset zoom to fit the image in the widget"""
        self.zoom_factor = 1.0
        self.pan_offset = (0, 0)
        self._update_view_transform()
        self._emit_view_changed()

    def update_display(self):
        """Rebuild overlay from image/masks and refresh view transform."""
        if self.image is None:
            return
        self._rebuild_overlay_pixmap()
        self._update_view_transform()

    def _rebuild_overlay_pixmap(self) -> None:
        """Expensive: composite image + masks into a cached pixmap."""
        if self.image is None:
            return

        display_image = self.image.copy()

        if self.masks is not None:
            display_image = self._overlay_masks(display_image, self.masks)

        if (
            self.preview_mask is not None
            and self.annotation_mode == AnnotationMode.CLICK_ADD
        ):
            display_image = self._overlay_preview_mask(display_image, self.preview_mask)

        h, w = display_image.shape[:2]
        bytes_per_line = 3 * w
        q_image = QImage(
            display_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )
        self.overlay_image = QPixmap.fromImage(q_image)

    def _update_view_transform(self) -> None:
        """Cheap: recompute scale/offset and repaint (pan/zoom/resize)."""
        if self.image is None:
            return

        h, w = self.image.shape[:2]
        widget_size = self.size()
        available_width = max(widget_size.width() - 4, 1)
        available_height = max(widget_size.height() - 4, 1)

        base_scale_factor = min(available_width / w, available_height / h)
        self.scale_factor = base_scale_factor * self.zoom_factor

        scaled_width = int(w * self.scale_factor)
        scaled_height = int(h * self.scale_factor)

        base_offset_x = (widget_size.width() - scaled_width) // 2
        base_offset_y = (widget_size.height() - scaled_height) // 2

        self.image_offset = (
            int(base_offset_x + self.pan_offset[0]),
            int(base_offset_y + self.pan_offset[1]),
        )

        self.setPixmap(QPixmap())
        self.update()

    def _overlay_masks(self, image: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """Overlay masks on image with transparency and optional cell IDs"""
        if masks is None or np.max(masks) == 0:
            return image

        # Safety check: ensure mask and image dimensions match
        if image.shape[:2] != masks.shape[:2]:
            print(
                f"Warning: Image shape {image.shape[:2]} != mask shape {masks.shape[:2]}. Skipping mask overlay."
            )
            return image

        overlay = image.copy()

        # Generate colors for each mask
        num_objects = int(np.max(masks))
        colors = self._generate_colors(num_objects)

        for obj_id in range(1, num_objects + 1):
            mask = masks == obj_id
            if not np.any(mask):
                continue

            if self.solo_cell_id is not None and obj_id != self.solo_cell_id:
                continue

            color = np.array(colors[obj_id - 1], dtype=np.uint8)
            alpha = self.mask_transparency
            if self.highlight_cell_id is not None and obj_id == self.highlight_cell_id:
                alpha = min(1.0, alpha + 0.35)

            overlay[mask] = ((1 - alpha) * overlay[mask] + alpha * color).astype(
                np.uint8
            )

        return overlay

    def _cell_id_font(self) -> QFont:
        """Font for labels — scales uniformly with label size and zoom."""
        # Height in image pixels at 100% label size; multiplied by scale_factor for screen.
        image_px = 14.0 * self.cell_id_scale
        screen_px = max(8, int(round(image_px * self.scale_factor)))
        font = QFont()
        font.setPixelSize(screen_px)
        font.setBold(True)
        font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        return font

    def _paint_cell_ids(self, painter: QPainter) -> None:
        """Draw cell ID labels in screen space (consistent typeface at all sizes)."""
        if not self.show_cell_ids or self.masks is None or self.image is None:
            return

        num_objects = int(np.max(self.masks))
        if num_objects <= 0:
            return

        painter.setFont(self._cell_id_font())
        metrics = painter.fontMetrics()
        outline = QColor(0, 0, 0, 230)
        fill = QColor(255, 255, 255)

        for obj_id in range(1, num_objects + 1):
            if self.solo_cell_id is not None and obj_id != self.solo_cell_id:
                continue

            mask = self.masks == obj_id
            if not np.any(mask):
                continue

            y_coords, x_coords = np.where(mask)
            cx = int(round(float(np.mean(x_coords))))
            cy = int(round(float(np.mean(y_coords))))
            wx, wy = self._image_to_widget_coords(cx, cy)

            text = str(obj_id)
            tw = metrics.horizontalAdvance(text)
            x = int(wx - tw / 2)
            y = int(wy + (metrics.ascent() - metrics.descent()) / 2)

            for ox, oy in (
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ):
                painter.setPen(outline)
                painter.drawText(x + ox, y + oy, text)

            painter.setPen(fill)
            painter.drawText(x, y, text)

    def resizeEvent(self, event):
        """Handle widget resize events by updating the display"""
        super().resizeEvent(event)
        if self.image is not None:
            self._update_view_transform()

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

        image_coords = self._widget_to_image_coords(pos.x(), pos.y())

        if (
            self.annotation_mode == AnnotationMode.VIEW
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._click_pending = True
            self._press_widget_pos = (pos.x(), pos.y())
            return

        if self.annotation_mode == AnnotationMode.CLICK_ADD:
            self.point_clicked.emit(image_coords)

        elif self.annotation_mode == AnnotationMode.BOX_ADD:
            if event.button() == Qt.MouseButton.LeftButton:
                self.drawing_box = True
                self.box_start = image_coords
                self.box_end = image_coords

        elif self.annotation_mode == AnnotationMode.PAINT_ADD:
            if event.button() in (
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.RightButton,
            ):
                erase = event.button() == Qt.MouseButton.RightButton
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    erase = not erase
                self.painting = True
                self.paint_erase = erase
                x, y = image_coords
                self._paint_stroke_point(x, y, erase)
                self._last_paint_point = (x, y)
                self.update()

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

        if (
            self._click_pending
            and self._press_widget_pos is not None
            and event.buttons() & Qt.MouseButton.LeftButton
            and self.annotation_mode == AnnotationMode.VIEW
        ):
            dx = pos.x() - self._press_widget_pos[0]
            dy = pos.y() - self._press_widget_pos[1]
            if (dx * dx + dy * dy) >= self._drag_pan_threshold**2:
                self._click_pending = False
                self.panning = True
                self.last_pan_point = (pos.x(), pos.y())
                self.setCursor(Qt.CursorShape.ClosedHandCursor)

        # Handle panning (abort if button was released without a release event)
        pan_buttons = (
            Qt.MouseButton.LeftButton | Qt.MouseButton.MiddleButton
        )
        if self.panning and not (event.buttons() & pan_buttons):
            self._reset_pointer_state()
            self._update_cursor_for_mode()
        elif self.panning and self.last_pan_point:
            dx = pos.x() - self.last_pan_point[0]
            dy = pos.y() - self.last_pan_point[1]

            new_pan_x = self.pan_offset[0] + dx
            new_pan_y = self.pan_offset[1] + dy

            if self.image is not None:
                widget_size = self.size()
                h, w = self.image.shape[:2]
                available_width = max(widget_size.width() - 4, 1)
                available_height = max(widget_size.height() - 4, 1)
                base_scale_factor = min(
                    available_width / w, available_height / h
                )
                total_scale = base_scale_factor * self.zoom_factor
                scaled_width = w * total_scale
                scaled_height = h * total_scale
                max_pan_x = scaled_width * 0.8
                max_pan_y = scaled_height * 0.8
                new_pan_x = max(-max_pan_x, min(max_pan_x, new_pan_x))
                new_pan_y = max(-max_pan_y, min(max_pan_y, new_pan_y))

            self.pan_offset = self._normalize_pan((new_pan_x, new_pan_y))
            self.last_pan_point = (pos.x(), pos.y())
            self._update_view_transform()
            self._emit_view_changed()
            return

        if self.annotation_mode == AnnotationMode.BOX_ADD and self.drawing_box:
            self.box_end = self._widget_to_image_coords(pos.x(), pos.y())
            self.update()  # Trigger repaint to show box
        elif (
            self.annotation_mode == AnnotationMode.PAINT_ADD
            and self.painting
            and event.buttons()
            & (Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton)
        ):
            erase = self.paint_erase
            x, y = self._widget_to_image_coords(pos.x(), pos.y())
            if self._last_paint_point is not None:
                self._paint_stroke_line(
                    self._last_paint_point[0],
                    self._last_paint_point[1],
                    x,
                    y,
                    erase,
                )
            else:
                self._paint_stroke_point(x, y, erase)
            self._last_paint_point = (x, y)
            self.update()
        elif self.annotation_mode == AnnotationMode.CLICK_ADD:
            # Emit hover signal for live preview
            image_coords = self._widget_to_image_coords(pos.x(), pos.y())
            self.mouse_hover.emit(image_coords)

    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if (
            self._click_pending
            and event.button() == Qt.MouseButton.LeftButton
            and self.annotation_mode == AnnotationMode.VIEW
            and self.masks is not None
        ):
            pos = event.position().toPoint()
            x, y = self._widget_to_image_coords(pos.x(), pos.y())
            if 0 <= y < self.masks.shape[0] and 0 <= x < self.masks.shape[1]:
                cell_id = int(self.masks[y, x])
                if cell_id > 0:
                    self.cell_picked.emit(cell_id)
            self._click_pending = False
            self._press_widget_pos = None
            return

        # Handle pan release
        if self.panning:
            self._reset_pointer_state()
            self._update_cursor_for_mode()
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

        if self.annotation_mode == AnnotationMode.PAINT_ADD and self.painting:
            self.painting = False
            self._last_paint_point = None
            if self.paint_mask is not None and np.any(self.paint_mask):
                painted = self.paint_mask > 0
                self._clear_paint_mask()
                self.update()
                self.paint_mask_ready.emit(painted)
            else:
                self._clear_paint_mask()
                self.update()
            if self.annotation_mode == AnnotationMode.PAINT_ADD:
                self.setCursor(Qt.CursorShape.CrossCursor)

    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming (all annotation modes)."""
        if self.image is None:
            super().wheelEvent(event)
            return

        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return

        zoom_in = delta > 0
        zoom_factor = 1.1 if zoom_in else 1.0 / 1.1
        new_zoom = self.zoom_factor * zoom_factor
        new_zoom = max(0.1, min(10.0, new_zoom))

        if new_zoom != self.zoom_factor:
            mouse_pos = event.position().toPoint()
            mouse_x, mouse_y = mouse_pos.x(), mouse_pos.y()
            image_coords_before = self._widget_to_image_coords(mouse_x, mouse_y)
            self.zoom_factor = new_zoom
            widget_coords_after = self._image_to_widget_coords(
                image_coords_before[0], image_coords_before[1]
            )
            pan_adjust_x = mouse_x - widget_coords_after[0]
            pan_adjust_y = mouse_y - widget_coords_after[1]
            self.pan_offset = self._normalize_pan(
                (
                    self.pan_offset[0] + pan_adjust_x,
                    self.pan_offset[1] + pan_adjust_y,
                )
            )
            self._update_view_transform()
            self._emit_view_changed()

        event.accept()

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
                int(target_rect[0]),
                int(target_rect[1]),
                int(target_rect[2]),
                int(target_rect[3]),
                self.overlay_image,
            )

            self._paint_cell_ids(painter)

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

        self._paint_brush_overlay(painter)

    def _paint_brush_overlay(self, painter: QPainter) -> None:
        if (
            self.paint_mask is None
            or not np.any(self.paint_mask)
            or self.image is None
        ):
            return

        h, w = self.image.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[self.paint_mask > 0] = (50, 220, 100, 140)

        q_image = QImage(
            np.ascontiguousarray(rgba).data, w, h, 4 * w, QImage.Format.Format_RGBA8888
        ).copy()
        scaled_width = int(w * self.scale_factor)
        scaled_height = int(h * self.scale_factor)
        target = QRect(
            int(self.image_offset[0]),
            int(self.image_offset[1]),
            scaled_width,
            scaled_height,
        )
        painter.drawImage(target, q_image)
