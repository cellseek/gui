"""Preview pane with optional crop rectangle for preprocessing."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QLabel


class PreprocessPreviewWidget(QLabel):
    """Shows an RGB frame and supports drag-to-crop in image coordinates."""

    crop_changed = pyqtSignal(object)  # Optional[Tuple[int,int,int,int]]

    def __init__(self):
        super().__init__()
        self.setMinimumSize(480, 360)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            "background-color: #1e1e1e; border: 1px solid #606060;"
        )
        self.setMouseTracking(True)

        self._image: Optional[np.ndarray] = None
        self._pixmap: Optional[QPixmap] = None
        self._scale = 1.0
        self._offset = (0, 0)
        self._crop: Optional[Tuple[int, int, int, int]] = None
        self._show_crop_overlay = False
        self._crop_mode = False
        self._dragging = False
        self._drag_start_img: Optional[Tuple[int, int]] = None
        self._drag_current_img: Optional[Tuple[int, int]] = None

    def set_crop_mode(self, enabled: bool) -> None:
        self._crop_mode = enabled
        if enabled:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self._dragging = False

    def set_crop_rect(self, rect: Optional[Tuple[int, int, int, int]]) -> None:
        self._crop = rect
        self.update()

    def set_show_crop_overlay(self, visible: bool) -> None:
        """Show crop rectangle overlay without clearing the stored crop."""
        self._show_crop_overlay = visible
        self.update()

    def get_crop_rect(self) -> Optional[Tuple[int, int, int, int]]:
        return self._crop

    def is_dragging_crop(self) -> bool:
        return self._dragging

    def set_image(self, rgb: Optional[np.ndarray]) -> None:
        self._image = rgb
        if rgb is None:
            self._pixmap = None
            self.update()
            return

        h, w = rgb.shape[:2]
        q_image = QImage(
            rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888
        ).copy()
        self._pixmap = QPixmap.fromImage(q_image)
        self._update_layout()
        self.update()

    def _update_layout(self) -> None:
        if self._pixmap is None or self._image is None:
            return
        pw, ph = self._pixmap.width(), self._pixmap.height()
        avail_w = max(self.width() - 8, 1)
        avail_h = max(self.height() - 8, 1)
        self._scale = min(avail_w / pw, avail_h / ph)
        sw, sh = int(pw * self._scale), int(ph * self._scale)
        self._offset = (
            (self.width() - sw) // 2,
            (self.height() - sh) // 2,
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_layout()
        self.update()

    def _widget_to_image(self, wx: int, wy: int) -> Tuple[int, int]:
        if self._image is None or self._scale <= 0:
            return (0, 0)
        ix = int((wx - self._offset[0]) / self._scale)
        iy = int((wy - self._offset[1]) / self._scale)
        h, w = self._image.shape[:2]
        return max(0, min(ix, w - 1)), max(0, min(iy, h - 1))

    def _image_to_widget(self, ix: int, iy: int) -> Tuple[int, int]:
        return (
            int(ix * self._scale + self._offset[0]),
            int(iy * self._scale + self._offset[1]),
        )

    def _normalize_rect(
        self, p1: Tuple[int, int], p2: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        if self._image is None:
            return (0, 0, 1, 1)
        x1, y1 = p1
        x2, y2 = p2
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        ih, iw = self._image.shape[:2]
        left = max(0, min(left, iw - 1))
        top = max(0, min(top, ih - 1))
        right = max(left + 1, min(right, iw))
        bottom = max(top + 1, min(bottom, ih))
        return (left, top, right - left, bottom - top)

    def mousePressEvent(self, event):
        if not self._crop_mode or self._image is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            self._dragging = True
            self._drag_start_img = self._widget_to_image(pos.x(), pos.y())
            self._drag_current_img = self._drag_start_img
            self.update()

    def mouseMoveEvent(self, event):
        if self._dragging and self._drag_start_img is not None:
            pos = event.position().toPoint()
            self._drag_current_img = self._widget_to_image(pos.x(), pos.y())
            self.update()

    def mouseReleaseEvent(self, event):
        if (
            self._dragging
            and event.button() == Qt.MouseButton.LeftButton
            and self._drag_start_img is not None
            and self._drag_current_img is not None
        ):
            rect = self._normalize_rect(self._drag_start_img, self._drag_current_img)
            self._crop = rect
            self.crop_changed.emit(rect)
            self._dragging = False
            self._drag_start_img = None
            self._drag_current_img = None
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        if self._pixmap is None:
            painter.setPen(QColor(160, 160, 160))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "No preview",
            )
            return

        sw = int(self._pixmap.width() * self._scale)
        sh = int(self._pixmap.height() * self._scale)
        ox, oy = self._offset
        painter.drawPixmap(ox, oy, sw, sh, self._pixmap)

        if not self._show_crop_overlay and not self._dragging:
            return

        rect = self._crop
        if self._dragging and self._drag_start_img and self._drag_current_img:
            rect = self._normalize_rect(self._drag_start_img, self._drag_current_img)

        if rect is not None:
            x, y, w, h = rect
            wx1, wy1 = self._image_to_widget(x, y)
            wx2, wy2 = self._image_to_widget(x + w, y + h)
            pen = QPen(QColor(0, 200, 255), 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawRect(
                min(wx1, wx2), min(wy1, wy2), abs(wx2 - wx1), abs(wy2 - wy1)
            )
            painter.fillRect(
                min(wx1, wx2),
                min(wy1, wy2),
                abs(wx2 - wx1),
                abs(wy2 - wy1),
                QColor(0, 120, 200, 40),
            )
