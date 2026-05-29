"""Timeline with draggable trim handles and playhead."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget


class TrimTimelineWidget(QWidget):
    """
    Visual timeline: dimmed excluded regions, highlighted trim range,
    draggable start/end handles, click/drag playhead.
    """

    clip_range_changed = pyqtSignal(int, int)  # start, end (inclusive indices)
    current_frame_changed = pyqtSignal(int)

    HANDLE_PX = 10
    MIN_TRIM_FRAMES = 1

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(44)
        self.setMaximumHeight(56)
        self.setMouseTracking(True)

        self._frame_count = 1
        self._clip_start = 0
        self._clip_end = 0
        self._current = 0
        self._drag_mode: Optional[str] = None  # "start" | "end" | "playhead"

    def set_frame_count(self, count: int) -> None:
        self._frame_count = max(1, count)
        self._clip_end = self._frame_count - 1
        self._clip_start = 0
        self._current = 0
        self.update()

    def set_clip_range(self, start: int, end: int) -> None:
        if self._frame_count <= 0:
            return
        start = max(0, min(start, self._frame_count - 1))
        end = max(start, min(end, self._frame_count - 1))
        if end - start + 1 < self.MIN_TRIM_FRAMES:
            end = min(self._frame_count - 1, start)
        self._clip_start = start
        self._clip_end = end
        self._current = max(start, min(self._current, end))
        self.update()

    def get_clip_range(self) -> tuple[int, int]:
        return self._clip_start, self._clip_end

    def set_current_frame(self, frame: int) -> None:
        frame = max(0, min(frame, self._frame_count - 1))
        if frame != self._current:
            self._current = frame
            self.update()

    def get_current_frame(self) -> int:
        return self._current

    def _track_rect(self):
        m = 12
        y = 10
        h = 22
        return m, y, max(1, self.width() - 2 * m), h

    def _frame_to_x(self, frame: int) -> int:
        x, _, w, _ = self._track_rect()
        if self._frame_count <= 1:
            return x
        t = frame / (self._frame_count - 1)
        return int(x + t * w)

    def _x_to_frame(self, px: int) -> int:
        x, _, w, _ = self._track_rect()
        if w <= 0:
            return 0
        t = (px - x) / w
        t = max(0.0, min(1.0, t))
        return int(round(t * (self._frame_count - 1)))

    def _hit_handle(self, px: int, py: int) -> Optional[str]:
        _, ty, _, th = self._track_rect()
        if not (ty <= py <= ty + th + 16):
            return None
        sx = self._frame_to_x(self._clip_start)
        ex = self._frame_to_x(self._clip_end)
        if abs(px - sx) <= self.HANDLE_PX + 4:
            return "start"
        if abs(px - ex) <= self.HANDLE_PX + 4:
            return "end"
        return None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(43, 43, 43))

        if self._frame_count <= 0:
            return

        x, y, w, h = self._track_rect()
        track_bottom = y + h

        # Full track background
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(60, 60, 60))
        painter.drawRoundedRect(x, y, w, h, 4, 4)

        sx = self._frame_to_x(self._clip_start)
        ex = self._frame_to_x(self._clip_end)
        sel_w = max(2, ex - sx)

        # Selected range
        painter.setBrush(QColor(0, 120, 212, 140))
        painter.drawRect(sx, y, sel_w, h)

        # Dim excluded regions
        painter.setBrush(QColor(0, 0, 0, 110))
        if sx > x:
            painter.drawRect(x, y, sx - x, h)
        if ex < x + w:
            painter.drawRect(ex, y, x + w - ex, h)

        # Playhead
        cx = self._frame_to_x(self._current)
        painter.setPen(QPen(QColor(255, 220, 80), 2))
        painter.drawLine(cx, y - 4, cx, track_bottom + 6)

        # Handles
        for hx, label in ((sx, "start"), (ex, "end")):
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.setBrush(QColor(0, 180, 255) if label == "start" else QColor(0, 140, 220))
            painter.drawRect(hx - 4, y - 2, 8, h + 4)

        # Labels
        painter.setPen(QColor(180, 180, 180))
        painter.drawText(x, track_bottom + 14, f"1")
        painter.drawText(
            x + w - 24, track_bottom + 14, str(self._frame_count)
        )

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        px = int(event.position().x())
        py = int(event.position().y())
        handle = self._hit_handle(px, py)
        if handle:
            self._drag_mode = handle
            return
        frame = self._x_to_frame(px)
        self._current = frame
        self._drag_mode = "playhead"
        self.current_frame_changed.emit(self._current)
        self.update()

    def mouseMoveEvent(self, event):
        px = int(event.position().x())
        if self._drag_mode == "start":
            f = self._x_to_frame(px)
            f = min(f, self._clip_end)
            if self._clip_end - f + 1 >= self.MIN_TRIM_FRAMES:
                self._clip_start = f
        elif self._drag_mode == "end":
            f = self._x_to_frame(px)
            f = max(f, self._clip_start)
            if f - self._clip_start + 1 >= self.MIN_TRIM_FRAMES:
                self._clip_end = f
        elif self._drag_mode == "playhead":
            new_f = self._x_to_frame(px)
            if new_f != self._current:
                self._current = new_f
                self.current_frame_changed.emit(self._current)
        else:
            handle = self._hit_handle(px, int(event.position().y()))
            if handle:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def mouseReleaseEvent(self, event):
        if self._drag_mode in ("start", "end"):
            self._emit_clip()
        self._drag_mode = None
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def _emit_clip(self) -> None:
        self._current = max(self._clip_start, min(self._current, self._clip_end))
        self.clip_range_changed.emit(self._clip_start, self._clip_end)
        self.update()

    def reset_clip(self) -> None:
        self.set_clip_range(0, self._frame_count - 1)
