"""
Custom range slider widget for selecting start and end values
"""

from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QMouseEvent, QPainter, QPen
from PyQt6.QtWidgets import QStyle, QStyleOptionSlider, QWidget


class RangeSlider(QWidget):
    """Custom range slider with two handles for start and end selection"""

    # Signals
    rangeChanged = pyqtSignal(int, int)  # start_value, end_value

    def __init__(self, parent=None):
        super().__init__(parent)

        # State
        self._minimum = 0
        self._maximum = 100
        self._start_value = 0
        self._end_value = 100

        # UI constants
        self._handle_radius = 8
        self._track_height = 6
        self._handle_selected = None  # 'start', 'end', or None
        self._dragging = False

        # Set size policy
        self.setMinimumHeight(40)
        self.setMouseTracking(True)

    def setMinimum(self, value: int):
        """Set minimum value"""
        self._minimum = value
        if self._start_value < value:
            self._start_value = value
        if self._end_value < value:
            self._end_value = value
        self.update()

    def setMaximum(self, value: int):
        """Set maximum value"""
        self._maximum = value
        if self._start_value > value:
            self._start_value = value
        if self._end_value > value:
            self._end_value = value
        self.update()

    def setRange(self, minimum: int, maximum: int):
        """Set minimum and maximum values"""
        self.setMinimum(minimum)
        self.setMaximum(maximum)

    def startValue(self) -> int:
        """Get start value"""
        return self._start_value

    def endValue(self) -> int:
        """Get end value"""
        return self._end_value

    def setStartValue(self, value: int):
        """Set start value"""
        value = max(self._minimum, min(value, self._end_value))
        if self._start_value != value:
            self._start_value = value
            self.update()
            self.rangeChanged.emit(self._start_value, self._end_value)

    def setEndValue(self, value: int):
        """Set end value"""
        value = max(self._start_value, min(value, self._maximum))
        if self._end_value != value:
            self._end_value = value
            self.update()
            self.rangeChanged.emit(self._start_value, self._end_value)

    def setValues(self, start: int, end: int):
        """Set both start and end values"""
        start = max(self._minimum, min(start, self._maximum))
        end = max(self._minimum, min(end, self._maximum))

        if start > end:
            start, end = end, start

        changed = (self._start_value != start) or (self._end_value != end)
        self._start_value = start
        self._end_value = end

        if changed:
            self.update()
            self.rangeChanged.emit(self._start_value, self._end_value)

    def _value_to_pixel(self, value: int) -> int:
        """Convert value to pixel position"""
        if self._maximum == self._minimum:
            return self._handle_radius

        usable_width = self.width() - 2 * self._handle_radius
        ratio = (value - self._minimum) / (self._maximum - self._minimum)
        return int(self._handle_radius + ratio * usable_width)

    def _pixel_to_value(self, pixel: int) -> int:
        """Convert pixel position to value"""
        usable_width = self.width() - 2 * self._handle_radius
        pixel = max(self._handle_radius, min(pixel, self.width() - self._handle_radius))
        ratio = (pixel - self._handle_radius) / usable_width if usable_width > 0 else 0
        return int(self._minimum + ratio * (self._maximum - self._minimum))

    def _get_handle_rect(self, position: int) -> QRect:
        """Get rectangle for a handle at given pixel position"""
        center_y = self.height() // 2
        return QRect(
            position - self._handle_radius,
            center_y - self._handle_radius,
            2 * self._handle_radius,
            2 * self._handle_radius,
        )

    def _get_clicked_handle(self, pos: QPoint) -> str:
        """Determine which handle was clicked, if any"""
        start_x = self._value_to_pixel(self._start_value)
        end_x = self._value_to_pixel(self._end_value)

        start_rect = self._get_handle_rect(start_x)
        end_rect = self._get_handle_rect(end_x)

        # Check end handle first (drawn on top)
        if end_rect.contains(pos):
            return "end"
        if start_rect.contains(pos):
            return "start"

        return None

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press"""
        if event.button() == Qt.MouseButton.LeftButton:
            self._handle_selected = self._get_clicked_handle(event.pos())
            if self._handle_selected:
                self._dragging = True
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            else:
                # Click on track - move nearest handle
                click_value = self._pixel_to_value(event.pos().x())
                start_dist = abs(click_value - self._start_value)
                end_dist = abs(click_value - self._end_value)

                if start_dist < end_dist:
                    self.setStartValue(click_value)
                else:
                    self.setEndValue(click_value)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move"""
        if self._dragging and self._handle_selected:
            new_value = self._pixel_to_value(event.pos().x())

            if self._handle_selected == "start":
                self.setStartValue(new_value)
            elif self._handle_selected == "end":
                self.setEndValue(new_value)
        else:
            # Update cursor if hovering over handle
            handle = self._get_clicked_handle(event.pos())
            if handle:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release"""
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._handle_selected = None

            # Update cursor
            handle = self._get_clicked_handle(event.pos())
            if handle:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)

    def paintEvent(self, event):
        """Paint the range slider"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Calculate positions
        center_y = self.height() // 2
        start_x = self._value_to_pixel(self._start_value)
        end_x = self._value_to_pixel(self._end_value)
        track_left = self._handle_radius
        track_right = self.width() - self._handle_radius

        # Draw background track
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(200, 200, 200)))
        track_rect = QRect(
            track_left,
            center_y - self._track_height // 2,
            track_right - track_left,
            self._track_height,
        )
        painter.drawRoundedRect(track_rect, 3, 3)

        # Draw selected range track
        painter.setBrush(QBrush(QColor(0, 120, 215)))
        selected_rect = QRect(
            start_x,
            center_y - self._track_height // 2,
            end_x - start_x,
            self._track_height,
        )
        painter.drawRoundedRect(selected_rect, 3, 3)

        # Draw start handle
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.drawEllipse(
            QPoint(start_x, center_y), self._handle_radius, self._handle_radius
        )

        # Draw end handle
        painter.drawEllipse(
            QPoint(end_x, center_y), self._handle_radius, self._handle_radius
        )

        # Draw handle highlights if selected
        if self._handle_selected == "start":
            painter.setPen(QPen(QColor(0, 120, 215), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(
                QPoint(start_x, center_y),
                self._handle_radius + 2,
                self._handle_radius + 2,
            )
        elif self._handle_selected == "end":
            painter.setPen(QPen(QColor(0, 120, 215), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(
                QPoint(end_x, center_y),
                self._handle_radius + 2,
                self._handle_radius + 2,
            )
