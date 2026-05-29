"""Inspector panel for a single cell selected on the canvas."""

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from services.cell_track_service import CellDetail


def _fmt(value: Optional[float], decimals: int = 1, suffix: str = "") -> str:
    if value is None:
        return "—"
    return f"{value:.{decimals}f}{suffix}"


class CellSidebarWidget(QWidget):
    jump_to_frame = pyqtSignal(int)
    solo_mode_changed = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self._detail: Optional[CellDetail] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QLabel("Cell Inspector")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        self.hint_label = QLabel(
            "Switch to View mode and click a cell on the current frame."
        )
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        layout.addWidget(self.hint_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scroll_body = QWidget()
        body_layout = QVBoxLayout(scroll_body)
        body_layout.setContentsMargins(0, 0, 4, 0)

        self.inspector_id = QLabel("No cell selected")
        self.inspector_id.setStyleSheet("font-weight: bold; font-size: 13px;")
        body_layout.addWidget(self.inspector_id)

        self.detail_label = QLabel("")
        self.detail_label.setWordWrap(True)
        self.detail_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self.detail_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.detail_label.setTextFormat(Qt.TextFormat.RichText)
        self.detail_label.setStyleSheet("font-size: 12px;")
        body_layout.addWidget(self.detail_label)
        body_layout.addStretch()

        scroll.setWidget(scroll_body)
        layout.addWidget(scroll, stretch=1)

        actions = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions)

        btn_row = QHBoxLayout()
        self.jump_first_btn = QPushButton("First frame")
        self.jump_last_btn = QPushButton("Last frame")
        self.jump_first_btn.clicked.connect(self._jump_first)
        self.jump_last_btn.clicked.connect(self._jump_last)
        btn_row.addWidget(self.jump_first_btn)
        btn_row.addWidget(self.jump_last_btn)
        actions_layout.addLayout(btn_row)

        self.solo_checkbox = QCheckBox("Solo (hide other cells on canvas)")
        self.solo_checkbox.toggled.connect(self.solo_mode_changed.emit)
        actions_layout.addWidget(self.solo_checkbox)

        layout.addWidget(actions)

        self.setMinimumWidth(240)
        self.setMaximumWidth(360)
        self._set_actions_enabled(False)

    def show_cell(self, detail: Optional[CellDetail]) -> None:
        self._detail = detail
        self._update_display()

    def clear(self) -> None:
        self._detail = None
        self._update_display()

    def _set_actions_enabled(self, enabled: bool) -> None:
        self.jump_first_btn.setEnabled(enabled)
        self.jump_last_btn.setEnabled(enabled)
        self.solo_checkbox.setEnabled(enabled)

    def _update_display(self) -> None:
        detail = self._detail
        if detail is None:
            self.inspector_id.setText("No cell selected")
            self.inspector_id.setStyleSheet("font-weight: bold; font-size: 13px;")
            self.detail_label.setText("")
            self.hint_label.setVisible(True)
            self._set_actions_enabled(False)
            return

        self.hint_label.setVisible(False)
        r, g, b = detail.color
        self.inspector_id.setText(f"Cell {detail.cell_id}")
        self.inspector_id.setStyleSheet(
            f"font-weight: bold; font-size: 13px; color: rgb({r},{g},{b});"
        )
        self._set_actions_enabled(True)
        self.jump_first_btn.setEnabled(detail.first_frame != detail.current_frame)
        self.jump_last_btn.setEnabled(detail.last_frame != detail.current_frame)
        self.detail_label.setText(self._format_detail(detail))

    def _format_detail(self, d: CellDetail) -> str:
        status = "On this frame" if d.present_current else "Not on this frame"
        if d.last_frame == d.first_frame and d.present_current:
            status = "New / only this frame so far"

        gap_text = "continuous" if d.gap_count == 0 else f"{d.gap_count} gap(s)"

        lines = [
            f"<b>Track</b>",
            f"{d.frame_range_label()} · {d.frame_count} appearance(s)",
            f"Continuity: {gap_text} · {status}",
            "",
            f"<b>Position (this frame)</b>",
            f"Centroid: ({_fmt(d.x_px)}, {_fmt(d.y_px)}) px",
            "",
            f"<b>Size & shape (this frame)</b>",
            f"Area: {_fmt(d.area_px2)} px²",
            f"Perimeter: {_fmt(d.perimeter_px)} px",
            f"Circularity: {_fmt(d.circularity, 3)}",
            f"Ellipse aspect ratio: {_fmt(d.ellipse_aspect_ratio, 2)}",
            f"Ellipse angle: {_fmt(d.ellipse_angle, 1)}°",
            f"Solidity: {_fmt(d.solidity, 3)}",
            "",
            f"<b>Motion (frames 1–{d.current_frame + 1})</b>",
            f"Displacement (first → now): {_fmt(d.displacement_px)} px",
            f"Step (prev → now): {_fmt(d.step_displacement_px)} px",
            f"Total path length: {_fmt(d.total_path_px)} px",
            f"Avg speed: {_fmt(d.average_speed_px_per_frame, 2)} px/frame",
        ]
        return "<br>".join(lines)

    def _jump_first(self):
        if self._detail is not None:
            self.jump_to_frame.emit(self._detail.first_frame)

    def _jump_last(self):
        if self._detail is not None:
            self.jump_to_frame.emit(self._detail.last_frame)
