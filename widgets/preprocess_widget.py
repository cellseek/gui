"""Pre-tracking video/image preprocessing screen."""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from services.storage_service import StorageService
from utils.frame_preprocess import PreprocessSettings
from widgets.preprocess_preview_widget import PreprocessPreviewWidget
from widgets.trim_timeline_widget import TrimTimelineWidget

_PREVIEW_MAX_DIM = 720
_RAW_CACHE_MAX = 48


class PreprocessWidget(QWidget):
    """Adjust, crop, and trim media before tracking."""

    preprocess_confirmed = pyqtSignal(object)
    status_update = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._storage: Optional[StorageService] = None
        self._settings = PreprocessSettings()
        self._preview_frame = 0
        self._source_count = 0
        self._playing = False
        self._scrubbing = False
        self._crop_editing = False
        self._crop_before_edit: Optional[Tuple[int, int, int, int]] = None

        self._raw_cache: OrderedDict[int, np.ndarray] = OrderedDict()

        self._preview_timer = QTimer(self)
        self._preview_timer.timeout.connect(self._on_play_tick)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(50)
        self._debounce.timeout.connect(lambda: self._refresh_preview(fast=False))

        self._scrub_debounce = QTimer(self)
        self._scrub_debounce.setSingleShot(True)
        self._scrub_debounce.setInterval(16)
        self._scrub_debounce.timeout.connect(self._finish_scrub)

        self._setup_ui()

    def _setup_ui(self) -> None:
        root = QHBoxLayout()
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        title = QLabel("Preview")
        title.setStyleSheet("font-size: 15px; font-weight: bold;")
        left_layout.addWidget(title)

        self.crop_banner = QLabel(
            "Drag on the image to select the area to keep, then release the mouse."
        )
        self.crop_banner.setWordWrap(True)
        self.crop_banner.setVisible(False)
        self.crop_banner.setStyleSheet(
            "background-color: #1a4d6e; color: #e8f4fc; padding: 10px 12px;"
            "border-radius: 4px; font-size: 12px;"
        )
        left_layout.addWidget(self.crop_banner)

        self.preview = PreprocessPreviewWidget()
        self.preview.crop_changed.connect(self._on_crop_changed)
        left_layout.addWidget(self.preview, stretch=1)

        transport = QHBoxLayout()
        self.prev_frame_btn = QPushButton("◀ Prev frame")
        self.prev_frame_btn.clicked.connect(lambda: self._step_preview_frame(-1))
        transport.addWidget(self.prev_frame_btn)

        self.next_frame_btn = QPushButton("Next frame ▶")
        self.next_frame_btn.clicked.connect(lambda: self._step_preview_frame(1))
        transport.addWidget(self.next_frame_btn)

        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setFixedWidth(88)
        self.play_btn.clicked.connect(self._toggle_play)
        transport.addWidget(self.play_btn)
        transport.addStretch()
        self.frame_label = QLabel("Frame 1 / 1")
        self.frame_label.setStyleSheet("color: #aaaaaa;")
        transport.addWidget(self.frame_label)
        left_layout.addLayout(transport)

        self.trim_label = QLabel("")
        self.trim_label.setStyleSheet("color: #9cdcfe; font-size: 11px;")
        left_layout.addWidget(self.trim_label)

        self.trim_timeline = TrimTimelineWidget()
        self.trim_timeline.setToolTip(
            "Drag handles to trim · click or drag on bar to scrub"
        )
        self.trim_timeline.clip_range_changed.connect(self._on_trim_changed)
        self.trim_timeline.current_frame_changed.connect(self._on_timeline_scrub)
        left_layout.addWidget(self.trim_timeline)

        splitter.addWidget(left)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setMinimumWidth(280)
        scroll.setMaximumWidth(360)

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setSpacing(10)

        tone_group = QGroupBox("Adjustments")
        tone_layout = QVBoxLayout(tone_group)
        self.reset_tone_btn = QPushButton("Reset adjustments")
        self.reset_tone_btn.clicked.connect(self._reset_adjustments)
        tone_layout.addWidget(self.reset_tone_btn)

        self.brightness_slider = self._add_slider(tone_layout, "Brightness", -100, 100, 0)
        self.contrast_slider = self._add_slider(
            tone_layout, "Contrast", 25, 300, 100, 100.0
        )
        self.gamma_slider = self._add_slider(tone_layout, "Gamma", 20, 300, 100, 100.0)
        panel_layout.addWidget(tone_group)

        crop_group = QGroupBox("Crop (optional)")
        crop_layout = QVBoxLayout(crop_group)

        self.crop_status = QLabel("Using full frame")
        self.crop_status.setWordWrap(True)
        self.crop_status.setStyleSheet("font-weight: bold;")
        crop_layout.addWidget(self.crop_status)

        crop_btn_row = QHBoxLayout()
        self.crop_set_btn = QPushButton("Set crop area…")
        self.crop_set_btn.clicked.connect(self._start_crop_edit)
        crop_btn_row.addWidget(self.crop_set_btn)

        self.crop_cancel_btn = QPushButton("Cancel")
        self.crop_cancel_btn.clicked.connect(self._cancel_crop_edit)
        self.crop_cancel_btn.setVisible(False)
        crop_btn_row.addWidget(self.crop_cancel_btn)
        crop_layout.addLayout(crop_btn_row)

        self.crop_remove_btn = QPushButton("Use full frame")
        self.crop_remove_btn.clicked.connect(self._reset_crop)
        self.crop_remove_btn.setVisible(False)
        crop_layout.addWidget(self.crop_remove_btn)

        panel_layout.addWidget(crop_group)

        panel_layout.addStretch()
        scroll.setWidget(panel)
        splitter.addWidget(scroll)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

        footer = QHBoxLayout()
        footer.addStretch()
        continue_btn = QPushButton("Continue to tracking →")
        continue_btn.setStyleSheet(
            "QPushButton { background-color: #28a745; padding: 10px 20px; }"
            "QPushButton:hover { background-color: #218838; }"
        )
        continue_btn.clicked.connect(self._on_continue)
        footer.addWidget(continue_btn)

        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        wrapper = QWidget()
        wrapper.setLayout(root)
        outer.addWidget(wrapper, stretch=1)
        outer.addLayout(footer)
        self.setLayout(outer)

    def _add_slider(
        self,
        parent: QVBoxLayout,
        title: str,
        low: int,
        high: int,
        default: int,
        divisor: float = 1.0,
    ) -> QSlider:
        row = QHBoxLayout()
        name = QLabel(title)
        name.setMinimumWidth(82)
        row.addWidget(name)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(low, high)
        slider.setValue(default)
        val_lbl = QLabel(self._fmt(default, divisor))
        val_lbl.setMinimumWidth(44)
        slider.valueChanged.connect(
            lambda v, lbl=val_lbl, d=divisor: (
                lbl.setText(self._fmt(v, d)),
                self._schedule_preview(),
            )
        )
        row.addWidget(slider, stretch=1)
        row.addWidget(val_lbl)
        parent.addLayout(row)
        return slider

    @staticmethod
    def _fmt(value: int, divisor: float) -> str:
        if divisor == 1.0:
            return str(value)
        return f"{value / divisor:.2f}"

    @staticmethod
    def _downscale_for_preview(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        m = max(h, w)
        if m <= _PREVIEW_MAX_DIM:
            return image
        scale = _PREVIEW_MAX_DIM / m
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)

    def _get_raw_cached(self, frame_index: int) -> Optional[np.ndarray]:
        if self._storage is None:
            return None
        if frame_index in self._raw_cache:
            self._raw_cache.move_to_end(frame_index)
            return self._raw_cache[frame_index]

        raw = self._storage.load_original_frame_raw_full(frame_index)
        if raw is None:
            return None

        self._raw_cache[frame_index] = raw
        if len(self._raw_cache) > _RAW_CACHE_MAX:
            self._raw_cache.popitem(last=False)
        return raw

    def load_storage(
        self,
        storage: StorageService,
        settings: Optional[PreprocessSettings] = None,
    ) -> None:
        self._stop_play()
        self._storage = storage
        self._raw_cache.clear()
        self._source_count = storage.get_source_frame_count()
        self._preview_frame = 0

        self.trim_timeline.set_frame_count(self._source_count)
        self.trim_timeline.set_current_frame(0)

        self._crop_editing = False
        self._crop_before_edit = None
        self._exit_crop_edit()

        if settings is not None:
            self.apply_settings(settings)
        else:
            self._settings = PreprocessSettings()
            self.trim_timeline.reset_clip()
            self._reset_adjustments()
            self.preview.set_crop_rect(None)
            self._refresh_preview(fast=False)

        kind = "video" if storage.is_video else "images"
        self.status_update.emit(f"Preprocess {kind}: {self._source_count} frames")

    def apply_settings(self, settings: PreprocessSettings) -> None:
        """Restore sliders, trim range, and crop from a previous session."""
        self._settings = settings.copy()
        self.brightness_slider.setValue(self._settings.brightness)
        self.contrast_slider.setValue(int(round(self._settings.contrast * 100)))
        self.gamma_slider.setValue(int(round(self._settings.gamma * 100)))

        end = self._settings.effective_clip_end(self._source_count)
        self.trim_timeline.set_clip_range(self._settings.clip_start, end)
        self.preview.set_crop_rect(self._settings.crop_rect)
        self._refresh_preview(fast=False)

    def clear(self) -> None:
        self._stop_play()
        self._storage = None
        self._raw_cache.clear()
        self.preview.set_image(None)

    def _clip_range(self) -> tuple[int, int]:
        return self.trim_timeline.get_clip_range()

    def _collect_settings(self) -> PreprocessSettings:
        start, end = self._clip_range()
        return PreprocessSettings(
            brightness=self.brightness_slider.value(),
            contrast=self.contrast_slider.value() / 100.0,
            gamma=self.gamma_slider.value() / 100.0,
            crop_rect=self.preview.get_crop_rect(),
            clip_start=start,
            clip_end=end,
        )

    def _schedule_preview(self) -> None:
        self._debounce.start()

    def _refresh_preview(self, fast: bool = False) -> None:
        if self._storage is None:
            return

        self._settings = self._collect_settings()
        raw = self._get_raw_cached(self._preview_frame)
        if raw is None:
            self.preview.set_image(None)
            return

        drawing_crop = self._crop_editing or self.preview.is_dragging_crop()

        if drawing_crop:
            cfg = self._settings.copy()
            cfg.crop_rect = None
            display = cfg.apply(raw)
            self.preview.set_image(display)
            self.preview.set_show_crop_overlay(True)
        else:
            display = self._settings.apply(raw)
            if fast:
                display = self._downscale_for_preview(display)
            self.preview.set_image(display)
            self.preview.set_show_crop_overlay(False)

        start, end = self._clip_range()
        used = end - start + 1
        self.trim_label.setText(
            f"Trim: frames {start + 1}–{end + 1} ({used} of {self._source_count})"
        )
        self.frame_label.setText(
            f"Frame {self._preview_frame + 1} of trim "
            f"({start + 1}–{end + 1}, {self._source_count} total)"
        )
        self.prev_frame_btn.setEnabled(self._preview_frame > start)
        self.next_frame_btn.setEnabled(self._preview_frame < end)

        self._update_crop_ui()

    def _step_preview_frame(self, delta: int) -> None:
        if self._storage is None or self._source_count <= 0:
            return
        start, end = self._clip_range()
        self._stop_play()
        new_frame = max(start, min(self._preview_frame + delta, end))
        if new_frame == self._preview_frame:
            return
        self._preview_frame = new_frame
        self.trim_timeline.set_current_frame(new_frame)
        self._refresh_preview(fast=False)

    def _on_timeline_scrub(self, frame: int) -> None:
        if frame == self._preview_frame and not self._scrubbing:
            return
        self._preview_frame = frame
        self._scrubbing = True
        self.trim_timeline.set_current_frame(frame)
        self._refresh_preview(fast=True)
        self._scrub_debounce.start()

    def _finish_scrub(self) -> None:
        self._scrubbing = False
        self._refresh_preview(fast=False)

    def _on_trim_changed(self, start: int, end: int) -> None:
        if not (start <= self._preview_frame <= end):
            self._preview_frame = start
            self.trim_timeline.set_current_frame(start)
        self._refresh_preview(fast=False)

    def _reset_adjustments(self) -> None:
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(100)
        self.gamma_slider.setValue(100)
        self._refresh_preview(fast=False)

    def _update_crop_ui(self) -> None:
        crop = self.preview.get_crop_rect()
        editing = self._crop_editing

        self.crop_banner.setVisible(editing)
        self.crop_set_btn.setVisible(not editing)
        self.crop_cancel_btn.setVisible(editing)
        self.crop_remove_btn.setVisible(crop is not None and not editing)

        if not editing:
            self.crop_set_btn.setText(
                "Change crop area…" if crop else "Set crop area…"
            )

        if editing:
            self.crop_status.setText("Selecting crop area…")
            self.crop_status.setStyleSheet("font-weight: bold; color: #9cdcfe;")
            self.preview.setStyleSheet(
                "background-color: #1e1e1e; border: 2px solid #0078d4;"
            )
        elif crop:
            _, _, w, h = crop
            self.crop_status.setText(f"Cropped to {w}×{h} px")
            self.crop_status.setStyleSheet("font-weight: bold; color: #8fd19e;")
            self.preview.setStyleSheet(
                "background-color: #1e1e1e; border: 1px solid #606060;"
            )
        else:
            self.crop_status.setText("Using full frame")
            self.crop_status.setStyleSheet("font-weight: bold; color: #aaaaaa;")
            self.preview.setStyleSheet(
                "background-color: #1e1e1e; border: 1px solid #606060;"
            )

    def _start_crop_edit(self) -> None:
        self._crop_before_edit = self.preview.get_crop_rect()
        self._crop_editing = True
        self.preview.set_crop_mode(True)
        self.preview.set_show_crop_overlay(True)
        self._update_crop_ui()
        self._refresh_preview(fast=False)

    def _exit_crop_edit(self) -> None:
        self._crop_editing = False
        self._crop_before_edit = None
        self.preview.set_crop_mode(False)
        self.preview.set_show_crop_overlay(False)
        self._update_crop_ui()

    def _cancel_crop_edit(self) -> None:
        self.preview.set_crop_rect(self._crop_before_edit)
        self._exit_crop_edit()
        self._refresh_preview(fast=False)

    def _reset_crop(self) -> None:
        if self._crop_editing:
            self._cancel_crop_edit()
        self.preview.set_crop_rect(None)
        self._exit_crop_edit()
        self._refresh_preview(fast=False)

    def _on_crop_changed(self, rect: object) -> None:
        if rect is not None:
            self.preview.set_crop_rect(rect)  # type: ignore[arg-type]
        self._exit_crop_edit()
        self._refresh_preview(fast=False)

    def _toggle_play(self) -> None:
        if self._playing:
            self._stop_play()
        else:
            if self._source_count <= 1:
                return
            self._playing = True
            self.play_btn.setText("⏸ Pause")
            self._preview_timer.start(100)

    def _stop_play(self) -> None:
        self._playing = False
        self._preview_timer.stop()
        self.play_btn.setText("▶ Play")

    def _on_play_tick(self) -> None:
        if self._source_count <= 0:
            return
        start, end = self._clip_range()
        next_f = self._preview_frame + 1
        if next_f > end:
            next_f = start
        self._preview_frame = next_f
        self.trim_timeline.set_current_frame(next_f)
        self._refresh_preview(fast=True)

    def _on_continue(self) -> None:
        self._stop_play()
        self.preprocess_confirmed.emit(self._collect_settings())
