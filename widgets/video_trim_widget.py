"""
Video trimming widget (embedded, not a dialog)
"""

from pathlib import Path

import cv2
from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from widgets.range_slider import RangeSlider


class VideoTrimWidget(QWidget):
    """Embedded widget for trimming video with preview and range slider"""

    # Signals
    trim_confirmed = pyqtSignal(float, float)  # start_sec, end_sec
    trim_cancelled = pyqtSignal()

    def __init__(self, video_path: str, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.duration_ms = 0
        self.fps = 0

        # Get video properties
        self._get_video_properties()

        # Setup UI
        self.setup_ui()

        # Initialize media player
        self.setup_media_player()

    def _get_video_properties(self):
        """Get video duration and FPS using OpenCV"""
        cap = cv2.VideoCapture(self.video_path)
        if cap.isOpened():
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.duration_ms = int(
                (total_frames / self.fps * 1000) if self.fps > 0 else 0
            )
            cap.release()
        else:
            self.duration_ms = 0
            self.fps = 30  # Default fallback

    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)

        # Header with title and close button
        header_layout = QHBoxLayout()

        title_label = QLabel("Trim Video")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        # Back button
        back_btn = QPushButton("✕ Cancel")
        back_btn.setStyleSheet(
            """
            QPushButton {
                background: transparent;
                border: 1px solid #666;
                padding: 6px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #f44336;
                color: white;
                border-color: #f44336;
            }
        """
        )
        back_btn.clicked.connect(self.on_cancel)
        header_layout.addWidget(back_btn)

        layout.addLayout(header_layout)

        # Video preview widget
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(400)
        self.video_widget.setStyleSheet("background-color: black; border-radius: 8px;")
        layout.addWidget(self.video_widget)

        # Playback controls
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(8)

        # Play/Pause button
        self.play_button = QPushButton()
        self.play_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.play_button.setFixedSize(40, 40)
        self.play_button.clicked.connect(self.toggle_playback)
        controls_layout.addWidget(self.play_button)

        # Position slider
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, self.duration_ms)
        self.position_slider.sliderMoved.connect(self.seek_to_position)
        self.position_slider.sliderPressed.connect(self.on_slider_pressed)
        self.position_slider.sliderReleased.connect(self.on_slider_released)
        controls_layout.addWidget(self.position_slider, 1)

        # Current time label
        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setMinimumWidth(100)
        self.time_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        controls_layout.addWidget(self.time_label)

        layout.addLayout(controls_layout)

        # Trim section
        trim_group_layout = QVBoxLayout()
        trim_group_layout.setSpacing(12)

        trim_title = QLabel("Trim Range")
        trim_title.setStyleSheet("font-weight: bold;")
        trim_group_layout.addWidget(trim_title)

        # Range slider
        self.range_slider = RangeSlider()
        self.range_slider.setRange(0, self.duration_ms)
        self.range_slider.setValues(0, self.duration_ms)
        self.range_slider.rangeChanged.connect(self.on_range_changed)
        self.range_slider.setMinimumHeight(50)
        trim_group_layout.addWidget(self.range_slider)

        # Range labels
        range_labels_layout = QHBoxLayout()
        self.start_time_label = QLabel("Start: 0:00")
        self.start_time_label.setStyleSheet("color: #0078d7;")
        range_labels_layout.addWidget(self.start_time_label)

        range_labels_layout.addStretch()

        self.end_time_label = QLabel(f"End: {self._format_time(self.duration_ms)}")
        self.end_time_label.setStyleSheet("color: #0078d7;")
        range_labels_layout.addWidget(self.end_time_label)

        trim_group_layout.addLayout(range_labels_layout)

        # Duration label
        self.duration_label = QLabel(
            f"Selected duration: {self._format_time(self.duration_ms)}"
        )
        self.duration_label.setStyleSheet("font-style: italic;")
        trim_group_layout.addWidget(self.duration_label)

        layout.addLayout(trim_group_layout)

        # Preview buttons
        preview_layout = QHBoxLayout()
        preview_layout.setSpacing(8)

        preview_start_btn = QPushButton("Preview Start")
        preview_start_btn.clicked.connect(self.preview_start)
        preview_layout.addWidget(preview_start_btn)

        preview_end_btn = QPushButton("Preview End")
        preview_end_btn.clicked.connect(self.preview_end)
        preview_layout.addWidget(preview_end_btn)

        preview_layout.addStretch()

        layout.addLayout(preview_layout)

        # Confirm button (prominent)
        confirm_button = QPushButton("✓ Confirm and Extract Frames")
        confirm_button.setMinimumHeight(44)
        confirm_button.clicked.connect(self.on_confirm)
        confirm_button.setStyleSheet(
            """
            QPushButton {
                background-color: #0078d7;
                color: white;
                padding: 8px 24px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
        """
        )
        layout.addWidget(confirm_button)

    def setup_media_player(self):
        """Setup media player and connect signals"""
        self.media_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)

        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)

        # Connect signals
        self.media_player.positionChanged.connect(self.on_position_changed)
        self.media_player.playbackStateChanged.connect(self.on_playback_state_changed)
        self.media_player.durationChanged.connect(self.on_duration_changed)

        # Load video
        self.media_player.setSource(QUrl.fromLocalFile(self.video_path))

        # Slider update tracking
        self.slider_being_dragged = False

    def toggle_playback(self):
        """Toggle between play and pause"""
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
        else:
            # If at end of trim range, restart from beginning of trim
            current_pos = self.media_player.position()
            end_pos = self.range_slider.endValue()

            if current_pos >= end_pos:
                self.media_player.setPosition(self.range_slider.startValue())

            self.media_player.play()

    def seek_to_position(self, position: int):
        """Seek to a specific position"""
        self.media_player.setPosition(position)

    def on_slider_pressed(self):
        """Called when position slider is pressed"""
        self.slider_being_dragged = True

    def on_slider_released(self):
        """Called when position slider is released"""
        self.slider_being_dragged = False

    def on_position_changed(self, position: int):
        """Update UI when playback position changes"""
        # Update position slider if not being dragged
        if not self.slider_being_dragged:
            self.position_slider.setValue(position)

        # Update time label
        current_time = self._format_time(position)
        total_time = self._format_time(self.duration_ms)
        self.time_label.setText(f"{current_time} / {total_time}")

        # Auto-pause at end of trim range
        if position >= self.range_slider.endValue():
            if (
                self.media_player.playbackState()
                == QMediaPlayer.PlaybackState.PlayingState
            ):
                self.media_player.pause()
                self.media_player.setPosition(self.range_slider.endValue())

    def on_playback_state_changed(self, state: QMediaPlayer.PlaybackState):
        """Update play button icon based on playback state"""
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_button.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause)
            )
        else:
            self.play_button.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
            )

    def on_duration_changed(self, duration: int):
        """Update UI when media duration is loaded"""
        if duration > 0 and self.duration_ms == 0:
            self.duration_ms = duration
            self.position_slider.setRange(0, duration)
            self.range_slider.setRange(0, duration)
            self.range_slider.setValues(0, duration)
            self.update_range_labels()

    def on_range_changed(self, start: int, end: int):
        """Called when trim range changes"""
        self.update_range_labels()

        # If currently playing and passed end point, pause
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            if self.media_player.position() > end:
                self.media_player.pause()
                self.media_player.setPosition(end)

    def update_range_labels(self):
        """Update the range time labels"""
        start = self.range_slider.startValue()
        end = self.range_slider.endValue()

        self.start_time_label.setText(f"Start: {self._format_time(start)}")
        self.end_time_label.setText(f"End: {self._format_time(end)}")

        duration = end - start
        self.duration_label.setText(f"Selected duration: {self._format_time(duration)}")

    def preview_start(self):
        """Preview the start trim point"""
        start_pos = self.range_slider.startValue()
        self.media_player.setPosition(start_pos)
        if self.media_player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.play()

    def preview_end(self):
        """Preview the end trim point"""
        end_pos = self.range_slider.endValue()
        # Go back 3 seconds from end (or to start if less than 3 seconds)
        preview_pos = max(self.range_slider.startValue(), end_pos - 3000)
        self.media_player.setPosition(preview_pos)
        if self.media_player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.play()

    def get_trim_range(self):
        """Get the selected trim range in seconds"""
        start_sec = self.range_slider.startValue() / 1000.0
        end_sec = self.range_slider.endValue() / 1000.0
        return start_sec, end_sec

    def on_confirm(self):
        """Handle confirm button click"""
        self.media_player.stop()
        start_sec, end_sec = self.get_trim_range()
        self.trim_confirmed.emit(start_sec, end_sec)

    def on_cancel(self):
        """Handle cancel button click"""
        self.media_player.stop()
        self.trim_cancelled.emit()

    def _format_time(self, milliseconds: int) -> str:
        """Format milliseconds as M:SS or H:MM:SS"""
        seconds = int(milliseconds / 1000)
        minutes = seconds // 60
        seconds = seconds % 60

        if minutes >= 60:
            hours = minutes // 60
            minutes = minutes % 60
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes}:{seconds:02d}"

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, "media_player"):
            self.media_player.stop()
