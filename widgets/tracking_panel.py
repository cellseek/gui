"""
Simplified tracking panel for XMem cell tracking
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

# Add project paths
current_dir = Path(__file__).parent.parent.parent
xmem_path = current_dir / "xmem"
sys.path.insert(0, str(xmem_path.resolve()))


class TrackingWorker(QThread):
    """Worker thread for running cell tracking"""

    progress_update = pyqtSignal(int, str)  # progress, status
    tracking_complete = pyqtSignal(dict)  # results
    error_occurred = pyqtSignal(str)  # error message

    def __init__(
        self,
        frames: List[np.ndarray],
        selected_mask: np.ndarray,
        parameters: Dict[str, Any],
        preloaded_xmem=None,
    ):
        super().__init__()
        self.frames = frames
        self.selected_mask = selected_mask
        self.parameters = parameters
        self.preloaded_xmem = preloaded_xmem
        self._cancelled = False

    def cancel(self):
        """Cancel the tracking"""
        self._cancelled = True

    def run(self):
        """Run tracking in background thread"""
        try:
            self.progress_update.emit(0, "Initializing XMem...")

            # Use preloaded XMem if available
            if self.preloaded_xmem is not None:
                tracker = self.preloaded_xmem
                self.progress_update.emit(5, "Using preloaded XMem model...")
            else:
                # Import XMem
                from xmem import XMem

                # Setup checkpoint paths
                xmem_path = Path(__file__).parent.parent.parent / "xmem"
                xmem_checkpoint = xmem_path / "checkpoints" / "XMem-s012.pth"

                # Create args object
                class TrackArgs:
                    def __init__(self):
                        self.device = self.parameters.get("device", "cuda:0")
                        self.sam_model_type = self.parameters.get("sam_model", "vit_h")
                        self.debug = False
                        self.mask_save = False

                track_args = TrackArgs()

                # Initialize tracker
                tracker = XMem(
                    xmem_checkpoint=str(xmem_checkpoint),
                    args=track_args,
                )

            if self._cancelled:
                return

            self.progress_update.emit(10, "Starting tracking...")

            # Reset tracker state to ensure clean start (important for XMem)
            if hasattr(tracker, "xmem") and hasattr(tracker.xmem, "clear_memory"):
                tracker.xmem.clear_memory()

            # Run tracking with custom progress tracking
            tracked_masks, tracked_logits, painted_frames = (
                self._run_tracking_with_progress(
                    tracker, self.frames, self.selected_mask
                )
            )

            if self._cancelled:
                return

            # Create results
            results = {
                "tracked_masks": tracked_masks,
                "tracked_logits": tracked_logits,
                "painted_frames": painted_frames,
                "frame_count": len(tracked_masks),
                "parameters": self.parameters.copy(),
            }

            self.progress_update.emit(100, "Tracking completed")
            self.tracking_complete.emit(results)

        except Exception as e:
            self.error_occurred.emit(f"Tracking failed: {str(e)}")

    def _run_tracking_with_progress(self, tracker, frames, template_mask):
        """Run tracking with proper progress updates"""
        masks = []
        logits = []
        painted_images = []

        total_frames = len(frames)

        for i in range(total_frames):
            if self._cancelled:
                break

            # Calculate progress (10% for initialization, 80% for tracking, 10% for completion)
            progress = 10 + int((i / total_frames) * 80)
            frame_info = f"Processing frame {i + 1}/{total_frames}"
            self.progress_update.emit(progress, frame_info)

            if i == 0:
                # First frame: initialize tracking with template mask
                # Ensure template mask has the right data type and format for XMem
                if template_mask.dtype != np.uint8:
                    template_mask = template_mask.astype(np.uint8)

                mask, logit, painted_image = tracker.xmem.track(
                    frames[i], template_mask
                )
            else:
                # Subsequent frames: track without mask
                mask, logit, painted_image = tracker.xmem.track(frames[i])

            masks.append(mask)
            logits.append(logit)
            painted_images.append(painted_image)

        return masks, logits, painted_images


class TrackingPanel(QWidget):
    """Simplified panel for cell tracking using XMem"""

    tracking_started = pyqtSignal()
    tracking_completed = pyqtSignal(dict)  # results
    tracking_error = pyqtSignal(str)  # error message

    def __init__(self):
        super().__init__()
        self.segmentation_results: Optional[Dict[str, Any]] = None
        self.tracking_results: Optional[Dict[str, Any]] = None
        self.tracking_worker: Optional[TrackingWorker] = None
        self.all_frames: List[np.ndarray] = []
        self.main_window = None
        self.preloaded_models = {}

        self.setup_ui()
        self.setEnabled(False)  # Disabled until segmentation results are available

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

    def setup_ui(self):
        """Setup the user interface"""
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Create content widget
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Header
        header_label = QLabel("Cell Tracking")
        header_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #ffffff; margin-bottom: 8px;"
        )
        layout.addWidget(header_label)

        # Info label
        info_label = QLabel(
            "Track selected cells from segmentation results across all frames using XMem."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(
            "color: #888888; margin-bottom: 16px; font-size: 12px;"
        )
        layout.addWidget(info_label)

        # Segmentation status
        self.segmentation_status_group = QGroupBox("Segmentation Status")
        seg_layout = QVBoxLayout(self.segmentation_status_group)

        self.segmentation_info_label = QLabel("No segmentation results available")
        self.segmentation_info_label.setStyleSheet("color: #cccccc; padding: 8px;")
        seg_layout.addWidget(self.segmentation_info_label)

        layout.addWidget(self.segmentation_status_group)

        # Tracking parameters
        params_group = QGroupBox("Tracking Parameters")
        params_layout = QVBoxLayout(params_group)

        # Device selection
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda:0", "cpu"])
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                if i > 0:
                    self.device_combo.addItem(f"cuda:{i}")
        device_layout.addWidget(self.device_combo)
        device_layout.addStretch()
        params_layout.addLayout(device_layout)

        layout.addWidget(params_group)

        # Tracking controls
        track_group = QGroupBox("Run Tracking")
        track_layout = QVBoxLayout(track_group)

        self.track_info_label = QLabel(
            "Track all detected cells across all frames using XMem"
        )
        self.track_info_label.setWordWrap(True)
        self.track_info_label.setStyleSheet(
            "color: #888888; font-size: 11px; margin-bottom: 8px;"
        )
        track_layout.addWidget(self.track_info_label)

        button_layout = QHBoxLayout()

        self.track_button = QPushButton("Start Tracking")
        self.track_button.clicked.connect(self.start_tracking)
        self.track_button.setEnabled(False)
        self.track_button.setStyleSheet(
            """
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
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
        button_layout.addWidget(self.track_button)

        self.cancel_track_button = QPushButton("Cancel")
        self.cancel_track_button.clicked.connect(self.cancel_tracking)
        self.cancel_track_button.setEnabled(False)
        self.cancel_track_button.setStyleSheet(
            """
            QPushButton {
                background-color: #666666;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #555555;
            }
        """
        )
        button_layout.addWidget(self.cancel_track_button)

        track_layout.addLayout(button_layout)
        layout.addWidget(track_group)

        # Progress section
        self.progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(self.progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 3px;
                text-align: center;
                padding: 1px;
                background-color: #2d2d30;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 2px;
            }
        """
        )
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(
            "color: #cccccc; font-size: 11px; padding: 4px;"
        )
        progress_layout.addWidget(self.status_label)

        self.progress_group.setVisible(False)
        layout.addWidget(self.progress_group)

        # Preview section
        self.preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(self.preview_group)

        # Preview label for video display
        self.preview_label = QLabel(
            "Tracking preview will be shown here after completion"
        )
        self.preview_label.setMinimumSize(600, 450)  # Increased size
        self.preview_label.setStyleSheet(
            """
            QLabel {
                border: 2px dashed #666666;
                border-radius: 8px;
                color: #888888;
                background-color: #2d2d30;
                padding: 20px;
                text-align: center;
            }
            """
        )
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self.preview_label)

        # Preview controls
        preview_controls_layout = QHBoxLayout()

        self.play_button = QPushButton("▶ Play")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.toggle_preview_playback)
        preview_controls_layout.addWidget(self.play_button)

        self.frame_label = QLabel("Frame: 0/0")
        preview_controls_layout.addWidget(self.frame_label)
        preview_controls_layout.addStretch()

        preview_layout.addLayout(preview_controls_layout)

        self.preview_group.setVisible(False)
        layout.addWidget(self.preview_group)

        # Initialize preview properties
        self.preview_frames = []
        self.current_frame_index = 0
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self.update_preview_frame)
        self.is_playing = False
        self.has_preview_data = False  # Flag to track if we have preview data

        layout.addStretch()

        # Set content widget to scroll area and add to main layout
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)

    def set_segmentation_results(self, results: Dict[str, Any]):
        """Set segmentation results from the segmentation panel"""
        self.segmentation_results = results

        if results:
            # Update UI
            cell_count = results.get("cell_count", 0)

            if cell_count > 0:
                self.segmentation_info_label.setText(
                    f"✓ Segmentation complete: {cell_count} cells detected, all will be tracked"
                )
                self.segmentation_info_label.setStyleSheet(
                    "color: #90EE90; padding: 8px;"
                )
                self.track_button.setEnabled(True)
            else:
                self.segmentation_info_label.setText(
                    f"⚠ Segmentation complete but no cells detected"
                )
                self.segmentation_info_label.setStyleSheet(
                    "color: #FFA500; padding: 8px;"
                )
                self.track_button.setEnabled(False)

            self.setEnabled(True)
        else:
            self.segmentation_info_label.setText("❌ No segmentation results available")
            self.segmentation_info_label.setStyleSheet("color: #FF6B6B; padding: 8px;")
            self.track_button.setEnabled(False)
            self.setEnabled(False)

    def set_all_frames(self, frames: List[np.ndarray]):
        """Set all frames for tracking"""
        self.all_frames = frames

    def start_tracking(self):
        """Start the cell tracking process"""
        if not self.all_frames:
            QMessageBox.warning(self, "Warning", "No frames available for tracking")
            return

        if not self.segmentation_results:
            QMessageBox.warning(self, "Warning", "No segmentation results available")
            return

        # Get all cell masks from segmentation results (use all cells, not just selected ones)
        masks = self.segmentation_results.get("masks")

        if masks is None:
            QMessageBox.warning(
                self,
                "Warning",
                "No segmentation masks available.",
            )
            return

        # Create combined mask for all detected cells
        selected_mask = np.zeros_like(masks, dtype=np.uint8)
        unique_ids = np.unique(masks[masks > 0])

        if len(unique_ids) == 0:
            QMessageBox.warning(
                self,
                "Warning",
                "No cells detected in segmentation masks.",
            )
            return

        # Remap cell IDs to be consecutive starting from 1 (required by XMem)
        # XMem expects labels to be 1, 2, 3, ... without gaps
        selected_mask = np.zeros_like(masks, dtype=np.uint8)
        for new_id, original_id in enumerate(unique_ids, start=1):
            selected_mask[masks == original_id] = new_id

        # Get parameters
        parameters = {
            "device": self.device_combo.currentText(),
        }

        # Start tracking worker
        preloaded_xmem = self.get_preloaded_model("xmem")
        self.tracking_worker = TrackingWorker(
            self.all_frames, selected_mask, parameters, preloaded_xmem
        )
        self.tracking_worker.progress_update.connect(self.on_tracking_progress)
        self.tracking_worker.tracking_complete.connect(self.on_tracking_complete)
        self.tracking_worker.error_occurred.connect(self.on_tracking_error)
        self.tracking_worker.start()

        # Update UI state
        self.track_button.setEnabled(False)
        self.cancel_track_button.setEnabled(True)
        self.progress_group.setVisible(True)

        # Emit signal
        self.tracking_started.emit()

    def cancel_tracking(self):
        """Cancel the current tracking"""
        if self.tracking_worker and self.tracking_worker.isRunning():
            self.tracking_worker.cancel()
            self.tracking_worker.wait()

        self.reset_ui_state()

    def on_tracking_progress(self, progress: int, status: str):
        """Handle tracking progress updates"""
        self.progress_bar.setValue(progress)
        self.status_label.setText(status)

    def on_tracking_complete(self, results: Dict[str, Any]):
        """Handle tracking completion"""
        self.tracking_results = results

        # Update UI
        frame_count = results.get("frame_count", 0)
        self.status_label.setText(f"Tracking completed: {frame_count} frames processed")

        # Show preview video if painted frames are available
        painted_frames = results.get("painted_frames", [])
        if painted_frames:
            self.show_preview_video(painted_frames)

        # Reset UI state
        self.reset_ui_state()

        # Emit signal
        self.tracking_completed.emit(results)

    def on_tracking_error(self, error_message: str):
        """Handle tracking errors"""
        self.reset_ui_state()
        self.tracking_error.emit(error_message)

    def reset_ui_state(self):
        """Reset UI to normal state"""
        self.track_button.setEnabled(True if self.segmentation_results else False)
        self.cancel_track_button.setEnabled(False)
        self.progress_group.setVisible(False)

    def show_preview_video(self, painted_frames: List[np.ndarray]):
        """Show preview video of tracking results"""
        if not painted_frames:
            return

        self.preview_frames = painted_frames
        self.current_frame_index = 0

        # Convert first frame to display
        self.update_preview_frame()

        # Show preview section and ensure it stays visible
        self.preview_group.setVisible(True)
        self.play_button.setEnabled(True)
        self.frame_label.setText(f"Frame: 1/{len(painted_frames)}")

        # Ensure the preview persists by storing a flag
        self.has_preview_data = True

    def toggle_preview_playback(self):
        """Toggle between play and pause"""
        if self.is_playing:
            self.pause_preview()
        else:
            self.play_preview()

    def play_preview(self):
        """Start playing the preview"""
        if not self.preview_frames:
            return

        self.is_playing = True
        self.play_button.setText("⏸ Pause")
        self.preview_timer.start(100)  # Update every 100ms (10 FPS)

    def pause_preview(self):
        """Pause the preview"""
        self.is_playing = False
        self.play_button.setText("▶ Play")
        self.preview_timer.stop()

    def update_preview_frame(self):
        """Update the currently displayed preview frame"""
        if not self.preview_frames:
            return

        frame = self.preview_frames[self.current_frame_index]

        # Ensure frame is in the correct format
        if isinstance(frame, np.ndarray):
            # Convert frame to uint8 if needed
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)

            # Convert numpy array to QPixmap
            if len(frame.shape) == 3:
                height, width, channel = frame.shape

                # Handle different channel orders
                if channel == 3:
                    # Assume BGR or RGB format
                    bytes_per_line = 3 * width
                    # OpenCV typically uses BGR, so we might need to convert
                    if frame.max() > 1:  # Assuming 0-255 range
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        rgb_frame = frame

                    q_image = QImage(
                        rgb_frame.data,
                        width,
                        height,
                        bytes_per_line,
                        QImage.Format.Format_RGB888,
                    )
                elif channel == 4:
                    # RGBA format
                    bytes_per_line = 4 * width
                    q_image = QImage(
                        frame.data,
                        width,
                        height,
                        bytes_per_line,
                        QImage.Format.Format_RGBA8888,
                    )
                else:
                    # Fallback to grayscale
                    gray_frame = (
                        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        if len(frame.shape) == 3
                        else frame
                    )
                    height, width = gray_frame.shape
                    bytes_per_line = width
                    q_image = QImage(
                        gray_frame.data,
                        width,
                        height,
                        bytes_per_line,
                        QImage.Format.Format_Grayscale8,
                    )
            else:
                # Grayscale image
                height, width = frame.shape
                bytes_per_line = width
                q_image = QImage(
                    frame.data,
                    width,
                    height,
                    bytes_per_line,
                    QImage.Format.Format_Grayscale8,
                )

            # Display video at original size instead of scaling
            pixmap = QPixmap.fromImage(q_image)
            self.preview_label.setPixmap(pixmap)
        else:
            # If frame is not a numpy array, show error message
            self.preview_label.setText(
                f"Frame {self.current_frame_index + 1}: Invalid format"
            )

        self.frame_label.setText(
            f"Frame: {self.current_frame_index + 1}/{len(self.preview_frames)}"
        )

        # Advance to next frame
        if self.is_playing:
            self.current_frame_index = (self.current_frame_index + 1) % len(
                self.preview_frames
            )

    def ensure_preview_visible(self):
        """Ensure preview is visible if we have preview data"""
        if self.has_preview_data and self.preview_frames:
            self.preview_group.setVisible(True)
            # Refresh the current frame display
            if not self.is_playing:
                self.update_preview_frame()

    def showEvent(self, event):
        """Called when the widget is shown"""
        super().showEvent(event)
        # Ensure preview stays visible when tab is switched back
        self.ensure_preview_visible()

    # Public interface
    def get_data(self) -> Dict[str, Any]:
        """Get tracking data for project saving"""
        data = {
            "parameters": {
                "device": self.device_combo.currentText(),
            },
        }

        if self.tracking_results:
            data["results"] = {
                "frame_count": self.tracking_results.get("frame_count", 0)
            }

        return data

    def set_data(self, data: Dict[str, Any]):
        """Set tracking data from project loading"""
        if "parameters" in data:
            params = data["parameters"]

            device = params.get("device", "cuda:0")
            index = self.device_combo.findText(device)
            if index >= 0:
                self.device_combo.setCurrentIndex(index)

    def reset(self):
        """Reset the panel to initial state"""
        self.segmentation_results = None
        self.tracking_results = None
        self.all_frames = []

        # Reset preview
        self.preview_frames = []
        self.current_frame_index = 0
        self.has_preview_data = False
        self.pause_preview()
        self.preview_group.setVisible(False)
        self.preview_label.clear()
        self.preview_label.setText(
            "Tracking preview will be shown here after completion"
        )
        self.frame_label.setText("Frame: 0/0")

        # Reset UI
        self.track_button.setEnabled(False)
        self.progress_group.setVisible(False)

        # Reset labels
        self.segmentation_info_label.setText("No segmentation results available")
        self.segmentation_info_label.setStyleSheet("color: #cccccc; padding: 8px;")
        self.status_label.setText("Ready")

        # Reset parameters to defaults
        self.device_combo.setCurrentIndex(0)

        # Disable panel
        self.setEnabled(False)
