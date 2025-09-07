"""
Export widget for cell tracking data export and analysis.

This widget provides a comprehensive interface for:
- Configuring export parameters
- Previewing export data
- Exporting to multiple formats (CSV, JSON, video, images)
- Viewing summary statistics
"""

import os
from pathlib import Path
from typing import Optional, Tuple

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from services.export_service import ExportService
from services.storage_service import StorageService


class ExportWorker(QThread):
    """Worker thread for export operations"""

    def __init__(self, export_service: ExportService, operation: str, **kwargs):
        super().__init__()
        self.export_service = export_service
        self.operation = operation
        self.kwargs = kwargs

    def run(self):
        """Execute the export operation"""
        if self.operation == "csv":
            self.export_service.export_to_csv(**self.kwargs)
        elif self.operation == "json":
            self.export_service.export_to_json(**self.kwargs)
        elif self.operation == "video":
            self.export_service.export_annotated_video(**self.kwargs)
        elif self.operation == "frames":
            self.export_service.export_individual_frames(**self.kwargs)


class ExportWidget(QWidget):
    """Main widget for exporting cell tracking data and analysis"""

    # Signals
    back_to_tracking = pyqtSignal()

    def __init__(self, storage_service: StorageService):
        super().__init__()
        self.storage_service = storage_service
        self.export_service = ExportService(storage_service)
        self.export_worker = None

        # Setup UI
        self.setup_ui()
        self.setup_connections()

        # Load initial data
        self.refresh_preview()

    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Title and navigation
        self.setup_header(layout)

        # Main content with tabs
        self.setup_tabs(layout)

        # Progress and status
        self.setup_progress_panel(layout)

    def setup_header(self, parent_layout):
        """Setup header with title and navigation"""
        header_layout = QHBoxLayout()

        # Title
        title_label = QLabel("Export Cell Tracking Data")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        # Back button
        self.back_button = QPushButton("← Back to Tracking")
        self.back_button.clicked.connect(self.back_to_tracking.emit)
        header_layout.addWidget(self.back_button)

        parent_layout.addLayout(header_layout)

    def setup_tabs(self, parent_layout):
        """Setup tabbed interface"""
        self.tab_widget = QTabWidget()
        parent_layout.addWidget(self.tab_widget)

        # Configuration Tab
        self.setup_config_tab()

        # Preview Tab
        self.setup_preview_tab()

        # Statistics Tab
        self.setup_statistics_tab()

    def setup_config_tab(self):
        """Setup export configuration tab"""
        config_widget = QWidget()
        layout = QVBoxLayout(config_widget)

        # Create splitter for side-by-side layout
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left panel: Export settings
        self.setup_export_settings(splitter)

        # Right panel: Output settings
        self.setup_output_settings(splitter)

        self.tab_widget.addTab(config_widget, "Configuration")

    def setup_export_settings(self, parent_splitter):
        """Setup export settings panel"""
        settings_widget = QWidget()
        layout = QVBoxLayout(settings_widget)

        # Frame range settings
        frame_group = QGroupBox("Frame Range")
        frame_layout = QFormLayout(frame_group)

        self.use_all_frames_check = QCheckBox("Export all frames")
        self.use_all_frames_check.setChecked(True)
        self.use_all_frames_check.toggled.connect(self.on_frame_range_toggled)
        frame_layout.addRow(self.use_all_frames_check)

        self.start_frame_spin = QSpinBox()
        self.start_frame_spin.setMinimum(0)
        self.start_frame_spin.setMaximum(9999)
        self.start_frame_spin.setEnabled(False)
        frame_layout.addRow("Start Frame:", self.start_frame_spin)

        self.end_frame_spin = QSpinBox()
        self.end_frame_spin.setMinimum(0)
        self.end_frame_spin.setMaximum(9999)
        self.end_frame_spin.setEnabled(False)
        frame_layout.addRow("End Frame:", self.end_frame_spin)

        layout.addWidget(frame_group)

        # Time settings
        time_group = QGroupBox("Time Parameters")
        time_layout = QFormLayout(time_group)

        self.time_per_frame_spin = QDoubleSpinBox()
        self.time_per_frame_spin.setMinimum(0.1)
        self.time_per_frame_spin.setMaximum(60.0)
        self.time_per_frame_spin.setValue(3.0)
        self.time_per_frame_spin.setSuffix(" minutes")
        self.time_per_frame_spin.valueChanged.connect(self.on_time_per_frame_changed)
        time_layout.addRow("Time per Frame:", self.time_per_frame_spin)

        layout.addWidget(time_group)

        # Export format selection
        format_group = QGroupBox("Export Formats")
        format_layout = QVBoxLayout(format_group)

        self.export_csv_check = QCheckBox("CSV Data File")
        self.export_csv_check.setChecked(True)
        format_layout.addWidget(self.export_csv_check)

        self.export_json_check = QCheckBox("JSON Data File")
        format_layout.addWidget(self.export_json_check)

        self.export_video_check = QCheckBox("Annotated Video")
        format_layout.addWidget(self.export_video_check)

        self.export_frames_check = QCheckBox("Individual Frames")
        format_layout.addWidget(self.export_frames_check)

        layout.addWidget(format_group)

        layout.addStretch()
        parent_splitter.addWidget(settings_widget)

    def setup_output_settings(self, parent_splitter):
        """Setup output settings panel"""
        output_widget = QWidget()
        layout = QVBoxLayout(output_widget)

        # Output directory
        output_group = QGroupBox("Output Location")
        output_layout = QVBoxLayout(output_group)

        dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Select output directory...")
        dir_layout.addWidget(self.output_dir_edit)

        self.browse_dir_button = QPushButton("Browse...")
        self.browse_dir_button.clicked.connect(self.browse_output_directory)
        dir_layout.addWidget(self.browse_dir_button)

        output_layout.addLayout(dir_layout)

        # File naming
        self.filename_prefix_edit = QLineEdit("celltrack_export")
        output_layout.addWidget(QLabel("Filename Prefix:"))
        output_layout.addWidget(self.filename_prefix_edit)

        layout.addWidget(output_group)

        # Video settings (when video export is enabled)
        video_group = QGroupBox("Video Settings")
        video_layout = QFormLayout(video_group)

        self.video_fps_spin = QSpinBox()
        self.video_fps_spin.setMinimum(1)
        self.video_fps_spin.setMaximum(60)
        self.video_fps_spin.setValue(5)
        video_layout.addRow("FPS:", self.video_fps_spin)

        self.video_format_combo = QComboBox()
        self.video_format_combo.addItems(["MP4", "AVI"])
        video_layout.addRow("Format:", self.video_format_combo)

        layout.addWidget(video_group)

        # Export buttons
        buttons_layout = QVBoxLayout()

        self.export_button = QPushButton("Start Export")
        self.export_button.setStyleSheet(
            """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """
        )
        self.export_button.clicked.connect(self.start_export)
        buttons_layout.addWidget(self.export_button)

        self.cancel_button = QPushButton("Cancel Export")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_export)
        buttons_layout.addWidget(self.cancel_button)

        layout.addLayout(buttons_layout)
        layout.addStretch()

        parent_splitter.addWidget(output_widget)

    def setup_preview_tab(self):
        """Setup data preview tab"""
        preview_widget = QWidget()
        layout = QVBoxLayout(preview_widget)

        # Preview controls
        controls_layout = QHBoxLayout()

        refresh_button = QPushButton("Refresh Preview")
        refresh_button.clicked.connect(self.refresh_preview)
        controls_layout.addWidget(refresh_button)

        controls_layout.addStretch()

        self.preview_frame_spin = QSpinBox()
        self.preview_frame_spin.setMinimum(0)
        self.preview_frame_spin.valueChanged.connect(self.update_frame_preview)
        controls_layout.addWidget(QLabel("Preview Frame:"))
        controls_layout.addWidget(self.preview_frame_spin)

        layout.addLayout(controls_layout)

        # Preview table
        self.preview_table = QTableWidget()
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.preview_table)

        self.tab_widget.addTab(preview_widget, "Data Preview")

    def setup_statistics_tab(self):
        """Setup statistics summary tab"""
        stats_widget = QWidget()
        layout = QVBoxLayout(stats_widget)

        # Refresh button
        refresh_stats_button = QPushButton("Refresh Statistics")
        refresh_stats_button.clicked.connect(self.refresh_statistics)
        layout.addWidget(refresh_stats_button)

        # Statistics display
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setFont(QFont("Consolas", 10))
        layout.addWidget(self.stats_text)

        self.tab_widget.addTab(stats_widget, "Statistics")

    def setup_progress_panel(self, parent_layout):
        """Setup progress and status panel"""
        progress_group = QGroupBox("Export Progress")
        layout = QVBoxLayout(progress_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready to export")
        layout.addWidget(self.status_label)

        parent_layout.addWidget(progress_group)

    def setup_connections(self):
        """Setup signal connections"""
        # Export service signals
        self.export_service.progress_updated.connect(self.progress_bar.setValue)
        self.export_service.status_updated.connect(self.status_label.setText)
        self.export_service.export_completed.connect(self.on_export_completed)

    # ========================================
    # EVENT HANDLERS
    # ========================================

    def on_frame_range_toggled(self, checked):
        """Handle frame range toggle"""
        self.start_frame_spin.setEnabled(not checked)
        self.end_frame_spin.setEnabled(not checked)

        if checked:
            # Update spin boxes with full range
            frame_count = self.storage_service.get_frame_count()
            self.start_frame_spin.setMaximum(frame_count - 1)
            self.end_frame_spin.setMaximum(frame_count - 1)
            self.start_frame_spin.setValue(0)
            self.end_frame_spin.setValue(frame_count - 1)

    def on_time_per_frame_changed(self, value):
        """Handle time per frame change"""
        self.export_service.set_time_per_frame(value)

    def browse_output_directory(self):
        """Browse for output directory"""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self.output_dir_edit.text()
        )
        if directory:
            self.output_dir_edit.setText(directory)

    def start_export(self):
        """Start the export process"""
        # Validate settings
        if not self.validate_export_settings():
            return

        # Get export parameters
        output_dir = self.output_dir_edit.text()
        filename_prefix = self.filename_prefix_edit.text()

        # Get frame range
        frame_range = None
        if not self.use_all_frames_check.isChecked():
            start_frame = self.start_frame_spin.value()
            end_frame = self.end_frame_spin.value()
            frame_range = (start_frame, end_frame)

        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.export_button.setEnabled(False)
        self.cancel_button.setEnabled(True)

        # Start exports
        self.run_exports(output_dir, filename_prefix, frame_range)

    def cancel_export(self):
        """Cancel ongoing export"""
        if self.export_worker and self.export_worker.isRunning():
            self.export_worker.terminate()
            self.export_worker.wait()

        self.on_export_completed(False, "Export cancelled by user")

    def validate_export_settings(self) -> bool:
        """Validate export settings"""
        # Check if any format is selected
        if not any(
            [
                self.export_csv_check.isChecked(),
                self.export_json_check.isChecked(),
                self.export_video_check.isChecked(),
                self.export_frames_check.isChecked(),
            ]
        ):
            QMessageBox.warning(
                self, "Export Error", "Please select at least one export format."
            )
            return False

        # Check output directory
        output_dir = self.output_dir_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(
                self, "Export Error", "Please select an output directory."
            )
            return False

        # Check if we have data to export
        frame_count = self.storage_service.get_frame_count()
        if frame_count == 0:
            QMessageBox.warning(self, "Export Error", "No frames loaded.")
            return False

        # Check frame range
        if not self.use_all_frames_check.isChecked():
            start_frame = self.start_frame_spin.value()
            end_frame = self.end_frame_spin.value()
            if start_frame > end_frame:
                QMessageBox.warning(
                    self,
                    "Export Error",
                    "Start frame must be less than or equal to end frame.",
                )
                return False

        return True

    def run_exports(
        self,
        output_dir: str,
        filename_prefix: str,
        frame_range: Optional[Tuple[int, int]],
    ):
        """Run the selected export operations"""
        # Create timestamped subdirectory
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = os.path.join(output_dir, f"{filename_prefix}_{timestamp}")
        os.makedirs(export_dir, exist_ok=True)

        # Export CSV
        if self.export_csv_check.isChecked():
            csv_path = os.path.join(export_dir, f"{filename_prefix}.csv")
            self.export_worker = ExportWorker(
                self.export_service,
                "csv",
                output_path=csv_path,
                frame_range=frame_range,
            )
            self.export_worker.start()
            self.export_worker.wait()

        # Export JSON
        if self.export_json_check.isChecked():
            json_path = os.path.join(export_dir, f"{filename_prefix}.json")
            self.export_worker = ExportWorker(
                self.export_service,
                "json",
                output_path=json_path,
                frame_range=frame_range,
            )
            self.export_worker.start()
            self.export_worker.wait()

        # Export Video
        if self.export_video_check.isChecked():
            video_ext = (
                ".mp4" if self.video_format_combo.currentText() == "MP4" else ".avi"
            )
            video_path = os.path.join(export_dir, f"{filename_prefix}_video{video_ext}")
            fps = self.video_fps_spin.value()

            self.export_worker = ExportWorker(
                self.export_service,
                "video",
                output_path=video_path,
                fps=fps,
                frame_range=frame_range,
            )
            self.export_worker.start()
            self.export_worker.wait()

        # Export Individual Frames
        if self.export_frames_check.isChecked():
            frames_dir = os.path.join(export_dir, "frames")
            self.export_worker = ExportWorker(
                self.export_service,
                "frames",
                output_dir=frames_dir,
                frame_range=frame_range,
            )
            self.export_worker.start()
            self.export_worker.wait()

    def on_export_completed(self, success: bool, message: str):
        """Handle export completion"""
        self.progress_bar.setVisible(False)
        self.export_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

        if success:
            QMessageBox.information(self, "Export Complete", message)
        else:
            QMessageBox.warning(self, "Export Error", message)

        self.status_label.setText("Ready to export")

    def refresh_preview(self):
        """Refresh the data preview"""
        frame_count = self.storage_service.get_frame_count()
        self.preview_frame_spin.setMaximum(max(0, frame_count - 1))

        if frame_count > 0:
            self.update_frame_preview()

    def update_frame_preview(self):
        """Update preview table with current frame data"""
        frame_index = self.preview_frame_spin.value()
        frame_data = self.export_service.extract_frame_data(frame_index)

        if not frame_data:
            self.preview_table.setRowCount(0)
            self.preview_table.setColumnCount(0)
            return

        # Setup table
        self.preview_table.setRowCount(len(frame_data))
        if frame_data:
            columns = list(frame_data[0].keys())
            self.preview_table.setColumnCount(len(columns))
            self.preview_table.setHorizontalHeaderLabels(columns)

            # Fill data
            for row, data in enumerate(frame_data):
                for col, key in enumerate(columns):
                    value = data[key]
                    if isinstance(value, float):
                        item = QTableWidgetItem(f"{value:.3f}")
                    else:
                        item = QTableWidgetItem(str(value))
                    self.preview_table.setItem(row, col, item)

            # Resize columns
            self.preview_table.resizeColumnsToContents()

    def refresh_statistics(self):
        """Refresh statistics display"""
        try:
            stats = self.export_service.get_summary_statistics()

            if not stats:
                self.stats_text.setText("No tracking data available for statistics.")
                return

            # Format statistics
            stats_text = "CELL TRACKING STATISTICS\n"
            stats_text += "=" * 50 + "\n\n"

            stats_text += f"Total Detections: {stats['total_detections']}\n"
            stats_text += f"Unique Cells: {stats['unique_cells']}\n"
            stats_text += (
                f"Frame Range: {stats['frame_range'][0]} - {stats['frame_range'][1]}\n"
            )
            stats_text += f"Time Span: {stats['time_span_minutes']:.1f} minutes\n\n"

            # Morphological statistics
            stats_text += "MORPHOLOGICAL STATISTICS\n"
            stats_text += "-" * 30 + "\n"

            for metric, data in stats["morphological_stats"].items():
                stats_text += f"\n{metric.replace('_', ' ').title()}:\n"
                stats_text += f"  Mean: {data['mean']:.3f}\n"
                stats_text += f"  Std:  {data['std']:.3f}\n"
                stats_text += f"  Min:  {data['min']:.3f}\n"
                stats_text += f"  Max:  {data['max']:.3f}\n"

            self.stats_text.setText(stats_text)

        except Exception as e:
            self.stats_text.setText(f"Error calculating statistics: {str(e)}")

    # ========================================
    # PUBLIC METHODS
    # ========================================

    def initialize(self):
        """Initialize the export widget with current data"""
        # Update frame range controls
        frame_count = self.storage_service.get_frame_count()
        if frame_count > 0:
            self.start_frame_spin.setMaximum(frame_count - 1)
            self.end_frame_spin.setMaximum(frame_count - 1)
            self.end_frame_spin.setValue(frame_count - 1)

        # Set default output directory
        if not self.output_dir_edit.text():
            default_dir = os.path.join(os.path.expanduser("~"), "CellSeek_Exports")
            self.output_dir_edit.setText(default_dir)

        # Refresh preview and statistics
        self.refresh_preview()
        self.refresh_statistics()
