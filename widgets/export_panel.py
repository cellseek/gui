"""
Export panel for saving results in various formats
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ExportWorker(QThread):
    """Worker thread for exporting results"""

    progress_update = pyqtSignal(int, str)  # progress, status
    export_complete = pyqtSignal(str)  # output_path
    error_occurred = pyqtSignal(str)  # error message

    def __init__(self, export_config: Dict[str, Any]):
        super().__init__()
        self.export_config = export_config
        self._cancelled = False

    def cancel(self):
        """Cancel the export"""
        self._cancelled = True

    def run(self):
        """Run export in background thread"""
        try:
            output_dir = Path(self.export_config["output_directory"])
            output_dir.mkdir(parents=True, exist_ok=True)

            self.progress_update.emit(0, "Starting export...")

            # Export tracking results
            if self.export_config.get("export_tracking", False):
                self._export_tracking_results()

            if self._cancelled:
                return

            # Export analysis results
            if self.export_config.get("export_analysis", False):
                self._export_analysis_results()

            if self._cancelled:
                return

            # Export images
            if self.export_config.get("export_images", False):
                self._export_images()

            if self._cancelled:
                return

            # Export videos
            if self.export_config.get("export_videos", False):
                self._export_videos()

            if self._cancelled:
                return

            # Create summary report
            if self.export_config.get("create_report", False):
                self._create_summary_report()

            self.progress_update.emit(100, "Export completed")
            self.export_complete.emit(str(output_dir))

        except Exception as e:
            self.error_occurred.emit(f"Export failed: {str(e)}")

    def _export_tracking_results(self):
        """Export tracking results"""
        self.progress_update.emit(20, "Exporting tracking results...")

        tracking_results = self.export_config.get("tracking_results")
        if not tracking_results:
            return

        output_dir = Path(self.export_config["output_directory"])

        # Export masks
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(exist_ok=True)

        tracked_masks = tracking_results.get("tracked_masks", [])
        for i, mask in enumerate(tracked_masks):
            if self._cancelled:
                return

            mask_path = masks_dir / f"mask_{i:05d}.png"
            cv2.imwrite(str(mask_path), mask * 255)

        # Export painted frames
        if self.export_config.get("include_painted_frames", True):
            painted_dir = output_dir / "painted_frames"
            painted_dir.mkdir(exist_ok=True)

            painted_frames = tracking_results.get("painted_frames", [])
            for i, frame in enumerate(painted_frames):
                if self._cancelled:
                    return

                frame_path = painted_dir / f"painted_{i:05d}.png"
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(frame_path), frame_bgr)

    def _export_analysis_results(self):
        """Export analysis results"""
        self.progress_update.emit(50, "Exporting analysis results...")

        analysis_results = self.export_config.get("analysis_results")
        if not analysis_results:
            return

        output_dir = Path(self.export_config["output_directory"])

        # Export data as CSV/Excel/JSON
        frame_data = analysis_results.get("frame_data")
        if frame_data is not None:
            data_format = self.export_config.get("data_format", "CSV")

            if data_format == "CSV":
                csv_path = output_dir / "cell_data.csv"
                frame_data.to_csv(csv_path, index=False)
            elif data_format == "Excel":
                excel_path = output_dir / "cell_data.xlsx"
                frame_data.to_excel(excel_path, index=False)
            elif data_format == "JSON":
                json_path = output_dir / "cell_data.json"
                frame_data.to_json(json_path, orient="records", indent=2)

        # Export summary statistics as JSON
        summary_stats = analysis_results.get("summary_statistics")
        if summary_stats:
            summary_path = output_dir / "summary_statistics.json"
            with open(summary_path, "w") as f:
                json.dump(summary_stats, f, indent=2, default=str)

        # Export movement statistics
        movement_stats = analysis_results.get("movement_statistics")
        if movement_stats:
            movement_path = output_dir / "movement_statistics.json"
            with open(movement_path, "w") as f:
                json.dump(movement_stats, f, indent=2, default=str)

    def _export_images(self):
        """Export original images"""
        self.progress_update.emit(70, "Exporting images...")

        # This would export original frames if needed
        # Implementation depends on available frame data
        pass

    def _export_videos(self):
        """Export result videos"""
        self.progress_update.emit(80, "Exporting videos...")

        tracking_results = self.export_config.get("tracking_results")
        if not tracking_results:
            return

        output_dir = Path(self.export_config["output_directory"])

        painted_frames = tracking_results.get("painted_frames", [])
        if not painted_frames:
            return

        # Create tracking video
        video_path = output_dir / "tracking_result.mp4"
        height, width = painted_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (width, height))

        for frame in painted_frames:
            if self._cancelled:
                break
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()

    def _create_summary_report(self):
        """Create a summary report"""
        self.progress_update.emit(90, "Creating summary report...")

        output_dir = Path(self.export_config["output_directory"])
        report_path = output_dir / "analysis_report.txt"

        # Create comprehensive report
        with open(report_path, "w") as f:
            f.write("CELLSEEK ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Add analysis summary if available
            analysis_results = self.export_config.get("analysis_results")
            if analysis_results:
                summary_stats = analysis_results.get("summary_statistics", {})

                f.write("EXPERIMENT OVERVIEW\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total Frames: {summary_stats.get('total_frames', 'N/A')}\n")
                f.write(f"Total Cells: {summary_stats.get('total_cells', 'N/A')}\n")
                f.write(
                    f"Total Detections: {summary_stats.get('total_detections', 'N/A')}\n\n"
                )

                # Add morphological summary
                f.write("MORPHOLOGICAL PROPERTIES\n")
                f.write("-" * 25 + "\n")
                for prop in ["area_um2", "perimeter_um", "circularity"]:
                    if prop in summary_stats:
                        stats = summary_stats[prop]
                        f.write(f"{prop.replace('_', ' ').title()}:\n")
                        f.write(f"  Mean: {stats.get('mean', 'N/A'):.3f}\n")
                        f.write(f"  Std Dev: {stats.get('std', 'N/A'):.3f}\n")
                        f.write(
                            f"  Range: {stats.get('min', 'N/A'):.3f} - {stats.get('max', 'N/A'):.3f}\n\n"
                        )
            else:
                f.write("No analysis results available.\n")


class FileListWidget(QListWidget):
    """Widget for displaying files to be exported"""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(150)
        self.setStyleSheet(
            """
            QListWidget {
                background-color: #353535;
                border: 1px solid #606060;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 4px;
                border-bottom: 1px solid #404040;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
            }
        """
        )

    def add_file(self, file_path: str, file_type: str, size: str = ""):
        """Add a file to the list"""
        item_text = f"{file_type}: {Path(file_path).name}"
        if size:
            item_text += f" ({size})"

        item = QListWidgetItem(item_text)
        item.setData(Qt.ItemDataRole.UserRole, file_path)
        self.addItem(item)

    def clear_files(self):
        """Clear all files"""
        self.clear()


class ExportPanel(QWidget):
    """Panel for exporting results in various formats"""

    def __init__(self):
        super().__init__()
        self.analysis_results: Optional[Dict[str, Any]] = None
        self.tracking_results: Optional[Dict[str, Any]] = None
        self.export_worker: Optional[ExportWorker] = None

        self.setup_ui()
        self.setEnabled(False)  # Disabled until analysis is complete

    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Create splitter for options and preview
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left side - Export options
        options_widget = self.create_options_widget()
        splitter.addWidget(options_widget)

        # Right side - File preview and progress
        preview_widget = self.create_preview_widget()
        splitter.addWidget(preview_widget)

        # Set splitter proportions (40% options, 60% preview)
        splitter.setSizes([400, 600])

        # Progress bar and controls
        progress_layout = QHBoxLayout()

        self.export_button = QPushButton("Start Export")
        self.export_button.clicked.connect(self.start_export)
        self.export_button.setEnabled(False)
        progress_layout.addWidget(self.export_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_export)
        self.cancel_button.setEnabled(False)
        progress_layout.addWidget(self.cancel_button)

        progress_layout.addStretch()

        self.open_folder_button = QPushButton("Open Output Folder")
        self.open_folder_button.clicked.connect(self.open_output_folder)
        self.open_folder_button.setEnabled(False)
        progress_layout.addWidget(self.open_folder_button)

        layout.addLayout(progress_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel()
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)

    def create_options_widget(self) -> QWidget:
        """Create the export options widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Output directory
        dir_group = QGroupBox("Output Directory")
        dir_layout = QVBoxLayout(dir_group)

        dir_select_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Select output directory...")
        dir_select_layout.addWidget(self.output_dir_edit)

        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_output_directory)
        dir_select_layout.addWidget(self.browse_button)

        dir_layout.addLayout(dir_select_layout)
        layout.addWidget(dir_group)

        # Export options
        options_group = QGroupBox("Export Options")
        options_layout = QVBoxLayout(options_group)

        # Tracking results
        self.export_tracking_checkbox = QCheckBox("Export Tracking Results")
        self.export_tracking_checkbox.setChecked(True)
        self.export_tracking_checkbox.toggled.connect(self.update_file_preview)
        options_layout.addWidget(self.export_tracking_checkbox)

        self.include_painted_checkbox = QCheckBox("Include Painted Frames")
        self.include_painted_checkbox.setChecked(True)
        self.include_painted_checkbox.toggled.connect(self.update_file_preview)
        options_layout.addWidget(self.include_painted_checkbox)

        # Analysis results
        self.export_analysis_checkbox = QCheckBox("Export Analysis Results")
        self.export_analysis_checkbox.setChecked(True)
        self.export_analysis_checkbox.toggled.connect(self.update_file_preview)
        options_layout.addWidget(self.export_analysis_checkbox)

        # Images and videos
        self.export_images_checkbox = QCheckBox("Export Original Images")
        self.export_images_checkbox.setChecked(False)
        self.export_images_checkbox.toggled.connect(self.update_file_preview)
        options_layout.addWidget(self.export_images_checkbox)

        self.export_videos_checkbox = QCheckBox("Export Result Videos")
        self.export_videos_checkbox.setChecked(True)
        self.export_videos_checkbox.toggled.connect(self.update_file_preview)
        options_layout.addWidget(self.export_videos_checkbox)

        # Summary report
        self.create_report_checkbox = QCheckBox("Create Summary Report")
        self.create_report_checkbox.setChecked(True)
        self.create_report_checkbox.toggled.connect(self.update_file_preview)
        options_layout.addWidget(self.create_report_checkbox)

        layout.addWidget(options_group)

        # Format options
        format_group = QGroupBox("Format Options")
        format_layout = QVBoxLayout(format_group)

        # Data format
        data_format_layout = QHBoxLayout()
        data_format_layout.addWidget(QLabel("Data Format:"))
        self.data_format_combo = QComboBox()
        self.data_format_combo.addItems(["CSV", "Excel", "JSON"])
        data_format_layout.addWidget(self.data_format_combo)
        format_layout.addLayout(data_format_layout)

        # Image format
        image_format_layout = QHBoxLayout()
        image_format_layout.addWidget(QLabel("Image Format:"))
        self.image_format_combo = QComboBox()
        self.image_format_combo.addItems(["PNG", "JPEG", "TIFF"])
        image_format_layout.addWidget(self.image_format_combo)
        format_layout.addLayout(image_format_layout)

        # Video format
        video_format_layout = QHBoxLayout()
        video_format_layout.addWidget(QLabel("Video Format:"))
        self.video_format_combo = QComboBox()
        self.video_format_combo.addItems(["MP4", "AVI", "MOV"])
        video_format_layout.addWidget(self.video_format_combo)
        format_layout.addLayout(video_format_layout)

        layout.addWidget(format_group)

        layout.addStretch()

        return widget

    def create_preview_widget(self) -> QWidget:
        """Create the file preview widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Preview header
        preview_group = QGroupBox("Export Preview")
        preview_layout = QVBoxLayout(preview_group)

        # File list
        self.file_list = FileListWidget()
        preview_layout.addWidget(self.file_list)

        # Export summary
        summary_layout = QHBoxLayout()

        self.file_count_label = QLabel("Files: 0")
        summary_layout.addWidget(self.file_count_label)

        summary_layout.addStretch()

        self.estimated_size_label = QLabel("Estimated size: 0 MB")
        summary_layout.addWidget(self.estimated_size_label)

        preview_layout.addLayout(summary_layout)

        layout.addWidget(preview_group)

        # Export log
        log_group = QGroupBox("Export Log")
        log_layout = QVBoxLayout(log_group)

        self.export_log = QTextEdit()
        self.export_log.setMaximumHeight(150)
        self.export_log.setReadOnly(True)
        self.export_log.setFont(QFont("Consolas", 8))
        log_layout.addWidget(self.export_log)

        layout.addWidget(log_group)

        return widget

    def browse_output_directory(self):
        """Browse for output directory"""
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir_edit.setText(directory)
            self.update_export_button_state()
            self.update_file_preview()

    def update_export_button_state(self):
        """Update the state of the export button"""
        has_output_dir = bool(self.output_dir_edit.text().strip())
        has_results = (
            self.analysis_results is not None or self.tracking_results is not None
        )
        self.export_button.setEnabled(has_output_dir and has_results)

    def update_file_preview(self):
        """Update the file preview list"""
        self.file_list.clear_files()

        output_dir = self.output_dir_edit.text().strip()
        if not output_dir:
            self.file_count_label.setText("Files: 0")
            self.estimated_size_label.setText("Estimated size: 0 MB")
            return

        file_count = 0
        estimated_size = 0  # MB

        # Tracking results files
        if self.export_tracking_checkbox.isChecked() and self.tracking_results:
            tracked_masks = self.tracking_results.get("tracked_masks", [])
            if tracked_masks:
                self.file_list.add_file(
                    f"{output_dir}/masks/", "Mask Images", f"{len(tracked_masks)} files"
                )
                file_count += len(tracked_masks)
                estimated_size += len(tracked_masks) * 0.1  # Estimate 0.1 MB per mask

            if self.include_painted_checkbox.isChecked():
                painted_frames = self.tracking_results.get("painted_frames", [])
                if painted_frames:
                    self.file_list.add_file(
                        f"{output_dir}/painted_frames/",
                        "Painted Frames",
                        f"{len(painted_frames)} files",
                    )
                    file_count += len(painted_frames)
                    estimated_size += (
                        len(painted_frames) * 0.5
                    )  # Estimate 0.5 MB per painted frame

        # Analysis results files
        if self.export_analysis_checkbox.isChecked() and self.analysis_results:
            data_format = self.data_format_combo.currentText()

            if data_format == "CSV":
                self.file_list.add_file(
                    f"{output_dir}/cell_data.csv", "Cell Data", "CSV"
                )
            elif data_format == "Excel":
                self.file_list.add_file(
                    f"{output_dir}/cell_data.xlsx", "Cell Data", "Excel"
                )
            elif data_format == "JSON":
                self.file_list.add_file(
                    f"{output_dir}/cell_data.json", "Cell Data", "JSON"
                )

            self.file_list.add_file(
                f"{output_dir}/summary_statistics.json", "Summary Statistics", "JSON"
            )
            self.file_list.add_file(
                f"{output_dir}/movement_statistics.json", "Movement Statistics", "JSON"
            )
            file_count += 3
            estimated_size += 1  # Estimate 1 MB for analysis files

        # Video files
        if self.export_videos_checkbox.isChecked() and self.tracking_results:
            painted_frames = self.tracking_results.get("painted_frames", [])
            if painted_frames:
                self.file_list.add_file(
                    f"{output_dir}/tracking_result.mp4", "Tracking Video", "MP4"
                )
                file_count += 1
                estimated_size += (
                    len(painted_frames) * 0.1
                )  # Estimate based on frame count

        # Summary report
        if self.create_report_checkbox.isChecked():
            self.file_list.add_file(
                f"{output_dir}/analysis_report.txt", "Summary Report", "Text"
            )
            file_count += 1
            estimated_size += 0.01  # Small text file

        self.file_count_label.setText(f"Files: {file_count}")
        self.estimated_size_label.setText(f"Estimated size: {estimated_size:.1f} MB")

    def start_export(self):
        """Start the export process"""
        output_dir = self.output_dir_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "Warning", "Please select an output directory")
            return

        # Create export configuration
        export_config = {
            "output_directory": output_dir,
            "export_tracking": self.export_tracking_checkbox.isChecked(),
            "include_painted_frames": self.include_painted_checkbox.isChecked(),
            "export_analysis": self.export_analysis_checkbox.isChecked(),
            "export_images": self.export_images_checkbox.isChecked(),
            "export_videos": self.export_videos_checkbox.isChecked(),
            "create_report": self.create_report_checkbox.isChecked(),
            "data_format": self.data_format_combo.currentText(),
            "image_format": self.image_format_combo.currentText(),
            "video_format": self.video_format_combo.currentText(),
        }

        # Add results data
        if self.tracking_results:
            export_config["tracking_results"] = self.tracking_results
        if self.analysis_results:
            export_config["analysis_results"] = self.analysis_results

        # Start export worker
        self.export_worker = ExportWorker(export_config)
        self.export_worker.progress_update.connect(self.on_export_progress)
        self.export_worker.export_complete.connect(self.on_export_complete)
        self.export_worker.error_occurred.connect(self.on_export_error)
        self.export_worker.start()

        # Update UI
        self.export_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.show_progress(True)

        # Clear and start log
        self.export_log.clear()
        self.export_log.append(f"Starting export to: {output_dir}")

    def cancel_export(self):
        """Cancel the current export"""
        if self.export_worker and self.export_worker.isRunning():
            self.export_worker.cancel()
            self.export_worker.wait()
            self.export_log.append("Export cancelled by user")

        self.reset_ui_state()

    def on_export_progress(self, progress: int, status: str):
        """Handle export progress updates"""
        self.progress_bar.setValue(progress)
        self.status_label.setText(status)
        self.export_log.append(f"[{progress}%] {status}")

    def on_export_complete(self, output_path: str):
        """Handle export completion"""
        self.reset_ui_state()
        self.open_folder_button.setEnabled(True)

        self.export_log.append(f"Export completed successfully!")
        self.export_log.append(f"Files saved to: {output_path}")

        QMessageBox.information(
            self,
            "Export Complete",
            f"Export completed successfully!\n\nFiles saved to:\n{output_path}",
        )

    def on_export_error(self, error_message: str):
        """Handle export errors"""
        self.reset_ui_state()
        self.export_log.append(f"ERROR: {error_message}")
        QMessageBox.critical(self, "Export Error", error_message)

    def show_progress(self, show: bool):
        """Show or hide progress indicators"""
        self.progress_bar.setVisible(show)
        self.status_label.setVisible(show)

        if show:
            self.progress_bar.setValue(0)

    def reset_ui_state(self):
        """Reset UI to normal state"""
        self.export_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.show_progress(False)
        self.update_export_button_state()

    def open_output_folder(self):
        """Open the output folder in file explorer"""
        output_dir = self.output_dir_edit.text().strip()
        if output_dir and Path(output_dir).exists():
            import platform
            import subprocess

            if platform.system() == "Windows":
                subprocess.run(f'explorer "{output_dir}"', shell=True)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", output_dir])
            else:  # Linux
                subprocess.run(["xdg-open", output_dir])

    def set_analysis_results(self, results: Dict[str, Any]):
        """Set analysis results for export"""
        self.analysis_results = results
        self.setEnabled(True)
        self.update_export_button_state()
        self.update_file_preview()

        # Set default output directory based on current time
        default_dir = (
            Path.home() / "CellSeek_Export" / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.output_dir_edit.setText(str(default_dir))

    def set_tracking_results(self, results: Dict[str, Any]):
        """Set tracking results for export"""
        self.tracking_results = results
        self.update_file_preview()

    # Public interface
    def get_data(self) -> Dict[str, Any]:
        """Get export data for project saving"""
        return {
            "output_directory": self.output_dir_edit.text(),
            "options": {
                "export_tracking": self.export_tracking_checkbox.isChecked(),
                "include_painted_frames": self.include_painted_checkbox.isChecked(),
                "export_analysis": self.export_analysis_checkbox.isChecked(),
                "export_images": self.export_images_checkbox.isChecked(),
                "export_videos": self.export_videos_checkbox.isChecked(),
                "create_report": self.create_report_checkbox.isChecked(),
            },
            "formats": {
                "data_format": self.data_format_combo.currentText(),
                "image_format": self.image_format_combo.currentText(),
                "video_format": self.video_format_combo.currentText(),
            },
        }

    def set_data(self, data: Dict[str, Any]):
        """Set export data from project loading"""
        if "output_directory" in data:
            self.output_dir_edit.setText(data["output_directory"])

        if "options" in data:
            options = data["options"]
            self.export_tracking_checkbox.setChecked(
                options.get("export_tracking", True)
            )
            self.include_painted_checkbox.setChecked(
                options.get("include_painted_frames", True)
            )
            self.export_analysis_checkbox.setChecked(
                options.get("export_analysis", True)
            )
            self.export_images_checkbox.setChecked(options.get("export_images", False))
            self.export_videos_checkbox.setChecked(options.get("export_videos", True))
            self.create_report_checkbox.setChecked(options.get("create_report", True))

        if "formats" in data:
            formats = data["formats"]

            data_format = formats.get("data_format", "CSV")
            index = self.data_format_combo.findText(data_format)
            if index >= 0:
                self.data_format_combo.setCurrentIndex(index)

            image_format = formats.get("image_format", "PNG")
            index = self.image_format_combo.findText(image_format)
            if index >= 0:
                self.image_format_combo.setCurrentIndex(index)

            video_format = formats.get("video_format", "MP4")
            index = self.video_format_combo.findText(video_format)
            if index >= 0:
                self.video_format_combo.setCurrentIndex(index)

        self.update_file_preview()

    def reset(self):
        """Reset the panel to initial state"""
        self.analysis_results = None
        self.tracking_results = None

        # Reset UI
        self.output_dir_edit.clear()
        self.file_list.clear_files()
        self.export_log.clear()
        self.open_folder_button.setEnabled(False)

        # Reset checkboxes to defaults
        self.export_tracking_checkbox.setChecked(True)
        self.include_painted_checkbox.setChecked(True)
        self.export_analysis_checkbox.setChecked(True)
        self.export_images_checkbox.setChecked(False)
        self.export_videos_checkbox.setChecked(True)
        self.create_report_checkbox.setChecked(True)

        # Reset format combos to defaults
        self.data_format_combo.setCurrentIndex(0)
        self.image_format_combo.setCurrentIndex(0)
        self.video_format_combo.setCurrentIndex(0)

        # Reset labels
        self.file_count_label.setText("Files: 0")
        self.estimated_size_label.setText("Estimated size: 0 MB")

        self.setEnabled(False)
