"""
Analysis panel for cell parameter calculation and visualization
"""

from typing import Any, Dict, Optional

import cv2
import numpy as np
import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class AnalysisWorker(QThread):
    """Worker thread for running cell analysis"""

    progress_update = pyqtSignal(str)  # status message
    analysis_complete = pyqtSignal(dict)  # results
    error_occurred = pyqtSignal(str)  # error message

    def __init__(self, tracking_results: Dict[str, Any], parameters: Dict[str, Any]):
        super().__init__()
        self.tracking_results = tracking_results
        self.parameters = parameters
        self._cancelled = False

    def cancel(self):
        """Cancel the analysis"""
        self._cancelled = True

    def run(self):
        """Run analysis in background thread"""
        try:
            self.progress_update.emit("Starting cell analysis...")

            tracked_masks = self.tracking_results.get("tracked_masks", [])
            if not tracked_masks:
                self.error_occurred.emit("No tracking masks available for analysis")
                return

            # Parameters
            time_per_frame = self.parameters.get("time_per_frame", 3.0)  # minutes
            pixel_size = self.parameters.get("pixel_size", 1.0)  # μm per pixel

            self.progress_update.emit("Calculating cell properties...")

            # Analyze each frame
            frame_data = []

            for frame_idx, mask in enumerate(tracked_masks):
                if self._cancelled:
                    return

                self.progress_update.emit(
                    f"Analyzing frame {frame_idx + 1}/{len(tracked_masks)}"
                )

                # Get unique cell IDs
                unique_cells = np.unique(mask[mask > 0])

                for cell_id in unique_cells:
                    cell_mask = mask == cell_id

                    # Calculate properties
                    props = self._calculate_cell_properties(cell_mask, pixel_size)

                    # Add frame and time info
                    props.update(
                        {
                            "frame": frame_idx,
                            "time_minutes": frame_idx * time_per_frame,
                            "cell_id": cell_id,
                        }
                    )

                    frame_data.append(props)

            if self._cancelled:
                return

            self.progress_update.emit("Creating summary statistics...")

            # Create DataFrame
            df = pd.DataFrame(frame_data)

            # Ensure correct data types for frame and cell_id
            if not df.empty:
                df["frame"] = df["frame"].astype(int)
                df["cell_id"] = df["cell_id"].astype(int)

            # Calculate summary statistics
            summary_stats = self._calculate_summary_statistics(df)

            # Calculate movement metrics
            movement_stats = self._calculate_movement_metrics(df)

            results = {
                "frame_data": df,
                "summary_statistics": summary_stats,
                "movement_statistics": movement_stats,
                "parameters": self.parameters.copy(),
            }

            self.progress_update.emit("Analysis completed")
            self.analysis_complete.emit(results)

        except Exception as e:
            self.error_occurred.emit(f"Analysis failed: {str(e)}")

    def _calculate_cell_properties(
        self, cell_mask: np.ndarray, pixel_size: float
    ) -> Dict[str, float]:
        """Calculate properties for a single cell"""
        # Basic properties
        area_pixels = np.sum(cell_mask)
        area_um2 = area_pixels * (pixel_size**2)

        # Find contours for shape analysis
        contours, _ = cv2.findContours(
            cell_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return {
                "area_pixels": area_pixels,
                "area_um2": area_um2,
                "perimeter_pixels": 0,
                "perimeter_um": 0,
                "circularity": 0,
                "aspect_ratio": 0,
                "centroid_x": 0,
                "centroid_y": 0,
            }

        # Use largest contour
        contour = max(contours, key=cv2.contourArea)

        # Perimeter
        perimeter_pixels = cv2.arcLength(contour, True)
        perimeter_um = perimeter_pixels * pixel_size

        # Circularity (4π * area / perimeter²)
        if perimeter_pixels > 0:
            circularity = 4 * np.pi * area_pixels / (perimeter_pixels**2)
        else:
            circularity = 0

        # Bounding box for aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / max(min(w, h), 1)

        # Centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            centroid_x = M["m10"] / M["m00"]
            centroid_y = M["m01"] / M["m00"]
        else:
            centroid_x = x + w // 2
            centroid_y = y + h // 2

        return {
            "area_pixels": area_pixels,
            "area_um2": area_um2,
            "perimeter_pixels": perimeter_pixels,
            "perimeter_um": perimeter_um,
            "circularity": circularity,
            "aspect_ratio": aspect_ratio,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
        }

    def _calculate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics across all frames"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_columns = [
            col for col in numeric_columns if col not in ["frame", "cell_id"]
        ]

        summary = {}
        for col in numeric_columns:
            summary[col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
                "median": df[col].median(),
            }

        # Additional metrics
        summary["total_frames"] = df["frame"].nunique()
        summary["total_cells"] = df["cell_id"].nunique()
        summary["total_detections"] = len(df)

        return summary

    def _calculate_movement_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate cell movement and migration metrics"""
        movement_data = []

        for cell_id in df["cell_id"].unique():
            cell_df = df[df["cell_id"] == cell_id].sort_values("frame")

            if len(cell_df) < 2:
                continue

            # Calculate displacement between consecutive frames
            x_coords = cell_df["centroid_x"].values
            y_coords = cell_df["centroid_y"].values
            times = cell_df["time_minutes"].values

            # Frame-to-frame displacements
            dx = np.diff(x_coords)
            dy = np.diff(y_coords)
            dt = np.diff(times)

            distances = np.sqrt(dx**2 + dy**2)
            velocities = distances / np.maximum(dt, 1e-6)  # Avoid division by zero

            # Total path length
            total_distance = np.sum(distances)

            # Net displacement (start to end)
            net_displacement = np.sqrt(
                (x_coords[-1] - x_coords[0]) ** 2 + (y_coords[-1] - y_coords[0]) ** 2
            )

            # Migration efficiency (net displacement / total distance)
            migration_efficiency = net_displacement / max(total_distance, 1e-6)

            movement_data.append(
                {
                    "cell_id": cell_id,
                    "total_distance": total_distance,
                    "net_displacement": net_displacement,
                    "migration_efficiency": migration_efficiency,
                    "mean_velocity": np.mean(velocities),
                    "max_velocity": np.max(velocities),
                    "frames_tracked": len(cell_df),
                }
            )

        return {
            "individual_cells": movement_data,
            "population_stats": {
                "mean_total_distance": np.mean(
                    [d["total_distance"] for d in movement_data]
                ),
                "mean_net_displacement": np.mean(
                    [d["net_displacement"] for d in movement_data]
                ),
                "mean_migration_efficiency": np.mean(
                    [d["migration_efficiency"] for d in movement_data]
                ),
                "mean_velocity": np.mean([d["mean_velocity"] for d in movement_data]),
            },
        }


class AnalysisPanel(QWidget):
    """Panel for cell analysis and visualization"""

    analysis_completed = pyqtSignal(dict)  # results

    def __init__(self):
        super().__init__()
        self.current_frame: Optional[Dict[str, Any]] = None
        self.tracking_results: Optional[Dict[str, Any]] = None
        self.analysis_results: Optional[Dict[str, Any]] = None
        self.analysis_worker: Optional[AnalysisWorker] = None

        self.setup_ui()
        self.setEnabled(False)  # Disabled until tracking is complete

    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Create tab widget for different analysis views
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Parameters and controls tab
        self.controls_tab = self.create_controls_tab()
        self.tab_widget.addTab(self.controls_tab, "📋 Parameters")

        # Data table tab
        self.data_tab = self.create_data_tab()
        self.tab_widget.addTab(self.data_tab, "📊 Data")

    def create_controls_tab(self) -> QWidget:
        """Create the controls and parameters tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Analysis parameters
        params_group = QGroupBox("Analysis Parameters")
        params_layout = QVBoxLayout(params_group)

        # Time per frame
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Time per frame:"))
        self.time_per_frame_spinbox = QDoubleSpinBox()
        self.time_per_frame_spinbox.setRange(0.1, 60.0)
        self.time_per_frame_spinbox.setValue(3.0)
        self.time_per_frame_spinbox.setSuffix(" minutes")
        time_layout.addWidget(self.time_per_frame_spinbox)
        time_layout.addStretch()
        params_layout.addLayout(time_layout)

        # Pixel size
        pixel_layout = QHBoxLayout()
        pixel_layout.addWidget(QLabel("Pixel size:"))
        self.pixel_size_spinbox = QDoubleSpinBox()
        self.pixel_size_spinbox.setRange(0.01, 10.0)
        self.pixel_size_spinbox.setValue(1.0)
        self.pixel_size_spinbox.setSuffix(" μm/pixel")
        self.pixel_size_spinbox.setDecimals(3)
        pixel_layout.addWidget(self.pixel_size_spinbox)
        pixel_layout.addStretch()
        params_layout.addLayout(pixel_layout)

        layout.addWidget(params_group)

        # Analysis controls
        analysis_group = QGroupBox("Analysis")
        analysis_layout = QVBoxLayout(analysis_group)

        self.analyze_button = QPushButton("Run Analysis")
        self.analyze_button.clicked.connect(self.run_analysis)
        self.analyze_button.setEnabled(False)
        analysis_layout.addWidget(self.analyze_button)

        # Progress indicators
        self.progress_label = QLabel()
        self.progress_label.setVisible(False)
        analysis_layout.addWidget(self.progress_label)

        layout.addWidget(analysis_group)

        # Tracking info
        info_group = QGroupBox("Tracking Information")
        info_layout = QVBoxLayout(info_group)

        self.frames_info_label = QLabel("Frames: 0")
        info_layout.addWidget(self.frames_info_label)

        self.cells_info_label = QLabel("Cells tracked: 0")
        info_layout.addWidget(self.cells_info_label)

        layout.addWidget(info_group)

        layout.addStretch()

        return widget

    def create_data_tab(self) -> QWidget:
        """Create the data table tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Controls
        controls_layout = QHBoxLayout()

        # Cell selector
        self.cell_combo = QComboBox()
        self.cell_combo.addItem("All Cells")
        self.cell_combo.currentTextChanged.connect(self.update_data_filter)
        controls_layout.addWidget(QLabel("Cell:"))
        controls_layout.addWidget(self.cell_combo)

        # Filter selector
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All Data", "By Frame"])
        self.filter_combo.currentTextChanged.connect(self.update_data_filter)
        controls_layout.addWidget(QLabel("Filter:"))
        controls_layout.addWidget(self.filter_combo)

        controls_layout.addStretch()

        self.export_data_button = QPushButton("Export Data")
        self.export_data_button.clicked.connect(self.export_data)
        self.export_data_button.setEnabled(False)
        controls_layout.addWidget(self.export_data_button)

        layout.addLayout(controls_layout)

        # Data table
        self.data_table = QTableWidget()
        self.data_table.setAlternatingRowColors(True)
        self.data_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.data_table)

        return widget

    def set_current_frame(self, frame_data: Dict[str, Any]):
        """Set the current frame"""
        self.current_frame = frame_data

    def set_tracking_results(self, results: Dict[str, Any]):
        """Set tracking results and prepare for analysis"""
        self.tracking_results = results

        # Update info labels
        frame_count = results.get("frame_count", 0)
        self.frames_info_label.setText(f"Frames: {frame_count}")

        # Estimate cell count from first frame mask
        tracked_masks = results.get("tracked_masks", [])
        if tracked_masks:
            unique_cells = len(np.unique(tracked_masks[0][tracked_masks[0] > 0]))
            self.cells_info_label.setText(f"Cells tracked: {unique_cells}")

        # Enable analysis
        self.analyze_button.setEnabled(frame_count > 0)

    def run_analysis(self):
        """Run the cell analysis"""
        if not self.tracking_results:
            return

        # Get parameters
        parameters = {
            "time_per_frame": self.time_per_frame_spinbox.value(),
            "pixel_size": self.pixel_size_spinbox.value(),
        }

        # Start analysis worker
        self.analysis_worker = AnalysisWorker(self.tracking_results, parameters)
        self.analysis_worker.progress_update.connect(self.on_progress_update)
        self.analysis_worker.analysis_complete.connect(self.on_analysis_complete)
        self.analysis_worker.error_occurred.connect(self.on_analysis_error)
        self.analysis_worker.start()

        # Update UI
        self.analyze_button.setEnabled(False)
        self.progress_label.setVisible(True)
        self.progress_label.setText("Starting analysis...")

    def on_progress_update(self, status: str):
        """Handle analysis progress updates"""
        self.progress_label.setText(status)

    def on_analysis_complete(self, results: Dict[str, Any]):
        """Handle analysis completion"""
        self.analysis_results = results

        # Update UI
        self.analyze_button.setEnabled(True)
        self.progress_label.setVisible(False)

        # Populate cell selector
        self.populate_cell_selector()

        # Populate data table
        self.populate_data_table()

        # Enable export buttons
        self.export_data_button.setEnabled(True)

        # Emit signal
        self.analysis_completed.emit(results)

    def on_analysis_error(self, error_message: str):
        """Handle analysis errors"""
        self.analyze_button.setEnabled(True)
        self.progress_label.setVisible(False)
        QMessageBox.critical(self, "Analysis Error", error_message)

    def populate_cell_selector(self):
        """Populate the cell selector with available cell IDs"""
        if not self.analysis_results:
            return

        df = self.analysis_results["frame_data"]
        if "cell_id" not in df.columns:
            print("Warning: No cell_id column in analysis data")
            return

        unique_cells = sorted(df["cell_id"].unique())

        # Store current selection to restore it if possible
        current_selection = self.cell_combo.currentText()

        # Clear and repopulate
        self.cell_combo.clear()
        self.cell_combo.addItem("All Cells")

        for cell_id in unique_cells:
            self.cell_combo.addItem(f"Cell {cell_id}")

        # If no previous selection or it was empty, default to "All Cells"
        if not current_selection or current_selection.strip() == "":
            self.cell_combo.setCurrentIndex(0)  # "All Cells"
        else:
            # Try to restore previous selection
            index = self.cell_combo.findText(current_selection)
            if index >= 0:
                self.cell_combo.setCurrentIndex(index)
            else:
                self.cell_combo.setCurrentIndex(0)  # Default to "All Cells"

    def populate_data_table(self):
        """Populate the data table with analysis results"""
        if not self.analysis_results:
            return

        # Get filtered data based on current selections
        df = self.get_filtered_data()

        # Set up table
        self.data_table.setRowCount(len(df))
        self.data_table.setColumnCount(len(df.columns))
        self.data_table.setHorizontalHeaderLabels(df.columns.tolist())

        # Fill data
        for i, (idx, row) in enumerate(df.iterrows()):
            for j, (col_name, value) in enumerate(zip(df.columns, row)):
                if col_name in ["frame", "cell_id"]:
                    # Display integers without decimal places
                    item = QTableWidgetItem(str(int(value)))
                elif isinstance(value, float):
                    item = QTableWidgetItem(f"{value:.3f}")
                else:
                    item = QTableWidgetItem(str(value))
                self.data_table.setItem(i, j, item)

        # Resize columns
        self.data_table.resizeColumnsToContents()

    def get_filtered_data(self) -> pd.DataFrame:
        """Get filtered data based on current filter selections"""
        if not self.analysis_results or "frame_data" not in self.analysis_results:
            return pd.DataFrame()

        df = self.analysis_results["frame_data"]

        # Ensure we have a valid dataframe
        if df.empty:
            return df

        # Filter by cell
        selected_cell = self.cell_combo.currentText()

        if selected_cell and selected_cell.strip() and selected_cell != "All Cells":
            # Extract cell ID from "Cell X" format
            try:
                cell_parts = selected_cell.strip().split()
                if len(cell_parts) >= 2 and cell_parts[0] == "Cell":
                    cell_id = int(cell_parts[-1])
                    # Check if cell_id exists in the dataframe
                    if "cell_id" in df.columns and cell_id in df["cell_id"].values:
                        df = df[df["cell_id"] == cell_id]
                    else:
                        # Cell ID not found, return empty dataframe
                        print(f"Warning: Cell ID {cell_id} not found in data")
                        return pd.DataFrame()
            except (ValueError, IndexError) as e:
                # If parsing fails, show all data
                print(f"Warning: Could not parse cell selection '{selected_cell}': {e}")

        # Additional filtering by frame if needed
        filter_type = self.filter_combo.currentText()
        # Currently filter_type doesn't do additional filtering beyond cell selection
        # but could be extended for frame-based filtering

        return df

    def update_data_filter(self, filter_type: str = None):
        """Update data table filter"""
        # Repopulate table with current filter settings
        self.populate_data_table()

    def export_data(self):
        """Export analysis data to CSV"""
        if not self.analysis_results:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Analysis Data", "", "CSV Files (*.csv);;All Files (*)"
        )

        if file_path:
            try:
                df = self.get_filtered_data()
                df.to_csv(file_path, index=False)
                QMessageBox.information(
                    self, "Export", f"Data exported to:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Export Error", f"Failed to export data:\n{str(e)}"
                )

    # Public interface
    def get_data(self) -> Dict[str, Any]:
        """Get analysis data for project saving"""
        data = {
            "parameters": {
                "time_per_frame": self.time_per_frame_spinbox.value(),
                "pixel_size": self.pixel_size_spinbox.value(),
            },
            "selected_cell": self.cell_combo.currentText(),
            "filter_type": self.filter_combo.currentText(),
        }

        return data

    def set_data(self, data: Dict[str, Any]):
        """Set analysis data from project loading"""
        if "parameters" in data:
            params = data["parameters"]
            self.time_per_frame_spinbox.setValue(params.get("time_per_frame", 3.0))
            self.pixel_size_spinbox.setValue(params.get("pixel_size", 1.0))

        if "selected_cell" in data:
            selected_cell = data["selected_cell"]
            index = self.cell_combo.findText(selected_cell)
            if index >= 0:
                self.cell_combo.setCurrentIndex(index)

        if "filter_type" in data:
            filter_type = data["filter_type"]
            index = self.filter_combo.findText(filter_type)
            if index >= 0:
                self.filter_combo.setCurrentIndex(index)

    def reset(self):
        """Reset the panel to initial state"""
        self.current_frame = None
        self.tracking_results = None
        self.analysis_results = None

        # Reset UI
        self.analyze_button.setEnabled(False)
        self.export_data_button.setEnabled(False)

        # Clear data
        self.data_table.setRowCount(0)
        self.data_table.setColumnCount(0)

        # Reset parameters
        self.time_per_frame_spinbox.setValue(3.0)
        self.pixel_size_spinbox.setValue(1.0)

        # Reset selectors
        self.cell_combo.clear()
        self.cell_combo.addItem("All Cells")
        self.filter_combo.setCurrentIndex(0)

        # Reset info labels
        self.frames_info_label.setText("Frames: 0")
        self.cells_info_label.setText("Cells tracked: 0")
