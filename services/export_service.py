"""
Export service for cell tracking analysis and data export.

This service provides functionality for:
- Calculating morphological and movement metrics from cell masks
- Exporting tracking data to CSV format
- Generating annotated videos and images
- Statistical analysis of cell populations
"""

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from services.storage_service import StorageService


class ExportService(QObject):
    """Service for exporting cell tracking data and analysis"""

    # Signals for progress updates
    progress_updated = pyqtSignal(int)  # Progress percentage (0-100)
    status_updated = pyqtSignal(str)  # Status message
    export_completed = pyqtSignal(bool, str)  # Success flag, message

    def __init__(self, storage_service: StorageService):
        super().__init__()
        self.storage_service = storage_service

        # Constants
        self.TIME_PER_FRAME = 3  # minutes per frame (configurable)
        self.MIN_CONTOUR_POINTS = 5

    def set_time_per_frame(self, minutes: float):
        """Set time interval between frames"""
        self.TIME_PER_FRAME = minutes

    # ========================================
    # MORPHOLOGICAL ANALYSIS FUNCTIONS
    # ========================================

    def calculate_mask_centroid(self, contour: np.ndarray) -> Tuple[float, float]:
        """Calculate the centroid (center of mass) of a cell mask contour."""
        try:
            # Calculate moments of the contour
            M = cv2.moments(contour)

            # Calculate centroid coordinates
            if M["m00"] != 0:  # Avoid division by zero
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                return float(cx), float(cy)
            else:
                # Fallback to geometric center if moments calculation fails
                x_coords = (
                    contour[:, 0, 0] if len(contour.shape) == 3 else contour[:, 0]
                )
                y_coords = (
                    contour[:, 0, 1] if len(contour.shape) == 3 else contour[:, 1]
                )
                return float(np.mean(x_coords)), float(np.mean(y_coords))
        except Exception:
            # Final fallback: return contour bounds center
            x_coords = contour[:, 0, 0] if len(contour.shape) == 3 else contour[:, 0]
            y_coords = contour[:, 0, 1] if len(contour.shape) == 3 else contour[:, 1]
            return float(np.mean(x_coords)), float(np.mean(y_coords))

    def calculate_contour_metrics(
        self, contour: np.ndarray
    ) -> Tuple[float, float, float, float, float, float]:
        """Calculate comprehensive morphological metrics from contour (in pixels)."""
        if len(contour) <= 2:
            return 0.0, 0.0, 0.0, 1.0, 0.0, 0.0

        # Basic measurements
        area = cv2.contourArea(contour)  # in pixels^2
        perimeter = cv2.arcLength(contour, True)  # in pixels

        # Circularity
        circularity = (4 * math.pi * area) / (perimeter**2) if perimeter > 0 else 0.0

        # Ellipse metrics
        aspect_ratio, ellipse_angle = self.calculate_ellipse_metrics(contour)

        # Solidity calculation
        solidity = self.calculate_solidity(contour, area)

        return area, perimeter, circularity, aspect_ratio, ellipse_angle, solidity

    def calculate_ellipse_metrics(self, contour: np.ndarray) -> Tuple[float, float]:
        """Calculate aspect ratio and angle from fitted ellipse."""
        if len(contour) < self.MIN_CONTOUR_POINTS:
            return 1.0, 0.0

        try:
            ellipse = cv2.fitEllipse(contour)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 1.0
            ellipse_angle = ellipse[2]
            return aspect_ratio, ellipse_angle
        except cv2.error:
            return 1.0, 0.0

    def calculate_solidity(self, contour: np.ndarray, cell_area: float) -> float:
        """Calculate solidity as the ratio of cell area to convex hull area."""
        if len(contour) <= 2 or cell_area <= 0:
            return 0.0

        try:
            # Calculate convex hull
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)

            # Calculate solidity (ratio of cell area to convex hull area)
            if hull_area > 0:
                solidity = cell_area / hull_area
                return min(solidity, 1.0)  # Solidity should not exceed 1.0
            else:
                return 0.0
        except Exception:
            return 0.0

    # ========================================
    # MASK TO CONTOUR CONVERSION
    # ========================================

    def extract_cell_contours(self, masks: np.ndarray) -> Dict[int, np.ndarray]:
        """Extract contours for each cell ID from mask array."""
        cell_contours = {}

        # Get unique cell IDs (excluding background 0)
        unique_ids = np.unique(masks)
        cell_ids = [id_val for id_val in unique_ids if id_val > 0]

        for cell_id in cell_ids:
            # Create binary mask for this cell
            cell_mask = (masks == cell_id).astype(np.uint8)

            # Find contours
            contours, _ = cv2.findContours(
                cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                # Use the largest contour if multiple found
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 0:
                    cell_contours[cell_id] = largest_contour

        return cell_contours

    # ========================================
    # DATA EXTRACTION AND ANALYSIS
    # ========================================

    def extract_frame_data(self, frame_index: int) -> List[Dict[str, Any]]:
        """Extract all cell data for a single frame."""
        masks = self.storage_service.get_mask_for_frame_original_size(frame_index)
        if masks is None:
            return []

        # Extract contours for each cell
        cell_contours = self.extract_cell_contours(masks)

        frame_data = []
        for cell_id, contour in cell_contours.items():
            try:
                # Calculate morphological metrics
                area, perimeter, circularity, aspect_ratio, ellipse_angle, solidity = (
                    self.calculate_contour_metrics(contour)
                )

                # Calculate centroid
                cx, cy = self.calculate_mask_centroid(contour)

                # Time stamp (frame index * time per frame)
                time_minutes = frame_index * self.TIME_PER_FRAME

                cell_data = {
                    "frame_id": frame_index,
                    "cell_id": cell_id,
                    "time_minutes": time_minutes,
                    "x_px": cx,
                    "y_px": cy,
                    "area_px2": area,
                    "perimeter_px": perimeter,
                    "circularity": circularity,
                    "ellipse_aspect_ratio": aspect_ratio,
                    "ellipse_angle": ellipse_angle,
                    "solidity": solidity,
                }

                frame_data.append(cell_data)

            except Exception as e:
                print(
                    f"Warning: Failed to process cell {cell_id} in frame {frame_index}: {e}"
                )
                continue

        return frame_data

    def extract_all_tracking_data(self) -> List[Dict[str, Any]]:
        """Extract tracking data for all frames with masks."""
        all_data = []
        frame_count = self.storage_service.get_frame_count()

        for frame_index in range(frame_count):
            if self.storage_service.has_mask_for_frame(frame_index):
                frame_data = self.extract_frame_data(frame_index)
                all_data.extend(frame_data)

                # Emit progress update
                progress = int(
                    (frame_index + 1) / frame_count * 50
                )  # 50% for data extraction
                self.progress_updated.emit(progress)
                self.status_updated.emit(
                    f"Analyzing frame {frame_index + 1}/{frame_count}"
                )

        return all_data

    # ========================================
    # EXPORT FUNCTIONS
    # ========================================

    def export_to_csv(
        self, output_path: str, frame_range: Optional[Tuple[int, int]] = None
    ) -> bool:
        """Export tracking data to CSV file."""
        try:
            self.status_updated.emit("Extracting tracking data...")

            # Get all tracking data
            all_data = self.extract_all_tracking_data()

            # Filter by frame range if specified
            if frame_range:
                start_frame, end_frame = frame_range
                all_data = [
                    d for d in all_data if start_frame <= d["frame_id"] <= end_frame
                ]

            if not all_data:
                self.export_completed.emit(False, "No tracking data found to export")
                return False

            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Write CSV file
            self.status_updated.emit("Writing CSV file...")
            with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
                fieldnames = [
                    "frame_id",
                    "cell_id",
                    "time_minutes",
                    "x_px",
                    "y_px",
                    "area_px2",
                    "perimeter_px",
                    "circularity",
                    "ellipse_aspect_ratio",
                    "ellipse_angle",
                    "solidity",
                ]

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for i, row in enumerate(all_data):
                    writer.writerow(row)

                    # Update progress
                    progress = 50 + int((i + 1) / len(all_data) * 50)  # 50-100%
                    self.progress_updated.emit(progress)

            self.export_completed.emit(
                True, f"CSV exported successfully to {output_path}"
            )
            return True

        except Exception as e:
            error_msg = f"Failed to export CSV: {str(e)}"
            self.export_completed.emit(False, error_msg)
            return False

    def export_to_json(
        self, output_path: str, frame_range: Optional[Tuple[int, int]] = None
    ) -> bool:
        """Export tracking data to JSON file."""
        try:
            self.status_updated.emit("Extracting tracking data...")

            # Get all tracking data
            all_data = self.extract_all_tracking_data()

            # Filter by frame range if specified
            if frame_range:
                start_frame, end_frame = frame_range
                all_data = [
                    d for d in all_data if start_frame <= d["frame_id"] <= end_frame
                ]

            if not all_data:
                self.export_completed.emit(False, "No tracking data found to export")
                return False

            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Organize data by frame and cell
            organized_data = {
                "metadata": {
                    "total_frames": self.storage_service.get_frame_count(),
                    "time_per_frame_minutes": self.TIME_PER_FRAME,
                    "export_timestamp": str(Path(output_path).stem),
                },
                "tracking_data": all_data,
            }

            # Write JSON file
            self.status_updated.emit("Writing JSON file...")
            with open(output_path, "w", encoding="utf-8") as jsonfile:
                json.dump(organized_data, jsonfile, indent=2)

            self.progress_updated.emit(100)
            self.export_completed.emit(
                True, f"JSON exported successfully to {output_path}"
            )
            return True

        except Exception as e:
            error_msg = f"Failed to export JSON: {str(e)}"
            self.export_completed.emit(False, error_msg)
            return False

    def export_annotated_video(
        self,
        output_path: str,
        fps: int = 5,
        frame_range: Optional[Tuple[int, int]] = None,
    ) -> bool:
        """Export annotated video with cell tracking overlays."""
        try:
            frame_count = self.storage_service.get_frame_count()

            # Determine frame range
            if frame_range:
                start_frame, end_frame = frame_range
            else:
                start_frame, end_frame = 0, frame_count - 1

            # Check if we have frames to export
            frames_to_export = []
            for i in range(start_frame, end_frame + 1):
                if self.storage_service.has_mask_for_frame(i):
                    frames_to_export.append(i)

            if not frames_to_export:
                self.export_completed.emit(
                    False, "No frames with masks found to export"
                )
                return False

            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Initialize video writer
            first_frame = self.storage_service.load_original_frame(frames_to_export[0])
            if first_frame is None:
                self.export_completed.emit(False, "Failed to load first frame")
                return False

            height, width = first_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            self.status_updated.emit("Creating annotated video...")

            for i, frame_index in enumerate(frames_to_export):
                # Load original frame
                frame = self.storage_service.load_original_frame(frame_index)
                if frame is None:
                    continue

                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Get masks and draw annotations
                masks = self.storage_service.get_mask_for_frame_original_size(
                    frame_index
                )
                if masks is not None:
                    annotated_frame = self.draw_cell_annotations(frame_bgr, masks)
                else:
                    annotated_frame = frame_bgr

                # Write frame to video
                video_writer.write(annotated_frame)

                # Update progress
                progress = int((i + 1) / len(frames_to_export) * 100)
                self.progress_updated.emit(progress)
                self.status_updated.emit(f"Processing frame {frame_index + 1}")

            video_writer.release()
            self.export_completed.emit(
                True, f"Video exported successfully to {output_path}"
            )
            return True

        except Exception as e:
            error_msg = f"Failed to export video: {str(e)}"
            self.export_completed.emit(False, error_msg)
            return False

    def draw_cell_annotations(self, frame: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """Draw cell annotations on frame."""
        annotated_frame = frame.copy()

        # Get unique cell IDs
        unique_ids = np.unique(masks)
        cell_ids = [id_val for id_val in unique_ids if id_val > 0]

        # Color map for different cells
        colors = [
            (0, 255, 0),  # Green
            (255, 0, 0),  # Blue
            (0, 0, 255),  # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]

        for i, cell_id in enumerate(cell_ids):
            color = colors[i % len(colors)]

            # Create binary mask for this cell
            cell_mask = (masks == cell_id).astype(np.uint8)

            # Find contours
            contours, _ = cv2.findContours(
                cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                # Draw contour
                cv2.drawContours(annotated_frame, contours, -1, color, 2)

                # Draw cell ID at centroid
                largest_contour = max(contours, key=cv2.contourArea)
                cx, cy = self.calculate_mask_centroid(largest_contour)
                cv2.putText(
                    annotated_frame,
                    str(cell_id),
                    (int(cx), int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

        return annotated_frame

    def export_individual_frames(
        self, output_dir: str, frame_range: Optional[Tuple[int, int]] = None
    ) -> bool:
        """Export individual annotated frames as images."""
        try:
            frame_count = self.storage_service.get_frame_count()

            # Determine frame range
            if frame_range:
                start_frame, end_frame = frame_range
            else:
                start_frame, end_frame = 0, frame_count - 1

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            exported_count = 0
            for frame_index in range(start_frame, end_frame + 1):
                if not self.storage_service.has_mask_for_frame(frame_index):
                    continue

                # Load original frame
                frame = self.storage_service.load_original_frame(frame_index)
                if frame is None:
                    continue

                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Get masks and create annotated frame
                masks = self.storage_service.get_mask_for_frame_original_size(
                    frame_index
                )
                if masks is not None:
                    annotated_frame = self.draw_cell_annotations(frame_bgr, masks)
                else:
                    annotated_frame = frame_bgr

                # Save frame
                frame_filename = f"frame_{frame_index:04d}_annotated.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, annotated_frame)

                exported_count += 1

                # Update progress
                progress = int(
                    (frame_index - start_frame + 1)
                    / (end_frame - start_frame + 1)
                    * 100
                )
                self.progress_updated.emit(progress)
                self.status_updated.emit(f"Exported frame {frame_index + 1}")

            if exported_count > 0:
                self.export_completed.emit(
                    True, f"Exported {exported_count} frames to {output_dir}"
                )
                return True
            else:
                self.export_completed.emit(
                    False, "No frames with masks found to export"
                )
                return False

        except Exception as e:
            error_msg = f"Failed to export frames: {str(e)}"
            self.export_completed.emit(False, error_msg)
            return False

    # ========================================
    # UTILITY FUNCTIONS
    # ========================================

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of the tracking data."""
        try:
            all_data = self.extract_all_tracking_data()

            if not all_data:
                return {}

            # Basic counts
            total_detections = len(all_data)
            unique_cells = len(set(d["cell_id"] for d in all_data))
            frame_range = (
                min(d["frame_id"] for d in all_data),
                max(d["frame_id"] for d in all_data),
            )

            # Morphological statistics
            areas = [d["area_px2"] for d in all_data]
            perimeters = [d["perimeter_px"] for d in all_data]
            circularities = [d["circularity"] for d in all_data]

            summary = {
                "total_detections": total_detections,
                "unique_cells": unique_cells,
                "frame_range": frame_range,
                "time_span_minutes": (frame_range[1] - frame_range[0])
                * self.TIME_PER_FRAME,
                "morphological_stats": {
                    "area_px2": {
                        "mean": np.mean(areas),
                        "std": np.std(areas),
                        "min": np.min(areas),
                        "max": np.max(areas),
                    },
                    "perimeter_px": {
                        "mean": np.mean(perimeters),
                        "std": np.std(perimeters),
                        "min": np.min(perimeters),
                        "max": np.max(perimeters),
                    },
                    "circularity": {
                        "mean": np.mean(circularities),
                        "std": np.std(circularities),
                        "min": np.min(circularities),
                        "max": np.max(circularities),
                    },
                },
            }

            return summary

        except Exception as e:
            print(f"Error calculating summary statistics: {e}")
            return {}
