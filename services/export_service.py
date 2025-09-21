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
    # CELL TRAJECTORY ANALYSIS
    # ========================================

    def calculate_cell_trajectories(self) -> Dict[int, Dict[str, Any]]:
        """Calculate trajectory metrics for each cell across all frames."""
        # Get all tracking data organized by cell
        all_data = self.extract_all_tracking_data()

        # Group data by cell_id
        cell_data_by_id = {}
        for row in all_data:
            cell_id = row["cell_id"]
            if cell_id not in cell_data_by_id:
                cell_data_by_id[cell_id] = []
            cell_data_by_id[cell_id].append(row)

        # Sort each cell's data by frame_id
        for cell_id in cell_data_by_id:
            cell_data_by_id[cell_id].sort(key=lambda x: x["frame_id"])

        # Calculate trajectory metrics for each cell
        cell_trajectories = {}

        for cell_id, cell_frames in cell_data_by_id.items():
            if len(cell_frames) < 2:
                # Can't calculate trajectory metrics with less than 2 points
                cell_trajectories[cell_id] = {
                    "total_distance": 0.0,
                    "displacement": 0.0,
                    "average_velocity": 0.0,
                    "average_speed": 0.0,
                    "max_speed": 0.0,
                    "frame_count": len(cell_frames),
                    "time_span_minutes": 0.0,
                }
                continue

            # Calculate distances between consecutive frames
            distances = []
            speeds = []

            for i in range(1, len(cell_frames)):
                prev_frame = cell_frames[i - 1]
                curr_frame = cell_frames[i]

                # Calculate distance between consecutive positions
                dx = curr_frame["x_px"] - prev_frame["x_px"]
                dy = curr_frame["y_px"] - prev_frame["y_px"]
                distance = math.sqrt(dx**2 + dy**2)
                distances.append(distance)

                # Calculate speed (distance per time)
                time_diff = (
                    curr_frame["frame_id"] - prev_frame["frame_id"]
                ) * self.TIME_PER_FRAME
                if time_diff > 0:
                    speed = distance / time_diff  # pixels per minute
                    speeds.append(speed)
                else:
                    speeds.append(0.0)

            # Total distance traveled
            total_distance = sum(distances)

            # Displacement (straight-line distance from start to end)
            start_pos = cell_frames[0]
            end_pos = cell_frames[-1]
            displacement = math.sqrt(
                (end_pos["x_px"] - start_pos["x_px"]) ** 2
                + (end_pos["y_px"] - start_pos["y_px"]) ** 2
            )

            # Time span
            time_span = (
                cell_frames[-1]["frame_id"] - cell_frames[0]["frame_id"]
            ) * self.TIME_PER_FRAME

            # Velocity (displacement per time)
            average_velocity = displacement / time_span if time_span > 0 else 0.0

            # Speed statistics
            average_speed = np.mean(speeds) if speeds else 0.0
            max_speed = np.max(speeds) if speeds else 0.0

            cell_trajectories[cell_id] = {
                "total_distance": total_distance,
                "displacement": displacement,
                "average_velocity": average_velocity,
                "average_speed": average_speed,
                "max_speed": max_speed,
                "frame_count": len(cell_frames),
                "time_span_minutes": time_span,
            }

        return cell_trajectories

    def calculate_cell_morphological_averages(self) -> Dict[int, Dict[str, float]]:
        """Calculate average morphological metrics for each cell."""
        # Get all tracking data organized by cell
        all_data = self.extract_all_tracking_data()

        # Group data by cell_id
        cell_data_by_id = {}
        for row in all_data:
            cell_id = row["cell_id"]
            if cell_id not in cell_data_by_id:
                cell_data_by_id[cell_id] = []
            cell_data_by_id[cell_id].append(row)

        # Calculate averages for each cell
        cell_averages = {}

        for cell_id, cell_frames in cell_data_by_id.items():
            if not cell_frames:
                continue

            # Extract morphological metrics
            areas = [frame["area_px2"] for frame in cell_frames]
            perimeters = [frame["perimeter_px"] for frame in cell_frames]
            circularities = [frame["circularity"] for frame in cell_frames]
            aspect_ratios = [frame["ellipse_aspect_ratio"] for frame in cell_frames]
            solidities = [frame["solidity"] for frame in cell_frames]

            cell_averages[cell_id] = {
                "average_area": np.mean(areas),
                "average_perimeter": np.mean(perimeters),
                "average_circularity": np.mean(circularities),
                "average_ellipse_aspect_ratio": np.mean(aspect_ratios),
                "average_solidity": np.mean(solidities),
                "std_area": np.std(areas),
                "std_perimeter": np.std(perimeters),
                "std_circularity": np.std(circularities),
                "std_ellipse_aspect_ratio": np.std(aspect_ratios),
                "std_solidity": np.std(solidities),
            }

        return cell_averages

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

    def export_cell_summary_to_csv(
        self, output_path: str, frame_range: Optional[Tuple[int, int]] = None
    ) -> bool:
        """Export per-cell summary statistics to CSV file."""
        try:
            self.status_updated.emit("Calculating cell trajectories...")

            # Get trajectory data
            cell_trajectories = self.calculate_cell_trajectories()

            self.status_updated.emit("Calculating morphological averages...")
            self.progress_updated.emit(25)

            # Get morphological averages
            cell_morphological_averages = self.calculate_cell_morphological_averages()

            if not cell_trajectories and not cell_morphological_averages:
                self.export_completed.emit(False, "No cell data found to export")
                return False

            # Combine trajectory and morphological data
            all_cell_ids = set(cell_trajectories.keys()) | set(
                cell_morphological_averages.keys()
            )

            self.status_updated.emit("Preparing summary data...")
            self.progress_updated.emit(50)

            summary_data = []
            for cell_id in sorted(all_cell_ids):
                # Get trajectory data (with defaults if missing)
                trajectory = cell_trajectories.get(cell_id, {})
                morphology = cell_morphological_averages.get(cell_id, {})

                row_data = {
                    "cell_id": cell_id,
                    "total_distance_px": trajectory.get("total_distance", 0.0),
                    "displacement_px": trajectory.get("displacement", 0.0),
                    "average_velocity_px_per_min": trajectory.get(
                        "average_velocity", 0.0
                    ),
                    "average_speed_px_per_min": trajectory.get("average_speed", 0.0),
                    "max_speed_px_per_min": trajectory.get("max_speed", 0.0),
                    "frame_count": trajectory.get("frame_count", 0),
                    "time_span_minutes": trajectory.get("time_span_minutes", 0.0),
                    "average_area_px2": morphology.get("average_area", 0.0),
                    "average_perimeter_px": morphology.get("average_perimeter", 0.0),
                    "average_circularity": morphology.get("average_circularity", 0.0),
                    "average_ellipse_aspect_ratio": morphology.get(
                        "average_ellipse_aspect_ratio", 0.0
                    ),
                    "average_solidity": morphology.get("average_solidity", 0.0),
                    "std_area_px2": morphology.get("std_area", 0.0),
                    "std_perimeter_px": morphology.get("std_perimeter", 0.0),
                    "std_circularity": morphology.get("std_circularity", 0.0),
                    "std_ellipse_aspect_ratio": morphology.get(
                        "std_ellipse_aspect_ratio", 0.0
                    ),
                    "std_solidity": morphology.get("std_solidity", 0.0),
                }

                summary_data.append(row_data)

            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Write CSV file
            self.status_updated.emit("Writing cell summary CSV file...")
            self.progress_updated.emit(75)

            with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
                fieldnames = [
                    "cell_id",
                    "total_distance_px",
                    "displacement_px",
                    "average_velocity_px_per_min",
                    "average_speed_px_per_min",
                    "max_speed_px_per_min",
                    "frame_count",
                    "time_span_minutes",
                    "average_area_px2",
                    "average_perimeter_px",
                    "average_circularity",
                    "average_ellipse_aspect_ratio",
                    "average_solidity",
                    "std_area_px2",
                    "std_perimeter_px",
                    "std_circularity",
                    "std_ellipse_aspect_ratio",
                    "std_solidity",
                ]

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for row in summary_data:
                    writer.writerow(row)

            self.progress_updated.emit(100)
            self.export_completed.emit(
                True, f"Cell summary exported successfully to {output_path}"
            )
            return True

        except Exception as e:
            error_msg = f"Failed to export cell summary: {str(e)}"
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
        """Draw cell annotations on frame using the same style as interactive frame widget."""
        annotated_frame = frame.copy()

        # Get unique cell IDs
        unique_ids = np.unique(masks)
        cell_ids = [id_val for id_val in unique_ids if id_val > 0]

        if not cell_ids:
            return annotated_frame

        # Generate colors using the same method as interactive frame widget
        colors = self._generate_colors(len(cell_ids))

        # Create overlay with transparency (30% opacity like default in frame widget)
        overlay = annotated_frame.copy()
        alpha = 0.3  # Same default transparency as frame widget

        for i, cell_id in enumerate(cell_ids):
            mask = masks == cell_id
            if np.any(mask):
                color = np.array(colors[i], dtype=np.uint8)
                # Apply the mask with transparency
                overlay[mask] = ((1 - alpha) * overlay[mask] + alpha * color).astype(
                    np.uint8
                )

        # Add cell ID text
        overlay = self._add_cell_id_text(overlay, masks, cell_ids)

        return overlay

    def _generate_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for masks using same method as interactive frame widget"""
        colors = []
        for i in range(num_colors):
            hue = (i * 137.508) % 360  # Golden angle approximation
            # Convert HSV to RGB (simplified)
            c = 1.0
            x = c * (1 - abs((hue / 60) % 2 - 1))
            m = 0

            if 0 <= hue < 60:
                r, g, b = c, x, 0
            elif 60 <= hue < 120:
                r, g, b = x, c, 0
            elif 120 <= hue < 180:
                r, g, b = 0, c, x
            elif 180 <= hue < 240:
                r, g, b = 0, x, c
            elif 240 <= hue < 300:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x

            colors.append((int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)))

        return colors

    def _add_cell_id_text(
        self, overlay: np.ndarray, masks: np.ndarray, cell_ids: List[int]
    ) -> np.ndarray:
        """Add cell ID text to the overlay using same style as interactive frame widget"""
        for cell_id in cell_ids:
            mask = masks == cell_id
            if np.any(mask):
                # Find centroid of the mask
                y_coords, x_coords = np.where(mask)
                if len(y_coords) > 0:
                    center_y = int(np.mean(y_coords))
                    center_x = int(np.mean(x_coords))

                    # White text with black outline for better visibility
                    text_color = (255, 255, 255)  # White text
                    outline_color = (0, 0, 0)  # Black outline

                    # Add black outline
                    cv2.putText(
                        overlay,
                        str(cell_id),
                        (center_x - 10, center_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        outline_color,
                        3,  # Thicker for outline
                        cv2.LINE_AA,
                    )

                    # Add white text on top
                    cv2.putText(
                        overlay,
                        str(cell_id),
                        (center_x - 10, center_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        text_color,
                        2,
                        cv2.LINE_AA,
                    )

        return overlay

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
            aspect_ratios = [d["ellipse_aspect_ratio"] for d in all_data]
            solidities = [d["solidity"] for d in all_data]

            # Calculate trajectory statistics
            cell_trajectories = self.calculate_cell_trajectories()
            if cell_trajectories:
                total_distances = [
                    traj["total_distance"] for traj in cell_trajectories.values()
                ]
                displacements = [
                    traj["displacement"] for traj in cell_trajectories.values()
                ]
                avg_velocities = [
                    traj["average_velocity"] for traj in cell_trajectories.values()
                ]
                avg_speeds = [
                    traj["average_speed"] for traj in cell_trajectories.values()
                ]
            else:
                total_distances = displacements = avg_velocities = avg_speeds = []

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
                    "ellipse_aspect_ratio": {
                        "mean": np.mean(aspect_ratios),
                        "std": np.std(aspect_ratios),
                        "min": np.min(aspect_ratios),
                        "max": np.max(aspect_ratios),
                    },
                    "solidity": {
                        "mean": np.mean(solidities),
                        "std": np.std(solidities),
                        "min": np.min(solidities),
                        "max": np.max(solidities),
                    },
                },
                "trajectory_stats": {
                    "total_distance_px": {
                        "mean": np.mean(total_distances) if total_distances else 0.0,
                        "std": np.std(total_distances) if total_distances else 0.0,
                        "min": np.min(total_distances) if total_distances else 0.0,
                        "max": np.max(total_distances) if total_distances else 0.0,
                    },
                    "displacement_px": {
                        "mean": np.mean(displacements) if displacements else 0.0,
                        "std": np.std(displacements) if displacements else 0.0,
                        "min": np.min(displacements) if displacements else 0.0,
                        "max": np.max(displacements) if displacements else 0.0,
                    },
                    "average_velocity_px_per_min": {
                        "mean": np.mean(avg_velocities) if avg_velocities else 0.0,
                        "std": np.std(avg_velocities) if avg_velocities else 0.0,
                        "min": np.min(avg_velocities) if avg_velocities else 0.0,
                        "max": np.max(avg_velocities) if avg_velocities else 0.0,
                    },
                    "average_speed_px_per_min": {
                        "mean": np.mean(avg_speeds) if avg_speeds else 0.0,
                        "std": np.std(avg_speeds) if avg_speeds else 0.0,
                        "min": np.min(avg_speeds) if avg_speeds else 0.0,
                        "max": np.max(avg_speeds) if avg_speeds else 0.0,
                    },
                },
            }

            return summary

        except Exception as e:
            print(f"Error calculating summary statistics: {e}")
            return {}

    def get_detailed_cell_summary(self) -> Dict[str, Any]:
        """Get detailed per-cell summary statistics for display in summary tab."""
        try:
            # Get trajectory data
            cell_trajectories = self.calculate_cell_trajectories()

            # Get morphological averages
            cell_morphological_averages = self.calculate_cell_morphological_averages()

            if not cell_trajectories and not cell_morphological_averages:
                return {}

            # Combine data
            all_cell_ids = set(cell_trajectories.keys()) | set(
                cell_morphological_averages.keys()
            )

            detailed_summary = {"cell_count": len(all_cell_ids), "cells": {}}

            for cell_id in sorted(all_cell_ids):
                trajectory = cell_trajectories.get(cell_id, {})
                morphology = cell_morphological_averages.get(cell_id, {})

                detailed_summary["cells"][cell_id] = {
                    "trajectory": {
                        "total_distance_px": trajectory.get("total_distance", 0.0),
                        "displacement_px": trajectory.get("displacement", 0.0),
                        "average_velocity_px_per_min": trajectory.get(
                            "average_velocity", 0.0
                        ),
                        "average_speed_px_per_min": trajectory.get(
                            "average_speed", 0.0
                        ),
                        "max_speed_px_per_min": trajectory.get("max_speed", 0.0),
                        "frame_count": trajectory.get("frame_count", 0),
                        "time_span_minutes": trajectory.get("time_span_minutes", 0.0),
                    },
                    "morphology": {
                        "average_area_px2": morphology.get("average_area", 0.0),
                        "average_perimeter_px": morphology.get(
                            "average_perimeter", 0.0
                        ),
                        "average_circularity": morphology.get(
                            "average_circularity", 0.0
                        ),
                        "average_ellipse_aspect_ratio": morphology.get(
                            "average_ellipse_aspect_ratio", 0.0
                        ),
                        "average_solidity": morphology.get("average_solidity", 0.0),
                        "std_area_px2": morphology.get("std_area", 0.0),
                        "std_perimeter_px": morphology.get("std_perimeter", 0.0),
                        "std_circularity": morphology.get("std_circularity", 0.0),
                        "std_ellipse_aspect_ratio": morphology.get(
                            "std_ellipse_aspect_ratio", 0.0
                        ),
                        "std_solidity": morphology.get("std_solidity", 0.0),
                    },
                }

            return detailed_summary

        except Exception as e:
            print(f"Error calculating detailed cell summary: {e}")
            return {}
