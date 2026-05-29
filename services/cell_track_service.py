"""Per-cell track summaries and detailed inspection metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from utils.cell_morphology import metrics_for_cell


@dataclass
class CellDetail:
    cell_id: int
    color: Tuple[int, int, int]
    first_frame: int
    last_frame: int
    frame_count: int
    gap_count: int
    present_current: bool
    current_frame: int
    # Position (display / mask resolution)
    x_px: Optional[float]
    y_px: Optional[float]
    # Morphology on current frame
    area_px2: Optional[float]
    perimeter_px: Optional[float]
    circularity: Optional[float]
    ellipse_aspect_ratio: Optional[float]
    ellipse_angle: Optional[float]
    solidity: Optional[float]
    # Motion up to current frame
    displacement_px: Optional[float]
    step_displacement_px: Optional[float]
    total_path_px: float
    average_speed_px_per_frame: Optional[float]

    @property
    def span_frames(self) -> int:
        return self.last_frame - self.first_frame + 1

    def frame_range_label(self) -> str:
        if self.first_frame == self.last_frame:
            return f"Frame {self.first_frame + 1}"
        return f"Frames {self.first_frame + 1}–{self.last_frame + 1}"


def generate_cell_colors(num_colors: int) -> List[Tuple[int, int, int]]:
    colors = []
    for i in range(num_colors):
        hue = (i * 137.508) % 360
        c = 1.0
        x = c * (1 - abs((hue / 60) % 2 - 1))
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
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def _count_gaps(frame_indices: List[int]) -> int:
    if len(frame_indices) <= 1:
        return 0
    gaps = 0
    for i in range(1, len(frame_indices)):
        if frame_indices[i] - frame_indices[i - 1] > 1:
            gaps += 1
    return gaps


def _color_for_id(cell_id: int) -> Tuple[int, int, int]:
    colors = generate_cell_colors(max(cell_id, 1))
    if cell_id <= 0:
        return (200, 200, 200)
    return colors[cell_id - 1] if cell_id <= len(colors) else (200, 200, 200)


class CellTrackService:
    """Analyze a single cell across frames for the inspector panel."""

    def analyze_cell(
        self,
        get_mask_for_frame: Callable[[int], Optional[np.ndarray]],
        frame_count: int,
        current_frame: int,
        cell_id: int,
    ) -> Optional[CellDetail]:
        if frame_count <= 0 or cell_id <= 0:
            return None

        frame_indices: List[int] = []
        centroids: List[Tuple[float, float]] = []

        # Scan all frames so First/Last frame jumps use the full track, not only
        # frames up to the current index.
        for frame_idx in range(frame_count):
            mask = get_mask_for_frame(frame_idx)
            if mask is None or mask.size == 0 or cell_id not in mask:
                continue
            metrics = metrics_for_cell(mask, cell_id)
            if metrics is None:
                continue
            frame_indices.append(frame_idx)
            centroids.append((metrics[6], metrics[7]))

        if not frame_indices:
            return None

        present_current = current_frame in frame_indices
        centroids_up_to_current: List[Tuple[float, float]] = [
            c for fi, c in zip(frame_indices, centroids) if fi <= current_frame
        ]
        current_mask = get_mask_for_frame(current_frame)

        x_px = y_px = None
        area_px2 = perimeter_px = circularity = None
        ellipse_aspect_ratio = ellipse_angle = solidity = None

        if present_current and current_mask is not None:
            metrics = metrics_for_cell(current_mask, cell_id)
            if metrics is not None:
                (
                    area_px2,
                    perimeter_px,
                    circularity,
                    ellipse_aspect_ratio,
                    ellipse_angle,
                    solidity,
                    x_px,
                    y_px,
                ) = metrics

        displacement_px = None
        step_displacement_px = None
        total_path_px = 0.0
        average_speed_px_per_frame = None

        if len(centroids_up_to_current) >= 2:
            start_x, start_y = centroids_up_to_current[0]
            end_x, end_y = centroids_up_to_current[-1]
            displacement_px = math.hypot(end_x - start_x, end_y - start_y)

            prev_x, prev_y = centroids_up_to_current[-2]
            step_displacement_px = math.hypot(end_x - prev_x, end_y - prev_y)

            for i in range(1, len(centroids_up_to_current)):
                dx = centroids_up_to_current[i][0] - centroids_up_to_current[i - 1][0]
                dy = centroids_up_to_current[i][1] - centroids_up_to_current[i - 1][1]
                total_path_px += math.hypot(dx, dy)

            indices_up_to_current = [fi for fi in frame_indices if fi <= current_frame]
            frame_span = indices_up_to_current[-1] - indices_up_to_current[0]
            if frame_span > 0:
                average_speed_px_per_frame = total_path_px / frame_span

        return CellDetail(
            cell_id=cell_id,
            color=_color_for_id(cell_id),
            first_frame=frame_indices[0],
            last_frame=frame_indices[-1],
            frame_count=len(frame_indices),
            gap_count=_count_gaps(frame_indices),
            present_current=present_current,
            current_frame=current_frame,
            x_px=x_px,
            y_px=y_px,
            area_px2=area_px2,
            perimeter_px=perimeter_px,
            circularity=circularity,
            ellipse_aspect_ratio=ellipse_aspect_ratio,
            ellipse_angle=ellipse_angle,
            solidity=solidity,
            displacement_px=displacement_px,
            step_displacement_px=step_displacement_px,
            total_path_px=total_path_px,
            average_speed_px_per_frame=average_speed_px_per_frame,
        )
