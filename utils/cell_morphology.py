"""Contour-based morphology metrics for a single cell in a mask."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

MIN_CONTOUR_POINTS = 5


def largest_contour_for_cell(masks: np.ndarray, cell_id: int) -> Optional[np.ndarray]:
    cell_mask = (masks == cell_id).astype(np.uint8)
    contours, _ = cv2.findContours(
        cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) <= 0:
        return None
    return largest


def contour_centroid(contour: np.ndarray) -> Tuple[float, float]:
    try:
        m = cv2.moments(contour)
        if m["m00"] != 0:
            return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])
    except Exception:
        pass
    xs = contour[:, 0, 0] if len(contour.shape) == 3 else contour[:, 0]
    ys = contour[:, 0, 1] if len(contour.shape) == 3 else contour[:, 1]
    return float(np.mean(xs)), float(np.mean(ys))


def contour_metrics(contour: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """Return area, perimeter, circularity, aspect_ratio, ellipse_angle, solidity."""
    if len(contour) <= 2:
        return 0.0, 0.0, 0.0, 1.0, 0.0, 0.0

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * math.pi * area) / (perimeter**2) if perimeter > 0 else 0.0

    aspect_ratio, ellipse_angle = _ellipse_metrics(contour)
    solidity = _solidity(contour, area)
    return area, perimeter, circularity, aspect_ratio, ellipse_angle, solidity


def _ellipse_metrics(contour: np.ndarray) -> Tuple[float, float]:
    if len(contour) < MIN_CONTOUR_POINTS:
        return 1.0, 0.0
    try:
        ellipse = cv2.fitEllipse(contour)
        major_axis = max(ellipse[1])
        minor_axis = min(ellipse[1])
        aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 1.0
        return aspect_ratio, ellipse[2]
    except cv2.error:
        return 1.0, 0.0


def _solidity(contour: np.ndarray, cell_area: float) -> float:
    if len(contour) <= 2 or cell_area <= 0:
        return 0.0
    try:
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            return min(cell_area / hull_area, 1.0)
    except Exception:
        pass
    return 0.0


def metrics_for_cell(
    masks: np.ndarray, cell_id: int
) -> Optional[Tuple[float, float, float, float, float, float, float, float]]:
    """Morphology + centroid for one cell, or None if not present."""
    contour = largest_contour_for_cell(masks, cell_id)
    if contour is None:
        return None
    area, perimeter, circularity, aspect_ratio, ellipse_angle, solidity = (
        contour_metrics(contour)
    )
    cx, cy = contour_centroid(contour)
    return area, perimeter, circularity, aspect_ratio, ellipse_angle, solidity, cx, cy
