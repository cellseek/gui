"""Read Zeiss CZI microscopy files and export frames for the GUI pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

try:
    import aicspylibczi
except ImportError:  # pragma: no cover - optional until requirements installed
    aicspylibczi = None

# Brightfield / phase-contrast channel for multi-channel CZI acquisitions.
BRIGHTFIELD_CHANNEL = 0


def is_czi_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() == ".czi"


def _require_aicspylibczi():
    if aicspylibczi is None:
        raise ImportError(
            "CZI support requires aicspylibczi. Install with: pip install aicspylibczi"
        )


def _dim_extent(dims_shape: dict, key: str, default_end: int = 1) -> int:
    dim = dims_shape.get(key, (0, default_end))
    if isinstance(dim, tuple):
        return dim[1] - dim[0]
    return int(dim)


def get_czi_metadata(path: str | Path) -> Tuple[int, int, Tuple[int, int]]:
    """Return (num_timepoints, num_channels, (height, width))."""
    _require_aicspylibczi()
    czi = aicspylibczi.CziFile(str(path))
    dims_shape = czi.get_dims_shape()[0]
    num_t = _dim_extent(dims_shape, "T")
    num_c = _dim_extent(dims_shape, "C")
    height = dims_shape.get("Y", (0, 0))[1]
    width = dims_shape.get("X", (0, 0))[1]
    return num_t, num_c, (height, width)


def _normalize_to_uint8(plane: np.ndarray) -> np.ndarray:
    """Map a 2D plane to uint8 using percentile scaling."""
    plane = np.squeeze(plane)
    if plane.dtype == np.uint8:
        return plane
    f_img = plane.astype(np.float32)
    lo, hi = np.percentile(f_img, (0.5, 99.5))
    if hi <= lo:
        return np.zeros(plane.shape, dtype=np.uint8)
    scaled = np.clip((f_img - lo) / (hi - lo) * 255.0, 0.0, 255.0)
    return scaled.astype(np.uint8)


def _read_plane(czi: "aicspylibczi.CziFile", t: int, c: int) -> np.ndarray:
    plane, _ = czi.read_image(C=c, T=t)
    return np.squeeze(plane)


def czi_frame_to_rgb(
    czi: "aicspylibczi.CziFile",
    t: int,
    channel: int = BRIGHTFIELD_CHANNEL,
) -> np.ndarray:
    """Convert one CZI timepoint to RGB uint8 using a single channel (default: brightfield)."""
    gray = _normalize_to_uint8(_read_plane(czi, t, channel))
    return np.stack([gray, gray, gray], axis=-1)


def extract_czi_frames(
    czi_path: str | Path,
    output_dir: str | Path | None = None,
    frame_interval: int = 1,
    channel: int = BRIGHTFIELD_CHANNEL,
) -> List[str]:
    """
    Export CZI timepoints as PNG frames from the brightfield channel (C=0).

    Returns sorted list of exported frame paths.
    """
    _require_aicspylibczi()
    czi_path = Path(czi_path)
    if output_dir is None:
        output_dir = Path("temp_frames") / f"czi_{czi_path.stem}"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    czi = aicspylibczi.CziFile(str(czi_path))
    num_t, num_c, _ = get_czi_metadata(czi_path)
    if channel >= num_c:
        raise ValueError(
            f"CZI has {num_c} channel(s); requested channel index {channel} is out of range"
        )

    frame_paths: List[str] = []
    for t in range(0, num_t, max(1, frame_interval)):
        rgb = czi_frame_to_rgb(czi, t, channel=channel)
        frame_path = output_dir / f"frame_{t:04d}.png"
        cv2.imwrite(str(frame_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        frame_paths.append(str(frame_path))

    return frame_paths
