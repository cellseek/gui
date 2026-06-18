"""Shared media extension lists and file-dialog helpers for import."""

from typing import FrozenSet

IMAGE_EXTENSIONS: FrozenSet[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif"}
)

VIDEO_EXTENSIONS: FrozenSet[str] = frozenset(
    {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
)

CZI_EXTENSIONS: FrozenSet[str] = frozenset({".czi"})

# Short list for UI copy (drop zone, README)
SUPPORTED_FORMATS_LABEL = (
    "Images: PNG, JPG, TIFF, BMP, GIF — "
    "Video: MP4, AVI, MOV, MKV, WMV, WEBM — "
    "Microscopy: CZI"
)


def _glob_list(extensions: FrozenSet[str]) -> str:
    return " ".join(f"*{ext}" for ext in sorted(extensions))


def media_files_dialog_filter() -> str:
    """QFileDialog filter covering all supported image and video types."""
    globs = (
        f"{_glob_list(IMAGE_EXTENSIONS)} "
        f"{_glob_list(VIDEO_EXTENSIONS)} "
        f"{_glob_list(CZI_EXTENSIONS)}"
    )
    return f"Media Files ({globs});;All Files (*)"


def is_image_path(path: str) -> bool:
    from pathlib import Path

    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def is_video_path(path: str) -> bool:
    from pathlib import Path

    return Path(path).suffix.lower() in VIDEO_EXTENSIONS
