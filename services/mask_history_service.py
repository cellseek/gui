"""Undo/redo stacks for mask edits on the current and subsequent frames."""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

# (frame_index, mask copy or None)
MaskSnapshot = List[Tuple[int, Optional[np.ndarray]]]


class MaskHistoryService:
    """Stores mask snapshots for undo/redo (newest at end of undo stack)."""

    def __init__(self, max_steps: int = 50) -> None:
        self._max_steps = max_steps
        self._undo: List[MaskSnapshot] = []
        self._redo: List[MaskSnapshot] = []

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()

    def can_undo(self) -> bool:
        return len(self._undo) > 0

    def can_redo(self) -> bool:
        return len(self._redo) > 0

    def push(self, snapshot: MaskSnapshot) -> None:
        if not snapshot:
            return
        self._undo.append(snapshot)
        self._redo.clear()
        if len(self._undo) > self._max_steps:
            self._undo.pop(0)

    def undo(self, capture_current: Callable[[], MaskSnapshot]) -> Optional[MaskSnapshot]:
        if not self._undo:
            return None
        self._redo.append(capture_current())
        return self._undo.pop()

    def redo(self, capture_current: Callable[[], MaskSnapshot]) -> Optional[MaskSnapshot]:
        if not self._redo:
            return None
        self._undo.append(capture_current())
        return self._redo.pop()
