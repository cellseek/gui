"""
Annotation service for handling annotation functionality
"""

from typing import Protocol, Tuple

import numpy as np
from PyQt6.QtWidgets import QInputDialog

from widgets.interactive_frame_widget import AnnotationMode


class AnnotationServiceDelegate(Protocol):
    """Protocol for objects that can delegate annotation operations"""

    def get_current_frame_masks(self) -> np.ndarray | None: ...
    def set_mask_for_frame(self, frame_index: int, masks: np.ndarray) -> None: ...
    def get_current_frame_index(self) -> int: ...
    def emit_status_update(self, message: str) -> None: ...
    def show_message_box(
        self, title: str, message: str, box_type: str = "information"
    ) -> None: ...
    def show_question_box(self, title: str, message: str) -> bool: ...
    def update_current_display_masks(self, masks: np.ndarray) -> None: ...


class AnnotationService:
    """Service for annotation functionality"""

    def __init__(self, delegate: AnnotationServiceDelegate):
        self.delegate = delegate

    def set_annotation_mode(self, image_widget, mode: AnnotationMode) -> None:
        """Set the annotation mode"""
        image_widget.set_annotation_mode(mode)

    def on_mask_clicked(self, point: Tuple[int, int]) -> None:
        """Handle mask removal"""
        current_masks = self.delegate.get_current_frame_masks()
        if current_masks is None:
            return

        x, y = point
        if 0 <= y < current_masks.shape[0] and 0 <= x < current_masks.shape[1]:
            mask_id = current_masks[y, x]
            if mask_id > 0:
                # Remove this mask
                current_masks[current_masks == mask_id] = 0
                current_index = self.delegate.get_current_frame_index()
                self.delegate.set_mask_for_frame(current_index, current_masks)
                self.delegate.update_current_display_masks(current_masks)

                self.delegate.emit_status_update(f"Removed mask {mask_id}")

    def on_cell_id_edit_requested(
        self, point: Tuple[int, int], current_cell_id: int
    ) -> None:
        """Handle cell ID editing request"""
        x, y = point

        if current_cell_id == 0:
            self.delegate.show_message_box(
                "Edit Cell ID", "No cell at this location to edit."
            )
            return

        # Get new cell ID from user
        new_id, ok = QInputDialog.getInt(
            None,  # We'll pass parent widget through delegate if needed
            "Edit Cell ID",
            f"Enter new ID for cell {current_cell_id}:",
            value=current_cell_id,
            min=1,
            max=9999,
        )

        if ok and new_id != current_cell_id:
            current_masks = self.delegate.get_current_frame_masks()
            if current_masks is not None:
                # Check if new ID already exists
                if new_id in current_masks:
                    merge_cells = self.delegate.show_question_box(
                        "ID Conflict",
                        f"Cell ID {new_id} already exists. Do you want to merge the cells?",
                    )
                    if not merge_cells:
                        return

                # Update the cell ID
                current_masks[current_masks == current_cell_id] = new_id
                current_index = self.delegate.get_current_frame_index()
                self.delegate.set_mask_for_frame(current_index, current_masks)
                self.delegate.update_current_display_masks(current_masks)

                self.delegate.emit_status_update(
                    f"Changed cell ID from {current_cell_id} to {new_id}"
                )
