"""
Annotation functionality mixin for frame-by-frame widget
"""

from typing import Tuple

from PyQt6.QtWidgets import QInputDialog, QMessageBox

from widgets.interactive_frame_widget import AnnotationMode


class AnnotationMixin:
    """Mixin for annotation functionality

    Requires the implementing class to provide StorageProtocol and UIProtocol interfaces.
    """

    def set_annotation_mode(self, mode: AnnotationMode) -> None:
        """Set the annotation mode"""
        self.curr_image_label.set_annotation_mode(mode)

    def on_mask_clicked(self, point: Tuple[int, int]) -> None:
        """Handle mask removal"""
        current_masks = self.get_current_frame_masks()
        if current_masks is None:
            return

        x, y = point
        if 0 <= y < current_masks.shape[0] and 0 <= x < current_masks.shape[1]:
            mask_id = current_masks[y, x]
            if mask_id > 0:
                # Remove this mask
                current_masks[current_masks == mask_id] = 0
                current_index = self.get_current_frame_index()
                self.set_mask_for_frame(current_index, current_masks)
                self.curr_image_label.set_masks(current_masks)

                self.status_update.emit(f"Removed mask {mask_id}")

    def on_cell_id_edit_requested(
        self, point: Tuple[int, int], current_cell_id: int
    ) -> None:
        """Handle cell ID editing request"""
        x, y = point

        if current_cell_id == 0:
            QMessageBox.information(
                self, "Edit Cell ID", "No cell at this location to edit."
            )
            return

        # Get new cell ID from user
        new_id, ok = QInputDialog.getInt(
            self,
            "Edit Cell ID",
            f"Enter new ID for cell {current_cell_id}:",
            value=current_cell_id,
            min=1,
            max=9999,
        )

        if ok and new_id != current_cell_id:
            current_masks = self.get_current_frame_masks()
            if current_masks is not None:
                # Check if new ID already exists
                if new_id in current_masks:
                    reply = QMessageBox.question(
                        self,
                        "ID Conflict",
                        f"Cell ID {new_id} already exists. Do you want to merge the cells?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    )
                    if reply == QMessageBox.StandardButton.No:
                        return

                # Update the cell ID
                current_masks[current_masks == current_cell_id] = new_id
                current_index = self.get_current_frame_index()
                self.set_mask_for_frame(current_index, current_masks)
                self.curr_image_label.set_masks(current_masks)

                self.status_update.emit(
                    f"Changed cell ID from {current_cell_id} to {new_id}"
                )
