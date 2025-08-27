import numpy as np
import torch
from PyQt6.QtCore import QThread
from segment_anything import SamPredictor, sam_model_registry


class SamWorker(QThread):

    def __init__(self):
        """Initialize SAM worker and model."""
        super().__init__()

        # Initialize SAM
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
        sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
        sam.to(device)

        # Set predictor
        self.predictor = SamPredictor(sam)

    def set_image(self, image: np.ndarray):
        """Set the image for SAM processing."""
        self.predictor.set_image(image)

    def predict_point(self, point: tuple) -> tuple:
        """Predict mask from point click.

        Args:
            point: (x, y) coordinates of the point

        Returns:
            tuple: (mask, score) - best mask and its score
        """

        input_point = np.array([point])
        input_label = np.array([1])

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        # Use the best mask (highest score)
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        score = scores[best_idx]

        return mask, score

    def predict_box(self, box: tuple) -> tuple:
        """Predict mask from bounding box.

        Args:
            box: (x1, y1, x2, y2) coordinates of the box

        Returns:
            tuple: (mask, score) - mask and its score
        """

        input_box = np.array([box])  # [x1, y1, x2, y2]

        masks, scores, logits = self.predictor.predict(
            box=input_box,
            multimask_output=False,
        )

        mask = masks[0]
        score = scores[0]

        return mask, score

    def predict_mask(self, mask_prompt: np.ndarray) -> tuple:
        """Predict mask from mask prompt.

        Args:
            mask_prompt: Binary mask array where 1 indicates positive prompt

        Returns:
            tuple: (mask, score) - refined mask and its score
        """
        import cv2

        # SAM expects a low-resolution mask input (256x256)
        # First, get the original image size that was set in the predictor
        original_size = self.predictor.original_size  # (H, W)
        
        # Resize the mask to SAM's input size (256x256)
        # The predictor automatically handles the transformation from low-res to original size
        mask_input = cv2.resize(mask_prompt.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
        
        # Convert to float32 and normalize to 0-1 range
        mask_input = mask_input.astype(np.float32) / 255.0

        masks, scores, logits = self.predictor.predict(
            mask_input=mask_input[None, :, :],  # Add batch dimension
            multimask_output=False,
        )

        mask = masks[0]
        score = scores[0]

        return mask, score
