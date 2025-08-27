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

        # Check if mask has any content
        if not np.any(mask_prompt > 0):
            # Return empty mask if no brush strokes
            return np.zeros_like(mask_prompt, dtype=bool), 0.0

        # SAM's mask input should be a low-resolution logit mask
        # Resize to 256x256 (SAM's expected input size)
        h, w = mask_prompt.shape
        mask_input = cv2.resize(mask_prompt.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
        
        # Convert to float32 logits
        # Positive values (>0) indicate positive prompts, negative values indicate negative prompts
        # We want high positive values for brush areas
        mask_logits = np.where(mask_input > 0, 10.0, -10.0).astype(np.float32)

        print(f"Debug: mask_input range: {mask_input.min()}-{mask_input.max()}")
        print(f"Debug: mask_logits range: {mask_logits.min()}-{mask_logits.max()}")
        print(f"Debug: positive pixels in logits: {np.sum(mask_logits > 0)}")

        try:
            masks, scores, logits = self.predictor.predict(
                mask_input=mask_logits[None, :, :],  # Add batch dimension
                multimask_output=False,  # Start with single mask to see behavior
            )

            mask = masks[0]
            score = scores[0]

            return mask, score
        
        except Exception as e:
            print(f"SAM mask prediction error: {e}")
            # Return the original brush mask as fallback
            return mask_prompt > 0, 0.5
