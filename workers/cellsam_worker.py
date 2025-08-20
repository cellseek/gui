import cv2
from cellsam import CellSAM


class CellSamWorker:
    """Worker thread for running CellSAM processing on first frame only"""

    def __init__(self):
        super().__init__()
        self._model = CellSAM(gpu=True)

    def run(self, first_frame_path):
        """Run CellSAM processing on first frame only"""
        try:

            # Load and process first frame
            img = cv2.imread(first_frame_path)
            if img is None:
                raise RuntimeError(f"Failed to load first frame: {first_frame_path}")

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Run segmentation on first frame
            masks, flows, styles = self._model.segment(img, diameter=None)

            # Store first frame results
            result = {
                "frame_path": first_frame_path,
                "masks": masks,
                "flows": flows,
                "styles": styles,
                "original_image": img,
            }

            return result

        except Exception as e:
            raise RuntimeError(f"CellSAM processing failed: {str(e)}")

    def cleanup(self):
        del self._model
