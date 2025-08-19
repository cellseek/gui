from typing import Optional, Protocol

import numpy as np


class InteractiveFrameWidgetProtocol(Protocol):
    """Protocol for InteractiveFrameWidget used by mixins"""

    def set_masks(self, masks: Optional[np.ndarray]) -> None:
        """Set masks on the interactive frame widget"""
        ...

    def set_annotation_mode(self, mode: object) -> None:
        """Set annotation mode"""
        ...
