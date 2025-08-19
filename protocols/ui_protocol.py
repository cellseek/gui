from typing import Protocol

from protocols.interactive_frame_widget_protocol import InteractiveFrameWidgetProtocol
from protocols.status_update_signal_protocol import StatusUpdateSignalProtocol


class UiProtocol(Protocol):
    """Protocol defining the UI interface expected by mixins"""

    status_update: StatusUpdateSignalProtocol

    # UI components that mixins interact with
    curr_image_label: InteractiveFrameWidgetProtocol

    def update_display(self) -> None:
        """Update the display"""
        ...
