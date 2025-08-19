from protocols.interactive_frame_widget_protocol import InteractiveFrameWidgetProtocol
from protocols.status_update_signal_protocol import StatusUpdateSignalProtocol
from protocols.storage_protocol import StorageProtocol
from protocols.ui_protocol import UiProtocol

FrameByFrameWidgetProtocol = (
    InteractiveFrameWidgetProtocol
    | StatusUpdateSignalProtocol
    | StorageProtocol
    | UiProtocol
)
