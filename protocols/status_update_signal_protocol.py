from typing import Protocol


class StatusUpdateSignalProtocol(Protocol):
    """Protocol for PyQt signal with emit method"""

    def emit(self, message: str) -> None:
        """Emit status update signal"""
        ...
