"""Background probe of video metadata (no frame extraction)."""

from PyQt6.QtCore import QThread, pyqtSignal

from utils.video_frame_source import VideoFrameSource


class VideoImportWorker(QThread):
    finished = pyqtSignal(str, int, int)  # path, frame_interval, logical_frame_count
    error_occurred = pyqtSignal(str)

    def __init__(self, video_path: str, frame_interval: int = 30):
        super().__init__()
        self._video_path = video_path
        self._frame_interval = frame_interval

    def run(self):
        try:
            source = VideoFrameSource(self._video_path, self._frame_interval)
            count = source.logical_frame_count
            source.close()
            if count <= 0:
                self.error_occurred.emit("Video contains no readable frames")
                return
            self.finished.emit(self._video_path, self._frame_interval, count)
        except Exception as e:
            self.error_occurred.emit(str(e))
