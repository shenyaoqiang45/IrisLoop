"""IrisLoop: USB 摄像头视频流采集工具。"""

__version__ = "0.1.0"

from .camera import UsbCamera, probe_cameras
from .capture import capture
from .recorder import VideoRecorder

__all__ = ["UsbCamera", "probe_cameras", "capture", "VideoRecorder"]
