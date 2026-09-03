"""IrisLoop: USB camera video capture toolkit."""

__version__ = "0.1.0"

from .camera import UsbCamera, probe_cameras
from .capture import capture
from .recorder import VideoRecorder

__all__ = ["UsbCamera", "probe_cameras", "capture", "VideoRecorder"]
