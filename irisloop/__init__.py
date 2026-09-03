"""IrisLoop: camera video capture toolkit (built-in webcam + USB camera)."""

__version__ = "0.1.0"

from .camera import Camera, UsbCamera, probe_cameras
from .capture import capture
from .recorder import VideoRecorder

__all__ = ["Camera", "UsbCamera", "probe_cameras", "capture", "VideoRecorder"]
