# src/dms/modules/camera.py
"""
Camera capture module.
Supports IMX219 CSI (GStreamer / nvarguscamerasrc) and USB cameras.
"""

import cv2
import time
from dms.config import (
    CAMERA_SOURCE, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, CAMERA_FLIP
)


def _csi_pipeline(width: int, height: int, fps: int, flip: int) -> str:
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"framerate={fps}/1, format=NV12 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink"
    )


class Camera:
    """
    Thread-safe camera wrapper.

    Usage:
        cam = Camera()
        cam.open()
        frame = cam.read()
        cam.release()
    """

    def __init__(
        self,
        source: str  = CAMERA_SOURCE,
        width:  int  = CAMERA_WIDTH,
        height: int  = CAMERA_HEIGHT,
        fps:    int  = CAMERA_FPS,
        flip:   int  = CAMERA_FLIP,
    ):
        self.source = source
        self.width  = width
        self.height = height
        self.fps    = fps
        self.flip   = flip
        self._cap   = None

    # -- public API ------------------------------------------------------------

    def open(self) -> None:
        """Open the camera and verify frames are flowing."""
        if self.source == "csi":
            pipeline   = _csi_pipeline(self.width, self.height, self.fps, self.flip)
            self._cap  = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            # Accept both integer index (0, 1) and device path ("/dev/video0")
            src = self.source if isinstance(self.source, str) and self.source.startswith("/") \
                  else int(self.source)
            self._cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS,          self.fps)

        if not self._cap.isOpened():
            raise RuntimeError(f"[Camera] Failed to open (source={self.source})")

        # Warm-up - drain a few frames
        ok = False
        for _ in range(10):
            ret, frame = self._cap.read()
            if ret and frame is not None:
                ok = True
                break
            time.sleep(0.05)

        if not ok:
            self._cap.release()
            raise RuntimeError(
                f"[Camera] Opened but no frames received (source={self.source})"
            )

        print(f"[Camera] OK  source={self.source}  "
              f"{self.width}x{self.height} @ {self.fps}fps")

    def read(self):
        """Return the latest frame or None on failure."""
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.release()